"""
Binance Spot adapter.

Why Binance is the reference implementation: it has the deepest triangular
graph in crypto (~1400 spot pairs over USDT, USDC, FDUSD, BTC, ETH, BNB and TRY
quotes), a $5 minimum notional that a $100 account can actually clear on three
consecutive legs, and 7.5 bps taker fees when settling in BNB. No other venue
combines all three.

Book synchronisation follows Binance's documented procedure exactly, and the
procedure matters -- doing it approximately produces a book that is subtly wrong
for the first few seconds after every reconnect:

    1. Open the diff stream and buffer events.
    2. Fetch a REST depth snapshot; note its ``lastUpdateId``.
    3. Discard buffered events whose ``u`` is <= ``lastUpdateId``.
    4. The first event applied must satisfy ``U <= lastUpdateId+1 <= u``.
       If no buffered event satisfies it, the snapshot is too old: go to 2.
    5. From then on each event's ``U`` must equal the previous event's ``u``+1.
       Any gap means restart from step 1.

Step 4 is the one everyone skips.
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Mapping, Sequence

from triangulum.core.decimal_math import D, ZERO
from triangulum.core.errors import (
    BadResponse,
    InsufficientBalance,
    MinNotionalError,
    OrderRejected,
    RateLimited,
)
from triangulum.core.types import (
    Asset,
    Balance,
    Fill,
    Liquidity,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Symbol,
    TimeInForce,
)
from triangulum.exchanges.base import AdapterCapabilities, ExchangeAdapter
from triangulum.exchanges.http import HttpClient, WebSocketClient, sign_hmac
from triangulum.exchanges.ratelimit import WeightedLimiter
from triangulum.exchanges.spec import SymbolSpec

logger = logging.getLogger(__name__)

__all__ = ["BinanceAdapter"]

REST_URL = "https://api.binance.com"
REST_TESTNET = "https://testnet.binance.vision"
WS_URL = "wss://stream.binance.com:9443/stream"
WS_TESTNET = "wss://testnet.binance.vision/stream"

# Binance error codes we translate into typed exceptions.
_CODE_MAP = {
    -1013: MinNotionalError,      # filter failure (LOT_SIZE / MIN_NOTIONAL)
    -2010: InsufficientBalance,   # NEW_ORDER_REJECTED (usually funds)
    -1021: OrderRejected,         # timestamp outside recvWindow
    -1003: RateLimited,
}

_STATUS_MAP = {
    "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "PENDING_CANCEL": OrderStatus.OPEN,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
    "EXPIRED_IN_MATCH": OrderStatus.EXPIRED,
}

_TIF_MAP = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.GTX: "GTX",
}


class BinanceAdapter(ExchangeAdapter):
    """Live Binance Spot adapter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._testnet = self.config.sandbox
        self._rest = HttpClient(
            REST_TESTNET if self._testnet else REST_URL,
            venue=self.name,
            timeout_sec=self.config.rest_timeout_sec,
            headers=self._auth_headers(),
        )
        self._ws: WebSocketClient | None = None
        self._ws_task: asyncio.Task | None = None
        self._subscribed: list[Symbol] = []

        # Binance enforces several independent windows simultaneously.
        self._limiter = (
            WeightedLimiter(self.name)
            .add_window("weight", 1150, 60.0)     # 1200 published; leave headroom
            .add_window("orders_10s", 45, 10.0)   # 50 published
            .add_window("orders_day", 150_000, 86_400.0)
        )

        # Book sync state, per symbol.
        self._buffered: dict[str, list[dict]] = {}
        self._synced: dict[str, bool] = {}
        self._last_update_id: dict[str, int] = {}

        self.capabilities = AdapterCapabilities(
            streaming_books=True, streaming_fills=True,
            post_only=True, fok=True, ioc=True,
            query_by_client_id=True, cancel_all=True,
            batch_orders=False, fee_in_response=True,
        )

    def _auth_headers(self) -> dict[str, str]:
        creds = self.config.credentials()
        return {"X-MBX-APIKEY": creds["api_key"]} if creds["api_key"] else {}

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        creds = self.config.credentials()
        params = {**params, "timestamp": int(time.time() * 1000), "recvWindow": 5000}
        query = HttpClient.urlencode(params)
        params["signature"] = sign_hmac(creds["api_secret"], query)
        return params

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        await self._rest.start()
        await self._limiter.acquire({"weight": 1})
        info = await self._rest.get("/api/v3/ping")
        self._connected = True
        self.books.mark_connected(self.name)
        logger.info("binance connected (testnet=%s)", self._testnet)

    async def disconnect(self) -> None:
        self._connected = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None
        await self._rest.close()
        self.books.mark_disconnected(self.name)

    # -- instruments -------------------------------------------------------

    async def load_instruments(self) -> Sequence[Symbol]:
        """
        Load ``exchangeInfo`` and translate the filter list into SymbolSpecs.

        The filters are where the real trading rules live and they are
        inconsistently named: ``LOT_SIZE.stepSize`` is the lot grid,
        ``PRICE_FILTER.tickSize`` is the price grid, and the minimum order value
        appears as ``MIN_NOTIONAL.minNotional`` on older pairs and
        ``NOTIONAL.minNotional`` on newer ones. Missing the second spelling
        means silently defaulting to a wrong minimum on a large fraction of the
        universe.
        """
        await self._limiter.acquire({"weight": 20})
        data = await self._rest.get("/api/v3/exchangeInfo")
        symbols: list[Symbol] = []

        allow = set(self.config.symbol_allowlist)
        deny = set(self.config.symbol_denylist)
        quotes = set(self.config.quote_assets)

        for entry in data.get("symbols", []):
            if entry.get("status") != "TRADING":
                continue
            if not entry.get("isSpotTradingAllowed", True):
                continue
            venue_symbol = entry["symbol"]
            base_code, quote_code = entry["baseAsset"], entry["quoteAsset"]

            if allow and venue_symbol not in allow:
                continue
            if venue_symbol in deny:
                continue
            if quotes and quote_code not in quotes:
                continue

            spec = SymbolSpec(
                venue_symbol=venue_symbol,
                base=base_code,
                quote=quote_code,
                base_precision=int(entry.get("baseAssetPrecision", 8)),
                quote_precision=int(entry.get("quoteAssetPrecision", 8)),
                maker_fee_bps=(
                    D(str(self.config.maker_fee_bps))
                    if self.config.maker_fee_bps is not None else None
                ),
                taker_fee_bps=(
                    D(str(self.config.taker_fee_bps))
                    if self.config.taker_fee_bps is not None else None
                ),
            )

            for filt in entry.get("filters", []):
                ftype = filt.get("filterType")
                if ftype == "LOT_SIZE":
                    spec.lot_step = D(filt["stepSize"])
                    spec.min_quantity = D(filt["minQty"])
                    spec.max_quantity = D(filt["maxQty"])
                elif ftype == "PRICE_FILTER":
                    spec.tick_size = D(filt["tickSize"])
                elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    # Both spellings exist across the universe.
                    value = filt.get("minNotional") or filt.get("notional")
                    if value:
                        spec.min_notional = D(value)
                    if filt.get("maxNotional"):
                        spec.max_notional = D(filt["maxNotional"])

            self.register_spec(spec)
            symbols.append(
                self.normalizer.register(self.name, venue_symbol, base_code, quote_code)
            )
            if len(symbols) >= self.config.max_symbols:
                break

        logger.info("binance: loaded %d instruments", len(symbols))
        return symbols

    # -- market data -------------------------------------------------------

    async def subscribe_books(self, symbols: Sequence[Symbol]) -> None:
        self._subscribed = list(symbols)
        for s in symbols:
            self.books.register(s)
            self._synced[s.venue_symbol] = False
            self._buffered[s.venue_symbol] = []
        if self._ws_task is None:
            self._ws_task = asyncio.create_task(self._stream(), name="binance-ws")

    async def _stream(self) -> None:
        streams = "/".join(
            f"{s.venue_symbol.lower()}@depth@100ms" for s in self._subscribed
        )
        url = f"{WS_TESTNET if self._testnet else WS_URL}?streams={streams}"
        self._ws = WebSocketClient(
            url, venue=self.name, ping_interval=self.config.ws_ping_interval_sec
        )
        await self._ws.connect()

        # Snapshot every subscribed symbol; events buffer meanwhile.
        snapshot_tasks = [self._snapshot(s) for s in self._subscribed]
        asyncio.create_task(self._gather_snapshots(snapshot_tasks))

        async for message in self._ws.messages():
            data = message.get("data") if "data" in message else message
            if not data or data.get("e") != "depthUpdate":
                continue
            self._on_depth(data)

    async def _gather_snapshots(self, tasks: list) -> None:
        # Serialised: 250 concurrent depth requests would trip the weight limit
        # instantly and get the whole connection banned.
        for task in tasks:
            try:
                await task
            except Exception:
                logger.exception("binance: snapshot failed")

    async def _snapshot(self, symbol: Symbol) -> None:
        await self._limiter.acquire({"weight": 5})
        data = await self._rest.get(
            "/api/v3/depth",
            params={"symbol": symbol.venue_symbol, "limit": max(20, self.config.book_depth)},
        )
        last_id = int(data["lastUpdateId"])
        bids = [(D(p), D(q)) for p, q in data.get("bids", [])]
        asks = [(D(p), D(q)) for p, q in data.get("asks", [])]

        self.books.apply_snapshot(symbol, bids, asks, sequence=last_id)
        self._last_update_id[symbol.venue_symbol] = last_id

        # Replay the buffer per the documented rule.
        buffered = self._buffered.get(symbol.venue_symbol, [])
        applied_first = False
        for event in buffered:
            u, U = int(event["u"]), int(event["U"])
            if u <= last_id:
                continue
            if not applied_first:
                if not (U <= last_id + 1 <= u):
                    # Snapshot is stale relative to the buffer: retry.
                    logger.debug("binance: stale snapshot for %s, retrying", symbol.canonical)
                    self._buffered[symbol.venue_symbol] = []
                    await asyncio.sleep(0.5)
                    return await self._snapshot(symbol)
                applied_first = True
            self._apply_depth_event(symbol, event)
        self._buffered[symbol.venue_symbol] = []
        self._synced[symbol.venue_symbol] = True

    def _on_depth(self, data: dict) -> None:
        venue_symbol = data.get("s", "")
        symbol = self.normalizer.lookup(self.name, venue_symbol)
        if symbol is None:
            return
        if not self._synced.get(venue_symbol):
            buf = self._buffered.setdefault(venue_symbol, [])
            if len(buf) < 2000:
                buf.append(data)
            return
        self._apply_depth_event(symbol, data)

    def _apply_depth_event(self, symbol: Symbol, data: dict) -> None:
        venue_symbol = symbol.venue_symbol
        U, u = int(data["U"]), int(data["u"])
        prev = self._last_update_id.get(venue_symbol, 0)

        if prev and U > prev + 1:
            # Gap: the book is no longer trustworthy.
            logger.warning(
                "binance: depth gap on %s (have %d, event starts %d) -- resyncing",
                symbol.canonical, prev, U,
            )
            self._synced[venue_symbol] = False
            self._buffered[venue_symbol] = []
            asyncio.create_task(self._snapshot(symbol))
            return

        bids = [(D(p), D(q)) for p, q in data.get("b", [])]
        asks = [(D(p), D(q)) for p, q in data.get("a", [])]
        self.books.apply_delta(
            symbol, bids, asks,
            sequence=u,
            ts_venue_ns=int(data.get("E", 0)) * 1_000_000,
        )
        self._last_update_id[venue_symbol] = u

    async def resnapshot(self, symbol: Symbol) -> None:
        self._synced[symbol.venue_symbol] = False
        await self._snapshot(symbol)

    # -- trading -----------------------------------------------------------

    async def submit(self, order: Order, *, timeout_sec: float = 5.0) -> Order:
        self.validate_order(order)
        await self._limiter.acquire({"weight": 1, "orders_10s": 1, "orders_day": 1})

        params: dict[str, Any] = {
            "symbol": order.symbol.venue_symbol,
            "side": order.side.value.upper(),
            "newClientOrderId": order.client_order_id,
            "quantity": _fmt(order.quantity),
            "newOrderRespType": "FULL",   # returns fills, so we get real fees
        }

        if order.order_type is OrderType.MARKET:
            params["type"] = "MARKET"
        elif order.order_type is OrderType.POST_ONLY:
            params["type"] = "LIMIT_MAKER"
            params["price"] = _fmt(order.price or ZERO)
        else:
            params["type"] = "LIMIT"
            params["price"] = _fmt(order.price or ZERO)
            params["timeInForce"] = _TIF_MAP[order.time_in_force]

        submitted_ns = self.clock.mono_ns()
        self.orders_submitted += 1
        try:
            data = await self._rest.post(
                "/api/v3/order", params=self._sign(params), retries=0
            )
        except BadResponse as exc:
            code = _extract_code(str(exc))
            self.orders_rejected += 1
            exc_type = _CODE_MAP.get(code, OrderRejected)
            raise exc_type(str(exc), venue=self.name, code=str(code)) from exc

        self.submit_latency.observe(self.clock.mono_ns() - submitted_ns)
        result = self._parse_order(order, data, submitted_ns)
        self._track(result)
        if result.status.any_fill:
            self.orders_filled += 1
        return result

    def _parse_order(self, original: Order, data: Mapping[str, Any],
                     submitted_ns: int) -> Order:
        status = _STATUS_MAP.get(data.get("status", ""), OrderStatus.OPEN)
        filled = D(data.get("executedQty", "0"))
        cummulative_quote = D(data.get("cummulativeQuoteQty", "0"))
        avg = (cummulative_quote / filled) if filled > 0 else ZERO

        # Sum the real fees the venue charged, per fill.
        total_fee = ZERO
        fee_asset: Asset | None = None
        for fill in data.get("fills", []) or []:
            total_fee += D(fill.get("commission", "0"))
            if fee_asset is None and fill.get("commissionAsset"):
                fee_asset = self.normalizer.asset(fill["commissionAsset"])

        return original.with_status(
            status,
            venue_order_id=str(data.get("orderId", "")),
            filled_quantity=filled,
            average_price=avg,
            fee_paid=total_fee,
            fee_asset=fee_asset,
            ts_submitted_ns=submitted_ns,
            ts_final_ns=self.clock.mono_ns() if status.terminal else 0,
        )

    async def cancel(self, order: Order) -> Order:
        await self._limiter.acquire({"weight": 1})
        try:
            data = await self._rest.delete(
                "/api/v3/order",
                params=self._sign({
                    "symbol": order.symbol.venue_symbol,
                    "origClientOrderId": order.client_order_id,
                }),
            )
        except BadResponse:
            # -2011 "Unknown order sent" means it is already gone. Treat a
            # cancel of a non-existent order as success, not as an error: during
            # an unwind we do not care why it is gone, only that it is.
            return order.with_status(OrderStatus.CANCELED, ts_final_ns=self.clock.mono_ns())
        result = self._parse_order(order, data, order.ts_submitted_ns)
        self._track(result)
        return result

    async def fetch_order(self, order: Order) -> Order:
        await self._limiter.acquire({"weight": 4})
        data = await self._rest.get(
            "/api/v3/order",
            params=self._sign({
                "symbol": order.symbol.venue_symbol,
                "origClientOrderId": order.client_order_id,
            }),
        )
        return self._parse_order(order, data, order.ts_submitted_ns)

    async def fetch_balances(self) -> Mapping[str, Balance]:
        await self._limiter.acquire({"weight": 20})
        data = await self._rest.get("/api/v3/account", params=self._sign({}))
        for entry in data.get("balances", []):
            free, locked = D(entry["free"]), D(entry["locked"])
            if free > 0 or locked > 0:
                asset = self.normalizer.asset(entry["asset"])
                self.set_balance(asset, free, locked)
        return self.all_balances()

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update({
            "synced_books": sum(1 for v in self._synced.values() if v),
            "limiter": self._limiter.stats(),
            "ws_messages": self._ws.messages_received if self._ws else 0,
            "ws_connects": self._ws.connects if self._ws else 0,
        })
        return base


def _fmt(value: Decimal) -> str:
    """Binance rejects scientific notation and trailing-zero mismatches."""
    return format(value.normalize(), "f")


def _extract_code(message: str) -> int:
    import re

    match = re.search(r'"code"\s*:\s*(-?\d+)', message)
    return int(match.group(1)) if match else 0
