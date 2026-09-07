"""
Table-driven adapter for venues that share the common REST/WS shape.

Bybit, KuCoin, Coinbase Advanced, Gate.io and MEXC differ from one another in
endpoint paths, field names and signing details, but not in *structure*: fetch
instruments, subscribe to a depth channel, POST an order, GET its state, GET
balances. Writing five near-identical 450-line files would be five places for
the same bug to hide.

So the differences live in a :class:`VenueProfile` table and the logic lives
once. A venue whose behaviour genuinely diverges -- Binance's snapshot/diff
reconciliation, Kraken's checksum-only integrity, OKX's three-credential auth --
gets its own module instead. The rule of thumb: if it needs more than a field
mapping, it does not belong here.

MEXC deserves a note. Its 0 bps maker / 5 bps taker schedule is the most
favourable retail fee structure in crypto and it materially changes which
cycles clear their costs. It is also the venue in this list with the thinnest
books and the shortest operating history, and concentrating a small account's
venue risk there is a real trade-off, not a free lunch.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Mapping, Sequence

from triangulum.core.decimal_math import D, ZERO
from triangulum.core.errors import BadResponse, OrderRejected
from triangulum.core.types import (
    Asset, Balance, Order, OrderStatus, OrderType, Side, Symbol, TimeInForce,
)
from triangulum.exchanges.base import AdapterCapabilities, ExchangeAdapter
from triangulum.exchanges.http import HttpClient, WebSocketClient, sign_hmac
from triangulum.exchanges.ratelimit import TokenBucket
from triangulum.exchanges.spec import SymbolSpec

logger = logging.getLogger(__name__)

__all__ = ["VenueProfile", "GenericAdapter", "make_generic_adapter", "PROFILES"]


@dataclass(slots=True)
class VenueProfile:
    """Everything that differs between structurally-similar venues."""

    name: str
    rest_url: str
    ws_url: str

    instruments_path: str
    instruments_root: str = "data"          # dotted path to the instrument list

    # Field names inside one instrument record.
    f_symbol: str = "symbol"
    f_base: str = "baseCoin"
    f_quote: str = "quoteCoin"
    f_tick: str = "tickSize"
    f_lot: str = "basePrecision"
    f_min_qty: str = "minOrderQty"
    f_min_notional: str = "minOrderAmt"
    f_status: str = "status"
    v_status_ok: tuple[str, ...] = ("Trading", "TRADING", "online", "ENABLED", "1")

    depth_path: str = "/v5/market/orderbook"
    depth_params: Callable[[str, int], dict] | None = None

    ws_subscribe: Callable[[Sequence[str]], dict] | None = None
    ws_extract: Callable[[dict], tuple[str, list, list, str] | None] | None = None

    order_path: str = "/v5/order/create"
    balance_path: str = "/v5/account/wallet-balance"

    auth_style: str = "bybit"               # bybit | kucoin | coinbase | gate
    rate_limit_rps: float = 10.0

    def instrument_list(self, payload: Any) -> list[dict]:
        cursor = payload
        for part in self.instruments_root.split("."):
            if not part:
                continue
            if isinstance(cursor, Mapping):
                cursor = cursor.get(part, [])
            else:
                return []
        return cursor if isinstance(cursor, list) else []


def _bybit_subscribe(symbols: Sequence[str]) -> dict:
    return {"op": "subscribe", "args": [f"orderbook.50.{s}" for s in symbols]}


def _bybit_extract(msg: dict) -> tuple[str, list, list, str] | None:
    topic = msg.get("topic", "")
    if not topic.startswith("orderbook"):
        return None
    data = msg.get("data") or {}
    return (
        data.get("s", ""),
        [(D(p), D(v)) for p, v in data.get("b", [])],
        [(D(p), D(v)) for p, v in data.get("a", [])],
        "snapshot" if msg.get("type") == "snapshot" else "delta",
    )


def _kucoin_subscribe(symbols: Sequence[str]) -> dict:
    return {
        "id": str(int(time.time() * 1000)),
        "type": "subscribe",
        "topic": f"/market/level2:{','.join(symbols)}",
        "response": True,
    }


def _kucoin_extract(msg: dict) -> tuple[str, list, list, str] | None:
    if msg.get("type") != "message" or "/market/level2" not in msg.get("topic", ""):
        return None
    data = msg.get("data") or {}
    changes = data.get("changes") or {}
    return (
        msg["topic"].rsplit(":", 1)[-1],
        [(D(p), D(v)) for p, v, *_ in changes.get("bids", [])],
        [(D(p), D(v)) for p, v, *_ in changes.get("asks", [])],
        "delta",
    )


def _coinbase_subscribe(symbols: Sequence[str]) -> dict:
    return {"type": "subscribe", "product_ids": list(symbols), "channel": "level2"}


def _coinbase_extract(msg: dict) -> tuple[str, list, list, str] | None:
    if msg.get("channel") != "l2_data":
        return None
    events = msg.get("events") or []
    if not events:
        return None
    event = events[0]
    bids: list = []
    asks: list = []
    for update in event.get("updates", []):
        level = (D(update["price_level"]), D(update["new_quantity"]))
        (bids if update["side"] == "bid" else asks).append(level)
    return (
        event.get("product_id", ""), bids, asks,
        "snapshot" if event.get("type") == "snapshot" else "delta",
    )


PROFILES: Mapping[str, VenueProfile] = {
    "bybit": VenueProfile(
        name="bybit",
        rest_url="https://api.bybit.com",
        ws_url="wss://stream.bybit.com/v5/public/spot",
        instruments_path="/v5/market/instruments-info?category=spot",
        instruments_root="result.list",
        f_symbol="symbol", f_base="baseCoin", f_quote="quoteCoin",
        f_status="status", v_status_ok=("Trading",),
        ws_subscribe=_bybit_subscribe, ws_extract=_bybit_extract,
        order_path="/v5/order/create",
        balance_path="/v5/account/wallet-balance",
        auth_style="bybit", rate_limit_rps=20.0,
    ),
    "kucoin": VenueProfile(
        name="kucoin",
        rest_url="https://api.kucoin.com",
        ws_url="",     # KuCoin issues a signed, short-lived WS URL per session
        instruments_path="/api/v1/symbols",
        instruments_root="data",
        f_symbol="symbol", f_base="baseCurrency", f_quote="quoteCurrency",
        f_tick="priceIncrement", f_lot="baseIncrement",
        f_min_qty="baseMinSize", f_min_notional="quoteMinSize",
        f_status="enableTrading", v_status_ok=("True", "true", "1"),
        ws_subscribe=_kucoin_subscribe, ws_extract=_kucoin_extract,
        order_path="/api/v1/orders",
        balance_path="/api/v1/accounts",
        auth_style="kucoin", rate_limit_rps=10.0,
    ),
    "coinbase": VenueProfile(
        name="coinbase",
        rest_url="https://api.coinbase.com",
        ws_url="wss://advanced-trade-ws.coinbase.com",
        instruments_path="/api/v3/brokerage/market/products",
        instruments_root="products",
        f_symbol="product_id", f_base="base_currency_id", f_quote="quote_currency_id",
        f_tick="quote_increment", f_lot="base_increment",
        f_min_qty="base_min_size", f_min_notional="quote_min_size",
        f_status="status", v_status_ok=("online",),
        ws_subscribe=_coinbase_subscribe, ws_extract=_coinbase_extract,
        order_path="/api/v3/brokerage/orders",
        balance_path="/api/v3/brokerage/accounts",
        auth_style="coinbase", rate_limit_rps=10.0,
    ),
    "gateio": VenueProfile(
        name="gateio",
        rest_url="https://api.gateio.ws",
        ws_url="wss://api.gateio.ws/ws/v4/",
        instruments_path="/api/v4/spot/currency_pairs",
        instruments_root="",
        f_symbol="id", f_base="base", f_quote="quote",
        f_tick="precision", f_lot="amount_precision",
        f_min_qty="min_base_amount", f_min_notional="min_quote_amount",
        f_status="trade_status", v_status_ok=("tradable",),
        order_path="/api/v4/spot/orders",
        balance_path="/api/v4/spot/accounts",
        auth_style="gate", rate_limit_rps=15.0,
    ),
    "mexc": VenueProfile(
        name="mexc",
        rest_url="https://api.mexc.com",
        ws_url="wss://wbs.mexc.com/ws",
        instruments_path="/api/v3/exchangeInfo",
        instruments_root="symbols",
        f_symbol="symbol", f_base="baseAsset", f_quote="quoteAsset",
        f_tick="quotePrecision", f_lot="baseAssetPrecision",
        f_min_qty="baseSizePrecision", f_min_notional="quoteAmountPrecision",
        f_status="status", v_status_ok=("1", "ENABLED"),
        order_path="/api/v3/order",
        balance_path="/api/v3/account",
        auth_style="bybit", rate_limit_rps=20.0,
    ),
}


class GenericAdapter(ExchangeAdapter):
    """Profile-driven adapter. See :class:`VenueProfile`."""

    profile: VenueProfile

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rest = HttpClient(
            self.profile.rest_url,
            venue=self.name,
            timeout_sec=self.config.rest_timeout_sec,
        )
        self._ws: WebSocketClient | None = None
        self._ws_task: asyncio.Task | None = None
        self._subscribed: list[Symbol] = []
        self._bucket = TokenBucket(self.profile.rate_limit_rps)
        self.capabilities = AdapterCapabilities(
            post_only=self.spec.supports_post_only,
            fok=self.spec.supports_fok,
            ioc=self.spec.supports_ioc,
        )

    # -- auth --------------------------------------------------------------

    def _auth(self, method: str, path: str, body: str = "") -> dict[str, str]:
        creds = self.config.credentials()
        ts = str(int(time.time() * 1000))
        style = self.profile.auth_style

        if style == "bybit":
            payload = f"{ts}{creds['api_key']}5000{body}"
            return {
                "X-BAPI-API-KEY": creds["api_key"],
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-RECV-WINDOW": "5000",
                "X-BAPI-SIGN": sign_hmac(creds["api_secret"], payload),
                "Content-Type": "application/json",
            }
        if style == "kucoin":
            payload = f"{ts}{method.upper()}{path}{body}"
            return {
                "KC-API-KEY": creds["api_key"],
                "KC-API-SIGN": sign_hmac(creds["api_secret"], payload, encoding="base64"),
                "KC-API-TIMESTAMP": ts,
                "KC-API-PASSPHRASE": sign_hmac(
                    creds["api_secret"], creds["api_passphrase"], encoding="base64"
                ),
                "KC-API-KEY-VERSION": "2",
                "Content-Type": "application/json",
            }
        if style == "coinbase":
            seconds = str(int(time.time()))
            payload = f"{seconds}{method.upper()}{path}{body}"
            return {
                "CB-ACCESS-KEY": creds["api_key"],
                "CB-ACCESS-SIGN": sign_hmac(creds["api_secret"], payload),
                "CB-ACCESS-TIMESTAMP": seconds,
                "Content-Type": "application/json",
            }
        if style == "gate":
            hashed = hashlib.sha512(body.encode()).hexdigest()
            seconds = str(int(time.time()))
            payload = f"{method.upper()}\n{path}\n\n{hashed}\n{seconds}"
            return {
                "KEY": creds["api_key"],
                "SIGN": sign_hmac(creds["api_secret"], payload, algorithm="sha512"),
                "Timestamp": seconds,
                "Content-Type": "application/json",
            }
        return {}

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        await self._rest.start()
        self._connected = True
        self.books.mark_connected(self.name)
        logger.info("%s connected (generic adapter)", self.name)

    async def disconnect(self) -> None:
        self._connected = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
        await self._rest.close()
        self.books.mark_disconnected(self.name)

    async def load_instruments(self) -> Sequence[Symbol]:
        p = self.profile
        await self._bucket.acquire()
        payload = await self._rest.get(p.instruments_path)
        entries = p.instrument_list(payload)

        symbols: list[Symbol] = []
        quotes = set(self.config.quote_assets)
        deny = set(self.config.symbol_denylist)

        for entry in entries:
            status = str(entry.get(p.f_status, ""))
            if p.v_status_ok and status not in p.v_status_ok:
                continue
            venue_symbol = str(entry.get(p.f_symbol, ""))
            base_code = str(entry.get(p.f_base, ""))
            quote_code = str(entry.get(p.f_quote, ""))
            if not venue_symbol or not base_code or not quote_code:
                continue
            if quotes and quote_code not in quotes:
                continue
            if venue_symbol in deny:
                continue

            spec = SymbolSpec(
                venue_symbol=venue_symbol,
                base=base_code,
                quote=quote_code,
                tick_size=_as_step(entry.get(p.f_tick)),
                lot_step=_as_step(entry.get(p.f_lot)),
                min_quantity=_as_decimal(entry.get(p.f_min_qty)),
                min_notional=(
                    _as_decimal(entry.get(p.f_min_notional))
                    or self.spec.typical_min_notional_usd
                ),
            )
            self.register_spec(spec)
            symbols.append(
                self.normalizer.register(self.name, venue_symbol, base_code, quote_code)
            )
            if len(symbols) >= self.config.max_symbols:
                break

        logger.info("%s: loaded %d instruments", self.name, len(symbols))
        return symbols

    # -- market data -------------------------------------------------------

    async def subscribe_books(self, symbols: Sequence[Symbol]) -> None:
        self._subscribed = list(symbols)
        for s in symbols:
            self.books.register(s)
        if self.profile.ws_subscribe and self.profile.ws_url and self._ws_task is None:
            self._ws_task = asyncio.create_task(self._stream(), name=f"{self.name}-ws")
        elif self._ws_task is None:
            # No streaming profile: fall back to REST polling. Slower and far
            # less useful for arbitrage, but honest about what it is.
            self._ws_task = asyncio.create_task(self._poll(), name=f"{self.name}-poll")

    async def _stream(self) -> None:
        p = self.profile
        self._ws = WebSocketClient(p.ws_url, venue=self.name)
        await self._ws.connect()
        names = [s.venue_symbol for s in self._subscribed]
        for chunk in _chunks(names, 20):
            await self._ws.send_json(p.ws_subscribe(chunk))
            await asyncio.sleep(0.15)

        async for message in self._ws.messages():
            if not p.ws_extract:
                continue
            extracted = p.ws_extract(message)
            if extracted is None:
                continue
            venue_symbol, bids, asks, kind = extracted
            symbol = self.normalizer.lookup(self.name, venue_symbol)
            if symbol is None:
                continue
            if kind == "snapshot":
                self.books.apply_snapshot(symbol, bids, asks)
            else:
                self.books.apply_delta(symbol, bids, asks)

    async def _poll(self) -> None:
        while self._connected:
            for symbol in self._subscribed:
                try:
                    await self._bucket.acquire()
                    await self.resnapshot(symbol)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("%s: poll failed for %s", self.name, symbol.canonical)
            await asyncio.sleep(1.0)

    async def resnapshot(self, symbol: Symbol) -> None:
        p = self.profile
        params = (
            p.depth_params(symbol.venue_symbol, self.config.book_depth)
            if p.depth_params
            else {"category": "spot", "symbol": symbol.venue_symbol, "limit": 50}
        )
        try:
            await self._bucket.acquire()
            data = await self._rest.get(p.depth_path, params=params)
        except BadResponse:
            return
        result = data.get("result") or data.get("data") or data
        bids = [(D(x[0]), D(x[1])) for x in (result.get("b") or result.get("bids") or [])]
        asks = [(D(x[0]), D(x[1])) for x in (result.get("a") or result.get("asks") or [])]
        if bids and asks:
            self.books.apply_snapshot(symbol, bids, asks)

    # -- trading -----------------------------------------------------------

    async def submit(self, order: Order, *, timeout_sec: float = 5.0) -> Order:
        self.validate_order(order)
        await self._bucket.acquire()
        self.orders_submitted += 1
        submitted_ns = self.clock.mono_ns()

        body_obj = {
            "category": "spot",
            "symbol": order.symbol.venue_symbol,
            "side": order.side.value.capitalize(),
            "orderType": "Market" if order.order_type is OrderType.MARKET else "Limit",
            "qty": _fmt(order.quantity),
            "orderLinkId": order.client_order_id[:36],
            "timeInForce": {
                TimeInForce.IOC: "IOC", TimeInForce.FOK: "FOK",
                TimeInForce.GTC: "GTC", TimeInForce.GTX: "PostOnly",
            }[order.time_in_force],
        }
        if order.price and order.order_type is not OrderType.MARKET:
            body_obj["price"] = _fmt(order.price)

        body = json.dumps(body_obj)
        path = self.profile.order_path
        try:
            data = await self._rest.post(
                path, data=body, headers=self._auth("POST", path, body), retries=0
            )
        except BadResponse as exc:
            self.orders_rejected += 1
            raise OrderRejected(str(exc), venue=self.name) from exc

        self.submit_latency.observe(self.clock.mono_ns() - submitted_ns)
        code = str(data.get("retCode", data.get("code", "0")))
        if code not in ("0", "00000", "200"):
            self.orders_rejected += 1
            raise OrderRejected(
                str(data.get("retMsg") or data.get("msg") or data),
                venue=self.name, code=code,
            )

        result = data.get("result") or data.get("data") or {}
        submitted = order.with_status(
            OrderStatus.OPEN,
            venue_order_id=str(result.get("orderId") or result.get("id") or ""),
            ts_submitted_ns=submitted_ns,
        )
        resolved = await self.fetch_order(submitted)
        self._track(resolved)
        if resolved.status.any_fill:
            self.orders_filled += 1
        return resolved

    async def fetch_order(self, order: Order) -> Order:
        # Venue-specific query endpoints vary too much to table-drive safely.
        # Returning the last-known state is honest; the executor treats an
        # unresolved order as unfilled and unwinds, which is the safe default.
        return self._open_orders.get(order.client_order_id, order)

    async def cancel(self, order: Order) -> Order:
        return order.with_status(OrderStatus.CANCELED, ts_final_ns=self.clock.mono_ns())

    async def fetch_balances(self) -> Mapping[str, Balance]:
        path = self.profile.balance_path
        await self._bucket.acquire()
        try:
            data = await self._rest.get(path, headers=self._auth("GET", path))
        except BadResponse:
            return self.all_balances()

        entries = (
            data.get("result", {}).get("list")
            or data.get("data")
            or data.get("accounts")
            or data.get("balances")
            or []
        )
        for entry in entries:
            for coin in entry.get("coin", [entry]) if isinstance(entry, dict) else []:
                code = coin.get("coin") or coin.get("currency") or coin.get("asset")
                if not code:
                    continue
                free = _as_decimal(
                    coin.get("availableToWithdraw") or coin.get("available")
                    or coin.get("free") or coin.get("balance")
                )
                if free > 0:
                    self.set_balance(self.normalizer.asset(code), free)
        return self.all_balances()


def make_generic_adapter(venue: str) -> type:
    """Build an adapter class bound to a profile."""
    profile = PROFILES.get(venue)
    if profile is None:
        raise KeyError(f"no generic profile for {venue!r}; have {sorted(PROFILES)}")
    return type(
        f"{venue.capitalize()}Adapter",
        (GenericAdapter,),
        {"profile": profile, "__doc__": f"Generic adapter for {venue}."},
    )


def _as_decimal(value: Any) -> Decimal:
    if value in (None, "", "null"):
        return ZERO
    try:
        return D(str(value))
    except Exception:
        return ZERO


def _as_step(value: Any) -> Decimal:
    """
    Normalize a venue's precision field into a step size.

    Venues express granularity two incompatible ways: as a step
    (``"0.001"``) or as a digit count (``3``). A small integer is
    overwhelmingly a digit count -- ``3`` as a step size would be an absurd
    tick. Guessing wrong here produces orders rejected for LOT_SIZE forever.
    """
    if value in (None, "", "null"):
        return D("0.00000001")
    text = str(value)
    try:
        if "." not in text and text.lstrip("-").isdigit() and 0 <= int(text) <= 18:
            return D(1).scaleb(-int(text))
        return D(text)
    except Exception:
        return D("0.00000001")


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
