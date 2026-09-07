"""
Kraken Spot adapter.

Kraken is deliberately NOT a triangular-arbitrage venue in this engine. Its
entry-tier taker fee is 40 bps, so a three-leg all-taker cycle costs 120 bps.
No dislocation of that size survives on a book as liquid as Kraken's; if you
ever see one, the book is stale or you have mis-parsed an asset code.

It earns its place for three other reasons:

1. **Deep fiat books.** EUR, GBP, USD, CHF, JPY and CAD pairs with real depth,
   which makes it the best cross-venue counterparty for spatial arbitrage
   against a crypto-only venue.
2. **A published checksum.** Book integrity is verifiable.
3. **Fiat on-ramp.** It is where capital enters and leaves.

Kraken's asset codes are the other reason this adapter exists as its own file:
the legacy ``X``/``Z`` prefixes (``XXBTZUSD`` is BTC/USD) mean a naive parser
will not even recognise its instruments as connected to the rest of the graph.
The normalizer handles it; this adapter feeds the normalizer the venue's own
``altname`` and ``wsname`` so the mapping is authoritative rather than guessed.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import time
import urllib.parse
from decimal import Decimal
from typing import Any, Mapping, Sequence

from triangulum.core.decimal_math import D, ZERO
from triangulum.core.errors import BadResponse, OrderRejected
from triangulum.core.types import (
    Asset, Balance, Order, OrderStatus, OrderType, Side, Symbol, TimeInForce,
)
from triangulum.exchanges.base import AdapterCapabilities, ExchangeAdapter
from triangulum.exchanges.http import HttpClient, WebSocketClient, sign_kraken
from triangulum.exchanges.ratelimit import TokenBucket
from triangulum.exchanges.spec import SymbolSpec

logger = logging.getLogger(__name__)

__all__ = ["KrakenAdapter"]

REST_URL = "https://api.kraken.com"
WS_URL = "wss://ws.kraken.com"


class KrakenAdapter(ExchangeAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rest = HttpClient(
            REST_URL, venue=self.name, timeout_sec=self.config.rest_timeout_sec
        )
        self._ws: WebSocketClient | None = None
        self._ws_task: asyncio.Task | None = None
        self._subscribed: list[Symbol] = []
        # Kraken's rate limiting is a decaying counter, not a window. One
        # request per second is the sustainable rate for a starter-tier account.
        self._bucket = TokenBucket(1.0, capacity=4.0)
        self._wsname_to_symbol: dict[str, Symbol] = {}
        self.capabilities = AdapterCapabilities(post_only=True, fok=True, ioc=True)
        self.checksum_failures = 0

    def _signed_headers(self, path: str, data: dict[str, Any]) -> tuple[dict, str]:
        creds = self.config.credentials()
        nonce = str(int(time.time() * 1_000_000))
        data = {**data, "nonce": nonce}
        body = urllib.parse.urlencode(data)
        signature = sign_kraken(creds["api_secret"], path, nonce, body)
        return (
            {
                "API-Key": creds["api_key"],
                "API-Sign": signature,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body,
        )

    async def _private(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        path = f"/0/private/{endpoint}"
        headers, body = self._signed_headers(path, data or {})
        await self._bucket.acquire()
        result = await self._rest.post(path, data=body, headers=headers)
        errors = result.get("error") or []
        if errors:
            raise OrderRejected("; ".join(errors), venue=self.name, code=errors[0])
        return result.get("result", {})

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        await self._rest.start()
        await self._bucket.acquire()
        await self._rest.get("/0/public/Time")
        self._connected = True
        self.books.mark_connected(self.name)
        logger.info("kraken connected")

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
        await self._bucket.acquire()
        data = await self._rest.get("/0/public/AssetPairs")
        pairs = data.get("result", {})
        symbols: list[Symbol] = []
        quotes = set(self.config.quote_assets)

        for pair_id, entry in pairs.items():
            if entry.get("status") != "online":
                continue
            wsname = entry.get("wsname", "")
            if not wsname or "/" not in wsname:
                continue
            base_code, _, quote_code = wsname.partition("/")
            if quotes and quote_code not in quotes:
                continue

            decimals = int(entry.get("pair_decimals", 5))
            lot_decimals = int(entry.get("lot_decimals", 8))
            spec = SymbolSpec(
                venue_symbol=pair_id,
                base=base_code,
                quote=quote_code,
                tick_size=D(1).scaleb(-decimals),
                lot_step=D(1).scaleb(-lot_decimals),
                min_quantity=D(entry.get("ordermin", "0") or "0"),
                min_notional=D(entry.get("costmin", "0") or "0")
                             or self.spec.typical_min_notional_usd,
            )
            self.register_spec(spec)
            symbol = self.normalizer.register(self.name, pair_id, base_code, quote_code)
            self._wsname_to_symbol[wsname] = symbol
            symbols.append(symbol)
            if len(symbols) >= self.config.max_symbols:
                break

        logger.info("kraken: loaded %d instruments", len(symbols))
        return symbols

    # -- market data -------------------------------------------------------

    async def subscribe_books(self, symbols: Sequence[Symbol]) -> None:
        self._subscribed = list(symbols)
        for s in symbols:
            self.books.register(s)
        if self._ws_task is None:
            self._ws_task = asyncio.create_task(self._stream(), name="kraken-ws")

    async def _stream(self) -> None:
        self._ws = WebSocketClient(WS_URL, venue=self.name)
        await self._ws.connect()
        wsnames = [
            name for name, sym in self._wsname_to_symbol.items()
            if any(s.key == sym.key for s in self._subscribed)
        ]
        for chunk in _chunks(wsnames, 25):
            await self._ws.send_json({
                "event": "subscribe",
                "pair": chunk,
                "subscription": {"name": "book", "depth": 25},
            })
            await asyncio.sleep(0.2)

        async for message in self._ws.messages():
            if isinstance(message, dict):
                continue          # subscription acks and heartbeats
            if not isinstance(message, list) or len(message) < 4:
                continue
            payload, _channel_name, wsname = message[1], message[-2], message[-1]
            symbol = self._wsname_to_symbol.get(wsname)
            if symbol is None:
                continue
            self._apply(symbol, payload)

    def _apply(self, symbol: Symbol, payload: dict) -> None:
        if "as" in payload or "bs" in payload:
            bids = [(D(p), D(v)) for p, v, *_ in payload.get("bs", [])]
            asks = [(D(p), D(v)) for p, v, *_ in payload.get("as", [])]
            self.books.apply_snapshot(symbol, bids, asks)
            return

        bids = [(D(p), D(v)) for p, v, *_ in payload.get("b", [])]
        asks = [(D(p), D(v)) for p, v, *_ in payload.get("a", [])]
        checksum = payload.get("c")
        # Kraken publishes no sequence number -- the CRC32 checksum IS the
        # integrity guard, which is why the checksum path is not optional here.
        result = self.books.apply_delta(
            symbol, bids, asks,
            sequence=0,
            checksum=int(checksum) if checksum is not None else None,
            checksum_algorithm="kraken",
        )
        if result is None:
            self.checksum_failures += 1
            asyncio.create_task(self.resnapshot(symbol))

    async def resnapshot(self, symbol: Symbol) -> None:
        await self._bucket.acquire()
        try:
            data = await self._rest.get(
                "/0/public/Depth",
                params={"pair": symbol.venue_symbol, "count": 25},
            )
        except BadResponse:
            return
        for _pair, payload in (data.get("result") or {}).items():
            bids = [(D(p), D(v)) for p, v, *_ in payload.get("bids", [])]
            asks = [(D(p), D(v)) for p, v, *_ in payload.get("asks", [])]
            self.books.apply_snapshot(symbol, bids, asks)

    # -- trading -----------------------------------------------------------

    async def submit(self, order: Order, *, timeout_sec: float = 5.0) -> Order:
        self.validate_order(order)
        self.orders_submitted += 1
        submitted_ns = self.clock.mono_ns()

        payload: dict[str, Any] = {
            "pair": order.symbol.venue_symbol,
            "type": order.side.value,
            "ordertype": "market" if order.order_type is OrderType.MARKET else "limit",
            "volume": _fmt(order.quantity),
            "userref": abs(hash(order.client_order_id)) % (2**31),
        }
        if order.order_type is not OrderType.MARKET:
            payload["price"] = _fmt(order.price or ZERO)
        flags = []
        if order.order_type is OrderType.POST_ONLY or order.time_in_force is TimeInForce.GTX:
            flags.append("post")
        if flags:
            payload["oflags"] = ",".join(flags)
        if order.time_in_force is TimeInForce.IOC:
            payload["timeinforce"] = "IOC"

        result = await self._private("AddOrder", payload)
        self.submit_latency.observe(self.clock.mono_ns() - submitted_ns)

        txids = result.get("txid") or []
        submitted = order.with_status(
            OrderStatus.OPEN,
            venue_order_id=txids[0] if txids else "",
            ts_submitted_ns=submitted_ns,
        )
        resolved = await self.fetch_order(submitted)
        self._track(resolved)
        if resolved.status.any_fill:
            self.orders_filled += 1
        return resolved

    async def fetch_order(self, order: Order) -> Order:
        if not order.venue_order_id:
            return order
        result = await self._private("QueryOrders", {"txid": order.venue_order_id})
        entry = result.get(order.venue_order_id)
        if not entry:
            return order
        status_map = {
            "pending": OrderStatus.PENDING, "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED, "canceled": OrderStatus.CANCELED,
            "expired": OrderStatus.EXPIRED,
        }
        filled = D(entry.get("vol_exec", "0"))
        total = D(entry.get("vol", "0"))
        status = status_map.get(entry.get("status", ""), OrderStatus.OPEN)
        if status is OrderStatus.FILLED and filled < total:
            status = OrderStatus.PARTIALLY_FILLED
        return order.with_status(
            status,
            filled_quantity=filled,
            average_price=D(entry.get("price", "0")),
            fee_paid=D(entry.get("fee", "0")),
            fee_asset=order.symbol.quote,
            ts_final_ns=self.clock.mono_ns(),
        )

    async def cancel(self, order: Order) -> Order:
        if order.venue_order_id:
            try:
                await self._private("CancelOrder", {"txid": order.venue_order_id})
            except OrderRejected:
                pass
        return order.with_status(OrderStatus.CANCELED, ts_final_ns=self.clock.mono_ns())

    async def fetch_balances(self) -> Mapping[str, Balance]:
        result = await self._private("Balance")
        for code, amount in result.items():
            value = D(amount)
            if value > 0:
                self.set_balance(self.normalizer.asset(code, self.name), value)
        return self.all_balances()

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base["checksum_failures"] = self.checksum_failures
        return base


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
