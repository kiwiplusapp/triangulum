"""
OKX Spot adapter.

Two properties make OKX the best second venue after Binance:

**8 bps maker.** Two basis points better than Binance per maker leg. On a
maker-taker-taker cycle that is 28 bps of total fee against Binance's 30, and
when the entire opportunity distribution lives between 1 and 8 bps, two bps of
structural cost is not a rounding difference -- it moves the break-even point
enough to change which cycles are tradeable at all.

**A published order-book checksum.** OKX sends a CRC32 over the top 25 levels
with every update. That turns book integrity from an assumption into a verified
property. Venues without a checksum can silently hand you a corrupted book after
a dropped frame, and you find out when a "free" cycle loses money.

Auth is HMAC-SHA256 over ``timestamp + method + requestPath + body``, base64,
with an additional signed passphrase -- three credentials rather than two.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Sequence

from triangulum.core.decimal_math import D, ZERO
from triangulum.core.errors import BadResponse, OrderRejected
from triangulum.core.types import (
    Asset, Balance, Order, OrderStatus, OrderType, Side, Symbol, TimeInForce,
)
from triangulum.exchanges.base import AdapterCapabilities, ExchangeAdapter
from triangulum.exchanges.http import HttpClient, WebSocketClient, sign_hmac
from triangulum.exchanges.ratelimit import WeightedLimiter
from triangulum.exchanges.spec import SymbolSpec

logger = logging.getLogger(__name__)

__all__ = ["OkxAdapter"]

REST_URL = "https://www.okx.com"
WS_PUBLIC = "wss://ws.okx.com:8443/ws/v5/public"
WS_PRIVATE = "wss://ws.okx.com:8443/ws/v5/private"

_STATUS_MAP = {
    "live": OrderStatus.OPEN,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "mmp_canceled": OrderStatus.CANCELED,
}


class OkxAdapter(ExchangeAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rest = HttpClient(
            REST_URL, venue=self.name, timeout_sec=self.config.rest_timeout_sec
        )
        self._ws: WebSocketClient | None = None
        self._ws_task: asyncio.Task | None = None
        self._subscribed: list[Symbol] = []
        self._limiter = (
            WeightedLimiter(self.name)
            .add_window("public", 20, 2.0)
            .add_window("orders", 60, 2.0)
        )
        self.capabilities = AdapterCapabilities(
            post_only=True, fok=True, ioc=True, batch_orders=True,
        )
        self.checksum_failures = 0

    def _headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        creds = self.config.credentials()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        signature = sign_hmac(
            creds["api_secret"], f"{ts}{method.upper()}{path}{body}", encoding="base64"
        )
        return {
            "OK-ACCESS-KEY": creds["api_key"],
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": creds["api_passphrase"],
            "Content-Type": "application/json",
        }

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        await self._rest.start()
        await self._limiter.acquire({"public": 1})
        await self._rest.get("/api/v5/public/time")
        self._connected = True
        self.books.mark_connected(self.name)
        logger.info("okx connected")

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
        await self._limiter.acquire({"public": 1})
        data = await self._rest.get(
            "/api/v5/public/instruments", params={"instType": "SPOT"}
        )
        symbols: list[Symbol] = []
        quotes = set(self.config.quote_assets)
        deny = set(self.config.symbol_denylist)

        for entry in data.get("data", []):
            if entry.get("state") != "live":
                continue
            inst = entry["instId"]
            base_code, quote_code = entry["baseCcy"], entry["quoteCcy"]
            if quotes and quote_code not in quotes:
                continue
            if inst in deny:
                continue

            spec = SymbolSpec(
                venue_symbol=inst,
                base=base_code,
                quote=quote_code,
                tick_size=D(entry.get("tickSz") or "0.01"),
                lot_step=D(entry.get("lotSz") or "0.00000001"),
                min_quantity=D(entry.get("minSz") or "0"),
                # OKX expresses the minimum in base units, not quote. Converting
                # it to a notional needs a price, so we keep the venue default
                # and let the sizing layer apply the base-unit minimum directly.
                min_notional=self.spec.typical_min_notional_usd,
            )
            self.register_spec(spec)
            symbols.append(
                self.normalizer.register(self.name, inst, base_code, quote_code)
            )
            if len(symbols) >= self.config.max_symbols:
                break

        logger.info("okx: loaded %d instruments", len(symbols))
        return symbols

    # -- market data -------------------------------------------------------

    async def subscribe_books(self, symbols: Sequence[Symbol]) -> None:
        self._subscribed = list(symbols)
        for s in symbols:
            self.books.register(s)
        if self._ws_task is None:
            self._ws_task = asyncio.create_task(self._stream(), name="okx-ws")

    async def _stream(self) -> None:
        self._ws = WebSocketClient(WS_PUBLIC, venue=self.name)
        await self._ws.connect()
        # ``books`` is the 400-level channel with checksums; ``books5`` has no
        # checksum, which is precisely why we do not use it.
        args = [{"channel": "books", "instId": s.venue_symbol} for s in self._subscribed]
        for chunk in _chunks(args, 50):
            await self._ws.send_json({"op": "subscribe", "args": chunk})
            await asyncio.sleep(0.1)

        async for message in self._ws.messages():
            if "event" in message:
                if message["event"] == "error":
                    logger.error("okx ws error: %s", message)
                continue
            arg = message.get("arg", {})
            if arg.get("channel") != "books":
                continue
            symbol = self.normalizer.lookup(self.name, arg.get("instId", ""))
            if symbol is None:
                continue
            action = message.get("action", "update")
            for payload in message.get("data", []):
                self._apply(symbol, payload, action)

    def _apply(self, symbol: Symbol, payload: dict, action: str) -> None:
        bids = [(D(p), D(s)) for p, s, *_ in payload.get("bids", [])]
        asks = [(D(p), D(s)) for p, s, *_ in payload.get("asks", [])]
        ts_ns = int(payload.get("ts", 0)) * 1_000_000
        seq = int(payload.get("seqId", 0) or 0)
        prev_seq = payload.get("prevSeqId")

        if action == "snapshot":
            self.books.apply_snapshot(
                symbol, bids, asks, sequence=seq, ts_venue_ns=ts_ns
            )
            return

        checksum = payload.get("checksum")
        result = self.books.apply_delta(
            symbol, bids, asks,
            sequence=seq,
            prev_sequence=int(prev_seq) if prev_seq is not None else None,
            ts_venue_ns=ts_ns,
            checksum=int(checksum) if checksum is not None else None,
            checksum_algorithm="okx",
        )
        if result is None:
            self.checksum_failures += 1
            asyncio.create_task(self.resnapshot(symbol))

    async def resnapshot(self, symbol: Symbol) -> None:
        await self._limiter.acquire({"public": 1})
        try:
            data = await self._rest.get(
                "/api/v5/market/books",
                params={"instId": symbol.venue_symbol, "sz": 50},
            )
        except BadResponse:
            return
        for payload in data.get("data", []):
            bids = [(D(p), D(s)) for p, s, *_ in payload.get("bids", [])]
            asks = [(D(p), D(s)) for p, s, *_ in payload.get("asks", [])]
            self.books.apply_snapshot(symbol, bids, asks)

    # -- trading -----------------------------------------------------------

    async def submit(self, order: Order, *, timeout_sec: float = 5.0) -> Order:
        self.validate_order(order)
        await self._limiter.acquire({"orders": 1})

        ord_type = "market"
        if order.order_type is OrderType.POST_ONLY:
            ord_type = "post_only"
        elif order.order_type is OrderType.LIMIT:
            ord_type = {
                TimeInForce.IOC: "ioc",
                TimeInForce.FOK: "fok",
                TimeInForce.GTC: "limit",
                TimeInForce.GTX: "post_only",
            }[order.time_in_force]

        body_obj = [{
            "instId": order.symbol.venue_symbol,
            "tdMode": "cash",
            "side": order.side.value,
            "ordType": ord_type,
            "sz": _fmt(order.quantity),
            "clOrdId": order.client_order_id.replace("-", "")[:32],
            **({"px": _fmt(order.price)} if order.price and ord_type != "market" else {}),
        }]
        body = json.dumps(body_obj)
        path = "/api/v5/trade/order"

        submitted_ns = self.clock.mono_ns()
        self.orders_submitted += 1
        data = await self._rest.post(
            path, data=body, headers=self._headers("POST", path, body), retries=0
        )
        self.submit_latency.observe(self.clock.mono_ns() - submitted_ns)

        entries = data.get("data", [])
        if not entries or entries[0].get("sCode") not in ("0", 0):
            self.orders_rejected += 1
            detail = entries[0] if entries else data
            raise OrderRejected(
                str(detail.get("sMsg", detail)), venue=self.name,
                code=str(detail.get("sCode", "")),
            )

        result = order.with_status(
            OrderStatus.OPEN,
            venue_order_id=entries[0].get("ordId", ""),
            ts_submitted_ns=submitted_ns,
        )
        # OKX acknowledges asynchronously; resolve the true state immediately.
        resolved = await self.fetch_order(result)
        self._track(resolved)
        if resolved.status.any_fill:
            self.orders_filled += 1
        return resolved

    async def fetch_order(self, order: Order) -> Order:
        await self._limiter.acquire({"orders": 1})
        path = "/api/v5/trade/order"
        params = {
            "instId": order.symbol.venue_symbol,
            "clOrdId": order.client_order_id.replace("-", "")[:32],
        }
        query = f"{path}?{HttpClient.urlencode(params)}"
        data = await self._rest.get(
            path, params=params, headers=self._headers("GET", query)
        )
        entries = data.get("data", [])
        if not entries:
            return order
        e = entries[0]
        filled = D(e.get("accFillSz") or "0")
        avg = D(e.get("avgPx") or "0")
        fee = abs(D(e.get("fee") or "0"))   # OKX reports fees as negative
        fee_ccy = e.get("feeCcy")
        return order.with_status(
            _STATUS_MAP.get(e.get("state", ""), OrderStatus.OPEN),
            venue_order_id=e.get("ordId", ""),
            filled_quantity=filled,
            average_price=avg,
            fee_paid=fee,
            fee_asset=self.normalizer.asset(fee_ccy) if fee_ccy else None,
            ts_final_ns=self.clock.mono_ns(),
        )

    async def cancel(self, order: Order) -> Order:
        await self._limiter.acquire({"orders": 1})
        path = "/api/v5/trade/cancel-order"
        body = json.dumps({
            "instId": order.symbol.venue_symbol,
            "clOrdId": order.client_order_id.replace("-", "")[:32],
        })
        try:
            await self._rest.post(path, data=body, headers=self._headers("POST", path, body))
        except BadResponse:
            pass
        return await self.fetch_order(order)

    async def fetch_balances(self) -> Mapping[str, Balance]:
        await self._limiter.acquire({"public": 1})
        path = "/api/v5/account/balance"
        data = await self._rest.get(path, headers=self._headers("GET", path))
        for account in data.get("data", []):
            for detail in account.get("details", []):
                free = D(detail.get("availBal") or "0")
                frozen = D(detail.get("frozenBal") or "0")
                if free > 0 or frozen > 0:
                    self.set_balance(self.normalizer.asset(detail["ccy"]), free, frozen)
        return self.all_balances()

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base["checksum_failures"] = self.checksum_failures
        return base


def _fmt(value: Decimal | None) -> str:
    return format((value or ZERO).normalize(), "f")


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
