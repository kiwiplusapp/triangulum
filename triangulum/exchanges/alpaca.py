"""
Alpaca adapter (US equities + crypto) -- and why equities cannot close a cycle.

You asked to avoid cryptocurrencies. This adapter is the honest attempt, and
the honest answer is in the geometry: **triangular arbitrage requires a cycle,
and single-listed equities do not form one.**

A cycle needs a path A -> B -> C -> A where each hop is a tradable instrument.
In FX and crypto every asset is quoted against several others, so the graph is
densely connected and cycles are everywhere. In equities the graph is a *star*:
AAPL trades against USD, MSFT trades against USD, and AAPL does not trade
against MSFT. Every path from AAPL back to AAPL goes through USD and retraces
its own steps -- that is a round trip, paying the spread twice, not an
arbitrage.

The genuine equity arbitrages all need infrastructure a $100 account cannot
reach:

    ADR / ordinary       requires FX plus foreign-market access plus the
                         conversion fee schedule; the spread is 5-20 bps and
                         two venues must be hit within seconds.
    ETF / NAV            requires creation-unit size, typically 50,000 shares.
    Index / constituent  requires simultaneous execution across 500 names.
    Merger / risk arb    is not arbitrage; it is an event bet with tail risk.
    Dual-listed pairs    (RDS A/B, BHP) needs multi-market data at $2k+/month.

So this adapter exists for the *statistical* strategy -- cointegrated pairs
traded on mean reversion -- which is a real, well-documented edge and also a
directional strategy with real drawdowns, not a riskless one. It is wired for
paper trading. Alpaca's paper environment is free and its data is real, which
makes it a legitimate place to test the statistical path without capital.
"""

from __future__ import annotations

import asyncio
import json
import logging
from decimal import Decimal
from typing import Any, Mapping, Sequence

from triangulum.core.decimal_math import D, ZERO
from triangulum.core.errors import BadResponse, OrderRejected
from triangulum.core.types import (
    Asset, AssetClass, Balance, Order, OrderStatus, OrderType, Side, Symbol, TimeInForce,
)
from triangulum.exchanges.base import AdapterCapabilities, ExchangeAdapter
from triangulum.exchanges.http import HttpClient, WebSocketClient
from triangulum.exchanges.ratelimit import TokenBucket
from triangulum.exchanges.spec import SymbolSpec

logger = logging.getLogger(__name__)

__all__ = ["AlpacaAdapter"]

REST_LIVE = "https://api.alpaca.markets"
REST_PAPER = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
WS_STOCKS = "wss://stream.data.alpaca.markets/v2/iex"

_STATUS_MAP = {
    "new": OrderStatus.OPEN,
    "accepted": OrderStatus.OPEN,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


class AlpacaAdapter(ExchangeAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        creds = self.config.credentials()
        headers = {
            "APCA-API-KEY-ID": creds["api_key"],
            "APCA-API-SECRET-KEY": creds["api_secret"],
            "Content-Type": "application/json",
        }
        self._rest = HttpClient(
            REST_PAPER if self.config.sandbox else REST_LIVE,
            venue=self.name, timeout_sec=self.config.rest_timeout_sec, headers=headers,
        )
        self._data = HttpClient(
            DATA_URL, venue=self.name,
            timeout_sec=self.config.rest_timeout_sec, headers=headers,
        )
        self._bucket = TokenBucket(3.0)
        self._ws: WebSocketClient | None = None
        self._ws_task: asyncio.Task | None = None
        self._subscribed: list[Symbol] = []
        self._usd = Asset("USD", AssetClass.FIAT)
        self.capabilities = AdapterCapabilities(
            streaming_books=True, streaming_fills=False,
            post_only=False,      # equities have no post-only in this API
            fok=True, ioc=True, fee_in_response=False,
        )

    async def connect(self) -> None:
        await self._rest.start()
        await self._data.start()
        await self._bucket.acquire()
        account = await self._rest.get("/v2/account")
        if account.get("trading_blocked"):
            raise OrderRejected("account is trading-blocked", venue=self.name)
        self._connected = True
        self.books.mark_connected(self.name)
        logger.info(
            "alpaca connected (paper=%s, equity=%s)",
            self.config.sandbox, account.get("equity"),
        )

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
        await self._data.close()
        self.books.mark_disconnected(self.name)

    async def load_instruments(self) -> Sequence[Symbol]:
        """
        Load tradable US equities as ``TICKER/USD`` pairs.

        Modelling an equity as a pair against USD is what lets it share the
        graph machinery -- but note that this produces a star topology, not a
        connected graph, so the cycle finder will correctly report that no
        cycles of length >= 3 exist. That is the intended, honest outcome.
        """
        await self._bucket.acquire()
        data = await self._rest.get(
            "/v2/assets", params={"status": "active", "asset_class": "us_equity"}
        )
        allow = set(self.config.symbol_allowlist)
        symbols: list[Symbol] = []

        for entry in data if isinstance(data, list) else []:
            if not entry.get("tradable"):
                continue
            ticker = entry.get("symbol", "")
            if allow and ticker not in allow:
                continue
            fractionable = entry.get("fractionable", False)
            spec = SymbolSpec(
                venue_symbol=ticker,
                base=ticker,
                quote="USD",
                tick_size=D("0.01"),
                # Fractional shares give a 1e-9 grid -- effectively zero
                # quantization drag, which is the one structural advantage
                # equities have over crypto for a small account.
                lot_step=D("0.000000001") if fractionable else D("1"),
                min_quantity=D("0.000000001") if fractionable else D("1"),
                min_notional=D("1"),
                maker_fee_bps=ZERO,
                taker_fee_bps=ZERO,
            )
            self.register_spec(spec)
            symbols.append(self.normalizer.register(self.name, ticker, ticker, "USD"))
            if len(symbols) >= self.config.max_symbols:
                break

        logger.info("alpaca: loaded %d equities", len(symbols))
        return symbols

    async def subscribe_books(self, symbols: Sequence[Symbol]) -> None:
        self._subscribed = list(symbols)
        for s in symbols:
            self.books.register(s)
        if self._ws_task is None:
            self._ws_task = asyncio.create_task(self._stream(), name="alpaca-ws")

    async def _stream(self) -> None:
        creds = self.config.credentials()
        self._ws = WebSocketClient(WS_STOCKS, venue=self.name)
        await self._ws.connect()
        await self._ws.send_json({
            "action": "auth",
            "key": creds["api_key"],
            "secret": creds["api_secret"],
        })
        await self._ws.send_json({
            "action": "subscribe",
            "quotes": [s.venue_symbol for s in self._subscribed[:30]],
        })

        async for message in self._ws.messages():
            events = message if isinstance(message, list) else [message]
            for event in events:
                if event.get("T") != "q":       # quote
                    continue
                symbol = self.normalizer.lookup(self.name, event.get("S", ""))
                if symbol is None:
                    continue
                bid, ask = D(str(event.get("bp", 0))), D(str(event.get("ap", 0)))
                if bid <= 0 or ask <= 0:
                    continue
                self.books.apply_snapshot(
                    symbol,
                    [(bid, D(str(event.get("bs", 0))) * 100)],
                    [(ask, D(str(event.get("as", 0))) * 100)],
                )

    async def submit(self, order: Order, *, timeout_sec: float = 5.0) -> Order:
        self.validate_order(order)
        await self._bucket.acquire()
        self.orders_submitted += 1
        submitted_ns = self.clock.mono_ns()

        body = {
            "symbol": order.symbol.venue_symbol,
            "qty": format(order.quantity.normalize(), "f"),
            "side": order.side.value,
            "type": "market" if order.order_type is OrderType.MARKET else "limit",
            "time_in_force": "ioc" if order.time_in_force is TimeInForce.IOC else "day",
            "client_order_id": order.client_order_id[:48],
        }
        if order.price and order.order_type is not OrderType.MARKET:
            body["limit_price"] = format(order.price, "f")

        try:
            data = await self._rest.post("/v2/orders", data=json.dumps(body), retries=0)
        except BadResponse as exc:
            self.orders_rejected += 1
            raise OrderRejected(str(exc), venue=self.name) from exc

        self.submit_latency.observe(self.clock.mono_ns() - submitted_ns)
        result = self._parse(order, data, submitted_ns)
        self._track(result)
        if result.status.any_fill:
            self.orders_filled += 1
        return result

    def _parse(self, original: Order, data: Mapping[str, Any], submitted_ns: int) -> Order:
        filled = D(str(data.get("filled_qty") or "0"))
        avg = D(str(data.get("filled_avg_price") or "0"))
        status = _STATUS_MAP.get(str(data.get("status", "")), OrderStatus.OPEN)
        return original.with_status(
            status,
            venue_order_id=str(data.get("id", "")),
            filled_quantity=filled,
            average_price=avg,
            fee_paid=ZERO,          # commission-free; cost is spread + PFOF
            ts_submitted_ns=submitted_ns,
            ts_final_ns=self.clock.mono_ns() if status.terminal else 0,
        )

    async def fetch_order(self, order: Order) -> Order:
        await self._bucket.acquire()
        try:
            data = await self._rest.get(
                f"/v2/orders:by_client_order_id",
                params={"client_order_id": order.client_order_id[:48]},
            )
        except BadResponse:
            return order
        return self._parse(order, data, order.ts_submitted_ns)

    async def cancel(self, order: Order) -> Order:
        if order.venue_order_id:
            try:
                await self._rest.delete(f"/v2/orders/{order.venue_order_id}")
            except BadResponse:
                pass
        return order.with_status(OrderStatus.CANCELED, ts_final_ns=self.clock.mono_ns())

    async def fetch_balances(self) -> Mapping[str, Balance]:
        await self._bucket.acquire()
        account = await self._rest.get("/v2/account")
        self.set_balance(self._usd, D(str(account.get("cash", "0"))))
        positions = await self._rest.get("/v2/positions")
        for position in positions if isinstance(positions, list) else []:
            qty = D(str(position.get("qty", "0")))
            if qty != 0:
                self.set_balance(self.normalizer.asset(position["symbol"]), qty)
        return self.all_balances()
