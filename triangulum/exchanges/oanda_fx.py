"""
OANDA FX adapter -- and an honest account of why FX triangular arbitrage does
not work for a retail participant.

THE STRUCTURAL PROBLEM
======================

A triangular arbitrage in FX means finding, say:

    EUR -> USD -> JPY -> EUR

and having the product of the three rates exceed 1. For that to happen, the
three prices must be set *independently*. On the interbank market they very
nearly are: EUR/USD, USD/JPY and EUR/JPY are quoted by different desks with
different inventories, and the cross does drift out of line -- for a few
microseconds, by a fraction of a pip, and the arbitrage is taken by someone with
a colocated server and a prime-brokerage relationship.

A retail broker does not work that way. It quotes EUR/JPY by *computing* it from
EUR/USD and USD/JPY and adding a markup. The third price is a deterministic
function of the first two. The triangle is therefore closed by construction, and
the product of the three rates is not merely "usually below 1" -- it is
*mechanically* below 1 by exactly the sum of the three spreads.

Concretely, with typical retail spreads:

    EUR/USD   0.8 pips  ->  ~0.7 bps
    USD/JPY   0.9 pips  ->  ~0.7 bps
    EUR/JPY   1.5 pips  ->  ~1.1 bps
                             -------
    round trip cost              2.5 bps against a mispricing of exactly 0.0

There is no configuration of this adapter, no amount of latency optimisation and
no execution cleverness that recovers a positive expectation from that. The
:meth:`OandaAdapter.demonstrate_closed_triangle` method computes it live from
the broker's own quotes so the conclusion is verifiable rather than asserted.

WHAT THIS ADAPTER IS ACTUALLY FOR
=================================

1. Statistical arbitrage: mean reversion on genuinely cointegrated pairs
   (EUR/CHF vs EUR/SEK, AUD/USD vs NZD/USD). Real, but it is a directional
   strategy with drawdowns, not arbitrage, and it needs far more than $100 to
   survive its own variance.
2. Cross-broker arbitrage, if you hold accounts at two brokers whose quotes
   diverge. Legal, real, and operationally punishing: the capital is split
   across venues, the divergence is small, and brokers actively discourage it.
3. Honest benchmarking of the crypto path against a zero-commission venue.

Practice mode ("fxPractice") is the default. Live requires the same triple lock
as everything else.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Mapping, Sequence

from triangulum.core.decimal_math import D, ONE, ZERO, bps, geometric_product, safe_div
from triangulum.core.errors import BadResponse, OrderRejected
from triangulum.core.types import (
    Asset, Balance, Order, OrderStatus, OrderType, Side, Symbol, TimeInForce,
)
from triangulum.exchanges.base import AdapterCapabilities, ExchangeAdapter
from triangulum.exchanges.http import HttpClient
from triangulum.exchanges.ratelimit import TokenBucket
from triangulum.exchanges.spec import SymbolSpec

logger = logging.getLogger(__name__)

__all__ = ["OandaAdapter", "TriangleClosureReport"]

REST_LIVE = "https://api-fxtrade.oanda.com"
REST_PRACTICE = "https://api-fxpractice.oanda.com"


class TriangleClosureReport:
    """
    Evidence that a currency triangle is closed at a broker's own quotes.

    ``product`` is the round-trip return of one unit through the three legs at
    the executable (bid/ask, not mid) prices. A value below 1 means the triangle
    costs money to traverse; the shortfall is exactly the summed spread.
    """

    __slots__ = ("legs", "product", "cost_bps", "mid_product", "mid_deviation_bps")

    def __init__(self, legs: list[tuple[str, str, Decimal]], product: Decimal,
                 mid_product: Decimal) -> None:
        self.legs = legs
        self.product = product
        self.cost_bps = bps(ONE - product)
        self.mid_product = mid_product
        self.mid_deviation_bps = bps(mid_product - ONE)

    @property
    def profitable(self) -> bool:
        return self.product > ONE

    def summary(self) -> str:
        path = " -> ".join(f"{a}/{b}" for a, b, _ in self.legs)
        return (
            f"{path}: executable product = {self.product:.8f} "
            f"({self.cost_bps:+.2f} bps cost), "
            f"mid product = {self.mid_product:.8f} "
            f"({self.mid_deviation_bps:+.3f} bps deviation). "
            f"{'PROFITABLE' if self.profitable else 'CLOSED -- no arbitrage exists'}"
        )


class OandaAdapter(ExchangeAdapter):
    """OANDA v20 REST adapter. Polling-based; OANDA's stream is per-account."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._account_id = self.config.credentials().get("api_passphrase", "")
        base = REST_PRACTICE if self.config.sandbox else REST_LIVE
        creds = self.config.credentials()
        self._rest = HttpClient(
            base,
            venue=self.name,
            timeout_sec=self.config.rest_timeout_sec,
            headers={
                "Authorization": f"Bearer {creds['api_key']}",
                "Content-Type": "application/json",
            },
        )
        self._bucket = TokenBucket(self.config.max_requests_per_second or 5.0)
        self._poll_task: asyncio.Task | None = None
        self._subscribed: list[Symbol] = []
        self.capabilities = AdapterCapabilities(
            streaming_books=False,      # polled, not streamed
            streaming_fills=False,
            post_only=False,            # FX brokers have no maker/taker model
            fok=True,
            ioc=True,
            batch_orders=False,
            fee_in_response=False,      # cost is in the spread, not a commission
        )

    async def connect(self) -> None:
        await self._rest.start()
        await self._bucket.acquire()
        data = await self._rest.get("/v3/accounts")
        accounts = data.get("accounts", [])
        if not self._account_id and accounts:
            self._account_id = accounts[0]["id"]
        self._connected = True
        self.books.mark_connected(self.name)
        logger.info(
            "oanda connected (practice=%s, account=%s)",
            self.config.sandbox, self._account_id,
        )

    async def disconnect(self) -> None:
        self._connected = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        await self._rest.close()
        self.books.mark_disconnected(self.name)

    async def load_instruments(self) -> Sequence[Symbol]:
        await self._bucket.acquire()
        data = await self._rest.get(f"/v3/accounts/{self._account_id}/instruments")
        symbols: list[Symbol] = []
        for entry in data.get("instruments", []):
            if entry.get("type") != "CURRENCY":
                continue
            name = entry["name"]                    # e.g. "EUR_USD"
            base_code, _, quote_code = name.partition("_")
            precision = int(entry.get("displayPrecision", 5))
            spec = SymbolSpec(
                venue_symbol=name,
                base=base_code,
                quote=quote_code,
                tick_size=D(1).scaleb(-precision),
                # OANDA trades in units of base currency, minimum 1.
                lot_step=D("1"),
                min_quantity=D("1"),
                min_notional=D("1"),
                maker_fee_bps=ZERO,
                taker_fee_bps=ZERO,     # the cost is entirely the spread
            )
            self.register_spec(spec)
            symbols.append(
                self.normalizer.register(self.name, name, base_code, quote_code)
            )
        logger.info("oanda: loaded %d currency instruments", len(symbols))
        return symbols

    # -- market data (polled) ----------------------------------------------

    async def subscribe_books(self, symbols: Sequence[Symbol]) -> None:
        self._subscribed = list(symbols)
        for s in symbols:
            self.books.register(s)
        if self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll(), name="oanda-poll")

    async def _poll(self) -> None:
        """
        Poll pricing.

        OANDA returns a top-of-book only -- there is no depth. That alone rules
        out the depth-aware sizing the crypto path relies on: you cannot know
        what a 3x-size order would fill at, only what 1 unit would.
        """
        while self._connected:
            try:
                await self._bucket.acquire()
                names = ",".join(s.venue_symbol for s in self._subscribed[:100])
                if not names:
                    await asyncio.sleep(1.0)
                    continue
                data = await self._rest.get(
                    f"/v3/accounts/{self._account_id}/pricing",
                    params={"instruments": names},
                )
                for price in data.get("prices", []):
                    symbol = self.normalizer.lookup(self.name, price.get("instrument", ""))
                    if symbol is None or price.get("tradeable") is False:
                        continue
                    bids = [
                        (D(b["price"]), D(b.get("liquidity", "1000000")))
                        for b in price.get("bids", [])
                    ]
                    asks = [
                        (D(a["price"]), D(a.get("liquidity", "1000000")))
                        for a in price.get("asks", [])
                    ]
                    if bids and asks:
                        self.books.apply_snapshot(symbol, bids, asks)
                await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("oanda: pricing poll failed")
                await asyncio.sleep(2.0)

    # -- the demonstration -------------------------------------------------

    def demonstrate_closed_triangle(
        self,
        a: str = "EUR",
        b: str = "USD",
        c: str = "JPY",
    ) -> TriangleClosureReport | None:
        """
        Compute a currency triangle from the broker's live quotes.

        Returns the executable product (crossing the spread on each leg) and the
        mid-price product. The mid product being within a fraction of a basis
        point of 1.0 is the proof that the cross is *derived*: independently
        quoted prices would deviate far more than that, and the deviation would
        fluctuate. A derived cross sits at 1.0 permanently.
        """
        legs: list[tuple[str, str, Decimal]] = []
        exec_rates: list[Decimal] = []
        mid_rates: list[Decimal] = []

        for frm, to in ((a, b), (b, c), (c, a)):
            symbol = self.normalizer.find_pair(
                self.name, self.normalizer.asset(frm), self.normalizer.asset(to)
            )
            if symbol is None:
                return None
            book = self.books.get(symbol)
            if book is None or not book.initialized:
                return None

            if symbol.base.code == frm:
                # Selling base for quote: hit the bid.
                exec_rate = book.best_bid
                mid_rate = book.mid
            else:
                # Buying base with quote: lift the ask -> rate is 1/ask.
                exec_rate = safe_div(ONE, book.best_ask)
                mid_rate = safe_div(ONE, book.mid)

            legs.append((frm, to, exec_rate))
            exec_rates.append(exec_rate)
            mid_rates.append(mid_rate)

        return TriangleClosureReport(
            legs=legs,
            product=geometric_product(exec_rates),
            mid_product=geometric_product(mid_rates),
        )

    # -- trading -----------------------------------------------------------

    async def submit(self, order: Order, *, timeout_sec: float = 5.0) -> Order:
        self.validate_order(order)
        await self._bucket.acquire()
        self.orders_submitted += 1
        submitted_ns = self.clock.mono_ns()

        units = int(order.quantity) * (1 if order.side is Side.BUY else -1)
        body: dict[str, Any] = {
            "order": {
                "instrument": order.symbol.venue_symbol,
                "units": str(units),
                "type": "MARKET" if order.order_type is OrderType.MARKET else "LIMIT",
                "timeInForce": "FOK" if order.time_in_force is TimeInForce.FOK else "IOC",
                "positionFill": "DEFAULT",
                "clientExtensions": {"id": order.client_order_id[:32]},
            }
        }
        if order.order_type is not OrderType.MARKET and order.price:
            body["order"]["price"] = format(order.price, "f")
            body["order"]["timeInForce"] = "GTC"

        import json as _json

        data = await self._rest.post(
            f"/v3/accounts/{self._account_id}/orders",
            data=_json.dumps(body), retries=0,
        )
        self.submit_latency.observe(self.clock.mono_ns() - submitted_ns)

        txn = data.get("orderFillTransaction")
        if txn is None:
            cancel = data.get("orderCancelTransaction")
            self.orders_rejected += 1
            raise OrderRejected(
                str(cancel.get("reason") if cancel else data),
                venue=self.name,
                code=str(cancel.get("reason", "")) if cancel else "",
            )

        filled = abs(D(txn.get("units", "0")))
        price = D(txn.get("price", "0"))
        # OANDA's "financing" and spread cost are not a commission line; the
        # cost is embedded in the fill price versus mid.
        result = order.with_status(
            OrderStatus.FILLED if filled >= order.quantity else OrderStatus.PARTIALLY_FILLED,
            venue_order_id=str(txn.get("orderID", "")),
            filled_quantity=filled,
            average_price=price,
            fee_paid=ZERO,
            ts_submitted_ns=submitted_ns,
            ts_final_ns=self.clock.mono_ns(),
        )
        self.orders_filled += 1
        self._track(result)
        return result

    async def cancel(self, order: Order) -> Order:
        await self._bucket.acquire()
        try:
            await self._rest.request(
                "PUT",
                f"/v3/accounts/{self._account_id}/orders/"
                f"@{order.client_order_id[:32]}/cancel",
            )
        except BadResponse:
            pass
        return order.with_status(OrderStatus.CANCELED, ts_final_ns=self.clock.mono_ns())

    async def fetch_order(self, order: Order) -> Order:
        return self._open_orders.get(order.client_order_id, order)

    async def fetch_balances(self) -> Mapping[str, Balance]:
        await self._bucket.acquire()
        data = await self._rest.get(f"/v3/accounts/{self._account_id}/summary")
        account = data.get("account", {})
        currency = account.get("currency", "USD")
        balance = D(account.get("balance", "0"))
        self.set_balance(self.normalizer.asset(currency), balance)
        return self.all_balances()
