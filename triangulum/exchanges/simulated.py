"""
Simulated venue: the matching engine behind paper mode and backtests.

A paper-trading simulator that fills every order at the touch is worse than no
simulator, because it produces a confident number that is wrong in a specific
direction: too high, always. This one is built to be *pessimistic* in exactly
the ways reality is pessimistic.

Four effects are modelled, in descending order of how much they matter:

1. **Latency and adverse selection.** Between deciding and arriving, the book
   moves. Crucially it does not move randomly: it moves *against you* more often
   than with you, because the same information that made your order attractive
   made the resting liquidity withdraw. The simulator applies a configurable
   adverse-drift term, defaulting to a value calibrated so that the touch is
   gone roughly a third of the time on a fast venue -- which matches what live
   IOC fill rates actually look like.

2. **Queue position for maker orders.** A post-only order joins the back of the
   queue at its price. It fills only when the volume ahead of it trades through.
   Modelled explicitly, because the alternative -- assuming maker orders fill
   whenever the price is touched -- overstates maker fill rates by a factor of
   two or more and makes MTT execution look far better than it is.

3. **Partial fills.** Walking the book with a size cap, exactly as the live
   sizing code does.

4. **Fee accounting in the correct asset.** Buy fees are typically charged in
   base, sell fees in quote. Getting this backwards produces a small, constant,
   direction-dependent P&L error that is maddening to find later.

The simulator holds real balances and refuses orders it cannot fund, so the
whole ledger and reconciliation path is exercised in paper exactly as in live.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Sequence

from triangulum.core.clock import Clock, SystemClock
from triangulum.core.config import VenueConfig
from triangulum.core.decimal_math import D, ONE, ZERO, bps, floor_to_step, safe_div
from triangulum.core.errors import InsufficientBalance, OrderRejected
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
from triangulum.exchanges.spec import SymbolSpec
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.normalizer import SymbolNormalizer
from triangulum.marketdata.orderbook import walk_book, walk_book_for_notional

logger = logging.getLogger(__name__)

__all__ = ["SimulatedExchange", "SimulationParams", "RestingOrder"]


@dataclass(slots=True)
class SimulationParams:
    """
    Knobs controlling how harsh the simulation is.

    The defaults are tuned to be slightly *worse* than a well-connected retail
    setup on a major venue. If the strategy is profitable under these, there is
    a real chance it survives contact with the venue. If it is only profitable
    under optimistic settings, you have learned that too, cheaply.
    """

    # Round-trip order latency, lognormal-ish: median plus a fat tail.
    latency_mean_ms: float = 45.0
    latency_stddev_ms: float = 25.0
    latency_max_ms: float = 400.0

    # Probability the touch has moved away by the time we arrive. This is the
    # single most important parameter in the file.
    adverse_move_probability: float = 0.35
    # When it moves, how far (in ticks).
    adverse_move_ticks_mean: float = 1.4

    # Probability an order is outright rejected for reasons outside our model
    # (venue hiccup, risk engine, momentary imbalance).
    spurious_reject_probability: float = 0.004

    # Fraction of a level's displayed size actually available to us.
    liquidity_availability: Decimal = D("0.6")

    # Maker queue model: what fraction of the level's size sits ahead of us when
    # we join. 1.0 = we are behind everything currently displayed.
    queue_position_fraction: Decimal = D("1.0")
    # Volume traded at a level per second, as a multiple of displayed size.
    # Drives how quickly the queue ahead of us drains.
    queue_drain_rate_per_sec: Decimal = D("0.35")

    # Fee tiers
    maker_fee_bps: Decimal = D("10")
    taker_fee_bps: Decimal = D("10")

    # Set to a fixed value for reproducible backtests.
    seed: int | None = 42

    def latency_sample(self, rng: random.Random) -> float:
        """Milliseconds. Lognormal so the tail is fat and always positive."""
        mu = self.latency_mean_ms
        sigma = max(1e-6, self.latency_stddev_ms)
        sample = rng.lognormvariate(
            mu=__import__("math").log(max(mu, 1e-6)) - 0.5 * (sigma / mu) ** 2,
            sigma=min(1.5, sigma / mu),
        )
        return min(self.latency_max_ms, max(1.0, sample))


@dataclass(slots=True)
class RestingOrder:
    """A post-only order waiting in the queue."""

    order: Order
    price: Decimal
    remaining: Decimal
    queue_ahead: Decimal          # size ahead of us at this price
    placed_ns: int
    filled: Decimal = ZERO

    def drain(self, volume: Decimal) -> Decimal:
        """
        Consume ``volume`` of trading at our price level.

        Volume first eats the queue ahead of us, and only what is left fills our
        order. This is why maker fill probability is so much lower than "did the
        price touch my level" suggests.
        """
        if self.queue_ahead > 0:
            eaten = min(self.queue_ahead, volume)
            self.queue_ahead -= eaten
            volume -= eaten
        if volume <= 0:
            return ZERO
        fill = min(self.remaining, volume)
        self.remaining -= fill
        self.filled += fill
        return fill


class SimulatedExchange(ExchangeAdapter):
    """
    Local matching engine trading against real (live or replayed) book data.

    Market data comes from the shared :class:`BookManager`, so the simulator
    sees exactly what the strategy sees. Only order handling is synthetic.
    """

    def __init__(
        self,
        config: VenueConfig,
        normalizer: SymbolNormalizer,
        books: BookManager,
        *,
        clock: Clock | None = None,
        params: SimulationParams | None = None,
        initial_balances: Mapping[str, Decimal] | None = None,
        source_venue: str = "",
    ) -> None:
        super().__init__(config, normalizer, books, clock=clock)
        self.params = params or SimulationParams()
        self._rng = random.Random(self.params.seed)
        self._resting: dict[str, RestingOrder] = {}
        self._fills: list[Fill] = []
        # Which venue's books to price against, when simulating a named venue.
        self.source_venue = source_venue or config.name

        self.capabilities = AdapterCapabilities(
            streaming_books=True, streaming_fills=True,
            post_only=True, fok=True, ioc=True,
            query_by_client_id=True, cancel_all=True,
            batch_orders=False, fee_in_response=True,
        )

        for code, amount in (initial_balances or {}).items():
            asset = normalizer.asset(code)
            self.set_balance(asset, D(amount))

        self.adverse_moves = 0
        self.spurious_rejects = 0
        self.maker_fills = 0
        self.maker_expiries = 0

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True
        self.books.mark_connected(self.name)
        logger.info("simulated venue %s connected", self.name)

    async def disconnect(self) -> None:
        self._connected = False
        self.books.mark_disconnected(self.name)

    async def load_instruments(self) -> Sequence[Symbol]:
        """
        Instruments mirror whatever the book manager already knows about.

        In paper mode the real adapter loads instruments and the simulator
        borrows them, so the lot steps and min-notionals are the venue's true
        ones -- which is essential, since those constraints are precisely what
        the simulation exists to test.
        """
        return [b.symbol for b in self.books.books_for_venue(self.source_venue)]

    async def subscribe_books(self, symbols: Sequence[Symbol]) -> None:
        return None

    # -- fees --------------------------------------------------------------

    def _fee_for(
        self, symbol: Symbol, side: Side, quantity: Decimal, price: Decimal,
        liquidity: Liquidity,
    ) -> tuple[Decimal, Asset]:
        """
        Fee amount and the asset it is charged in.

        Spot convention: a BUY is charged in the asset received (base), a SELL
        in the asset received (quote). So a BUY of 1.0 BTC at 10 bps delivers
        0.999 BTC, not 1.0 BTC minus a USDT debit. This asymmetry is why the
        cycle rate calculation must apply fees inside the conversion rather than
        subtracting them at the end.
        """
        spec = self.symbol_spec(symbol)
        rate_bps = (
            (spec.maker_fee_bps if spec.maker_fee_bps is not None else self.params.maker_fee_bps)
            if liquidity is Liquidity.MAKER
            else (spec.taker_fee_bps if spec.taker_fee_bps is not None else self.params.taker_fee_bps)
        )
        rate = rate_bps / D(10_000)
        if side is Side.BUY:
            return quantity * rate, symbol.base
        return quantity * price * rate, symbol.quote

    # -- balance management ------------------------------------------------

    def _require_funds(self, order: Order, price: Decimal) -> None:
        if order.side is Side.BUY:
            need = order.quantity * price
            have = self.available(order.symbol.quote)
            if have < need:
                raise InsufficientBalance(
                    f"need {need} {order.symbol.quote} but have {have}",
                    venue=self.name, code="INSUFFICIENT_BALANCE",
                )
        else:
            need = order.quantity
            have = self.available(order.symbol.base)
            if have < need:
                raise InsufficientBalance(
                    f"need {need} {order.symbol.base} but have {have}",
                    venue=self.name, code="INSUFFICIENT_BALANCE",
                )

    def _settle(self, symbol: Symbol, side: Side, quantity: Decimal,
                price: Decimal, fee: Decimal, fee_asset: Asset) -> None:
        base_bal = self.balance(symbol.base)
        quote_bal = self.balance(symbol.quote)
        notional = quantity * price

        if side is Side.BUY:
            received = quantity - (fee if fee_asset == symbol.base else ZERO)
            self.set_balance(symbol.base, base_bal.free + received, base_bal.locked)
            self.set_balance(symbol.quote, quote_bal.free - notional, quote_bal.locked)
        else:
            received = notional - (fee if fee_asset == symbol.quote else ZERO)
            self.set_balance(symbol.base, base_bal.free - quantity, base_bal.locked)
            self.set_balance(symbol.quote, quote_bal.free + received, quote_bal.locked)

    # -- the matching engine -----------------------------------------------

    async def submit(self, order: Order, *, timeout_sec: float = 5.0) -> Order:
        self.validate_order(order)
        self.orders_submitted += 1
        submitted_ns = self.clock.mono_ns()

        # 1. Network latency out.
        latency_ms = self.params.latency_sample(self._rng)
        await self.clock.sleep(latency_ms / 2000.0)

        # 2. Spurious venue rejection.
        if self._rng.random() < self.params.spurious_reject_probability:
            self.spurious_rejects += 1
            self.orders_rejected += 1
            rejected = order.with_status(
                OrderStatus.REJECTED,
                ts_submitted_ns=submitted_ns,
                ts_final_ns=self.clock.mono_ns(),
            )
            self._track(rejected)
            return rejected

        book = self.books.get(
            Symbol(order.symbol.base, order.symbol.quote,
                   self.source_venue, order.symbol.venue_symbol)
        ) or self.books.get(order.symbol)
        if book is None or not book.initialized:
            self.orders_rejected += 1
            return order.with_status(OrderStatus.REJECTED, ts_final_ns=self.clock.mono_ns())

        spec = self.symbol_spec(order.symbol)
        snapshot = book.snapshot()

        if order.order_type is OrderType.POST_ONLY or order.time_in_force is TimeInForce.GTX:
            result = await self._handle_maker(order, snapshot, spec, submitted_ns)
        else:
            result = await self._handle_taker(order, snapshot, spec, submitted_ns, latency_ms)

        # 3. Network latency back.
        await self.clock.sleep(latency_ms / 2000.0)
        self.submit_latency.observe(int(latency_ms * 1e6))
        self._track(result)
        if result.status.any_fill:
            self.orders_filled += 1
        elif result.status is OrderStatus.REJECTED:
            self.orders_rejected += 1
        return result

    async def _handle_taker(
        self, order: Order, snapshot, spec: SymbolSpec,
        submitted_ns: int, latency_ms: float,
    ) -> Order:
        """Marketable order: walk the book, applying adverse selection first."""
        levels = list(snapshot.side(order.side))
        if not levels:
            return order.with_status(OrderStatus.REJECTED, ts_final_ns=self.clock.mono_ns())

        # Adverse selection: the touch may have vanished while we were in flight.
        levels = self._apply_adverse_move(levels, order.side, spec)

        # Respect the limit price -- an IOC limit only fills at or better.
        if order.price is not None and order.price > 0:
            if order.side is Side.BUY:
                levels = [l for l in levels if l.price <= order.price]
            else:
                levels = [l for l in levels if l.price >= order.price]

        if not levels:
            # Priced through: no fill. This is the common IOC outcome and the
            # reason fill probability is the thing worth modelling.
            status = (
                OrderStatus.CANCELED
                if order.time_in_force in (TimeInForce.IOC, TimeInForce.FOK)
                else OrderStatus.OPEN
            )
            return order.with_status(
                status, ts_submitted_ns=submitted_ns, ts_final_ns=self.clock.mono_ns()
            )

        walk = walk_book(
            levels, order.quantity,
            max_levels=8,
            consumption_cap=self.params.liquidity_availability,
        )

        if walk.filled_quantity <= 0:
            return order.with_status(
                OrderStatus.CANCELED, ts_submitted_ns=submitted_ns,
                ts_final_ns=self.clock.mono_ns(),
            )

        # FOK is all-or-nothing.
        if order.time_in_force is TimeInForce.FOK and not walk.complete:
            return order.with_status(
                OrderStatus.CANCELED, ts_submitted_ns=submitted_ns,
                ts_final_ns=self.clock.mono_ns(),
            )

        filled = floor_to_step(walk.filled_quantity, spec.lot_step)
        if filled <= 0:
            return order.with_status(
                OrderStatus.CANCELED, ts_submitted_ns=submitted_ns,
                ts_final_ns=self.clock.mono_ns(),
            )

        price = walk.average_price
        try:
            self._require_funds(
                order.with_status(order.status, quantity=filled), price
            )
        except InsufficientBalance:
            return order.with_status(
                OrderStatus.REJECTED, ts_submitted_ns=submitted_ns,
                ts_final_ns=self.clock.mono_ns(),
            )

        fee, fee_asset = self._fee_for(order.symbol, order.side, filled, price, Liquidity.TAKER)
        self._settle(order.symbol, order.side, filled, price, fee, fee_asset)
        self._record_fill(order, filled, price, fee, fee_asset, Liquidity.TAKER)

        status = OrderStatus.FILLED if filled >= order.quantity else OrderStatus.PARTIALLY_FILLED
        if status is OrderStatus.PARTIALLY_FILLED and order.time_in_force is TimeInForce.IOC:
            # IOC: the unfilled remainder is cancelled, so the order is terminal.
            pass
        return order.with_status(
            status,
            filled_quantity=filled,
            average_price=price,
            fee_paid=fee,
            fee_asset=fee_asset,
            ts_submitted_ns=submitted_ns,
            ts_final_ns=self.clock.mono_ns(),
        )

    def _apply_adverse_move(self, levels, side: Side, spec: SymbolSpec):
        """
        Remove the touch with probability ``adverse_move_probability``.

        Modelled as level removal rather than a price shift because that is what
        actually happens: the resting order is cancelled or taken by someone
        faster, and the next price is the new touch. Removing levels also
        naturally produces the right *distribution* of realized slippage --
        usually zero extra cost, occasionally a full level.
        """
        if self._rng.random() >= self.params.adverse_move_probability:
            return levels
        self.adverse_moves += 1
        n = max(1, int(self._rng.expovariate(1.0 / max(0.5, self.params.adverse_move_ticks_mean))))
        return levels[min(n, len(levels)):] or []

    async def _handle_maker(
        self, order: Order, snapshot, spec: SymbolSpec, submitted_ns: int,
    ) -> Order:
        """
        Post-only order: reject if it would cross, otherwise rest in the queue.

        Returns OPEN; the caller polls via :meth:`fetch_order` while
        :meth:`step_resting` drains the queue. That two-phase shape mirrors the
        live adapters and lets the executor's maker-leg timeout logic be tested.
        """
        price = order.price or ZERO
        if price <= 0:
            return order.with_status(OrderStatus.REJECTED, ts_final_ns=self.clock.mono_ns())

        best_bid, best_ask = snapshot.best_bid, snapshot.best_ask
        crosses = (
            (order.side is Side.BUY and best_ask > 0 and price >= best_ask)
            or (order.side is Side.SELL and best_bid > 0 and price <= best_bid)
        )
        if crosses:
            # Post-only that would take liquidity is cancelled, not filled --
            # this is the behaviour that makes maker legs safe to attempt.
            return order.with_status(
                OrderStatus.CANCELED, ts_submitted_ns=submitted_ns,
                ts_final_ns=self.clock.mono_ns(),
            )

        same_side = snapshot.bids if order.side is Side.BUY else snapshot.asks
        displayed_at_price = next(
            (l.size for l in same_side if l.price == price), ZERO
        )
        resting = RestingOrder(
            order=order,
            price=price,
            remaining=order.quantity,
            queue_ahead=displayed_at_price * self.params.queue_position_fraction,
            placed_ns=self.clock.wall_ns(),
        )
        self._resting[order.client_order_id] = resting
        return order.with_status(
            OrderStatus.OPEN, ts_submitted_ns=submitted_ns,
        )

    def step_resting(self, elapsed_sec: float) -> list[Fill]:
        """
        Advance time for resting maker orders.

        Traded volume at each level is approximated as
        ``displayed_size * drain_rate * elapsed``, which drains the queue ahead
        of us before it fills us. Called by the engine loop each tick.
        """
        produced: list[Fill] = []
        for coid, resting in list(self._resting.items()):
            order = resting.order
            book = self.books.get(order.symbol)
            if book is None or not book.initialized:
                continue

            snapshot = book.snapshot()
            best_bid, best_ask = snapshot.best_bid, snapshot.best_ask

            # If the market has traded through our price, we are filled outright.
            through = (
                (order.side is Side.BUY and best_ask > 0 and best_ask < resting.price)
                or (order.side is Side.SELL and best_bid > 0 and best_bid > resting.price)
            )
            volume = (
                resting.remaining + resting.queue_ahead
                if through
                else self._simulated_volume(snapshot, order.side, resting.price, elapsed_sec)
            )

            filled = resting.drain(volume)
            if filled <= 0:
                continue

            spec = self.symbol_spec(order.symbol)
            filled = floor_to_step(filled, spec.lot_step)
            if filled <= 0:
                continue

            fee, fee_asset = self._fee_for(
                order.symbol, order.side, filled, resting.price, Liquidity.MAKER
            )
            try:
                self._require_funds(order.with_status(order.status, quantity=filled), resting.price)
            except InsufficientBalance:
                del self._resting[coid]
                continue

            self._settle(order.symbol, order.side, filled, resting.price, fee, fee_asset)
            fill = self._record_fill(
                order, filled, resting.price, fee, fee_asset, Liquidity.MAKER
            )
            produced.append(fill)
            self.maker_fills += 1

            total_filled = resting.filled
            status = (
                OrderStatus.FILLED if resting.remaining <= 0 else OrderStatus.PARTIALLY_FILLED
            )
            updated = order.with_status(
                status,
                filled_quantity=total_filled,
                average_price=resting.price,
                fee_paid=fee,
                fee_asset=fee_asset,
                ts_final_ns=self.clock.mono_ns() if status is OrderStatus.FILLED else 0,
            )
            resting.order = updated
            self._track(updated)
            if resting.remaining <= 0:
                del self._resting[coid]
        return produced

    def _simulated_volume(self, snapshot, side: Side, price: Decimal,
                          elapsed_sec: float) -> Decimal:
        """Volume traded at our price level over ``elapsed_sec``."""
        same_side = snapshot.bids if side is Side.BUY else snapshot.asks
        displayed = next((l.size for l in same_side if l.price == price), ZERO)
        if displayed <= 0:
            return ZERO
        return displayed * self.params.queue_drain_rate_per_sec * D(str(elapsed_sec))

    def _record_fill(self, order: Order, quantity: Decimal, price: Decimal,
                     fee: Decimal, fee_asset: Asset, liquidity: Liquidity) -> Fill:
        fill = Fill(
            order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            price=price,
            quantity=quantity,
            fee=fee,
            fee_asset=fee_asset,
            liquidity=liquidity,
            ts_ns=self.clock.wall_ns(),
        )
        self._fills.append(fill)
        return fill

    # -- order management --------------------------------------------------

    async def cancel(self, order: Order) -> Order:
        resting = self._resting.pop(order.client_order_id, None)
        if resting is not None:
            self.maker_expiries += 1
            status = (
                OrderStatus.PARTIALLY_FILLED if resting.filled > 0 else OrderStatus.CANCELED
            )
            cancelled = order.with_status(
                status,
                filled_quantity=resting.filled,
                average_price=resting.price if resting.filled > 0 else ZERO,
                ts_final_ns=self.clock.mono_ns(),
            )
            self._track(cancelled)
            return cancelled
        cancelled = order.with_status(OrderStatus.CANCELED, ts_final_ns=self.clock.mono_ns())
        self._track(cancelled)
        return cancelled

    async def fetch_order(self, order: Order) -> Order:
        resting = self._resting.get(order.client_order_id)
        if resting is not None:
            return resting.order
        return self._open_orders.get(order.client_order_id, order)

    async def fetch_balances(self) -> Mapping[str, Balance]:
        return self.all_balances()

    # -- introspection -----------------------------------------------------

    @property
    def fills(self) -> Sequence[Fill]:
        return list(self._fills)

    def equity_in(self, asset: Asset, prices: Mapping[str, Decimal]) -> Decimal:
        """Mark all balances into ``asset`` using the supplied price map."""
        total = ZERO
        for code, bal in self.all_balances().items():
            if code == asset.code:
                total += bal.total
                continue
            px = prices.get(f"{code}/{asset.code}")
            if px:
                total += bal.total * px
                continue
            inverse = prices.get(f"{asset.code}/{code}")
            if inverse and inverse > 0:
                total += bal.total / inverse
        return total

    def stats(self) -> dict[str, object]:
        base = super().stats()
        base.update({
            "simulated": True,
            "adverse_moves": self.adverse_moves,
            "spurious_rejects": self.spurious_rejects,
            "maker_fills": self.maker_fills,
            "maker_expiries": self.maker_expiries,
            "resting_orders": len(self._resting),
            "total_fills": len(self._fills),
            "maker_fill_rate": (
                self.maker_fills / (self.maker_fills + self.maker_expiries)
                if (self.maker_fills + self.maker_expiries) else 0.0
            ),
        })
        return base
