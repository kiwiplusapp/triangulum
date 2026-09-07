"""
Cycle sizing: turning a detected opportunity into executable order quantities.

This module is where most of a naive arbitrage bot's theoretical profit goes to
die, because it is where four constraints collide:

**Depth.** The edge was computed at a reference notional. At any other size the
rate differs, and it always differs *against* you.

**Lot quantization.** Every leg's quantity must land exactly on the venue's lot
grid. Rounding down loses, on average, half a lot -- see
``decimal_math.quantization_drag_bps``. At $100 that term is often larger than
the entire edge.

**Min notional.** Every leg must clear the venue minimum. A cycle whose second
leg falls below it is not a smaller opportunity, it is no opportunity: the
venue rejects the order and you are left holding leg one.

**Propagation.** Leg 2's input is leg 1's *actual* output, not its planned one.
Rounding leg 1 down means leg 2 has less to work with, which may round it down
further. The shortfall compounds forward, so sizing must be computed as a
forward pass and then *verified* backwards.

The algorithm:

    1. Start with the capital we are willing to commit.
    2. Forward pass: for each leg, walk the book for what the running amount
       buys, apply the fee, quantize to the lot grid, carry the result forward.
    3. If any leg fails min-notional or quantizes to zero, the cycle is
       unroutable at this size -- try a larger size once, then abandon.
    4. Compute the realized end amount and the true net edge, which is the only
       number worth acting on.

Step 4's answer is routinely *negative* on cycles whose gross edge looked
positive. That is the module working correctly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from triangulum.core.decimal_math import (
    D, ONE, ZERO, bps, ceil_to_step, floor_to_step, quantization_drag_bps, safe_div,
)
from triangulum.core.types import (
    Asset, Leg, Liquidity, OrderType, Side, Symbol, TimeInForce,
)
from triangulum.exchanges.spec import SymbolSpec
from triangulum.execution.fees import FeeEngine
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.orderbook import WalkResult, walk_book, walk_book_for_notional

logger = logging.getLogger(__name__)

__all__ = ["SizedLeg", "SizingResult", "CycleSizer", "SizingFailure"]


class SizingFailure:
    NONE = ""
    NO_BOOK = "no_book"
    THIN = "insufficient_depth"
    MIN_NOTIONAL = "below_min_notional"
    LOT_ZERO = "quantizes_to_zero"
    MAX_QTY = "above_max_quantity"
    NEGATIVE_EDGE = "negative_after_costs"
    NO_CAPITAL = "no_capital"


@dataclass(slots=True)
class SizedLeg:
    """One leg with concrete, venue-legal execution parameters."""

    leg: Leg
    spec: SymbolSpec

    input_amount: Decimal          # units of from_asset committed
    quantity: Decimal              # base units, lot-aligned
    limit_price: Decimal
    expected_output: Decimal       # units of to_asset, net of fee
    expected_fee: Decimal
    fee_asset: Asset

    walk: WalkResult
    touch_price: Decimal
    slippage_bps: Decimal
    drag_bps: Decimal
    notional_quote: Decimal
    liquidity: Liquidity = Liquidity.TAKER

    @property
    def effective_rate(self) -> Decimal:
        return safe_div(self.expected_output, self.input_amount)

    @property
    def levels(self) -> int:
        return self.walk.levels_consumed


@dataclass(slots=True)
class SizingResult:
    """Outcome of sizing a whole cycle."""

    legs: tuple[SizedLeg, ...] = ()
    start_asset: Asset | None = None
    start_amount: Decimal = ZERO
    end_amount: Decimal = ZERO

    gross_edge_bps: Decimal = ZERO
    fee_bps: Decimal = ZERO
    slippage_bps: Decimal = ZERO
    drag_bps: Decimal = ZERO
    net_edge_bps: Decimal = ZERO

    ok: bool = False
    failure: str = SizingFailure.NONE
    failed_leg: int = -1
    detail: str = ""

    @property
    def profit(self) -> Decimal:
        return self.end_amount - self.start_amount

    def summary(self) -> str:
        if not self.ok:
            return f"UNROUTABLE ({self.failure} at leg {self.failed_leg}): {self.detail}"
        return (
            f"{self.start_amount:.6f} -> {self.end_amount:.6f} "
            f"({self.net_edge_bps:+.2f} bps net = {self.gross_edge_bps:+.2f} gross "
            f"- {self.fee_bps:.2f} fees - {self.slippage_bps:.2f} slip "
            f"- {self.drag_bps:.2f} drag)"
        )


class CycleSizer:
    """Sizes cycles against live books and real venue rules."""

    def __init__(
        self,
        books: BookManager,
        fees: FeeEngine,
        *,
        specs: dict[str, SymbolSpec] | None = None,
        max_levels: int = 8,
        consumption_cap: Decimal = D("0.35"),
        min_notional_buffer: Decimal = D("1.15"),
        taker_offset_ticks: int = 2,
        maker_offset_ticks: int = 1,
    ) -> None:
        self.books = books
        self.fees = fees
        self._specs = specs or {}
        self.max_levels = max_levels
        self.consumption_cap = consumption_cap
        self.min_notional_buffer = min_notional_buffer
        self.taker_offset_ticks = taker_offset_ticks
        self.maker_offset_ticks = maker_offset_ticks

    def register_spec(self, symbol: Symbol, spec: SymbolSpec) -> None:
        self._specs[symbol.key] = spec

    def spec_for(self, symbol: Symbol) -> SymbolSpec:
        spec = self._specs.get(symbol.key)
        if spec is None:
            spec = SymbolSpec(
                venue_symbol=symbol.venue_symbol,
                base=symbol.base.code,
                quote=symbol.quote.code,
            )
            self._specs[symbol.key] = spec
        return spec

    # -- the forward pass --------------------------------------------------

    def size(
        self,
        legs: Sequence[Leg],
        start_asset: Asset,
        start_amount: Decimal,
        *,
        liquidities: Sequence[Liquidity] = (),
        now_ns: int = 0,
    ) -> SizingResult:
        """
        Size a cycle, propagating each leg's realized output into the next.

        ``liquidities`` lets the caller declare which legs will be worked as
        maker; it changes the fee and, downstream, the price the leg is quoted
        at. Defaults to all-taker.
        """
        if start_amount <= 0:
            return SizingResult(failure=SizingFailure.NO_CAPITAL, failed_leg=0)

        liquidity_plan = list(liquidities) or [Liquidity.TAKER] * len(legs)
        sized: list[SizedLeg] = []
        running = start_amount
        total_fee_bps = ZERO
        total_slip_bps = ZERO
        total_drag_bps = ZERO

        for index, leg in enumerate(legs):
            liquidity = liquidity_plan[index] if index < len(liquidity_plan) else Liquidity.TAKER
            sized_leg, failure, detail = self._size_leg(leg, running, liquidity, now_ns)
            if sized_leg is None:
                return SizingResult(
                    start_asset=start_asset,
                    start_amount=start_amount,
                    ok=False, failure=failure, failed_leg=index, detail=detail,
                )
            sized.append(sized_leg)
            # Fee cost of this leg, measured against the leg's own gross output
            # so it is directly additive with the slippage and drag terms.
            gross_output = sized_leg.expected_output + sized_leg.expected_fee
            total_fee_bps += bps(safe_div(sized_leg.expected_fee, gross_output))
            total_slip_bps += sized_leg.slippage_bps
            total_drag_bps += sized_leg.drag_bps
            running = sized_leg.expected_output

        end_amount = running
        net_bps = bps(safe_div(end_amount - start_amount, start_amount))

        # Gross edge = what the cycle would have returned with no costs at all.
        # Drag belongs in this sum: quantization loss is already baked into
        # ``end_amount`` (each leg was floored to its lot grid), so omitting it
        # here makes the reconstructed gross inconsistent across capital levels
        # -- it would appear to *change* with size when it is a property of the
        # prices alone.
        gross_bps = net_bps + total_fee_bps + total_slip_bps + total_drag_bps

        result = SizingResult(
            legs=tuple(sized),
            start_asset=start_asset,
            start_amount=start_amount,
            end_amount=end_amount,
            gross_edge_bps=gross_bps,
            fee_bps=total_fee_bps,
            slippage_bps=total_slip_bps,
            drag_bps=total_drag_bps,
            net_edge_bps=net_bps,
            ok=True,
        )
        if net_bps <= 0:
            result.ok = False
            result.failure = SizingFailure.NEGATIVE_EDGE
            result.detail = (
                f"net {net_bps:.2f} bps after {total_fee_bps:.2f} fees, "
                f"{total_slip_bps:.2f} slippage, {total_drag_bps:.2f} drag"
            )
        return result

    def _size_leg(
        self, leg: Leg, input_amount: Decimal, liquidity: Liquidity, now_ns: int,
    ) -> tuple[SizedLeg | None, str, str]:
        book = self.books.get(leg.symbol)
        if book is None or not book.initialized or book.crossed:
            return None, SizingFailure.NO_BOOK, f"no usable book for {leg.symbol.key}"

        spec = self.spec_for(leg.symbol)
        snapshot = book.snapshot(self.max_levels)
        levels = snapshot.side(leg.side)
        if not levels:
            return None, SizingFailure.THIN, "empty book side"

        touch = levels[0].price
        fee_rate = self.fees.rate_bps(leg.symbol.venue, liquidity, now_ns) / D(10_000)

        if leg.side is Side.BUY:
            # Spend ``input_amount`` of quote to acquire base.
            walk = walk_book_for_notional(
                levels, input_amount,
                max_levels=self.max_levels, consumption_cap=self.consumption_cap,
            )
            if walk.filled_quantity <= 0:
                return None, SizingFailure.THIN, "no depth for requested notional"
            raw_quantity = walk.filled_quantity
            quantity = floor_to_step(raw_quantity, spec.lot_step)
            if quantity <= 0:
                return None, SizingFailure.LOT_ZERO, (
                    f"{raw_quantity} quantizes to zero on a {spec.lot_step} grid"
                )
            price = walk.average_price
            notional = quantity * price
            fee_amount = quantity * fee_rate
            output = quantity - fee_amount       # fee charged in base
            fee_asset = leg.symbol.base
            limit_price = _aggressive_price(
                touch, spec.tick_size, leg.side, liquidity,
                self.taker_offset_ticks, self.maker_offset_ticks,
            )
        else:
            # Spend ``input_amount`` of base to receive quote.
            raw_quantity = input_amount
            quantity = floor_to_step(raw_quantity, spec.lot_step)
            if quantity <= 0:
                return None, SizingFailure.LOT_ZERO, (
                    f"{raw_quantity} quantizes to zero on a {spec.lot_step} grid"
                )
            walk = walk_book(
                levels, quantity,
                max_levels=self.max_levels, consumption_cap=self.consumption_cap,
            )
            if walk.filled_quantity <= 0:
                return None, SizingFailure.THIN, "no depth for requested quantity"
            if walk.exhausted:
                # Re-quantize down to what the book can actually absorb.
                quantity = floor_to_step(walk.filled_quantity, spec.lot_step)
                if quantity <= 0:
                    return None, SizingFailure.THIN, "book too thin after quantization"
            price = walk.average_price
            notional = quantity * price
            fee_amount = notional * fee_rate
            output = notional - fee_amount       # fee charged in quote
            fee_asset = leg.symbol.quote
            limit_price = _aggressive_price(
                touch, spec.tick_size, leg.side, liquidity,
                self.taker_offset_ticks, self.maker_offset_ticks,
            )

        if spec.max_quantity and quantity > spec.max_quantity:
            return None, SizingFailure.MAX_QTY, (
                f"quantity {quantity} exceeds venue max {spec.max_quantity}"
            )
        if spec.min_quantity and quantity < spec.min_quantity:
            return None, SizingFailure.MIN_NOTIONAL, (
                f"quantity {quantity} below venue min {spec.min_quantity}"
            )
        required = spec.min_notional * self.min_notional_buffer
        if notional < required:
            return None, SizingFailure.MIN_NOTIONAL, (
                f"notional {notional:.4f} below {spec.min_notional} x "
                f"{self.min_notional_buffer} buffer = {required:.4f}"
            )

        drag = quantization_drag_bps(notional, spec.lot_step, touch)

        return SizedLeg(
            leg=leg, spec=spec,
            input_amount=input_amount,
            quantity=quantity,
            limit_price=limit_price,
            expected_output=output,
            expected_fee=fee_amount,
            fee_asset=fee_asset,
            walk=walk,
            touch_price=touch,
            slippage_bps=walk.slippage_bps(touch),
            drag_bps=drag,
            notional_quote=notional,
            liquidity=liquidity,
        ), SizingFailure.NONE, ""

    # -- capital search ----------------------------------------------------

    def size_best(
        self,
        legs: Sequence[Leg],
        start_asset: Asset,
        available: Decimal,
        *,
        fractions: Sequence[Decimal] = (D("0.95"), D("0.70"), D("0.50"), D("0.30")),
        liquidities: Sequence[Liquidity] = (),
        now_ns: int = 0,
    ) -> SizingResult:
        """
        Try progressively smaller commitments and keep the best net edge.

        Smaller is not automatically better: a smaller cycle suffers *more*
        quantization drag (the half-lot cost is fixed while the notional
        shrinks) but *less* book-walk slippage. The optimum is interior, which
        is why this searches rather than assuming.
        """
        best: SizingResult | None = None
        for fraction in fractions:
            amount = available * fraction
            if amount <= 0:
                continue
            result = self.size(
                legs, start_asset, amount, liquidities=liquidities, now_ns=now_ns
            )
            if not result.ok:
                if best is None:
                    best = result
                continue
            if best is None or not best.ok or result.net_edge_bps > best.net_edge_bps:
                best = result
        return best or SizingResult(failure=SizingFailure.NO_CAPITAL)


def _aggressive_price(
    touch: Decimal,
    tick: Decimal,
    side: Side,
    liquidity: Liquidity,
    taker_ticks: int,
    maker_ticks: int,
) -> Decimal:
    """
    Limit price for the leg.

    Taker legs are priced *through* the touch by a couple of ticks. Pricing
    exactly at the touch looks cheaper but under-fills constantly: by the time
    the order lands the touch has often moved one tick, and a no-fill on leg 1
    costs the whole cycle while two ticks of a typical instrument costs a
    fraction of a basis point.

    Maker legs are priced *behind* the touch so the venue accepts them as
    post-only rather than cancelling them for crossing.
    """
    if tick <= 0:
        return touch
    if liquidity is Liquidity.MAKER:
        offset = tick * D(maker_ticks)
        return touch - offset if side is Side.BUY else touch + offset
    offset = tick * D(taker_ticks)
    return touch + offset if side is Side.BUY else touch - offset
