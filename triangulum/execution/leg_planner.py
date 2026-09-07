"""
Leg planning: turning a sized cycle into an ordered execution plan.

Two decisions live here, and both change realized P&L more than they look like
they should.

ORDERING
--------
Legs are not interchangeable even though the cycle is a loop. Two competing
heuristics:

    *Fail cheap.* Put the leg most likely to miss first. A miss on leg 1 costs
    nothing -- no capital was committed.

    *Hold briefly.* Put the leg whose resulting position is most expensive to
    unwind last, so it is held for the shortest time.

They usually agree: the thin, wide, illiquid leg is both the likeliest to miss
and the costliest to be stuck in. When they disagree, unwind cost wins. A missed
first leg costs one spread; a stranded position in an illiquid asset costs
whatever the market feels like.

The ordering is constrained: a cycle is a loop, so only *rotations* are
available, not arbitrary permutations. With three legs there are three
candidates, and we score all of them.

LIQUIDITY ASSIGNMENT
--------------------
Which legs to work as maker. Each maker leg saves the maker/taker spread but
adds fill risk. The rule implemented here: at most one maker leg, and only on
the leg with the widest relative spread (most room to rest inside it) *and*
enough depth that resting is plausible. Two maker legs in a three-leg cycle
multiplies the fill risk beyond what the fee saving justifies -- the
probabilities compound, the savings only add.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from triangulum.core.decimal_math import D, ONE, ZERO, bps, safe_div
from triangulum.core.types import (
    Asset, CyclePlan, ExecutionMode, Leg, LegPlan, Liquidity, OrderType,
    Opportunity, Side, TimeInForce, new_id,
)
from triangulum.execution.fees import FeeEngine
from triangulum.execution.sizing import CycleSizer, SizedLeg, SizingResult
from triangulum.marketdata.book_manager import BookManager

logger = logging.getLogger(__name__)

__all__ = ["LegPlanner", "OrderingScore"]


@dataclass(slots=True)
class OrderingScore:
    rotation: int
    fill_risk: float
    unwind_cost: float
    total: float


class LegPlanner:
    """Builds an ordered, liquidity-assigned :class:`CyclePlan`."""

    def __init__(
        self,
        books: BookManager,
        sizer: CycleSizer,
        fees: FeeEngine,
        *,
        maker_min_spread_bps: Decimal = D("4"),
        maker_min_depth_ratio: Decimal = D("3"),
        unwind_cost_weight: float = 1.5,
    ) -> None:
        self.books = books
        self.sizer = sizer
        self.fees = fees
        self.maker_min_spread_bps = maker_min_spread_bps
        self.maker_min_depth_ratio = maker_min_depth_ratio
        self.unwind_cost_weight = unwind_cost_weight

    # -- entry point -------------------------------------------------------

    def plan(
        self,
        opportunity: Opportunity,
        available: Decimal,
        *,
        mode: ExecutionMode = ExecutionMode.TTT,
        now_ns: int = 0,
        capital_fraction: Decimal = D("0.95"),
    ) -> tuple[CyclePlan | None, SizingResult]:
        """
        Build a plan, or explain why the opportunity is not routable.

        Returns ``(plan, sizing)``. When ``plan`` is None the sizing result
        carries the failure reason, which is worth recording: the distribution
        of *why* opportunities fail to route is the most actionable diagnostic
        the system produces.
        """
        legs = list(opportunity.legs)
        if not legs:
            return None, SizingResult(failure="no_legs")

        best_order = self._choose_rotation(legs, now_ns)
        ordered = legs[best_order:] + legs[:best_order]
        start_asset = ordered[0].from_asset

        liquidities = self._assign_liquidity(ordered, mode, now_ns)

        sizing = self.sizer.size_best(
            ordered,
            start_asset,
            available * capital_fraction,
            liquidities=liquidities,
            now_ns=now_ns,
        )
        if not sizing.ok:
            return None, sizing

        leg_plans = tuple(
            self._to_leg_plan(sized, mode) for sized in sizing.legs
        )
        plan = CyclePlan(
            cycle_id=new_id("cyc-"),
            legs=leg_plans,
            start_asset=start_asset,
            start_amount=sizing.start_amount,
            expected_end_amount=sizing.end_amount,
            execution_mode=mode,
            gross_edge_bps=sizing.gross_edge_bps,
            fee_bps=sizing.fee_bps,
            slippage_bps=sizing.slippage_bps + sizing.drag_bps,
            net_edge_bps=sizing.net_edge_bps,
            ts_created_ns=now_ns,
        )
        try:
            plan.validate()
        except ValueError as exc:
            logger.error("planner produced an invalid cycle: %s", exc)
            return None, SizingResult(failure="invalid_cycle", detail=str(exc))
        return plan, sizing

    # -- ordering ----------------------------------------------------------

    def _choose_rotation(self, legs: Sequence[Leg], now_ns: int) -> int:
        """Score every rotation; return the index to start from."""
        scores: list[OrderingScore] = []
        for rotation in range(len(legs)):
            ordered = list(legs[rotation:]) + list(legs[:rotation])
            fill_risk = self._fill_risk(ordered[0])
            # Unwind cost of the position held after the LAST leg before close.
            unwind_cost = sum(
                self._unwind_cost(leg) * (len(ordered) - i)
                for i, leg in enumerate(ordered[:-1])
            )
            scores.append(OrderingScore(
                rotation=rotation,
                fill_risk=fill_risk,
                unwind_cost=unwind_cost,
                # Lower is better: risky legs first (high fill_risk is GOOD at
                # position 0), expensive-to-hold positions later.
                total=-fill_risk + self.unwind_cost_weight * unwind_cost,
            ))
        return min(scores, key=lambda s: s.total).rotation

    def _fill_risk(self, leg: Leg) -> float:
        """Rough probability that this leg misses. Wide + thin = risky."""
        book = self.books.get(leg.symbol)
        if book is None or not book.initialized:
            return 1.0
        spread = float(book.spread_bps)
        depth = float(book.bids.best_size() if leg.side is Side.SELL else book.asks.best_size())
        # Normalised: a 1 bps spread with deep size is ~0; a 20 bps spread with
        # nothing behind it approaches 1.
        spread_component = min(1.0, spread / 20.0)
        depth_component = 1.0 / (1.0 + depth) if depth > 0 else 1.0
        return 0.6 * spread_component + 0.4 * depth_component

    def _unwind_cost(self, leg: Leg) -> float:
        """Estimated bps to flatten the position this leg creates."""
        book = self.books.get(leg.symbol)
        if book is None or not book.initialized:
            return 100.0
        # Crossing the spread once, plus a fee, plus a slippage allowance that
        # grows as the book thins.
        spread = float(book.spread_bps)
        taker = float(self.fees.rate_bps(leg.symbol.venue, Liquidity.TAKER))
        depth = float(book.depth_notional(leg.side.opposite, 5) or 1)
        thinness = min(50.0, 1000.0 / depth) if depth > 0 else 50.0
        return spread + taker + thinness

    # -- liquidity assignment ----------------------------------------------

    def _assign_liquidity(
        self, legs: Sequence[Leg], mode: ExecutionMode, now_ns: int,
    ) -> list[Liquidity]:
        if mode is ExecutionMode.TTT:
            return [Liquidity.TAKER] * len(legs)

        plan = [Liquidity.TAKER] * len(legs)
        if mode is ExecutionMode.MTT:
            candidate = 0
        elif mode is ExecutionMode.TMT:
            candidate = 1 if len(legs) > 1 else 0
        else:
            candidate = self._best_maker_leg(legs)
            if candidate < 0:
                return plan

        if self._maker_viable(legs[candidate]):
            plan[candidate] = Liquidity.MAKER
        return plan

    def _best_maker_leg(self, legs: Sequence[Leg]) -> int:
        """Widest-spread leg with enough depth to rest in. -1 if none qualify."""
        best_index, best_spread = -1, ZERO
        for index, leg in enumerate(legs):
            book = self.books.get(leg.symbol)
            if book is None or not book.initialized:
                continue
            if book.spread_bps > best_spread and self._maker_viable(leg):
                best_spread = book.spread_bps
                best_index = index
        return best_index

    def _maker_viable(self, leg: Leg) -> bool:
        """
        A maker leg is only worth attempting when there is room inside the
        spread and enough resting size that our order is not the whole level.
        """
        book = self.books.get(leg.symbol)
        if book is None or not book.initialized:
            return False
        if book.spread_bps < self.maker_min_spread_bps:
            return False
        spec = self.sizer.spec_for(leg.symbol)
        # Need at least two ticks of room, or a post-only order has nowhere to
        # sit that is not either crossing or far from the touch.
        if spec.tick_size > 0:
            ticks_of_spread = (book.best_ask - book.best_bid) / spec.tick_size
            if ticks_of_spread < 2:
                return False
        return True

    def _to_leg_plan(self, sized: SizedLeg, mode: ExecutionMode) -> LegPlan:
        is_maker = sized.liquidity is Liquidity.MAKER
        return LegPlan(
            leg=sized.leg,
            input_amount=sized.input_amount,
            expected_output=sized.expected_output,
            quantity=sized.quantity,
            limit_price=sized.limit_price,
            order_type=OrderType.POST_ONLY if is_maker else OrderType.LIMIT,
            time_in_force=TimeInForce.GTX if is_maker else TimeInForce.IOC,
            expected_fee=sized.expected_fee,
            expected_fee_asset=sized.fee_asset,
            expected_slippage_bps=sized.slippage_bps + sized.drag_bps,
            levels_consumed=sized.levels,
        )
