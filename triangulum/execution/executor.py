"""
Cycle executor.

The hardest constraint in triangular arbitrage: **there is no atomic multi-leg
order.** No spot venue offers one. You send leg 1, wait, send leg 2, wait, send
leg 3. Between each pair of legs you are holding a directional position nobody
asked for, exposed to the market for as long as the next leg takes.

That exposure is the entire risk of the strategy, and it is why this file is
structured around failure rather than success:

**Fail before committing, not after.** Every check that can be made before the
first order is made before the first order: freshness, sizing, min-notional,
balance, risk limits, EV. An abort at that stage costs nothing. An abort after
leg 1 costs a position.

**Order the legs by unwind cost.** Legs are not interchangeable. The one most
likely to fail should go *first* (fail cheaply), and the one whose position is
most expensive to unwind should go *last* (least time held). These two
heuristics usually agree; when they conflict, unwind cost wins, because a failed
first leg costs one spread while a stranded illiquid position costs whatever the
market decides.

**A latency budget that shrinks.** The whole cycle gets a fixed budget. Each leg
consumes from it. When it runs out mid-cycle, we stop opening legs and start
closing them -- a half-built cycle at t+900ms is a directional bet, and holding
it hoping the third leg fills is how a 2 bps strategy takes a 200 bps loss.

**Partial fills are the normal case.** An IOC that fills 60% is not an error. The
remaining legs are re-sized to the *actual* amount received, not the planned
one. Sizing leg 2 off leg 1's planned output is a bug that only manifests under
partial fills -- which is to say, exactly when it hurts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Sequence

from triangulum.core.clock import Clock, LatencyBudget, SystemClock
from triangulum.core.decimal_math import D, ONE, ZERO, bps, safe_div
from triangulum.core.errors import (
    CycleAborted, ExecutionError, LegFailed, OrderRejected, VenueError,
)
from triangulum.core.eventbus import EventBus, Topics
from triangulum.core.types import (
    Asset, CycleOutcome, CyclePlan, ExecutionResult, Fill, LegPlan, Liquidity,
    Order, OrderStatus, OrderType, Side, TimeInForce, new_id,
)
from triangulum.exchanges.base import ExchangeAdapter
from triangulum.execution.fees import FeeEngine
from triangulum.execution.sizing import CycleSizer
from triangulum.execution.unwinder import Unwinder

logger = logging.getLogger(__name__)

__all__ = ["CycleExecutor", "ExecutionContext"]


@dataclass(slots=True)
class ExecutionContext:
    """Mutable state carried through one cycle's execution."""

    plan: CyclePlan
    budget: LatencyBudget
    fills: list[Fill] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    legs_filled: int = 0
    running_amount: Decimal = ZERO
    current_asset: Asset | None = None
    aborted_reason: str = ""
    ts_start_ns: int = 0

    @property
    def committed(self) -> bool:
        return self.legs_filled > 0

    @property
    def complete(self) -> bool:
        return self.legs_filled == len(self.plan.legs)


class CycleExecutor:
    """Executes a sized cycle plan, leg by leg, under a latency budget."""

    def __init__(
        self,
        adapters: Mapping[str, ExchangeAdapter],
        sizer: CycleSizer,
        fees: FeeEngine,
        *,
        unwinder: Unwinder | None = None,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        cycle_budget_ms: float = 900.0,
        leg_timeout_ms: float = 250.0,
        maker_leg_timeout_ms: float = 1500.0,
        maker_max_requeues: int = 2,
        dry_run: bool = False,
    ) -> None:
        self.adapters = adapters
        self.sizer = sizer
        self.fees = fees
        self.unwinder = unwinder
        self.bus = bus
        self.clock = clock or SystemClock()
        self.cycle_budget_ms = cycle_budget_ms
        self.leg_timeout_ms = leg_timeout_ms
        self.maker_leg_timeout_ms = maker_leg_timeout_ms
        self.maker_max_requeues = maker_max_requeues
        self.dry_run = dry_run

        self.cycles_attempted = 0
        self.cycles_completed = 0
        self.cycles_aborted = 0
        self.cycles_unwound = 0
        self.cycles_stuck = 0
        self._in_flight: set[str] = set()

    # -- entry point -------------------------------------------------------

    async def execute(self, plan: CyclePlan) -> ExecutionResult:
        """Execute a cycle. Never raises -- failures come back as a result."""
        plan.validate()
        self.cycles_attempted += 1
        self._in_flight.add(plan.cycle_id)

        context = ExecutionContext(
            plan=plan,
            budget=LatencyBudget.from_ms(self.cycle_budget_ms, self.clock),
            running_amount=plan.start_amount,
            current_asset=plan.start_asset,
            ts_start_ns=self.clock.wall_ns(),
        )
        self._publish(Topics.CYCLE_STARTED, plan)

        try:
            await self._run_legs(context)
        except CycleAborted as exc:
            context.aborted_reason = str(exc)
        except asyncio.CancelledError:
            context.aborted_reason = "cancelled"
            raise
        except Exception as exc:
            logger.exception("cycle %s failed unexpectedly", plan.cycle_id)
            context.aborted_reason = f"{type(exc).__name__}: {exc}"
        finally:
            self._in_flight.discard(plan.cycle_id)

        result = await self._finalize(context)
        self._publish(
            Topics.CYCLE_COMPLETED
            if result.outcome is CycleOutcome.COMPLETED
            else Topics.CYCLE_FAILED,
            result,
        )
        return result

    # -- leg loop ----------------------------------------------------------

    async def _run_legs(self, ctx: ExecutionContext) -> None:
        legs = ctx.plan.legs
        for index, leg_plan in enumerate(legs):
            remaining = len(legs) - index

            if ctx.budget.expired:
                raise CycleAborted(
                    f"latency budget exhausted before leg {index} "
                    f"({ctx.budget.elapsed_ns / 1e6:.0f}ms elapsed)"
                )

            # Re-size against the amount we ACTUALLY hold. On leg 0 this equals
            # the plan; on later legs it reflects partial fills upstream.
            effective = self._resize_leg(leg_plan, ctx)
            if effective is None:
                raise CycleAborted(
                    f"leg {index} unroutable at realized size {ctx.running_amount}"
                )

            timeout = min(
                ctx.budget.slice_for_leg(remaining),
                (self.maker_leg_timeout_ms if effective.order_type is OrderType.POST_ONLY
                 else self.leg_timeout_ms) / 1000.0,
            )
            if timeout <= 0:
                raise CycleAborted(f"no time left for leg {index}")

            order = await self._execute_leg(effective, index, timeout, ctx)

            if order is None or not order.status.any_fill:
                if index == 0:
                    # Nothing committed. The cheapest possible failure.
                    raise CycleAborted(f"leg 0 did not fill; no capital committed")
                raise LegFailed(
                    f"leg {index} did not fill after leg(s) already executed",
                    leg_index=index,
                )

            ctx.legs_filled += 1
            received = self._received_amount(effective, order)
            ctx.running_amount = received
            ctx.current_asset = effective.leg.to_asset
            self._publish(Topics.CYCLE_LEG_FILLED, {
                "cycle_id": ctx.plan.cycle_id,
                "leg": index,
                "filled": str(order.filled_quantity),
                "price": str(order.average_price),
                "received": str(received),
            })

    def _resize_leg(self, leg_plan: LegPlan, ctx: ExecutionContext) -> LegPlan | None:
        """
        Rebuild a leg's parameters against the amount actually held.

        Skipped for leg 0 when the running amount still equals the plan, which
        is the common case and saves a book walk in the hot path.
        """
        if ctx.legs_filled == 0 and ctx.running_amount == ctx.plan.start_amount:
            return leg_plan

        liquidity = (
            Liquidity.MAKER if leg_plan.order_type is OrderType.POST_ONLY
            else Liquidity.TAKER
        )
        result = self.sizer.size(
            [leg_plan.leg],
            ctx.current_asset or leg_plan.leg.from_asset,
            ctx.running_amount,
            liquidities=[liquidity],
            now_ns=self.clock.wall_ns(),
        )
        if not result.ok or not result.legs:
            logger.info(
                "cycle %s: leg re-size failed (%s) at amount %s",
                ctx.plan.cycle_id, result.failure, ctx.running_amount,
            )
            return None

        sized = result.legs[0]
        return LegPlan(
            leg=sized.leg,
            input_amount=sized.input_amount,
            expected_output=sized.expected_output,
            quantity=sized.quantity,
            limit_price=sized.limit_price,
            order_type=leg_plan.order_type,
            time_in_force=leg_plan.time_in_force,
            expected_fee=sized.expected_fee,
            expected_fee_asset=sized.fee_asset,
            expected_slippage_bps=sized.slippage_bps,
            levels_consumed=sized.levels,
            book_sequence=leg_plan.book_sequence,
        )

    async def _execute_leg(
        self, leg_plan: LegPlan, index: int, timeout_sec: float, ctx: ExecutionContext,
    ) -> Order | None:
        adapter = self.adapters.get(leg_plan.leg.venue)
        if adapter is None:
            raise CycleAborted(f"no adapter for venue {leg_plan.leg.venue}")

        order = Order(
            symbol=leg_plan.leg.symbol,
            side=leg_plan.leg.side,
            quantity=leg_plan.quantity,
            order_type=leg_plan.order_type,
            price=leg_plan.limit_price,
            time_in_force=leg_plan.time_in_force,
            client_order_id=f"{ctx.plan.cycle_id}-L{index}",
            ts_created_ns=self.clock.wall_ns(),
            tag=f"cycle:{ctx.plan.cycle_id}",
        )

        if self.dry_run:
            logger.info("DRY RUN leg %d: %s", index, _describe(order))
            return order.with_status(
                OrderStatus.FILLED,
                filled_quantity=order.quantity,
                average_price=order.price or ZERO,
            )

        self._publish(Topics.ORDER_SUBMITTED, order)
        try:
            if leg_plan.order_type is OrderType.POST_ONLY:
                result = await self._work_maker_leg(adapter, order, timeout_sec)
            else:
                result = await asyncio.wait_for(
                    adapter.submit(order, timeout_sec=timeout_sec), timeout=timeout_sec + 1.0
                )
        except asyncio.TimeoutError:
            # We do not know whether it landed. Ask, rather than assume.
            logger.warning("leg %d timed out; querying venue for true state", index)
            try:
                result = await adapter.fetch_order(order)
            except VenueError:
                raise LegFailed(
                    f"leg {index} timed out and its state is unknown",
                    leg_index=index,
                ) from None
        except OrderRejected as exc:
            logger.info("leg %d rejected: %s", index, exc)
            self._publish(Topics.ORDER_REJECTED, {"order": order, "error": str(exc)})
            return None
        except VenueError as exc:
            logger.warning("leg %d venue error: %s", index, exc)
            return None

        ctx.orders.append(result)
        if result.status.any_fill:
            fill = Fill(
                order_id=result.client_order_id,
                symbol=result.symbol,
                side=result.side,
                price=result.average_price,
                quantity=result.filled_quantity,
                fee=result.fee_paid,
                fee_asset=result.fee_asset or result.symbol.quote,
                liquidity=(
                    Liquidity.MAKER if leg_plan.order_type is OrderType.POST_ONLY
                    else Liquidity.TAKER
                ),
                ts_ns=self.clock.wall_ns(),
            )
            ctx.fills.append(fill)
            self.fees.record_fill(fill, self.clock.wall_ns())
            self.fees.check_estimate(fill, leg_plan.expected_fee)
            self._publish(Topics.ORDER_FILLED, fill)
        return result

    async def _work_maker_leg(
        self, adapter: ExchangeAdapter, order: Order, timeout_sec: float,
    ) -> Order:
        """
        Post-only leg: rest, wait, requeue if the market moves away.

        The maker leg is where the fee saving lives and where the fill risk
        lives. Requeueing chases the touch a bounded number of times; past that
        we accept the no-fill rather than chasing indefinitely, because a maker
        leg that keeps needing to be repriced is a market that is moving, and a
        moving market is exactly when a stale cycle turns into a loss.
        """
        deadline = self.clock.mono_ns() + int(timeout_sec * 1e9)
        current = order

        for attempt in range(self.maker_max_requeues + 1):
            current = await adapter.submit(current, timeout_sec=timeout_sec)
            if current.status.terminal and current.status.any_fill:
                return current
            if current.status is OrderStatus.REJECTED:
                return current

            while self.clock.mono_ns() < deadline:
                await self.clock.sleep(0.02)
                # The simulator drains resting queues on this call; live
                # adapters ignore it and rely on fetch_order.
                stepper = getattr(adapter, "step_resting", None)
                if stepper is not None:
                    stepper(0.02)
                current = await adapter.fetch_order(current)
                if current.status.any_fill:
                    return current
                if current.status.terminal:
                    break

            if self.clock.mono_ns() >= deadline:
                break
            if attempt < self.maker_max_requeues:
                await adapter.cancel(current)
                current = Order(
                    symbol=order.symbol, side=order.side, quantity=order.remaining,
                    order_type=order.order_type, price=order.price,
                    time_in_force=order.time_in_force,
                    client_order_id=f"{order.client_order_id}-r{attempt + 1}",
                    ts_created_ns=self.clock.wall_ns(), tag=order.tag,
                )

        if not current.status.terminal:
            current = await adapter.cancel(current)
        return current

    def _received_amount(self, leg_plan: LegPlan, order: Order) -> Decimal:
        """
        How much of the target asset a fill actually delivered.

        BUY:  filled base quantity minus the base-denominated fee.
        SELL: filled notional minus the quote-denominated fee.
        """
        if order.side is Side.BUY:
            gross = order.filled_quantity
            fee = order.fee_paid if (order.fee_asset == order.symbol.base) else ZERO
            return gross - fee
        gross = order.filled_quantity * order.average_price
        fee = order.fee_paid if (order.fee_asset == order.symbol.quote) else ZERO
        return gross - fee

    # -- finalization ------------------------------------------------------

    async def _finalize(self, ctx: ExecutionContext) -> ExecutionResult:
        plan = ctx.plan
        latency = self.clock.mono_ns() - (ctx.budget.start_ns or self.clock.mono_ns())
        total_fees = sum((f.fee for f in ctx.fills), ZERO)

        if ctx.complete:
            self.cycles_completed += 1
            realized = ctx.running_amount
            pnl = realized - plan.start_amount
            return ExecutionResult(
                cycle_id=plan.cycle_id,
                outcome=CycleOutcome.COMPLETED,
                plan=plan,
                fills=tuple(ctx.fills),
                realized_start_amount=plan.start_amount,
                realized_end_amount=realized,
                realized_pnl=pnl,
                realized_pnl_bps=bps(safe_div(pnl, plan.start_amount)),
                total_fees=total_fees,
                legs_filled=ctx.legs_filled,
                latency_ns=latency,
                ts_start_ns=ctx.ts_start_ns,
                ts_end_ns=self.clock.wall_ns(),
            )

        if not ctx.committed:
            self.cycles_aborted += 1
            return ExecutionResult(
                cycle_id=plan.cycle_id,
                outcome=CycleOutcome.ABORTED_PRE_TRADE,
                plan=plan,
                error=ctx.aborted_reason or "aborted before committing capital",
                latency_ns=latency,
                ts_start_ns=ctx.ts_start_ns,
                ts_end_ns=self.clock.wall_ns(),
            )

        # Committed but incomplete: we are holding something unintended.
        residual = {
            (ctx.current_asset.code if ctx.current_asset else "?"): ctx.running_amount
        }
        outcome = CycleOutcome.PARTIAL_STUCK
        realized_end = ZERO

        if self.unwinder is not None:
            unwind = await self.unwinder.unwind(
                asset=ctx.current_asset,
                amount=ctx.running_amount,
                target=plan.start_asset,
                budget=ctx.budget,
                reason=ctx.aborted_reason or "leg failure",
                cycle_id=plan.cycle_id,
            )
            if unwind.success:
                self.cycles_unwound += 1
                outcome = CycleOutcome.PARTIAL_UNWOUND
                realized_end = unwind.recovered_amount
                residual = {}
                ctx.fills.extend(unwind.fills)
            else:
                self.cycles_stuck += 1
                logger.critical(
                    "cycle %s STUCK holding %s %s -- unwind failed: %s",
                    plan.cycle_id, ctx.running_amount,
                    ctx.current_asset, unwind.error,
                )
        else:
            self.cycles_stuck += 1

        pnl = realized_end - plan.start_amount if realized_end > 0 else ZERO
        return ExecutionResult(
            cycle_id=plan.cycle_id,
            outcome=outcome,
            plan=plan,
            fills=tuple(ctx.fills),
            realized_start_amount=plan.start_amount,
            realized_end_amount=realized_end,
            realized_pnl=pnl,
            realized_pnl_bps=bps(safe_div(pnl, plan.start_amount)) if pnl else ZERO,
            total_fees=total_fees,
            legs_filled=ctx.legs_filled,
            latency_ns=latency,
            error=ctx.aborted_reason,
            ts_start_ns=ctx.ts_start_ns,
            ts_end_ns=self.clock.wall_ns(),
            residual_inventory=residual,
        )

    # -- plumbing ----------------------------------------------------------

    def _publish(self, topic: str, payload: object) -> None:
        if self.bus is not None:
            self.bus.publish(topic, payload, ts_ns=self.clock.wall_ns(), source="executor")

    @property
    def in_flight(self) -> int:
        return len(self._in_flight)

    def stats(self) -> dict[str, object]:
        attempted = max(1, self.cycles_attempted)
        return {
            "attempted": self.cycles_attempted,
            "completed": self.cycles_completed,
            "aborted_pre_trade": self.cycles_aborted,
            "unwound": self.cycles_unwound,
            "stuck": self.cycles_stuck,
            "completion_rate": round(self.cycles_completed / attempted, 4),
            "in_flight": self.in_flight,
            "dry_run": self.dry_run,
        }


def _describe(order: Order) -> str:
    return (
        f"{order.side.value} {order.quantity} {order.symbol.canonical} "
        f"@ {order.price} [{order.order_type.value}/{order.time_in_force.value}]"
    )
