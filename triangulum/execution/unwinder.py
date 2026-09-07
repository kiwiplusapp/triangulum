"""
Emergency unwinder.

When a cycle fails mid-flight you are holding an asset you did not want, in a
size you did not choose, at a moment you did not pick. Every second held is
directional risk on a strategy whose entire premise was that it takes none.

The unwinder's job is to get back to the start asset, fast, and to be honest
when it cannot.

Design commitments:

**Speed over price.** The unwind crosses the spread aggressively -- several
ticks through the touch. Trying to save two basis points on the exit while
holding an unwanted position is exactly backwards: the position's variance per
second dwarfs the spread you are haggling over.

**Direct first, then routed.** If a direct market from the held asset to the
target exists, use it. If not (you are stranded in an asset with no pair against
your base currency), find the shortest path through the graph and traverse it.
Being stranded is rarer than it sounds but it does happen -- typically when a
cycle's second leg fills and the third leg's instrument is halted.

**Escalate, never hide.** A failed unwind is a CRITICAL event that trips the
kill switch. The engine stops opening new cycles and demands a human. An engine
that keeps trading while holding an un-closeable position is not managing risk,
it is compounding it.

**Bounded attempts.** Three tries with increasing aggression, then stop and
escalate. An unwinder that retries forever in a fast market can turn one bad
position into several.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Sequence

from triangulum.core.clock import Clock, LatencyBudget, SystemClock
from triangulum.core.decimal_math import D, ONE, ZERO, bps, floor_to_step, safe_div
from triangulum.core.errors import OrderRejected, UnwindFailed, VenueError
from triangulum.core.eventbus import EventBus, Topics
from triangulum.core.types import (
    Asset, Fill, Leg, Liquidity, Order, OrderStatus, OrderType, Side, Symbol, TimeInForce,
)
from triangulum.exchanges.base import ExchangeAdapter
from triangulum.execution.sizing import CycleSizer
from triangulum.graph.currency_graph import CurrencyGraph
from triangulum.marketdata.book_manager import BookManager

logger = logging.getLogger(__name__)

__all__ = ["Unwinder", "UnwindResult"]


@dataclass(slots=True)
class UnwindResult:
    success: bool
    recovered_amount: Decimal = ZERO
    fills: tuple[Fill, ...] = ()
    hops: int = 0
    cost_bps: Decimal = ZERO
    attempts: int = 0
    error: str = ""
    stranded_asset: str = ""
    stranded_amount: Decimal = ZERO

    def summary(self) -> str:
        if self.success:
            return (
                f"unwound in {self.hops} hop(s) over {self.attempts} attempt(s), "
                f"recovered {self.recovered_amount:.6f} at {self.cost_bps:.1f} bps cost"
            )
        return (
            f"UNWIND FAILED after {self.attempts} attempt(s): {self.error} "
            f"-- holding {self.stranded_amount} {self.stranded_asset}"
        )


class Unwinder:
    """Flattens unwanted inventory back into the start asset."""

    def __init__(
        self,
        adapters: Mapping[str, ExchangeAdapter],
        books: BookManager,
        graph: CurrencyGraph,
        sizer: CycleSizer,
        *,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        max_attempts: int = 3,
        aggressiveness_ticks: int = 5,
        max_hops: int = 3,
    ) -> None:
        self.adapters = adapters
        self.books = books
        self.graph = graph
        self.sizer = sizer
        self.bus = bus
        self.clock = clock or SystemClock()
        self.max_attempts = max_attempts
        self.aggressiveness_ticks = aggressiveness_ticks
        self.max_hops = max_hops

        self.unwinds_attempted = 0
        self.unwinds_succeeded = 0
        self.unwinds_failed = 0
        self.total_cost_bps = ZERO

    async def unwind(
        self,
        *,
        asset: Asset | None,
        amount: Decimal,
        target: Asset,
        budget: LatencyBudget | None = None,
        reason: str = "",
        cycle_id: str = "",
    ) -> UnwindResult:
        """Convert ``amount`` of ``asset`` back into ``target``."""
        if asset is None or amount <= 0:
            return UnwindResult(success=True, recovered_amount=ZERO)
        if asset.code == target.code:
            return UnwindResult(success=True, recovered_amount=amount)

        self.unwinds_attempted += 1
        logger.warning(
            "unwinding %s %s -> %s (cycle=%s, reason=%s)",
            amount, asset.code, target.code, cycle_id, reason,
        )
        self._publish(Topics.CYCLE_UNWOUND, {
            "cycle_id": cycle_id, "asset": asset.code,
            "amount": str(amount), "target": target.code, "reason": reason,
        })

        path = self._find_path(asset, target)
        if not path:
            self.unwinds_failed += 1
            return UnwindResult(
                success=False,
                error=(
                    f"no route from {asset.code} to {target.code} in the current "
                    f"graph; the position cannot be closed automatically"
                ),
                stranded_asset=asset.code,
                stranded_amount=amount,
            )

        collected: list[Fill] = []
        running = amount
        current = asset
        original_value = self._value_in(target, asset, amount)

        for hop_index, leg in enumerate(path):
            filled = False
            for attempt in range(self.max_attempts):
                if budget is not None and budget.expired:
                    logger.error("unwind budget expired at hop %d", hop_index)
                    break
                try:
                    order, received = await self._execute_hop(
                        leg, running, attempt, cycle_id, hop_index
                    )
                except (OrderRejected, VenueError) as exc:
                    logger.warning(
                        "unwind hop %d attempt %d failed: %s", hop_index, attempt, exc
                    )
                    await self.clock.sleep(0.05 * (attempt + 1))
                    continue

                if order is not None and order.status.any_fill:
                    collected.append(_fill_of(order, self.clock.wall_ns()))
                    running = received
                    current = leg.to_asset
                    filled = True
                    break
                await self.clock.sleep(0.05 * (attempt + 1))

            if not filled:
                self.unwinds_failed += 1
                result = UnwindResult(
                    success=False,
                    fills=tuple(collected),
                    hops=hop_index,
                    attempts=self.max_attempts,
                    error=f"hop {hop_index} ({leg}) would not fill",
                    stranded_asset=current.code,
                    stranded_amount=running,
                )
                # This is the state that must never be silent.
                self._publish(Topics.KILL_SWITCH, {
                    "reason": "unwind_failed",
                    "detail": result.error,
                    "stranded_asset": current.code,
                    "stranded_amount": str(running),
                    "cycle_id": cycle_id,
                })
                logger.critical("UNWIND FAILED: %s", result.summary())
                return result

        cost_bps = (
            bps(safe_div(original_value - running, original_value))
            if original_value > 0 else ZERO
        )
        self.unwinds_succeeded += 1
        self.total_cost_bps += cost_bps
        result = UnwindResult(
            success=True,
            recovered_amount=running,
            fills=tuple(collected),
            hops=len(path),
            cost_bps=cost_bps,
            attempts=1,
        )
        logger.info("unwind complete: %s", result.summary())
        return result

    # -- routing -----------------------------------------------------------

    def _find_path(self, frm: Asset, to: Asset) -> list[Leg]:
        """
        Shortest route from ``frm`` to ``to``, breadth-first.

        BFS rather than best-rate: when unwinding, hop count matters far more
        than price, because every additional hop is another chance to fail
        while still holding the position.
        """
        self.graph.reindex_if_dirty()

        direct = self.graph.best_edge(frm, to)
        if direct is not None:
            return [direct.to_leg()]

        from collections import deque

        queue: deque[tuple[str, list[Leg]]] = deque([(frm.code, [])])
        visited: set[str] = {frm.code}

        while queue:
            node, path = queue.popleft()
            if len(path) >= self.max_hops:
                continue
            for edge in self.graph.edges_from(node):
                target = edge.to.code
                if target in visited:
                    continue
                extended = path + [edge.to_leg()]
                if target == to.code:
                    return extended
                visited.add(target)
                queue.append((target, extended))
        return []

    async def _execute_hop(
        self, leg: Leg, amount: Decimal, attempt: int, cycle_id: str, hop_index: int,
    ) -> tuple[Order | None, Decimal]:
        adapter = self.adapters.get(leg.venue)
        if adapter is None:
            raise VenueError(f"no adapter for {leg.venue}", venue=leg.venue)

        book = self.books.get(leg.symbol)
        if book is None or not book.initialized:
            raise VenueError(f"no book for {leg.symbol.key}", venue=leg.venue)

        spec = self.sizer.spec_for(leg.symbol)
        # Escalate aggression with each attempt.
        ticks = self.aggressiveness_ticks * (attempt + 1)
        touch = book.best_ask if leg.side is Side.BUY else book.best_bid
        offset = spec.tick_size * D(ticks)
        price = touch + offset if leg.side is Side.BUY else max(spec.tick_size, touch - offset)

        if leg.side is Side.BUY:
            quantity = floor_to_step(safe_div(amount, price), spec.lot_step)
        else:
            quantity = floor_to_step(amount, spec.lot_step)

        if quantity <= 0:
            # The residual is smaller than one lot. It cannot be traded and is
            # written off as dust rather than blocking the unwind forever.
            logger.info(
                "unwind hop %d: residual %s %s is below one lot (%s) -- dust",
                hop_index, amount, leg.from_asset.code, spec.lot_step,
            )
            return None, ZERO

        order = Order(
            symbol=leg.symbol,
            side=leg.side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=price,
            time_in_force=TimeInForce.IOC,
            client_order_id=f"{cycle_id or new_cycle_tag()}-UW{hop_index}a{attempt}",
            ts_created_ns=self.clock.wall_ns(),
            tag=f"unwind:{cycle_id}",
        )
        result = await adapter.submit(order, timeout_sec=1.0)

        if not result.status.any_fill:
            return result, ZERO
        if result.side is Side.BUY:
            fee = result.fee_paid if result.fee_asset == result.symbol.base else ZERO
            return result, result.filled_quantity - fee
        gross = result.filled_quantity * result.average_price
        fee = result.fee_paid if result.fee_asset == result.symbol.quote else ZERO
        return result, gross - fee

    def _value_in(self, target: Asset, asset: Asset, amount: Decimal) -> Decimal:
        edge = self.graph.best_edge(asset, target)
        if edge is not None and edge.rate > 0:
            return amount * edge.rate
        value = self.graph.value_of(asset.code)
        return amount * value if value > 0 else amount

    def _publish(self, topic: str, payload: object) -> None:
        if self.bus is not None:
            self.bus.publish(topic, payload, ts_ns=self.clock.wall_ns(), source="unwinder")

    def stats(self) -> dict[str, object]:
        return {
            "attempted": self.unwinds_attempted,
            "succeeded": self.unwinds_succeeded,
            "failed": self.unwinds_failed,
            "success_rate": round(
                self.unwinds_succeeded / max(1, self.unwinds_attempted), 4
            ),
            "mean_cost_bps": (
                float(self.total_cost_bps / self.unwinds_succeeded)
                if self.unwinds_succeeded else 0.0
            ),
        }


def _fill_of(order: Order, ts_ns: int) -> Fill:
    return Fill(
        order_id=order.client_order_id,
        symbol=order.symbol,
        side=order.side,
        price=order.average_price,
        quantity=order.filled_quantity,
        fee=order.fee_paid,
        fee_asset=order.fee_asset or order.symbol.quote,
        liquidity=Liquidity.TAKER,
        ts_ns=ts_ns,
    )


def new_cycle_tag() -> str:
    from triangulum.core.types import new_id
    return new_id("uw-")
