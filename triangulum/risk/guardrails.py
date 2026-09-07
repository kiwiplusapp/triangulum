"""
Pre-trade risk guardrails.

Every check here runs before a single order is sent, in a fixed order, cheapest
first. They are hard limits, not suggestions: a strategy cannot argue its way
past them, and there is no "override" parameter.

The ordering is deliberate. The kill switch is checked first because when it is
engaged nothing else matters, and the expensive checks (exposure computation,
notional limits) run last so a halted engine spends no CPU on them.

A note on why these are not "conservative defaults you should tune up": the
limits protect against the failure modes that actually kill small accounts, and
every one of them is a mode where the loss is unbounded without the limit. A
runaway loop submitting orders is bounded only by the order-rate limit. A
mis-sized cycle is bounded only by the notional cap. A model that has silently
broken is bounded only by the consecutive-loss breaker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Sequence

from triangulum.core.clock import Clock, SystemClock
from triangulum.core.config import RiskConfig
from triangulum.core.decimal_math import D, ZERO, bps, safe_div
from triangulum.core.errors import LimitBreached
from triangulum.core.eventbus import EventBus, Topics
from triangulum.core.ringbuffer import TimedRingBuffer
from triangulum.core.types import Asset, CyclePlan, ExecutionResult

logger = logging.getLogger(__name__)

__all__ = ["Guardrails", "RiskCheck", "CheckResult"]


class RiskCheck:
    KILL_SWITCH = "kill_switch"
    CIRCUIT_OPEN = "circuit_open"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    ORDER_RATE = "order_rate"
    CYCLE_RATE = "cycle_rate"
    CYCLE_NOTIONAL = "cycle_notional"
    VENUE_EXPOSURE = "venue_exposure"
    ASSET_EXPOSURE = "asset_exposure"
    CONCURRENT_CYCLES = "concurrent_cycles"
    BOOK_STALENESS = "book_staleness"
    NEGATIVE_EDGE = "negative_edge"


@dataclass(slots=True)
class CheckResult:
    passed: bool
    check: str = ""
    reason: str = ""
    value: object = None
    threshold: object = None

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise LimitBreached(
                self.reason, limit=self.check, value=self.value, threshold=self.threshold
            )


_OK = CheckResult(passed=True)


class Guardrails:
    """Hard pre-trade limits. All checks run before any order is sent."""

    def __init__(
        self,
        config: RiskConfig,
        *,
        starting_equity: Decimal,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        max_concurrent_cycles: int = 2,
    ) -> None:
        self.config = config
        self.bus = bus
        self.clock = clock or SystemClock()
        self.max_concurrent_cycles = max_concurrent_cycles

        self.starting_equity = starting_equity
        self.peak_equity = starting_equity
        self.current_equity = starting_equity
        self.day_start_equity = starting_equity
        self._day_start_ns = self.clock.wall_ns()

        self.consecutive_losses = 0
        self.kill_switch_engaged = False
        self.kill_switch_reason = ""

        self._order_times = TimedRingBuffer(4096)
        self._cycle_times = TimedRingBuffer(2048)
        self._in_flight = 0

        self.checks_run = 0
        self.rejections: dict[str, int] = {}

    # -- equity tracking ---------------------------------------------------

    def update_equity(self, equity: Decimal) -> None:
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        # Roll the daily baseline at UTC midnight.
        now = self.clock.wall_ns()
        if now - self._day_start_ns >= 86_400 * 1_000_000_000:
            self.day_start_equity = equity
            self._day_start_ns = now
            logger.info("risk: daily baseline rolled to %s", equity)

    @property
    def daily_pnl_pct(self) -> Decimal:
        return safe_div(
            self.current_equity - self.day_start_equity, self.day_start_equity
        ) * D(100)

    @property
    def drawdown_pct(self) -> Decimal:
        if self.peak_equity <= 0:
            return ZERO
        return safe_div(self.peak_equity - self.current_equity, self.peak_equity) * D(100)

    @property
    def total_return_pct(self) -> Decimal:
        return safe_div(
            self.current_equity - self.starting_equity, self.starting_equity
        ) * D(100)

    # -- the gate ----------------------------------------------------------

    def check_all(
        self,
        plan: CyclePlan,
        *,
        exposures: Mapping[str, Decimal] | None = None,
        venue_exposures: Mapping[str, Decimal] | None = None,
    ) -> CheckResult:
        """Run every pre-trade check. Cheapest first, fail fast."""
        self.checks_run += 1
        now = self.clock.wall_ns()

        for result in (
            self._check_kill_switch(),
            self._check_concurrent(),
            self._check_daily_loss(),
            self._check_drawdown(),
            self._check_consecutive_losses(),
            self._check_rates(now),
            self._check_edge(plan),
            self._check_notional(plan),
            self._check_exposures(plan, exposures, venue_exposures),
        ):
            if not result.passed:
                self.rejections[result.check] = self.rejections.get(result.check, 0) + 1
                logger.info("risk rejected %s: %s", plan.cycle_id, result.reason)
                if self.bus:
                    self.bus.publish(Topics.RISK_LIMIT_BREACHED, {
                        "cycle_id": plan.cycle_id,
                        "check": result.check,
                        "reason": result.reason,
                    })
                return result
        return _OK

    def _check_kill_switch(self) -> CheckResult:
        if self.kill_switch_engaged:
            return CheckResult(
                False, RiskCheck.KILL_SWITCH,
                f"kill switch engaged: {self.kill_switch_reason}",
            )
        return _OK

    def _check_concurrent(self) -> CheckResult:
        if self._in_flight >= self.max_concurrent_cycles:
            return CheckResult(
                False, RiskCheck.CONCURRENT_CYCLES,
                f"{self._in_flight} cycles already in flight "
                f"(max {self.max_concurrent_cycles})",
                self._in_flight, self.max_concurrent_cycles,
            )
        return _OK

    def _check_daily_loss(self) -> CheckResult:
        loss = -self.daily_pnl_pct
        if loss >= D(str(self.config.max_daily_loss_pct)):
            self.engage_kill_switch(
                f"daily loss limit hit: {loss:.2f}% >= "
                f"{self.config.max_daily_loss_pct}%"
            )
            return CheckResult(
                False, RiskCheck.DAILY_LOSS,
                f"daily loss {loss:.2f}% exceeds {self.config.max_daily_loss_pct}%",
                float(loss), self.config.max_daily_loss_pct,
            )
        return _OK

    def _check_drawdown(self) -> CheckResult:
        dd = self.drawdown_pct
        if dd >= D(str(self.config.max_drawdown_pct)):
            self.engage_kill_switch(
                f"max drawdown hit: {dd:.2f}% >= {self.config.max_drawdown_pct}%"
            )
            return CheckResult(
                False, RiskCheck.DRAWDOWN,
                f"drawdown {dd:.2f}% exceeds {self.config.max_drawdown_pct}%",
                float(dd), self.config.max_drawdown_pct,
            )
        return _OK

    def _check_consecutive_losses(self) -> CheckResult:
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return CheckResult(
                False, RiskCheck.CONSECUTIVE_LOSSES,
                f"{self.consecutive_losses} consecutive losing cycles "
                f"(max {self.config.max_consecutive_losses}) -- the model may be "
                f"mis-specified for current conditions",
                self.consecutive_losses, self.config.max_consecutive_losses,
            )
        return _OK

    def _check_rates(self, now_ns: int) -> CheckResult:
        minute_ago = now_ns - 60_000_000_000
        orders = self._order_times.count_since(minute_ago)
        if orders >= self.config.max_orders_per_minute:
            return CheckResult(
                False, RiskCheck.ORDER_RATE,
                f"{orders} orders in the last minute "
                f"(max {self.config.max_orders_per_minute})",
                orders, self.config.max_orders_per_minute,
            )
        cycles = self._cycle_times.count_since(minute_ago)
        if cycles >= self.config.max_cycles_per_minute:
            return CheckResult(
                False, RiskCheck.CYCLE_RATE,
                f"{cycles} cycles in the last minute "
                f"(max {self.config.max_cycles_per_minute})",
                cycles, self.config.max_cycles_per_minute,
            )
        return _OK

    def _check_edge(self, plan: CyclePlan) -> CheckResult:
        if plan.net_edge_bps <= 0:
            return CheckResult(
                False, RiskCheck.NEGATIVE_EDGE,
                f"planned net edge is {plan.net_edge_bps:.2f} bps",
                float(plan.net_edge_bps), 0,
            )
        return _OK

    def _check_notional(self, plan: CyclePlan) -> CheckResult:
        limit = self.current_equity * D(str(self.config.max_cycle_notional_pct)) / D(100)
        if plan.start_amount > limit:
            return CheckResult(
                False, RiskCheck.CYCLE_NOTIONAL,
                f"cycle notional {plan.start_amount:.2f} exceeds "
                f"{self.config.max_cycle_notional_pct}% of equity ({limit:.2f})",
                float(plan.start_amount), float(limit),
            )
        return _OK

    def _check_exposures(
        self,
        plan: CyclePlan,
        exposures: Mapping[str, Decimal] | None,
        venue_exposures: Mapping[str, Decimal] | None,
    ) -> CheckResult:
        if self.current_equity <= 0:
            return _OK

        if venue_exposures:
            cap = D(str(self.config.max_venue_exposure_pct))
            for venue in plan.venues:
                current = venue_exposures.get(venue, ZERO)
                pct = safe_div(current + plan.start_amount, self.current_equity) * D(100)
                if pct > cap:
                    return CheckResult(
                        False, RiskCheck.VENUE_EXPOSURE,
                        f"{venue} exposure would reach {pct:.1f}% of equity "
                        f"(max {cap}%)",
                        float(pct), float(cap),
                    )

        if exposures:
            cap = D(str(self.config.max_asset_exposure_pct))
            # Only PERSISTENT holdings count against the per-asset cap.
            #
            # A cycle transiently converts the whole working amount into each
            # intermediate asset for a few hundred milliseconds; that is the
            # mechanism, not a position. Counting it as exposure makes every
            # cycle larger than the cap look like a limit breach, which at a
            # 60% cap means the engine refuses every cycle it can actually
            # afford. Transient size is bounded by max_cycle_notional_pct,
            # checked above; this cap governs what is still held afterwards.
            for asset, amount in exposures.items():
                if asset == plan.start_asset.code:
                    continue
                if amount <= 0:
                    continue
                pct = safe_div(amount, self.current_equity) * D(100)
                if pct > cap:
                    return CheckResult(
                        False, RiskCheck.ASSET_EXPOSURE,
                        f"{asset} already holds {pct:.1f}% of equity (max {cap}%) "
                        f"-- flatten it before opening new cycles",
                        float(pct), float(cap),
                    )
        return _OK

    # -- lifecycle hooks ---------------------------------------------------

    def on_cycle_started(self, plan: CyclePlan) -> None:
        self._in_flight += 1
        now = self.clock.wall_ns()
        self._cycle_times.push(now)
        for _ in plan.legs:
            self._order_times.push(now)

    def on_cycle_finished(self, result: ExecutionResult) -> None:
        self._in_flight = max(0, self._in_flight - 1)
        if not result.outcome.committed_capital:
            return
        if result.realized_pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def engage_kill_switch(self, reason: str) -> None:
        if self.kill_switch_engaged:
            return
        self.kill_switch_engaged = True
        self.kill_switch_reason = reason
        logger.critical("KILL SWITCH ENGAGED: %s", reason)
        if self.bus:
            self.bus.publish(Topics.KILL_SWITCH, {"reason": reason, "engaged": True})

    def release_kill_switch(self, *, acknowledged_by: str = "operator") -> None:
        """
        Manual release only.

        Deliberately has no automatic counterpart. A kill switch that
        re-arms itself on a timer is not a kill switch -- whatever tripped it
        will still be true a minute later, and the second trip costs more than
        the first.
        """
        if not self.kill_switch_engaged:
            return
        logger.warning(
            "kill switch released by %s (was: %s)", acknowledged_by, self.kill_switch_reason
        )
        self.kill_switch_engaged = False
        self.kill_switch_reason = ""
        self.consecutive_losses = 0
        if self.bus:
            self.bus.publish(Topics.KILL_SWITCH, {
                "engaged": False, "released_by": acknowledged_by,
            })

    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict[str, object]:
        now = self.clock.wall_ns()
        minute_ago = now - 60_000_000_000
        return {
            "kill_switch": self.kill_switch_engaged,
            "kill_switch_reason": self.kill_switch_reason,
            "equity": str(self.current_equity),
            "starting_equity": str(self.starting_equity),
            "peak_equity": str(self.peak_equity),
            "daily_pnl_pct": float(round(self.daily_pnl_pct, 4)),
            "drawdown_pct": float(round(self.drawdown_pct, 4)),
            "total_return_pct": float(round(self.total_return_pct, 4)),
            "consecutive_losses": self.consecutive_losses,
            "cycles_in_flight": self._in_flight,
            "orders_last_minute": self._order_times.count_since(minute_ago),
            "cycles_last_minute": self._cycle_times.count_since(minute_ago),
            "checks_run": self.checks_run,
            "rejections_by_check": dict(self.rejections),
            "limits": {
                "max_daily_loss_pct": self.config.max_daily_loss_pct,
                "max_drawdown_pct": self.config.max_drawdown_pct,
                "max_consecutive_losses": self.config.max_consecutive_losses,
                "max_cycle_notional_pct": self.config.max_cycle_notional_pct,
            },
        }
