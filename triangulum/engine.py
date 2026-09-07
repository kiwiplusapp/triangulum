"""
The engine: the loop that ties every component together.

One loop serves all three modes. Backtest, paper and live differ only in which
clock and which adapters are injected -- there is no ``if backtest:`` anywhere
below. That is the property that makes a backtest worth reading: a bug in the
sizing code produces the same wrong answer in all three, so the backtest cannot
flatter the strategy by exercising different code.

The cycle, once per scan interval:

    1. rebuild graph edges from the current books
    2. classify the market regime
    3. scan strategies for opportunities
    4. for each: plan -> risk check -> EV gate -> execute
    5. record the outcome into the ledger and back into the learner
    6. publish state for the dashboard

Ordering note: the risk check runs *before* the EV gate, not after. The gate is
the expensive step -- feature extraction plus two model evaluations -- and there
is no point computing an expected value for a cycle that a hard limit forbids.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Sequence

from triangulum.core.clock import Clock, EwmaLatency, SystemClock
from triangulum.core.config import Config
from triangulum.core.decimal_math import D, ZERO, bps, safe_div
from triangulum.core.errors import LimitBreached, TriangulumError
from triangulum.core.eventbus import EventBus, Topics
from triangulum.core.types import (
    Asset, CycleOutcome, ExecutionMode, ExecutionResult, Opportunity, RunMode,
)
from triangulum.exchanges.base import ExchangeAdapter
from triangulum.execution.executor import CycleExecutor
from triangulum.execution.fees import FeeEngine
from triangulum.execution.leg_planner import LegPlanner
from triangulum.execution.sizing import CycleSizer
from triangulum.graph.currency_graph import CurrencyGraph
from triangulum.learning.bandit import ThompsonBandit
from triangulum.learning.drift import DriftMonitor
from triangulum.learning.ev_gate import EVGate
from triangulum.learning.features import FeatureExtractor
from triangulum.learning.regime import RegimeDetector
from triangulum.marketdata.book_manager import BookManager
from triangulum.portfolio.ledger import Ledger
from triangulum.risk.guardrails import Guardrails
from triangulum.strategy.base import Strategy

logger = logging.getLogger(__name__)

__all__ = ["Engine", "EngineState"]


@dataclass(slots=True)
class EngineState:
    """Live snapshot, published to the dashboard and the MCP bridge."""

    running: bool = False
    mode: str = "paper"
    started_ns: int = 0
    scans: int = 0
    opportunities_seen: int = 0
    cycles_planned: int = 0
    cycles_executed: int = 0
    last_scan_ns: int = 0
    regime: str = "unknown"
    equity: Decimal = ZERO
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    cycle_returns_bps: list[float] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    recent_opportunities: list[dict] = field(default_factory=list)
    recent_decisions: list[dict] = field(default_factory=list)

    @property
    def uptime_sec(self) -> float:
        return 0.0 if not self.started_ns else 0.0


class Engine:
    """Orchestrates market data, detection, decision and execution."""

    def __init__(
        self,
        config: Config,
        *,
        books: BookManager,
        graph: CurrencyGraph,
        strategies: Sequence[Strategy],
        adapters: Mapping[str, ExchangeAdapter],
        planner: LegPlanner,
        executor: CycleExecutor,
        gate: EVGate,
        guardrails: Guardrails,
        ledger: Ledger,
        fees: FeeEngine,
        extractor: FeatureExtractor,
        bandit: ThompsonBandit | None = None,
        regime: RegimeDetector | None = None,
        drift: DriftMonitor | None = None,
        bus: EventBus | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.books = books
        self.graph = graph
        self.strategies = list(strategies)
        self.adapters = adapters
        self.planner = planner
        self.executor = executor
        self.gate = gate
        self.guardrails = guardrails
        self.ledger = ledger
        self.fees = fees
        self.extractor = extractor
        self.bandit = bandit
        self.regime = regime or RegimeDetector()
        self.drift = drift or DriftMonitor()
        self.bus = bus or EventBus()
        self.clock = clock or SystemClock()

        self.state = EngineState(mode=config.mode)
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._symbols: list = []
        self._last_scan_ns = 0
        self._equity_sample_ns = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self.state.running = True
        self.state.started_ns = self.clock.wall_ns()
        self._stop.clear()

        self.ledger.deposit(self.config.base_currency, self.config.capital)
        self.state.equity = self.config.capital
        self.guardrails.update_equity(self.config.capital)
        self.state.equity_curve.append(
            (self.state.started_ns, float(self.config.capital))
        )

        self.bus.publish(Topics.ENGINE_STARTED, {
            "mode": self.config.mode,
            "capital": str(self.config.capital),
            "venues": list(self.config.venue_names),
        })
        logger.info(
            "engine started in %s mode with %s %s across %s",
            self.config.mode, self.config.capital, self.config.base_currency,
            ", ".join(self.config.venue_names),
        )

        self._tasks = [
            asyncio.create_task(self._scan_loop(), name="scan"),
            asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
            asyncio.create_task(self._maker_loop(), name="maker-queue"),
        ]

    async def stop(self) -> None:
        logger.info("engine stopping")
        self.bus.publish(Topics.ENGINE_STOPPING, {})
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        self.state.running = False

        # Flatten anything left open. An engine that exits holding inventory
        # has not stopped, it has abandoned a position.
        for adapter in self.adapters.values():
            try:
                await adapter.cancel_all()
            except TriangulumError:
                logger.exception("failed to cancel orders on %s", adapter.name)

    async def run_until_stopped(self) -> None:
        await self.start()
        try:
            await self._stop.wait()
        finally:
            await self.stop()

    def set_symbols(self, symbols: Sequence) -> None:
        """
        Register the tradable universe and enumerate cycle templates.

        The graph MUST be built before enumeration: the enumerator walks
        ``graph.edges_from``, so enumerating against an unbuilt graph silently
        yields zero templates and the engine then scans an empty set forever.
        """
        self._symbols = list(symbols)
        self.graph.build(self._symbols, self.clock.wall_ns())
        stats = self.graph.stats()
        logger.info(
            "graph built: %s usable edges over %s assets (%s excluded)",
            stats["edges_usable"], stats["nodes"], sum(stats["excluded"].values()),
        )
        for strategy in self.strategies:
            enumerate_fn = getattr(strategy, "enumerate", None)
            if enumerate_fn is not None:
                count = enumerate_fn()
                logger.info("%s enumerated %d cycle templates", strategy.name, count)

    # -- the main loop -----------------------------------------------------

    async def _scan_loop(self) -> None:
        interval = self.config.strategy.scan_interval_ms / 1000.0
        while not self._stop.is_set():
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("scan iteration failed")
            await self.clock.sleep(interval)

    async def _scan_once(self) -> None:
        now = self.clock.wall_ns()
        self.state.scans += 1
        self.state.last_scan_ns = now

        # 1. refresh the graph
        self.graph.build(self._symbols, now)

        # 2. regime
        for book in self.books.tradeable_books(now):
            self.regime.observe(book.symbol.key, book.mid, book.spread_bps, now)
        current_regime = self.regime.classify(now)
        self.state.regime = current_regime
        for book in self.books.tradeable_books(now):
            self.extractor.observe_volatility(
                book.symbol.key, self.regime.symbol_volatility_z(book.symbol.key)
            )

        # 3. venue latency, for the feature set
        for name, adapter in self.adapters.items():
            latency = getattr(adapter, "submit_latency", None)
            if latency is not None and latency.count:
                self.extractor.observe_latency(
                    name, latency.mean_ms, latency.stddev_ns / 1e6
                )

        # 4. scan
        opportunities: list[Opportunity] = []
        for strategy in self.strategies:
            if strategy.enabled:
                opportunities.extend(strategy.scan(now))
        self.state.opportunities_seen += len(opportunities)

        if opportunities:
            self._record_opportunities(opportunities)

        # 5. act -- unless we are halted. Checking here rather than inside
        # _consider avoids planning, sizing and gating work the kill switch
        # would reject anyway, and stops the log filling with one rejection
        # line per opportunity per scan for as long as the halt lasts.
        if not self.guardrails.kill_switch_engaged:
            for opportunity in opportunities[: self.config.strategy.max_concurrent_cycles]:
                await self._consider(opportunity, now, current_regime)
        elif self.state.scans % 200 == 0:
            logger.warning(
                "halted (%s) -- passed by %d opportunities this scan",
                self.guardrails.kill_switch_reason, len(opportunities),
            )

        # 6. mark equity periodically
        if now - self._equity_sample_ns > 1_000_000_000:
            self._equity_sample_ns = now
            self._mark_equity(now)

    async def _consider(
        self, opportunity: Opportunity, now_ns: int, regime: str,
    ) -> None:
        # Pick an execution configuration.
        arm = None
        mode = ExecutionMode(self.config.execution.mode)
        capital_fraction = D(str(self.config.execution.capital_fraction_per_cycle))
        if self.bandit and self.config.learning.bandit_enabled:
            arm = self.bandit.select(regime=regime, now_ns=now_ns)
            mode = ExecutionMode(arm.execution_mode)
            capital_fraction = D(str(arm.capital_fraction))

        available = self.ledger.balance(opportunity.start_asset.code)
        if available <= 0:
            return

        plan, sizing = self.planner.plan(
            opportunity, available, mode=mode, now_ns=now_ns,
            capital_fraction=capital_fraction,
            required_start_asset=opportunity.start_asset,
        )
        if plan is None:
            self.bus.publish(Topics.OPPORTUNITY_REJECTED, {
                "opportunity_id": opportunity.opportunity_id,
                "path": opportunity.path,
                "stage": "sizing",
                "reason": sizing.failure,
                "detail": sizing.detail,
            })
            return

        self.state.cycles_planned += 1
        if arm is not None:
            plan = _with_arm(plan, arm.name)

        # Hard limits first -- they are cheap and non-negotiable.
        risk = self.guardrails.check_all(
            plan,
            exposures=dict(self.ledger.balances()),
            venue_exposures={
                v: sum(self.ledger.venue_balances(v).values(), ZERO)
                for v in plan.venues
            },
        )
        if not risk.passed:
            return

        # Then the learned decision.
        decision = self.gate.evaluate(opportunity, plan, now_ns=now_ns)
        self._record_decision(opportunity, plan, decision)
        if not decision.accept:
            return

        if decision.size_multiplier < 1.0:
            # Re-PLAN at the reduced size rather than scaling the existing plan.
            # Multiplying a leg quantity by 0.25 takes it off the venue's lot
            # grid, and the order is then rejected outright for LOT_SIZE -- an
            # exploration trade that cannot be placed teaches nothing and costs
            # a scan. Re-planning runs the full quantization path.
            scaled_plan, scaled_sizing = self.planner.plan(
                opportunity, available, mode=mode, now_ns=now_ns,
                capital_fraction=capital_fraction * D(str(decision.size_multiplier)),
                required_start_asset=opportunity.start_asset,
            )
            if scaled_plan is None:
                logger.debug(
                    "exploration trade unroutable at %.0f%% size (%s)",
                    decision.size_multiplier * 100, scaled_sizing.failure,
                )
                return
            plan = scaled_plan if arm is None else _with_arm(scaled_plan, arm.name)

        self.guardrails.on_cycle_started(plan)
        self.state.cycles_executed += 1
        result = await self.executor.execute(plan)
        self.guardrails.on_cycle_finished(result)

        self._settle(result, opportunity, decision, regime, now_ns)

    def _settle(
        self, result: ExecutionResult, opportunity: Opportunity,
        decision, regime: str, now_ns: int,
    ) -> None:
        self.ledger.record_cycle(result)

        filled = result.outcome is CycleOutcome.COMPLETED
        realized = float(result.realized_pnl_bps)

        self.gate.observe(
            decision, filled=filled, realized_bps=realized,
            now_ns=now_ns, cycle_key=opportunity.path,
        )
        drift_event = self.drift.observe(decision.fill_probability, 1 if filled else 0)
        if drift_event is not None:
            logger.warning("concept drift: %s", drift_event)
            self.bus.publish(Topics.MODEL_DRIFT, {"event": str(drift_event)})
            response = self.drift.response()
            self.gate.confidence_multiplier = response["confidence_multiplier"]
            self.gate.exploration_floor = response["exploration_floor"]

        if self.bandit and result.plan.bandit_arm:
            self.bandit.update(
                result.plan.bandit_arm, filled=filled,
                realized_bps=realized, regime=regime,
            )

        if result.outcome.committed_capital:
            self.state.cycle_returns_bps.append(realized)
            self.state.outcomes.append(result.outcome.value)
            if len(self.state.cycle_returns_bps) > 50_000:
                self.state.cycle_returns_bps = self.state.cycle_returns_bps[-25_000:]
                self.state.outcomes = self.state.outcomes[-25_000:]

        if result.outcome is CycleOutcome.PARTIAL_STUCK:
            residual = ", ".join(
                f"{amount} {asset}"
                for asset, amount in result.residual_inventory.items()
            ) or "an unknown amount"
            self.guardrails.engage_kill_switch(
                f"cycle {result.cycle_id} could not be unwound and left {residual} "
                f"unhedged. No new cycles until a human flattens it and releases "
                f"the halt."
            )

        self._mark_equity(now_ns)
        logger.info("%s", result.summary())

    # -- support loops -----------------------------------------------------

    async def _maker_loop(self) -> None:
        """Drain resting maker orders in the simulator. No-op for live adapters."""
        step = 0.05
        while not self._stop.is_set():
            for adapter in self.adapters.values():
                stepper = getattr(adapter, "step_resting", None)
                if stepper is not None:
                    try:
                        stepper(step)
                    except Exception:
                        logger.exception("maker queue step failed")
            await self.clock.sleep(step)

    async def _heartbeat_loop(self) -> None:
        interval = self.config.ops.heartbeat_interval_sec
        while not self._stop.is_set():
            await self.clock.sleep(interval)
            self.bus.publish(Topics.ENGINE_HEARTBEAT, self.snapshot())

    # -- state -------------------------------------------------------------

    def _mark_equity(self, now_ns: int) -> None:
        prices = self._price_map()
        equity = self.ledger.equity(prices)
        self.state.equity = equity
        self.guardrails.update_equity(equity)
        self.state.equity_curve.append((now_ns, float(equity)))
        if len(self.state.equity_curve) > 100_000:
            # Decimate the oldest half rather than dropping it: the shape of
            # the early curve still matters for drawdown computation.
            head = self.state.equity_curve[:50_000:2]
            self.state.equity_curve = head + self.state.equity_curve[50_000:]
        self.bus.publish(Topics.EQUITY_SNAPSHOT, {
            "ts": now_ns, "equity": float(equity),
        })

    def _price_map(self) -> dict[str, Decimal]:
        prices: dict[str, Decimal] = {}
        for book in self.books:
            if book.initialized and book.mid > 0:
                prices[book.symbol.canonical] = book.mid
        return prices

    def _record_opportunities(self, opportunities: Sequence[Opportunity]) -> None:
        for opportunity in opportunities[:8]:
            self.state.recent_opportunities.append({
                "id": opportunity.opportunity_id,
                "path": opportunity.path,
                "edge_bps": float(opportunity.gross_edge_bps),
                "age_ms": opportunity.max_book_age_ns / 1e6,
                "venues": list(opportunity.venues),
                "ts": opportunity.ts_detected_ns,
            })
        limit = self.config.dashboard.max_opportunity_rows
        if len(self.state.recent_opportunities) > limit:
            self.state.recent_opportunities = self.state.recent_opportunities[-limit:]
        self.bus.publish(Topics.OPPORTUNITY_DETECTED, self.state.recent_opportunities[-1])

    def _record_decision(self, opportunity, plan, decision) -> None:
        row = {
            "path": opportunity.path,
            "cycle_id": plan.cycle_id,
            "net_edge_bps": float(plan.net_edge_bps),
            **decision.to_dict(),
        }
        self.state.recent_decisions.append(row)
        if len(self.state.recent_decisions) > 200:
            self.state.recent_decisions = self.state.recent_decisions[-200:]

    def snapshot(self) -> dict[str, object]:
        """Full engine state. The dashboard and the MCP bridge both read this."""
        return {
            "mode": self.config.mode,
            "running": self.state.running,
            "started_ns": self.state.started_ns,
            "scans": self.state.scans,
            "regime": self.state.regime,
            "equity": float(self.state.equity),
            "starting_equity": float(self.config.capital),
            "base_currency": self.config.base_currency,
            "opportunities_seen": self.state.opportunities_seen,
            "cycles_planned": self.state.cycles_planned,
            "cycles_executed": self.state.cycles_executed,
            "graph": self.graph.stats(),
            "books": self.books.stats(),
            "risk": self.guardrails.stats(),
            "executor": self.executor.stats(),
            "ledger": self.ledger.stats(),
            "gate": self.gate.stats(),
            "bandit": self.bandit.stats() if self.bandit else None,
            "regime_detail": self.regime.stats(),
            "drift": self.drift.stats(),
            "fees": self.fees.stats(),
            "strategies": [
                getattr(s, "stats_dict", s.stats.to_dict)() for s in self.strategies
            ],
            "venues": {name: a.stats() for name, a in self.adapters.items()},
            "recent_opportunities": self.state.recent_opportunities[-50:],
            "recent_decisions": self.state.recent_decisions[-50:],
            "equity_curve": self.state.equity_curve[-2000:],
        }


def _with_arm(plan, arm_name: str):
    from dataclasses import replace
    return replace(plan, bandit_arm=arm_name)

