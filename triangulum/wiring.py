"""
Dependency wiring.

One place where every component is constructed and connected. Nothing else in
the codebase imports a concrete implementation of anything it does not own,
which is what keeps the engine testable: a test builds the same stack with a
simulated clock and a synthetic market and exercises the real code path.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Mapping

from triangulum.core.clock import Clock, SimulatedClock, SystemClock
from triangulum.core.config import Config
from triangulum.core.decimal_math import D, ZERO
from triangulum.core.eventbus import EventBus, Topics
from triangulum.core.types import ExecutionMode, RunMode
from triangulum.engine import Engine
from triangulum.exchanges.base import ExchangeAdapter
from triangulum.exchanges.simulated import SimulatedExchange, SimulationParams
from triangulum.exchanges.spec import get_venue_spec
from triangulum.execution.executor import CycleExecutor
from triangulum.execution.fees import FeeEngine
from triangulum.execution.leg_planner import LegPlanner
from triangulum.execution.sizing import CycleSizer
from triangulum.execution.unwinder import Unwinder
from triangulum.graph.currency_graph import CurrencyGraph
from triangulum.learning.bandit import ThompsonBandit
from triangulum.learning.calibration import PlattCalibrator
from triangulum.learning.drift import DriftMonitor
from triangulum.learning.ev_gate import EVGate
from triangulum.learning.features import FEATURE_NAMES, FeatureExtractor
from triangulum.learning.online_lr import FTRLProximal, OnlineRidge
from triangulum.learning.regime import RegimeDetector
from triangulum.learning.store import ModelStore
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.normalizer import SymbolNormalizer
from triangulum.marketdata.recorder import TickRecorder
from triangulum.portfolio.ledger import Ledger
from triangulum.risk.guardrails import Guardrails
from triangulum.strategy.cross_exchange import CrossExchangeStrategy
from triangulum.strategy.triangular import TriangularStrategy

logger = logging.getLogger(__name__)

__all__ = ["Stack", "build_stack", "record_market_data"]


@dataclass
class Stack:
    """Everything the engine needs, plus the handles to shut it down."""

    config: Config
    clock: Clock
    bus: EventBus
    books: BookManager
    normalizer: SymbolNormalizer
    graph: CurrencyGraph
    adapters: dict[str, ExchangeAdapter]
    engine: Engine
    fees: FeeEngine
    gate: EVGate
    store: ModelStore
    dashboard: object = None
    market: object = None
    recorder: TickRecorder | None = None

    async def shutdown(self) -> None:
        self.checkpoint()
        if self.recorder:
            await self.recorder.stop()
        if self.dashboard:
            self.dashboard.stop()
        for adapter in self.adapters.values():
            try:
                await adapter.disconnect()
            except Exception:
                logger.exception("failed to disconnect %s", adapter.name)

    def checkpoint(self) -> None:
        """Persist the learned models. Called on shutdown and on a timer."""
        try:
            feature_hash = ModelStore.feature_hash(FEATURE_NAMES)
            self.store.save(
                "fill_model", self.gate.fill_model.to_dict(),
                samples=self.gate.fill_model.samples,
                metrics=self.gate.fill_model.stats(),
                feature_hash=feature_hash,
            )
            self.store.save(
                "slippage_model", self.gate.slippage_model.to_dict(),
                samples=self.gate.slippage_model.samples,
                metrics=self.gate.slippage_model.stats(),
                feature_hash=feature_hash,
            )
            self.store.save(
                "calibrator", self.gate.calibrator.to_dict(),
                samples=self.gate.calibrator.samples,
                feature_hash=feature_hash,
            )
            if self.engine.bandit:
                self.store.save(
                    "bandit", self.engine.bandit.to_dict(),
                    samples=self.engine.bandit.selections,
                )
        except Exception:
            logger.exception("checkpoint failed")

    def report(self) -> str:
        from triangulum.backtest.metrics import compute_metrics

        state = self.engine.state
        metrics = compute_metrics(
            state.equity_curve,
            state.cycle_returns_bps,
            outcomes=state.outcomes,
            total_fees=float(self.engine.ledger.total_fees),
            daily_target_bps=self.config.daily_target_bps,
            monthly_target_pct=self.config.monthly_target_pct,
        )
        return metrics.report()


async def build_stack(
    config: Config,
    *,
    mode: RunMode = RunMode.PAPER,
    demo: bool = False,
    market_params: object = None,
) -> Stack:
    """Construct and connect the whole system."""
    bus = EventBus()
    clock: Clock = SystemClock()
    normalizer = SymbolNormalizer()
    books = BookManager(
        bus=bus, clock=clock,
        max_depth=max(v.book_depth for v in config.enabled_venues) if config.enabled_venues else 20,
        max_age_ns=int(config.strategy.max_book_age_ms * 1e6),
    )

    # -- fees --------------------------------------------------------------
    fees = FeeEngine(use_discount_asset=True)
    for venue_config in config.enabled_venues:
        if venue_config.maker_fee_bps is not None and venue_config.taker_fee_bps is not None:
            fees.override(
                venue_config.name,
                D(str(venue_config.maker_fee_bps)),
                D(str(venue_config.taker_fee_bps)),
            )

    # -- adapters and market data -----------------------------------------
    adapters: dict[str, ExchangeAdapter] = {}
    market = None
    symbols: list = []

    if demo or mode is RunMode.BACKTEST:
        from triangulum.simulation import MarketParams, SyntheticMarket

        venue_config = config.enabled_venues[0]
        market = SyntheticMarket(
            books, normalizer, venue=venue_config.name,
            params=market_params or MarketParams(),
        )
        market.run(60)          # warm the books before the engine looks at them
        symbols = list(market.symbols)

        sim = SimulatedExchange(
            venue_config, normalizer, books,
            clock=clock,
            params=SimulationParams(),
            initial_balances={config.base_currency: config.capital},
            source_venue=venue_config.name,
        )
        for symbol in symbols:
            lot, tick, min_notional = market.specs[symbol.key]
            from triangulum.exchanges.spec import SymbolSpec
            sim.register_spec(SymbolSpec(
                venue_symbol=symbol.venue_symbol,
                base=symbol.base.code, quote=symbol.quote.code,
                tick_size=tick, lot_step=lot, min_notional=min_notional,
            ))
        await sim.connect()
        adapters[venue_config.name] = sim
    else:
        from triangulum.exchanges.registry import build_adapters

        adapters = build_adapters(config, normalizer, books, clock=clock)
        for name, adapter in adapters.items():
            await adapter.connect()
            loaded = await adapter.load_instruments()
            symbols.extend(loaded)
            await adapter.subscribe_books(loaded)
            logger.info("%s: subscribed to %d symbols", name, len(loaded))

    # -- graph -------------------------------------------------------------
    primary = config.enabled_venues[0].name
    spec = get_venue_spec(primary)
    graph = CurrencyGraph(
        books,
        base_currency=config.base_currency,
        reference_notional=config.capital * D(str(config.execution.capital_fraction_per_cycle)),
        max_book_age_ns=int(config.strategy.max_book_age_ms * 1e6),
        maker_fee_bps=spec.default_maker_bps,
        taker_fee_bps=spec.default_taker_bps,
        max_levels=config.execution.max_levels_to_walk,
        consumption_cap=D(str(config.execution.max_book_consumption)),
        cycle_legs_for_budget=config.strategy.min_cycle_length,
        drag_budget_bps=D(str(config.strategy.drag_budget_bps)),
        enforce_lot_value=config.strategy.enforce_lot_value_screen,
    )
    for venue_config in config.enabled_venues:
        venue_spec = get_venue_spec(venue_config.name)
        graph.set_venue_fees(
            venue_config.name,
            D(str(venue_config.maker_fee_bps)) if venue_config.maker_fee_bps is not None
            else venue_spec.default_maker_bps,
            D(str(venue_config.taker_fee_bps)) if venue_config.taker_fee_bps is not None
            else venue_spec.default_taker_bps,
        )

    # -- sizing ------------------------------------------------------------
    sizer = CycleSizer(
        books, fees,
        max_levels=config.execution.max_levels_to_walk,
        consumption_cap=D(str(config.execution.max_book_consumption)),
        min_notional_buffer=D(str(config.execution.min_notional_buffer)),
        taker_offset_ticks=config.execution.taker_price_offset_ticks,
        maker_offset_ticks=config.execution.maker_price_offset_ticks,
    )
    for adapter in adapters.values():
        for venue_symbol, symbol_spec in adapter.all_specs().items():
            symbol = normalizer.lookup(adapter.name, venue_symbol)
            if symbol is not None:
                sizer.register_spec(symbol, symbol_spec)
                graph.set_symbol_rules(
                    symbol, symbol_spec.lot_step, symbol_spec.min_notional
                )

    # -- learning ----------------------------------------------------------
    store = ModelStore(config.learning.model_dir)
    feature_hash = ModelStore.feature_hash(FEATURE_NAMES)

    fill_model = FTRLProximal(
        alpha=config.learning.ftrl_alpha, beta=config.learning.ftrl_beta,
        l1=config.learning.ftrl_l1, l2=config.learning.ftrl_l2,
        dimensions=1 << config.learning.feature_hash_bits,
    )
    saved = store.load("fill_model", expect_feature_hash=feature_hash)
    if saved:
        fill_model = FTRLProximal.from_dict(saved)

    slippage_model = OnlineRidge(
        learning_rate=config.learning.slippage_lr, l2=config.learning.slippage_l2,
    )
    saved = store.load("slippage_model", expect_feature_hash=feature_hash)
    if saved:
        slippage_model = OnlineRidge.from_dict(saved)

    calibrator = PlattCalibrator()
    saved = store.load("calibrator", expect_feature_hash=feature_hash)
    if saved:
        calibrator = PlattCalibrator.from_dict(saved)

    extractor = FeatureExtractor(books)
    gate = EVGate(
        fill_model, slippage_model, extractor,
        calibrator=calibrator,
        min_ev_bps=config.learning.min_expected_value_bps,
        confidence_multiplier=config.learning.ev_confidence_multiplier,
        min_samples_before_trust=config.learning.min_samples_before_trust,
        exploration_floor=config.learning.exploration_floor,
        feature_bits=config.learning.feature_hash_bits,
    )

    bandit = None
    if config.learning.bandit_enabled:
        bandit = ThompsonBandit(
            decay=config.learning.bandit_decay,
            prior_alpha=config.learning.bandit_prior_alpha,
            prior_beta=config.learning.bandit_prior_beta,
            exploration_floor=config.learning.exploration_floor,
        )
        saved = store.load("bandit")
        if saved:
            bandit.load_state(saved)

    # -- strategies --------------------------------------------------------
    strategies = []
    if config.strategy.triangular_enabled:
        strategies.append(TriangularStrategy(
            graph,
            venue=primary,
            min_length=config.strategy.min_cycle_length,
            max_length=config.strategy.max_cycle_length,
            start_assets=config.strategy.start_assets,
            allow_cross_venue=config.strategy.cross_exchange_enabled,
            clock=clock,
            min_edge_bps=D(str(config.strategy.min_gross_edge_bps)),
            max_book_age_ns=int(config.strategy.max_book_age_ms * 1e6),
            cooldown_ns=int(config.strategy.cycle_cooldown_ms * 1e6),
        ))
    if config.strategy.cross_exchange_enabled and len(adapters) > 1:
        strategies.append(CrossExchangeStrategy(
            graph, books,
            venues=tuple(adapters),
            clock=clock,
            min_edge_bps=D(str(config.strategy.min_gross_edge_bps)),
        ))

    # -- execution ---------------------------------------------------------
    ledger = Ledger(config.base_currency, clock=clock)
    guardrails = Guardrails(
        config.risk,
        starting_equity=config.capital,
        bus=bus, clock=clock,
        max_concurrent_cycles=config.strategy.max_concurrent_cycles,
    )
    unwinder = Unwinder(
        adapters, books, graph, sizer,
        bus=bus, clock=clock,
        max_attempts=config.execution.unwind_max_attempts,
        aggressiveness_ticks=config.execution.unwind_aggressiveness_ticks,
    )
    planner = LegPlanner(books, sizer, fees)
    executor = CycleExecutor(
        adapters, sizer, fees,
        unwinder=unwinder if config.execution.unwind_enabled else None,
        bus=bus, clock=clock,
        cycle_budget_ms=config.execution.cycle_budget_ms,
        leg_timeout_ms=config.execution.leg_timeout_ms,
        maker_leg_timeout_ms=config.execution.maker_leg_timeout_ms,
        maker_max_requeues=config.execution.maker_max_requeues,
        dry_run=config.execution.dry_run_orders,
    )

    engine = Engine(
        config,
        books=books, graph=graph, strategies=strategies, adapters=adapters,
        planner=planner, executor=executor, gate=gate, guardrails=guardrails,
        ledger=ledger, fees=fees, extractor=extractor, bandit=bandit,
        regime=RegimeDetector(),
        drift=DriftMonitor(
            delta=config.learning.drift_delta,
            min_samples=config.learning.drift_min_samples,
        ),
        bus=bus, clock=clock,
    )
    engine.set_symbols(symbols)

    stack = Stack(
        config=config, clock=clock, bus=bus, books=books, normalizer=normalizer,
        graph=graph, adapters=adapters, engine=engine, fees=fees, gate=gate,
        store=store, market=market,
    )

    # -- dashboard ---------------------------------------------------------
    if config.dashboard.enabled:
        from triangulum.api.server import DashboardServer
        from triangulum.ops.logging_setup import get_ring_handler

        def snapshot() -> dict:
            data = engine.snapshot()
            data["daily_target_bps"] = config.daily_target_bps
            if market is not None:
                data["synthetic_market"] = market.stats()
            return data

        dashboard = DashboardServer(
            host=config.dashboard.host,
            port=config.dashboard.port,
            auth_token=config.dashboard.auth_token,
            snapshot_provider=snapshot,
            broadcast_interval_ms=config.dashboard.broadcast_interval_ms,
            allow_kill_switch=config.dashboard.allow_kill_switch,
        )
        dashboard.register_command(
            "engage_kill_switch",
            lambda msg: (
                guardrails.engage_kill_switch(
                    f"dashboard: {msg.get('reason', 'operator')}"
                ),
                {"engaged": True},
            )[1],
        )
        dashboard.register_command(
            "release_kill_switch",
            lambda msg: (
                guardrails.release_kill_switch(acknowledged_by="dashboard"),
                {"engaged": False},
            )[1],
        )
        dashboard.get_routes["/api/metrics"] = lambda: _metrics_payload(stack)
        dashboard.get_routes["/api/config"] = config.redacted

        ring = get_ring_handler()
        if ring is not None:
            ring.sink = lambda level, message, ts: dashboard.push_log(level, message, ts)

        dashboard.start()
        stack.dashboard = dashboard

    # -- recording ---------------------------------------------------------
    if config.storage.record_opportunities or config.storage.record_books:
        recorder = TickRecorder(
            config.storage.recording_dir,
            flush_interval_sec=config.storage.flush_interval_sec,
        )
        await recorder.start()
        stack.recorder = recorder

        async def _record(event) -> None:
            payload = event.payload
            if isinstance(payload, dict) and "path" in payload:
                recorder.record_meta(payload, stream=primary)

        asyncio.create_task(
            bus.run_handler(Topics.OPPORTUNITY_DETECTED, _record, name="recorder")
        )

    return stack


def _metrics_payload(stack: Stack) -> dict:
    from triangulum.backtest.metrics import compute_metrics
    from triangulum.backtest.monte_carlo import bootstrap_paths

    state = stack.engine.state
    metrics = compute_metrics(
        state.equity_curve, state.cycle_returns_bps,
        outcomes=state.outcomes,
        total_fees=float(stack.engine.ledger.total_fees),
        daily_target_bps=stack.config.daily_target_bps,
    )
    payload = {"metrics": metrics.to_dict()}
    if len(state.cycle_returns_bps) >= 30:
        payload["monte_carlo"] = bootstrap_paths(
            state.cycle_returns_bps,
            starting_equity=float(stack.config.capital),
            runs=1000,
        ).to_dict()
    return payload


async def record_market_data(config: Config, *, minutes: float, out_dir: str) -> int:
    """Connect to live venues and record books for later training/backtesting."""
    config.storage.record_books = True
    config.storage.recording_dir = out_dir
    config.dashboard.enabled = False

    stack = await build_stack(config, mode=RunMode.PAPER)
    recorder = stack.recorder
    if recorder is None:
        recorder = TickRecorder(out_dir)
        await recorder.start()
        stack.recorder = recorder

    logger.info("recording for %.1f minutes into %s", minutes, out_dir)
    deadline = asyncio.get_running_loop().time() + minutes * 60
    try:
        while asyncio.get_running_loop().time() < deadline:
            for book in stack.books:
                if book.initialized:
                    recorder.record_book(book.snapshot(10))
            await asyncio.sleep(0.25)
    finally:
        await stack.shutdown()
    logger.info("recording complete: %s", recorder.stats())
    return 0
