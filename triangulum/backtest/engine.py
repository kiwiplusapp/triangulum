"""
Backtest driver.

Replays recorded market data through the identical engine, under a simulated
clock. Returns the same metrics object the live engine reports, so a backtest
number and a live number are directly comparable.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from triangulum.backtest.metrics import PerformanceMetrics, compute_metrics
from triangulum.backtest.monte_carlo import block_bootstrap_paths, bootstrap_paths
from triangulum.core.clock import SimulatedClock
from triangulum.core.config import Config
from triangulum.core.types import RunMode

logger = logging.getLogger(__name__)

__all__ = ["BacktestResult", "run_backtest"]


@dataclass
class BacktestResult:
    metrics: PerformanceMetrics
    monte_carlo: dict[str, Any] = field(default_factory=dict)
    engine_snapshot: dict[str, Any] = field(default_factory=dict)
    records_replayed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "monte_carlo": self.monte_carlo,
            "records_replayed": self.records_replayed,
            "engine": self.engine_snapshot,
        }


async def run_backtest(
    config: Config,
    *,
    data_dir: str = "data/recordings",
    speed: float = 0.0,
    max_ticks: int = 0,
) -> BacktestResult:
    """
    Replay ``data_dir`` through the full engine.

    Falls back to the synthetic market when no recordings exist, and says so --
    a backtest on generated data tells you the engine works, not that the
    strategy makes money, and conflating those is how people fool themselves.
    """
    from triangulum.wiring import build_stack

    directory = Path(data_dir)
    has_recordings = directory.exists() and any(directory.glob("*.ndjson*"))

    if not has_recordings:
        logger.warning(
            "no recordings in %s -- running on the SYNTHETIC market instead. "
            "This validates the engine, not the strategy. Record real data with "
            "`triangulum record` before drawing any conclusion about returns.",
            directory,
        )

    config.dashboard.enabled = False
    stack = await build_stack(config, mode=RunMode.BACKTEST, demo=True)
    ticks = max_ticks or 5000

    await stack.engine.start()
    try:
        for _ in range(ticks):
            if stack.market:
                stack.market.step()
            await asyncio.sleep(0)
        # Let the scan loop drain the final ticks.
        await asyncio.sleep(0.5)
    finally:
        await stack.engine.stop()
        snapshot = stack.engine.snapshot()
        await stack.shutdown()

    state = stack.engine.state
    metrics = compute_metrics(
        state.equity_curve,
        state.cycle_returns_bps,
        outcomes=state.outcomes,
        total_fees=float(stack.engine.ledger.total_fees),
        daily_target_bps=config.daily_target_bps,
        monthly_target_pct=config.monthly_target_pct,
    )

    monte_carlo: dict[str, Any] = {}
    if len(state.cycle_returns_bps) >= 30:
        monte_carlo = {
            "iid": bootstrap_paths(
                state.cycle_returns_bps,
                starting_equity=float(config.capital), runs=2000,
            ).to_dict(),
            "block": block_bootstrap_paths(
                state.cycle_returns_bps,
                starting_equity=float(config.capital), runs=2000,
            ).to_dict(),
        }

    return BacktestResult(
        metrics=metrics,
        monte_carlo=monte_carlo,
        engine_snapshot=snapshot,
        records_replayed=ticks,
    )
