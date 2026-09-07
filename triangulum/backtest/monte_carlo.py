"""
Monte Carlo analysis of a trade sequence.

A backtest produces one path. That path is a single draw from a distribution,
and the most dangerous thing a backtest can do is present it as though it were
the expected outcome.

Two resampling schemes:

**IID bootstrap** shuffles trades with replacement. Answers: given this
distribution of per-trade outcomes, what range of equity curves is consistent
with it? Assumes trades are independent, which for arbitrage is nearly true --
each cycle is a separate event with its own book state.

**Block bootstrap** resamples contiguous blocks, preserving short-range
autocorrelation. Necessary when trades cluster: a bad afternoon produces a run
of correlated losses, and IID resampling would break up those runs and
systematically understate the drawdown.

The output that matters is not the mean -- the mean is just the backtest again.
It is the 5th percentile of final equity and the 95th percentile of maximum
drawdown. Those are the numbers to size against, because they are the outcomes
you have to survive.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["MonteCarloResult", "bootstrap_paths", "block_bootstrap_paths"]


@dataclass(slots=True)
class MonteCarloResult:
    runs: int
    starting_equity: float

    final_equity_p5: float = 0.0
    final_equity_p25: float = 0.0
    final_equity_median: float = 0.0
    final_equity_p75: float = 0.0
    final_equity_p95: float = 0.0

    max_drawdown_median: float = 0.0
    max_drawdown_p95: float = 0.0
    max_drawdown_worst: float = 0.0

    probability_of_loss: float = 0.0
    probability_of_ruin: float = 0.0
    ruin_threshold_pct: float = 50.0

    method: str = "iid"

    def to_dict(self) -> dict[str, object]:
        return {
            "runs": self.runs,
            "method": self.method,
            "final_equity": {
                "p5": round(self.final_equity_p5, 4),
                "p25": round(self.final_equity_p25, 4),
                "median": round(self.final_equity_median, 4),
                "p75": round(self.final_equity_p75, 4),
                "p95": round(self.final_equity_p95, 4),
            },
            "max_drawdown_pct": {
                "median": round(self.max_drawdown_median, 3),
                "p95": round(self.max_drawdown_p95, 3),
                "worst": round(self.max_drawdown_worst, 3),
            },
            "probability_of_loss": round(self.probability_of_loss, 4),
            "probability_of_ruin": round(self.probability_of_ruin, 4),
            "ruin_threshold_pct": self.ruin_threshold_pct,
        }

    def report(self) -> str:
        return "\n".join([
            f"Monte Carlo ({self.runs} paths, {self.method} resampling)",
            f"  Final equity   p5 {self.final_equity_p5:.4f}  "
            f"median {self.final_equity_median:.4f}  p95 {self.final_equity_p95:.4f}",
            f"  Max drawdown   median {self.max_drawdown_median:.2f}%  "
            f"p95 {self.max_drawdown_p95:.2f}%  worst {self.max_drawdown_worst:.2f}%",
            f"  P(losing money) {self.probability_of_loss:.1%}   "
            f"P(down {self.ruin_threshold_pct:.0f}%) {self.probability_of_ruin:.2%}",
        ])


def _simulate(
    returns_bps: Sequence[float], starting_equity: float,
) -> tuple[float, float]:
    """Apply a return sequence multiplicatively; return (final, max drawdown %)."""
    equity = starting_equity
    peak = starting_equity
    max_dd = 0.0
    for r in returns_bps:
        equity *= (1 + r / 10_000)
        if equity > peak:
            peak = equity
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
    return equity, max_dd


def _summarize(
    finals: list[float], drawdowns: list[float], starting_equity: float,
    runs: int, method: str, ruin_threshold_pct: float,
) -> MonteCarloResult:
    finals.sort()
    drawdowns.sort()

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, int(len(values) * p)))
        return values[index]

    ruin_level = starting_equity * (1 - ruin_threshold_pct / 100)
    return MonteCarloResult(
        runs=runs,
        starting_equity=starting_equity,
        final_equity_p5=pct(finals, 0.05),
        final_equity_p25=pct(finals, 0.25),
        final_equity_median=pct(finals, 0.50),
        final_equity_p75=pct(finals, 0.75),
        final_equity_p95=pct(finals, 0.95),
        max_drawdown_median=pct(drawdowns, 0.50),
        max_drawdown_p95=pct(drawdowns, 0.95),
        max_drawdown_worst=drawdowns[-1] if drawdowns else 0.0,
        probability_of_loss=(
            sum(1 for f in finals if f < starting_equity) / len(finals)
            if finals else 0.0
        ),
        probability_of_ruin=(
            sum(1 for f in finals if f <= ruin_level) / len(finals)
            if finals else 0.0
        ),
        ruin_threshold_pct=ruin_threshold_pct,
        method=method,
    )


def bootstrap_paths(
    returns_bps: Sequence[float],
    *,
    starting_equity: float = 100.0,
    runs: int = 2000,
    path_length: int | None = None,
    seed: int | None = 42,
    ruin_threshold_pct: float = 50.0,
) -> MonteCarloResult:
    """IID bootstrap: resample individual trades with replacement."""
    if not returns_bps:
        return MonteCarloResult(runs=0, starting_equity=starting_equity)

    rng = random.Random(seed)
    length = path_length or len(returns_bps)
    finals: list[float] = []
    drawdowns: list[float] = []

    for _ in range(runs):
        path = [rng.choice(returns_bps) for _ in range(length)]
        final, dd = _simulate(path, starting_equity)
        finals.append(final)
        drawdowns.append(dd)

    return _summarize(finals, drawdowns, starting_equity, runs, "iid", ruin_threshold_pct)


def block_bootstrap_paths(
    returns_bps: Sequence[float],
    *,
    starting_equity: float = 100.0,
    runs: int = 2000,
    block_size: int | None = None,
    seed: int | None = 42,
    ruin_threshold_pct: float = 50.0,
) -> MonteCarloResult:
    """
    Moving-block bootstrap: preserves short-range dependence.

    Default block size is ``n^(1/3)``, the standard rule of thumb, which
    balances preserving autocorrelation against having enough distinct blocks
    to resample from.
    """
    if not returns_bps:
        return MonteCarloResult(runs=0, starting_equity=starting_equity)

    rng = random.Random(seed)
    n = len(returns_bps)
    size = block_size or max(2, int(round(n ** (1 / 3))))
    blocks = [returns_bps[i:i + size] for i in range(0, max(1, n - size + 1))]
    if not blocks:
        blocks = [list(returns_bps)]

    finals: list[float] = []
    drawdowns: list[float] = []
    needed = math.ceil(n / size)

    for _ in range(runs):
        path: list[float] = []
        for _ in range(needed):
            path.extend(rng.choice(blocks))
        final, dd = _simulate(path[:n], starting_equity)
        finals.append(final)
        drawdowns.append(dd)

    return _summarize(
        finals, drawdowns, starting_equity, runs, f"block({size})", ruin_threshold_pct
    )
