"""
Performance metrics.

Two principles govern this module:

**Report the metric that can embarrass you.** Sharpe is easy to inflate on a
high-frequency strategy by annualising from a short sample. Hit rate looks
wonderful on a strategy whose losses are ten times its wins. So this module
reports the uncomfortable ones alongside: profit factor, worst loss, the
distribution of outcomes, and how long the sample actually is.

**Annualisation is a lie you must label.** A strategy observed for three days
has no annual Sharpe. Every annualised figure here carries the sample length
alongside it, and ``sample_adequacy`` states plainly whether the number should
be believed. A Sharpe of 8 from 40 trades is noise wearing a suit.

The specific trap for arbitrage strategies: they produce many small wins and
rare large losses. That distribution makes Sharpe look extraordinary right up
until the first real loss, because Sharpe uses standard deviation, which barely
notices a fat left tail. Sortino, Calmar, and the tail statistics are the
honest view, and they are all here.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from triangulum.core.decimal_math import D, ZERO, bps, safe_div

__all__ = ["PerformanceMetrics", "compute_metrics", "drawdown_series", "TargetTracking"]


@dataclass(slots=True)
class TargetTracking:
    """
    Progress against the configured return target.

    Tracked honestly and prominently. The engine does not chase the target --
    relaxing the EV gate to hit a number is precisely how a positive-expectancy
    system is turned into a negative one -- but the gap between target and
    reality should never be in doubt.
    """

    daily_target_bps: float
    monthly_target_pct: float
    realized_daily_bps: float = 0.0
    realized_monthly_pct: float = 0.0
    days_observed: float = 0.0
    days_target_met: int = 0

    @property
    def daily_attainment(self) -> float:
        """Realized as a fraction of target. 1.0 = on target."""
        return (
            self.realized_daily_bps / self.daily_target_bps
            if self.daily_target_bps else 0.0
        )

    # A sample shorter than this cannot be annualised without producing a
    # number that is arithmetic rather than information.
    MIN_DAYS_TO_ANNUALISE = 0.25

    @property
    def annualisable(self) -> bool:
        return self.days_observed >= self.MIN_DAYS_TO_ANNUALISE

    @property
    def implied_annual_pct(self) -> float:
        """
        What the realized daily rate compounds to over a year.

        Guarded twice. First against a sample too short to extrapolate from --
        a 40-second observation implies a "daily" rate in the thousands of
        percent, and compounding that over 365 days overflows a float before it
        finishes being meaningless. Second against the overflow itself, because
        a legitimate-looking daily rate of 5% still produces 10^7 percent a
        year and the exponent is what breaks, not the premise.
        """
        if not self.annualisable:
            return float("nan")
        return _compound(self.realized_daily_bps)

    @property
    def target_implied_annual_pct(self) -> float:
        return _compound(self.daily_target_bps)

    def to_dict(self) -> dict[str, object]:
        return {
            "daily_target_bps": self.daily_target_bps,
            "realized_daily_bps": round(self.realized_daily_bps, 3),
            "attainment": round(self.daily_attainment, 4),
            "days_observed": round(self.days_observed, 2),
            "days_target_met": self.days_target_met,
            "implied_annual_pct": (
                round(self.implied_annual_pct, 1) if self.annualisable else None
            ),
            "target_implied_annual_pct": round(self.target_implied_annual_pct, 1),
            "annualisable": self.annualisable,
        }


def _compound(daily_bps: float, *, days: int = 365, cap: float = 1e12) -> float:
    """Compound a daily bps rate over a year, saturating instead of overflowing."""
    daily = daily_bps / 10_000
    if daily <= -1:
        return -100.0
    try:
        # Work in logs: exp(365 * ln(1+r)) overflows far later than (1+r)**365,
        # and the log is exact for the small rates this system actually sees.
        exponent = days * math.log1p(daily)
        if exponent > 27:            # e^27 ~ 5e11, already past any real meaning
            return cap
        return (math.exp(exponent) - 1) * 100
    except (OverflowError, ValueError):
        return cap


@dataclass(slots=True)
class PerformanceMetrics:
    # Returns
    total_return_pct: float = 0.0
    final_equity: float = 0.0
    starting_equity: float = 0.0
    net_pnl: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    mar: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_hours: float = 0.0
    current_drawdown_pct: float = 0.0

    # Trade statistics
    total_cycles: int = 0
    completed_cycles: int = 0
    aborted_cycles: int = 0
    unwound_cycles: int = 0
    stuck_cycles: int = 0
    winning_cycles: int = 0
    losing_cycles: int = 0

    hit_rate: float = 0.0
    completion_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_bps: float = 0.0

    mean_win_bps: float = 0.0
    mean_loss_bps: float = 0.0
    largest_win_bps: float = 0.0
    largest_loss_bps: float = 0.0

    # Tail
    var_95_bps: float = 0.0
    cvar_95_bps: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Costs
    total_fees: float = 0.0
    fees_as_pct_of_gross: float = 0.0

    # Sample quality
    days_observed: float = 0.0
    cycles_per_day: float = 0.0
    sample_adequacy: str = "insufficient"

    target: TargetTracking | None = None

    def to_dict(self) -> dict[str, object]:
        data = {
            "total_return_pct": round(self.total_return_pct, 4),
            "net_pnl": round(self.net_pnl, 6),
            "final_equity": round(self.final_equity, 6),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "calmar": round(self.calmar, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "max_drawdown_duration_hours": round(self.max_drawdown_duration_hours, 2),
            "total_cycles": self.total_cycles,
            "completed_cycles": self.completed_cycles,
            "aborted_cycles": self.aborted_cycles,
            "unwound_cycles": self.unwound_cycles,
            "stuck_cycles": self.stuck_cycles,
            "hit_rate": round(self.hit_rate, 4),
            "completion_rate": round(self.completion_rate, 4),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy_bps": round(self.expectancy_bps, 4),
            "mean_win_bps": round(self.mean_win_bps, 3),
            "mean_loss_bps": round(self.mean_loss_bps, 3),
            "largest_loss_bps": round(self.largest_loss_bps, 3),
            "var_95_bps": round(self.var_95_bps, 3),
            "cvar_95_bps": round(self.cvar_95_bps, 3),
            "skewness": round(self.skewness, 3),
            "total_fees": round(self.total_fees, 6),
            "fees_as_pct_of_gross": round(self.fees_as_pct_of_gross, 2),
            "days_observed": round(self.days_observed, 3),
            "cycles_per_day": round(self.cycles_per_day, 1),
            "sample_adequacy": self.sample_adequacy,
        }
        if self.target:
            data["target"] = self.target.to_dict()
        return data

    def report(self) -> str:
        lines = [
            "=" * 64,
            "PERFORMANCE",
            "=" * 64,
            f"  Return           {self.total_return_pct:>+10.3f}%   "
            f"(net {self.net_pnl:+.4f})",
            f"  Sharpe           {self.sharpe:>10.2f}    "
            f"Sortino {self.sortino:.2f}   Calmar {self.calmar:.2f}",
            f"  Max drawdown     {self.max_drawdown_pct:>10.3f}%   "
            f"for {self.max_drawdown_duration_hours:.1f}h",
            "",
            f"  Cycles           {self.total_cycles:>10}    "
            f"{self.completed_cycles} completed, {self.aborted_cycles} aborted, "
            f"{self.stuck_cycles} stuck",
            f"  Hit rate         {self.hit_rate:>10.1%}    "
            f"profit factor {self.profit_factor:.2f}",
            f"  Expectancy       {self.expectancy_bps:>+10.3f} bps per cycle",
            f"  Win / loss       {self.mean_win_bps:>+10.2f} / "
            f"{self.mean_loss_bps:+.2f} bps",
            f"  Worst loss       {self.largest_loss_bps:>+10.2f} bps   "
            f"CVaR95 {self.cvar_95_bps:+.2f} bps",
            "",
            f"  Fees             {self.total_fees:>10.4f}    "
            f"{self.fees_as_pct_of_gross:.1f}% of gross profit",
            f"  Sample           {self.days_observed:>10.2f} days  "
            f"({self.cycles_per_day:.0f} cycles/day)  -> {self.sample_adequacy.upper()}",
        ]
        if self.target:
            t = self.target
            realized_annual = (
                f"{t.implied_annual_pct:,.0f}%/yr"
                if t.annualisable else "not annualisable"
            )
            lines += [
                "",
                "-" * 64,
                "TARGET TRACKING",
                "-" * 64,
                f"  Daily target     {t.daily_target_bps:>10.1f} bps "
                f"({t.daily_target_bps/100:.2f}%/day = "
                f"{t.target_implied_annual_pct:,.0f}%/yr)",
                f"  Realized         {t.realized_daily_bps:>+10.1f} bps "
                f"({t.realized_daily_bps/100:+.3f}%/day = {realized_annual})",
                f"  Attainment       {t.daily_attainment:>10.1%}",
            ]
            if not t.annualisable:
                lines.append(
                    f"  NOTE: {t.days_observed:.3f} days observed. Any annualised "
                    f"figure from a sample this short is arithmetic, not evidence."
                )
        lines.append("=" * 64)
        return "\n".join(lines)


def compute_metrics(
    equity_curve: Sequence[tuple[int, float]],
    cycle_returns_bps: Sequence[float],
    *,
    outcomes: Sequence[str] = (),
    total_fees: float = 0.0,
    daily_target_bps: float = 0.0,
    monthly_target_pct: float = 0.0,
    periods_per_year: float = 0.0,
) -> PerformanceMetrics:
    """
    Compute the full metric set.

    ``equity_curve`` is ``[(timestamp_ns, equity)]``. ``cycle_returns_bps`` is
    the per-cycle realized return in basis points, including zeros for cycles
    that aborted without committing capital -- excluding them would report the
    expectancy of "cycles we chose to complete", which is not a number you can
    trade.
    """
    m = PerformanceMetrics()
    if not equity_curve:
        return m

    m.starting_equity = equity_curve[0][1]
    m.final_equity = equity_curve[-1][1]
    m.net_pnl = m.final_equity - m.starting_equity
    m.total_return_pct = (
        (m.final_equity / m.starting_equity - 1) * 100 if m.starting_equity > 0 else 0.0
    )

    span_ns = equity_curve[-1][0] - equity_curve[0][0]
    m.days_observed = span_ns / (86_400 * 1e9) if span_ns > 0 else 0.0

    # -- returns series --
    returns = []
    for (_, prev), (_, current) in zip(equity_curve, equity_curve[1:]):
        if prev > 0:
            returns.append(current / prev - 1)

    if len(returns) >= 2:
        mean_return = statistics.fmean(returns)
        stdev = statistics.pstdev(returns)
        # Annualise from the observed sampling frequency, or from the caller's
        # override. Both are labelled by ``sample_adequacy``.
        if not periods_per_year:
            periods_per_year = (
                len(returns) / m.days_observed * 365 if m.days_observed > 0 else 0.0
            )
        scale = math.sqrt(periods_per_year) if periods_per_year > 0 else 0.0

        m.sharpe = (mean_return / stdev) * scale if stdev > 1e-12 else 0.0
        downside = [r for r in returns if r < 0]
        downside_dev = statistics.pstdev(downside) if len(downside) >= 2 else 0.0
        m.sortino = (
            (mean_return / downside_dev) * scale if downside_dev > 1e-12 else 0.0
        )

    # -- drawdown --
    drawdowns, max_dd, max_duration = drawdown_series(equity_curve)
    m.max_drawdown_pct = max_dd * 100
    m.max_drawdown_duration_hours = max_duration / 3600
    m.current_drawdown_pct = drawdowns[-1] * 100 if drawdowns else 0.0

    annual_return = (
        m.total_return_pct / m.days_observed * 365 if m.days_observed > 0 else 0.0
    )
    m.calmar = annual_return / m.max_drawdown_pct if m.max_drawdown_pct > 1e-9 else 0.0
    m.mar = m.calmar

    # -- trade statistics --
    m.total_cycles = len(cycle_returns_bps)
    wins = [r for r in cycle_returns_bps if r > 0]
    losses = [r for r in cycle_returns_bps if r < 0]
    m.winning_cycles = len(wins)
    m.losing_cycles = len(losses)
    m.hit_rate = len(wins) / m.total_cycles if m.total_cycles else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else (
        float("inf") if gross_profit > 0 else 0.0
    )
    m.expectancy_bps = (
        statistics.fmean(cycle_returns_bps) if cycle_returns_bps else 0.0
    )
    m.mean_win_bps = statistics.fmean(wins) if wins else 0.0
    m.mean_loss_bps = statistics.fmean(losses) if losses else 0.0
    m.largest_win_bps = max(wins) if wins else 0.0
    m.largest_loss_bps = min(losses) if losses else 0.0

    if outcomes:
        m.completed_cycles = sum(1 for o in outcomes if o == "completed")
        m.aborted_cycles = sum(1 for o in outcomes if o == "aborted")
        m.unwound_cycles = sum(1 for o in outcomes if o == "partial_unwound")
        m.stuck_cycles = sum(1 for o in outcomes if o == "partial_stuck")
        m.completion_rate = m.completed_cycles / len(outcomes) if outcomes else 0.0

    # -- tail risk --
    if len(cycle_returns_bps) >= 20:
        ordered = sorted(cycle_returns_bps)
        index = max(0, int(len(ordered) * 0.05))
        m.var_95_bps = ordered[index]
        tail = ordered[:index + 1]
        m.cvar_95_bps = statistics.fmean(tail) if tail else 0.0
        m.skewness = _skewness(cycle_returns_bps)
        m.kurtosis = _kurtosis(cycle_returns_bps)

    # -- costs --
    m.total_fees = total_fees
    gross = m.net_pnl + total_fees
    m.fees_as_pct_of_gross = (total_fees / gross * 100) if gross > 1e-12 else 0.0

    # -- sample adequacy --
    m.cycles_per_day = (
        m.total_cycles / m.days_observed if m.days_observed > 0 else 0.0
    )
    m.sample_adequacy = _adequacy(m.total_cycles, m.days_observed)

    if daily_target_bps:
        realized_daily = (
            m.total_return_pct * 100 / m.days_observed if m.days_observed > 0 else 0.0
        )
        m.target = TargetTracking(
            daily_target_bps=daily_target_bps,
            monthly_target_pct=monthly_target_pct,
            realized_daily_bps=realized_daily,
            realized_monthly_pct=realized_daily * 30 / 100,
            days_observed=m.days_observed,
        )
    return m


def drawdown_series(
    equity_curve: Sequence[tuple[int, float]]
) -> tuple[list[float], float, float]:
    """Returns (drawdown fractions, max drawdown, longest underwater seconds)."""
    peak = equity_curve[0][1] if equity_curve else 0.0
    peak_ts = equity_curve[0][0] if equity_curve else 0
    drawdowns: list[float] = []
    max_dd = 0.0
    max_duration_ns = 0

    for ts, equity in equity_curve:
        if equity > peak:
            peak = equity
            peak_ts = ts
        dd = (peak - equity) / peak if peak > 0 else 0.0
        drawdowns.append(dd)
        max_dd = max(max_dd, dd)
        if dd > 0:
            max_duration_ns = max(max_duration_ns, ts - peak_ts)

    return drawdowns, max_dd, max_duration_ns / 1e9


def _adequacy(cycles: int, days: float) -> str:
    """
    A blunt statement about whether the numbers mean anything.

    The thresholds come from the standard error of a Sharpe estimate: with N
    observations the standard error is roughly sqrt((1 + S^2/2)/N), so a Sharpe
    of 2 measured over 100 trades has a standard error near 0.15 -- usable --
    while over 20 trades it is 0.33, which is not.
    """
    if cycles < 30 or days < 1:
        return "insufficient"
    if cycles < 100 or days < 7:
        return "preliminary"
    if cycles < 500 or days < 30:
        return "indicative"
    return "adequate"


def _skewness(values: Sequence[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    if sd < 1e-12:
        return 0.0
    return sum(((v - mean) / sd) ** 3 for v in values) / n


def _kurtosis(values: Sequence[float]) -> float:
    """Excess kurtosis: 0 for a normal distribution."""
    n = len(values)
    if n < 4:
        return 0.0
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    if sd < 1e-12:
        return 0.0
    return sum(((v - mean) / sd) ** 4 for v in values) / n - 3.0
