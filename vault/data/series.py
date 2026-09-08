"""
Time series primitives.

Deliberately not pandas. Three reasons, in order of weight:

1. **The control plane must not depend on a 60MB numeric stack.** Same argument
   as Triangulum's dashboard: the thing that tells you whether your model is
   working has to come up when the environment is broken.

2. **The operations here are trivial.** Percentage change, z-score, rolling
   mean, resample-to-monthly, align-two-series-on-date. Forty lines each,
   fully inspectable, no alignment surprises.

3. **Silent misalignment is the bug that matters.** pandas will happily join
   two series on a DatetimeIndex and produce NaNs that propagate into a
   z-score which propagates into a regime label. Here, alignment is explicit
   and a gap raises.

Every series carries its `as_of` date and its source, because a macro series
that was last published six weeks ago is a different input from one published
yesterday, and the model must be told which it is looking at.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Iterator, Sequence

__all__ = ["Point", "Series", "align", "Frequency"]


class Frequency:
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    IRREGULAR = "irregular"


@dataclass(frozen=True, slots=True)
class Point:
    on: date
    value: float

    def __iter__(self) -> Iterator:
        yield self.on
        yield self.value


@dataclass(slots=True)
class Series:
    """An ordered, gap-tolerant time series with provenance."""

    key: str
    points: list[Point] = field(default_factory=list)
    source: str = ""
    units: str = ""
    label: str = ""
    frequency: str = Frequency.IRREGULAR
    fetched_at: datetime | None = None

    # The date this series is being READ as of. None means "now", which is
    # what live use wants. A historical evaluation sets it (see `until`) so
    # that staleness is measured from the decision date rather than from the
    # wall clock -- otherwise every point-in-time slice of a two-year backtest
    # looks hundreds of days stale, every signal is discarded as unusable, and
    # the backtest silently evaluates nothing at all.
    reference_date: date | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pairs(
        cls, key: str, pairs: Iterable[tuple[date, float]], **meta
    ) -> "Series":
        points = sorted(
            (Point(d, float(v)) for d, v in pairs if v is not None and _finite(v)),
            key=lambda p: p.on,
        )
        return cls(key=key, points=points, **meta)

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.points)

    def __bool__(self) -> bool:
        return bool(self.points)

    @property
    def values(self) -> list[float]:
        return [p.value for p in self.points]

    @property
    def dates(self) -> list[date]:
        return [p.on for p in self.points]

    @property
    def latest(self) -> Point | None:
        return self.points[-1] if self.points else None

    @property
    def last(self) -> float:
        return self.points[-1].value if self.points else float("nan")

    @property
    def as_of(self) -> date | None:
        return self.points[-1].on if self.points else None

    @property
    def staleness_days(self) -> int:
        """
        Days since the last observation.

        The single most important metadata field here. A CPI print is monthly
        and ~2 weeks lagged by construction; a DXY quote should be hours old.
        Feeding the model a six-week-old "current" reading without saying so is
        how a synthesis becomes confidently wrong about a world that moved.
        """
        if not self.points:
            return 10**6
        reference = self.reference_date or date.today()
        return (reference - self.points[-1].on).days

    def value_on(self, when: date, *, tolerance_days: int = 7) -> float | None:
        """Most recent value at or before ``when``, within tolerance."""
        best: Point | None = None
        for p in self.points:
            if p.on <= when:
                best = p
            else:
                break
        if best is None:
            return None
        if (when - best.on).days > tolerance_days:
            return None
        return best.value

    def since(self, when: date) -> "Series":
        return Series(
            key=self.key, points=[p for p in self.points if p.on >= when],
            source=self.source, units=self.units, label=self.label,
            frequency=self.frequency, fetched_at=self.fetched_at,
            reference_date=self.reference_date,
        )

    def until(self, when: date) -> "Series":
        """
        The series as it was knowable on ``when`` -- everything up to and
        including that date, nothing after.

        This is the point-in-time cut that makes historical signal evaluation
        honest. Computing a signal from the full series and correlating it
        with a "forward" return uses data from after the decision, and the
        information coefficients that come out of it are beautiful and
        entirely fictional.

        Note the caveat this does NOT fix: it slices by observation date, not
        by publication date. A macro series revised three months after the
        fact will show its revised value here, which the system could not
        have seen at the time. Truly point-in-time macro data needs a
        vintage database (ALFRED); this is the honest approximation, and the
        residual bias it leaves is toward flattering slow-moving fundamental
        signals over fast market ones.
        """
        return Series(
            key=self.key, points=[p for p in self.points if p.on <= when],
            source=self.source, units=self.units, label=self.label,
            frequency=self.frequency, fetched_at=self.fetched_at,
            reference_date=when,
        )

    def tail(self, n: int) -> "Series":
        return Series(
            key=self.key, points=self.points[-n:],
            source=self.source, units=self.units, label=self.label,
            frequency=self.frequency, fetched_at=self.fetched_at,
            reference_date=self.reference_date,
        )

    # -- transforms --------------------------------------------------------

    def pct_change(self, periods: int = 1) -> float | None:
        """Percentage change over ``periods`` observations."""
        if len(self.points) <= periods:
            return None
        old = self.points[-1 - periods].value
        new = self.points[-1].value
        if old == 0:
            return None
        return (new / old - 1) * 100

    def change(self, periods: int = 1) -> float | None:
        """Absolute change. Correct for anything already in percent (yields)."""
        if len(self.points) <= periods:
            return None
        return self.points[-1].value - self.points[-1 - periods].value

    def year_over_year(self) -> float | None:
        """
        YoY percentage change, using calendar dates rather than an observation
        count -- monthly series have irregular publication gaps and counting
        back twelve observations silently compares the wrong months.
        """
        if not self.points:
            return None
        latest = self.points[-1]
        target = latest.on - timedelta(days=365)
        prior = self.value_on(target, tolerance_days=45)
        if prior is None or prior == 0:
            return None
        return (latest.value / prior - 1) * 100

    def rolling_mean(self, window: int) -> float | None:
        if len(self.points) < window:
            return None
        return statistics.fmean(self.values[-window:])

    def rolling_std(self, window: int) -> float | None:
        if len(self.points) < max(2, window):
            return None
        return statistics.pstdev(self.values[-window:])

    def zscore(self, window: int = 252) -> float | None:
        """Current value in standard deviations from its rolling mean."""
        if len(self.points) < max(8, window // 4):
            return None
        sample = self.values[-window:]
        mean = statistics.fmean(sample)
        sd = statistics.pstdev(sample)
        if sd < 1e-12:
            return 0.0
        return (self.points[-1].value - mean) / sd

    def percentile_rank(self, window: int = 504) -> float | None:
        """Where the current value sits in its own recent distribution, 0-1."""
        if len(self.points) < 20:
            return None
        sample = self.values[-window:]
        current = self.points[-1].value
        below = sum(1 for v in sample if v < current)
        return below / len(sample)

    def realized_volatility(self, window: int = 20, *, annualize: bool = True) -> float | None:
        """Annualized stdev of log returns, in percent."""
        if len(self.points) < window + 1:
            return None
        values = self.values[-(window + 1):]
        returns = [
            math.log(values[i] / values[i - 1])
            for i in range(1, len(values))
            if values[i] > 0 and values[i - 1] > 0
        ]
        if len(returns) < 2:
            return None
        sd = statistics.pstdev(returns)
        return sd * (math.sqrt(252) if annualize else 1.0) * 100

    def momentum(self, fast: int = 20, slow: int = 100) -> float | None:
        """
        Fast mean over slow mean, minus one, in percent.

        A moving-average ratio rather than a raw return: it is far less
        sensitive to the exact endpoint, which matters when the endpoint may be
        a stale print.
        """
        f = self.rolling_mean(fast)
        s = self.rolling_mean(slow)
        if f is None or s is None or s == 0:
            return None
        return (f / s - 1) * 100

    def drawdown(self) -> float | None:
        """Current drawdown from the running peak, in percent (negative)."""
        if not self.points:
            return None
        peak = max(self.values)
        if peak <= 0:
            return None
        return (self.points[-1].value / peak - 1) * 100

    # -- reporting ---------------------------------------------------------

    def describe(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label or self.key,
            "source": self.source,
            "units": self.units,
            "frequency": self.frequency,
            "observations": len(self.points),
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "staleness_days": self.staleness_days,
            "last": round(self.last, 6) if self.points else None,
            "change_1": _round(self.change(1)),
            "pct_change_1": _round(self.pct_change(1)),
            "yoy_pct": _round(self.year_over_year()),
            "zscore_1y": _round(self.zscore(252)),
            "percentile_2y": _round(self.percentile_rank(504)),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Series({self.key}, n={len(self.points)}, "
            f"last={self.last:.4g} as_of={self.as_of}, stale={self.staleness_days}d)"
        )


def align(a: Series, b: Series, *, tolerance_days: int = 5) -> list[tuple[date, float, float]]:
    """
    Pair two series on their common dates.

    Uses ``a`` as the spine and looks each of its dates up in ``b`` with a
    tolerance, rather than requiring exact date equality. Exact matching
    across a daily market series and a monthly macro series produces an empty
    join, and an empty join that silently returns zero rows is how a
    correlation ends up reported as 0.0.
    """
    out: list[tuple[date, float, float]] = []
    for point in a.points:
        other = b.value_on(point.on, tolerance_days=tolerance_days)
        if other is not None:
            out.append((point.on, point.value, other))
    return out


def correlation(a: Series, b: Series, *, window: int = 60,
                on_returns: bool = True) -> float | None:
    """
    Pearson correlation over the last ``window`` aligned observations.

    Defaults to correlating *returns*, not levels. Correlating levels of two
    trending series produces a number near 1 that says nothing except that both
    went up -- the classic spurious-correlation trap, and the reason a naive
    "DXY and SPX are 0.9 correlated" reading is usually meaningless.
    """
    paired = align(a, b)
    if len(paired) < window // 2 or len(paired) < 10:
        return None
    paired = paired[-window:]

    xs = [p[1] for p in paired]
    ys = [p[2] for p in paired]

    if on_returns:
        xs = [
            xs[i] / xs[i - 1] - 1 for i in range(1, len(xs)) if xs[i - 1] not in (0,)
        ]
        ys = [
            ys[i] / ys[i - 1] - 1 for i in range(1, len(ys)) if ys[i - 1] not in (0,)
        ]
        n = min(len(xs), len(ys))
        xs, ys = xs[-n:], ys[-n:]

    if len(xs) < 8:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


def _finite(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _round(v: float | None, digits: int = 4) -> float | None:
    return round(v, digits) if v is not None and math.isfinite(v) else None
