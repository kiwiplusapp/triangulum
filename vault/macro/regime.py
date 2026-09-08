"""
Regime classification: the growth/inflation quadrant.

The organising idea is old and durable: at any moment, growth and inflation are
each either accelerating or decelerating, and the four combinations sort
cross-asset returns better than almost any single indicator. It underpins risk
parity, the "four quadrants" framing, and most macro allocation frameworks.

    GOLDILOCKS   growth up,   inflation down   -> equities, credit, duration
    REFLATION    growth up,   inflation up     -> commodities, value, banks
    STAGFLATION  growth down, inflation up     -> cash, gold, energy; worst for 60/40
    DEFLATION    growth down, inflation down   -> long duration, dollar, quality

Two things this implementation is careful about, because both are where naive
versions go wrong:

**Second derivatives, not levels.** The quadrant is about *acceleration*. 3%
inflation falling from 5% is a completely different regime from 3% rising from
1%, and a classifier keyed on the level calls them the same thing.

**Publication lag is real.** Core PCE arrives about six weeks after the month
it describes. A classifier that treats the latest print as "now" is describing
a world that is a month and a half old. So every component carries its own
staleness, and the confidence score is reduced when the inputs are old --
rather than the regime silently being reported as though it were current.

The output is a probability distribution over four regimes, not a single label.
Regimes are not discrete in reality; transitions take months and the middle of
one is exactly when a single hard label is most confidently wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from vault.data.series import Series

__all__ = ["Quadrant", "RegimeRead", "classify_regime", "sahm_rule"]


class Quadrant:
    GOLDILOCKS = "goldilocks"
    REFLATION = "reflation"
    STAGFLATION = "stagflation"
    DEFLATION = "deflation"
    UNKNOWN = "unknown"

    ALL = (GOLDILOCKS, REFLATION, STAGFLATION, DEFLATION)

    DESCRIPTION: Mapping[str, str] = {
        GOLDILOCKS: "growth accelerating, inflation decelerating",
        REFLATION: "growth accelerating, inflation accelerating",
        STAGFLATION: "growth decelerating, inflation accelerating",
        DEFLATION: "growth decelerating, inflation decelerating",
        UNKNOWN: "insufficient data to classify",
    }

    # What each regime has historically favoured. Included so the synthesis
    # layer has a prior to reason from -- and so a thesis that contradicts it
    # has to say why, rather than drifting there unnoticed.
    HISTORICAL_TILT: Mapping[str, str] = {
        GOLDILOCKS: "equities and credit lead; duration supported; dollar soft",
        REFLATION: "commodities, energy, value and banks lead; duration hurts",
        STAGFLATION: "cash, gold and energy; the worst regime for a 60/40 book",
        DEFLATION: "long duration, quality, the dollar; credit spreads widen",
    }


@dataclass(slots=True)
class Component:
    """One input to the growth or inflation score."""

    key: str
    label: str
    value: float | None
    contribution: float          # signed, in [-1, 1]
    weight: float
    staleness_days: int
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.value is not None and math.isfinite(self.value)


@dataclass(slots=True)
class RegimeRead:
    """A classification with its evidence and its honest confidence."""

    quadrant: str
    probabilities: dict[str, float]
    growth_score: float          # [-1, 1], positive = accelerating
    inflation_score: float
    confidence: float            # [0, 1]
    components: list[Component] = field(default_factory=list)
    max_staleness_days: int = 0
    missing: list[str] = field(default_factory=list)
    transitioning: bool = False

    @property
    def description(self) -> str:
        return Quadrant.DESCRIPTION.get(self.quadrant, "")

    @property
    def historical_tilt(self) -> str:
        return Quadrant.HISTORICAL_TILT.get(self.quadrant, "")

    @property
    def runner_up(self) -> tuple[str, float]:
        ranked = sorted(self.probabilities.items(), key=lambda kv: -kv[1])
        return ranked[1] if len(ranked) > 1 else ("", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quadrant": self.quadrant,
            "description": self.description,
            "historical_tilt": self.historical_tilt,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "growth_score": round(self.growth_score, 4),
            "inflation_score": round(self.inflation_score, 4),
            "confidence": round(self.confidence, 4),
            "transitioning": self.transitioning,
            "max_staleness_days": self.max_staleness_days,
            "missing_inputs": self.missing,
            "components": [
                {
                    "key": c.key, "label": c.label,
                    "value": round(c.value, 4) if c.usable else None,
                    "contribution": round(c.contribution, 4),
                    "weight": c.weight, "stale_days": c.staleness_days,
                    "note": c.note,
                }
                for c in self.components
            ],
        }

    def summary(self) -> str:
        second, second_p = self.runner_up
        line = (
            f"{self.quadrant.upper()} ({self.probabilities.get(self.quadrant, 0):.0%}) "
            f"-- {self.description}. growth {self.growth_score:+.2f}, "
            f"inflation {self.inflation_score:+.2f}, confidence {self.confidence:.0%}"
        )
        if self.transitioning:
            line += f" [TRANSITIONING toward {second} at {second_p:.0%}]"
        return line


def _squash(value: float, scale: float) -> float:
    """Map an unbounded reading into [-1, 1] with tanh."""
    if value is None or not math.isfinite(value):
        return 0.0
    return math.tanh(value / scale)


def classify_regime(series: Mapping[str, Series]) -> RegimeRead:
    """
    Classify the current growth/inflation quadrant.

    Growth components are weighted toward the high-frequency and the leading:
    jobless claims are weekly and lead; housing permits lead by roughly two
    quarters; payrolls are monthly and coincident-to-lagging. Weighting them
    equally would let a stale payrolls print outvote four weeks of fresh claims.
    """
    growth: list[Component] = []
    inflation: list[Component] = []
    missing: list[str] = []

    def add(bucket: list[Component], key: str, label: str, weight: float,
            value: float | None, scale: float, *, invert: bool = False,
            note: str = "") -> None:
        source = series.get(key)
        stale = source.staleness_days if source else 10**6
        if value is None or not math.isfinite(value):
            missing.append(key)
            bucket.append(Component(key, label, None, 0.0, weight, stale, note))
            return
        contribution = _squash(value, scale) * (-1 if invert else 1)
        bucket.append(Component(key, label, value, contribution, weight, stale, note))

    # ---- growth ----------------------------------------------------------
    claims = series.get("ICSA")
    add(growth, "ICSA", "Initial jobless claims, 13w change", 0.25,
        _pct_change_weeks(claims, 13), 12.0, invert=True,
        note="Rising claims = deteriorating labor market. Weekly, leads.")

    payrolls = series.get("PAYEMS")
    add(growth, "PAYEMS", "Payrolls, 3m vs 12m momentum", 0.20,
        _momentum_gap(payrolls, 3, 12), 0.35,
        note="3-month average against 12-month. Positive = accelerating hiring.")

    unemployment = series.get("UNRATE")
    add(growth, "UNRATE", "Unemployment, 6m change", 0.15,
        unemployment.change(6) if unemployment else None, 0.4, invert=True,
        note="Rising unemployment = decelerating growth.")

    permits = series.get("PERMIT")
    add(growth, "PERMIT", "Building permits, 6m change", 0.15,
        permits.pct_change(6) if permits else None, 10.0,
        note="The most rate-sensitive leading indicator. Leads by ~2 quarters.")

    industrial = series.get("INDPRO")
    add(growth, "INDPRO", "Industrial production, 6m change", 0.10,
        industrial.pct_change(6) if industrial else None, 2.5,
        note="Goods-side activity.")

    curve = series.get("T10Y3M")
    add(growth, "T10Y3M", "10Y-3M curve", 0.15,
        curve.last if curve and curve.points else None, 1.2,
        note="Inversion is the single best-documented recession lead.")

    # ---- inflation -------------------------------------------------------
    core_cpi = series.get("CPILFESL")
    add(inflation, "CPILFESL", "Core CPI, 3m vs 12m", 0.30,
        _inflation_acceleration(core_cpi), 1.0,
        note="3-month annualized minus year-over-year. Positive = reaccelerating.")

    core_pce = series.get("PCEPILFE")
    add(inflation, "PCEPILFE", "Core PCE, 3m vs 12m", 0.25,
        _inflation_acceleration(core_pce), 1.0,
        note="The Fed's actual target measure.")

    breakeven = series.get("T10YIE")
    add(inflation, "T10YIE", "10Y breakeven, 3m change", 0.20,
        breakeven.change(63) if breakeven else None, 0.25,
        note="Market-implied inflation. Fast-moving and forward-looking.")

    forward = series.get("T5YIFR")
    add(inflation, "T5YIFR", "5y5y forward, 3m change", 0.10,
        forward.change(63) if forward else None, 0.2,
        note="Long-run expectations. A move here is what the Fed fears.")

    oil = series.get("DCOILWTICO")
    add(inflation, "DCOILWTICO", "WTI, 3m change", 0.15,
        oil.pct_change(63) if oil else None, 22.0,
        note="Headline inflation input and a growth signal at once.")

    # ---- scores ----------------------------------------------------------
    growth_score = _weighted(growth)
    inflation_score = _weighted(inflation)

    probabilities = _quadrant_probabilities(growth_score, inflation_score)
    quadrant = max(probabilities.items(), key=lambda kv: kv[1])[0]
    top = probabilities[quadrant]
    ranked = sorted(probabilities.values(), reverse=True)
    transitioning = len(ranked) > 1 and (ranked[0] - ranked[1]) < 0.15

    all_components = growth + inflation
    usable = [c for c in all_components if c.usable]
    coverage = sum(c.weight for c in usable) / max(
        1e-9, sum(c.weight for c in all_components)
    )
    max_stale = max((c.staleness_days for c in usable), default=10**6)

    # Confidence falls with missing inputs, with stale inputs, and when two
    # quadrants are close. A confident label on old, partial data is the
    # failure mode this whole class exists to prevent.
    staleness_penalty = min(1.0, max(0.0, (max_stale - 45) / 120)) if max_stale < 10**5 else 1.0
    separation = min(1.0, (top - 0.25) / 0.5)
    confidence = max(0.0, coverage * (1 - 0.5 * staleness_penalty) * (0.4 + 0.6 * separation))

    return RegimeRead(
        quadrant=quadrant if usable else Quadrant.UNKNOWN,
        probabilities=probabilities,
        growth_score=growth_score,
        inflation_score=inflation_score,
        confidence=round(confidence, 4),
        components=all_components,
        max_staleness_days=max_stale if max_stale < 10**5 else -1,
        missing=missing,
        transitioning=transitioning,
    )


def _weighted(components: list[Component]) -> float:
    usable = [c for c in components if c.usable]
    total_weight = sum(c.weight for c in usable)
    if total_weight <= 0:
        return 0.0
    return sum(c.contribution * c.weight for c in usable) / total_weight


def _quadrant_probabilities(growth: float, inflation: float) -> dict[str, float]:
    """
    Turn two continuous scores into a distribution over four quadrants.

    A softmax over the four sign combinations, with the temperature tied to how
    far the scores are from zero. Near the origin -- where growth and inflation
    are both roughly flat -- the distribution correctly approaches uniform,
    which is exactly when a hard label would be least justified.
    """
    magnitude = math.hypot(growth, inflation)
    temperature = max(0.15, 1.0 - magnitude)

    raw = {
        Quadrant.GOLDILOCKS: growth - inflation,
        Quadrant.REFLATION: growth + inflation,
        Quadrant.STAGFLATION: -growth + inflation,
        Quadrant.DEFLATION: -growth - inflation,
    }
    scaled = {k: v / temperature for k, v in raw.items()}
    peak = max(scaled.values())
    exponentiated = {k: math.exp(v - peak) for k, v in scaled.items()}
    total = sum(exponentiated.values())
    return {k: v / total for k, v in exponentiated.items()}


def _pct_change_weeks(series: Series | None, weeks: int) -> float | None:
    if series is None or len(series) <= weeks:
        return None
    return series.pct_change(weeks)


def _momentum_gap(series: Series | None, fast: int, slow: int) -> float | None:
    """Fast-window growth rate minus slow-window growth rate, annualized-ish."""
    if series is None or len(series) <= slow:
        return None
    fast_change = series.pct_change(fast)
    slow_change = series.pct_change(slow)
    if fast_change is None or slow_change is None:
        return None
    return (fast_change / fast * 12) - (slow_change / slow * 12)


def _inflation_acceleration(index: Series | None) -> float | None:
    """
    3-month annualized rate minus the year-over-year rate.

    The standard way to see a turn before it shows in the headline: when the
    recent 3-month run rate exceeds the trailing year, inflation is
    reaccelerating even while the year-over-year number is still falling.
    """
    if index is None or len(index) < 14:
        return None
    values = index.values
    three_month = (values[-1] / values[-4]) ** 4 - 1 if values[-4] > 0 else None
    yoy = index.year_over_year()
    if three_month is None or yoy is None:
        return None
    return three_month * 100 - yoy


def sahm_rule(unemployment: Series | None) -> dict[str, Any]:
    """
    The Sahm rule: a 0.50pp rise in the 3-month average unemployment rate above
    its 12-month low has signalled the start of every US recession since 1970,
    with no false positives on the historical sample.

    Reported separately from the quadrant because it is a *threshold* signal,
    not a continuous one, and because its track record earns it a line of its own.
    """
    if unemployment is None or len(unemployment) < 15:
        return {"available": False, "reason": "insufficient history"}

    values = unemployment.values
    current_3m = sum(values[-3:]) / 3
    trailing = [sum(values[i - 2:i + 1]) / 3 for i in range(2, len(values))][-12:]
    if not trailing:
        return {"available": False, "reason": "insufficient history"}
    low_12m = min(trailing)
    gap = current_3m - low_12m
    return {
        "available": True,
        "value": round(gap, 3),
        "threshold": 0.50,
        "triggered": gap >= 0.50,
        "current_3m_avg": round(current_3m, 3),
        "min_12m": round(low_12m, 3),
        "as_of": unemployment.as_of.isoformat() if unemployment.as_of else None,
        "note": (
            "A 0.50pp rise in the 3-month average unemployment rate above its "
            "12-month low. No false positives since 1970, but note the sample "
            "is roughly a dozen recessions -- treat it as strong evidence, not "
            "as a law."
        ),
    }
