"""
Scoring: Brier, calibration, and skill against the honest baseline.

## Why Brier and not hit rate

Hit rate throws away the confidence. A system that calls everything at 0.51 and
is right 55% of the time and one that calls everything at 0.95 and is right 55%
of the time have identical hit rates and are not remotely equally useful -- the
second is dangerously overconfident and will be sized far too large.

The Brier score is the mean squared error of the stated probabilities:

    BS = mean((p_i - o_i)^2)     where o_i is 1 if correct, 0 if not

Lower is better. The reference point that matters:

    **0.25 is what you get by always saying 0.50.**

Any system scoring above 0.25 is worse than admitting it has no view. That is
the bar, and it is a genuinely hard one -- most discretionary forecasting fails
it, which is precisely why almost nobody measures.

## Murphy's decomposition

Brier decomposes into three interpretable pieces:

    BS = reliability - resolution + uncertainty

    reliability  how far stated probabilities are from realised frequencies.
                 Lower is better. This is calibration.
    resolution   how much the forecasts vary from the base rate, weighted by
                 how often. Higher is better. This is discrimination -- the
                 ability to tell high-probability situations from low ones.
    uncertainty  the base rate's own variance, p(1-p). Not controllable.

The decomposition separates two failures that a single number conflates: a
system can be perfectly calibrated and useless (always predicting the base
rate: good reliability, zero resolution), or highly discriminating and badly
scaled (right ordering, wrong numbers: poor reliability, good resolution). The
fixes are completely different.

## Sample size

A Brier score on twelve forecasts is noise. The standard error of a Brier
estimate is roughly sqrt(var/n), and with n=12 the confidence interval spans
most of the useful range. Every report here carries an explicit adequacy
verdict, and the capital gate refuses to size anything until the sample is
large enough for the number to mean something.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

from vault.thesis.schema import Outcome, ThesisRecord

__all__ = [
    "CalibrationScore", "score_records", "reliability_buckets",
    "wilson_interval", "sequential_wilson_interval", "sequential_z",
]

BASELINE_BRIER = 0.25          # always saying 0.5


@dataclass(slots=True)
class CalibrationScore:
    """The full scoreboard."""

    n: int = 0
    correct: int = 0
    wrong: int = 0
    invalidated: int = 0

    hit_rate: float = 0.0
    hit_rate_low: float = 0.0        # Wilson 95% lower bound, fixed-sample
    hit_rate_high: float = 0.0
    # Always-valid lower bound. Wider than hit_rate_low, and the ONLY one the
    # capital gate is allowed to test against -- see sequential_wilson_interval.
    hit_rate_low_sequential: float = 0.0
    hit_rate_high_sequential: float = 0.0
    mean_probability: float = 0.0

    brier: float = 0.0
    brier_baseline: float = BASELINE_BRIER
    skill: float = 0.0               # 1 - brier/baseline; >0 beats a coin flip

    reliability: float = 0.0
    resolution: float = 0.0
    uncertainty: float = 0.0

    log_loss: float = 0.0
    overconfidence: float = 0.0      # mean probability minus hit rate

    buckets: list[dict[str, Any]] = field(default_factory=list)
    by_asset: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_regime: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_horizon: dict[str, dict[str, Any]] = field(default_factory=dict)

    adequacy: str = "insufficient"
    beats_baseline: bool = False
    significant: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "scoreable": self.n,
            "correct": self.correct,
            "wrong": self.wrong,
            "invalidated": self.invalidated,
            "hit_rate": round(self.hit_rate, 4),
            "hit_rate_ci": [round(self.hit_rate_low, 4), round(self.hit_rate_high, 4)],
            "hit_rate_ci_sequential": [
                round(self.hit_rate_low_sequential, 4),
                round(self.hit_rate_high_sequential, 4),
            ],
            "mean_probability": round(self.mean_probability, 4),
            "brier": round(self.brier, 5),
            "brier_baseline": self.brier_baseline,
            "skill": round(self.skill, 4),
            "reliability": round(self.reliability, 5),
            "resolution": round(self.resolution, 5),
            "uncertainty": round(self.uncertainty, 5),
            "log_loss": round(self.log_loss, 5),
            "overconfidence": round(self.overconfidence, 4),
            "adequacy": self.adequacy,
            "beats_baseline": self.beats_baseline,
            "significant": self.significant,
            "reliability_buckets": self.buckets,
            "by_asset": self.by_asset,
            "by_regime": self.by_regime,
            "by_horizon": self.by_horizon,
        }

    def report(self) -> str:
        lines = [
            "=" * 66,
            "CALIBRATION",
            "=" * 66,
        ]
        if self.n == 0:
            lines += [
                "  No resolved calls yet.",
                "",
                "  This is the honest state of a new system, and it is the state",
                "  the reference terminal was in: 0 runs / 0 resolved. Until this",
                "  number grows, no claim about performance means anything and",
                "  the capital gate holds position size at zero.",
                "=" * 66,
            ]
            return "\n".join(lines)

        verdict = (
            "BEATS the coin-flip baseline" if self.beats_baseline
            else "does NOT beat the coin-flip baseline"
        )
        significance = " (statistically significant)" if self.significant else " (not yet significant)"

        lines += [
            f"  Resolved calls   {self.n:>10}    "
            f"{self.correct} correct, {self.wrong} wrong, {self.invalidated} stopped out",
            f"  Hit rate         {self.hit_rate:>9.1%}    "
            f"95% CI [{self.hit_rate_low:.1%}, {self.hit_rate_high:.1%}]",
            f"  Mean confidence  {self.mean_probability:>9.1%}    "
            f"overconfidence {self.overconfidence:+.1%}",
            "",
            f"  Brier score      {self.brier:>10.4f}    "
            f"baseline {self.brier_baseline:.4f} -- {verdict}{significance}",
            f"  Skill            {self.skill:>+10.4f}    (1 - brier/baseline)",
            "",
            "  Murphy decomposition:",
            f"    reliability    {self.reliability:>10.4f}    lower is better -- calibration",
            f"    resolution     {self.resolution:>10.4f}    higher is better -- discrimination",
            f"    uncertainty    {self.uncertainty:>10.4f}    the base rate's own variance",
            "",
            f"  Sample adequacy  {self.adequacy:>10}",
        ]
        if self.buckets:
            lines += ["", "  Reliability (stated -> realised):"]
            for bucket in self.buckets:
                bar = "#" * max(1, int(bucket["count"] / max(1, self.n) * 30))
                lines.append(
                    f"    {bucket['predicted']:.2f} -> {bucket['observed']:.2f}  "
                    f"n={bucket['count']:<4} {bar}"
                )
        lines.append("=" * 66)
        return "\n".join(lines)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because it stays inside [0, 1] and
    remains sensible at small n -- which is the entire regime this system
    operates in for its first few months. The normal approximation on 8
    successes out of 10 gives an upper bound above 1.0, which is not a number.

    This is a FIXED-SAMPLE interval. It is correct for one look at one sample
    size, and it is what gets reported as "the 95% CI". It is NOT what the
    capital gate tests against -- see ``sequential_wilson_interval``.
    """
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def sequential_z(n: int, alpha: float = 0.05) -> float:
    """
    Time-uniform critical value: the sub-Gaussian normal-mixture boundary.

        z(n) = sqrt( 2 * ln( sqrt(n + 1) / alpha ) )

    Grows like sqrt(log n), which is exactly the rate needed to keep the
    error probability bounded across an UNBOUNDED number of looks rather than
    one. At n=50 it is 3.15 against the fixed 1.96; at n=1000, 3.59.
    """
    if n <= 0:
        return 0.0
    return math.sqrt(2 * math.log(math.sqrt(n + 1) / alpha))


def sequential_wilson_interval(
    successes: int, n: int, alpha: float = 0.05,
) -> tuple[float, float]:
    """
    An always-valid ("anytime") confidence sequence on the hit rate.

    ## Why this exists, and why the capital gate uses it

    The gate is re-evaluated after EVERY resolved call. A one-sided 95% bound
    has a 5% false-positive rate *per look*; checked after each of 200 calls,
    a forecaster with no edge whatsoever will cross it eventually almost
    surely. This is optional stopping, and it is not a subtle effect:

        measured over 2,000 simulated runs of a TRUE 50% agent, 300 calls each,
        testing after every call --

            fixed 95% Wilson bound   -> unlocks capital in 14.1% of runs
            time-uniform bound       ->  unlocks capital in  0.5% of runs

    One in seven no-edge agents gets funded under the fixed bound. That is not
    a rounding error, it is the difference between a capital gate and a
    formality, and it is precisely the hole a lucky opening streak walks
    through. The simulation in ``vault/simulate.py`` has a "lucky" profile with
    no edge and twelve rigged opening wins; against the fixed bound it unlocked
    $356 at call 50.

    A confidence sequence is the standard instrument for this: it is valid at
    every sample size simultaneously, so you may look as often as you like and
    stop whenever you like without inflating the error rate. The price is
    width -- and width, for a capital gate, is the direction you want to err.

    Construction: the Wilson score interval evaluated with the time-uniform
    critical value from :func:`sequential_z` in place of 1.96. This is
    conservative rather than exact (the mixture boundary is itself a bound,
    and substituting it into the score interval does not preserve exactness),
    which is the acceptable direction for a gate that decides whether real
    money moves.

    Cost in time-to-capital, measured: a true 65% forecaster reaches a lower
    bound above 0.5 at a median of 94 calls rather than 50. Roughly double the
    evidence for the same permission. That is the trade, stated plainly.
    """
    return wilson_interval(successes, n, sequential_z(n, alpha))


def reliability_buckets(
    probabilities: Sequence[float], outcomes: Sequence[int], *, edges: Sequence[float] = (),
) -> list[dict[str, Any]]:
    """
    Group forecasts by stated confidence and report the realised frequency.

    Bucket edges default to a coarse grid because the schema caps probabilities
    at [0.05, 0.95] and a fine grid on a few dozen samples produces buckets of
    size one, which look like perfect or catastrophic calibration and are
    neither.
    """
    edges = tuple(edges) or (0.0, 0.55, 0.65, 0.75, 0.85, 1.0)
    buckets: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        members = [
            (p, o) for p, o in zip(probabilities, outcomes)
            if low <= p < high or (i == len(edges) - 2 and p == high)
        ]
        if not members:
            continue
        stated = statistics.fmean(p for p, _ in members)
        realised = statistics.fmean(o for _, o in members)
        buckets.append({
            "range": [round(low, 2), round(high, 2)],
            "predicted": round(stated, 4),
            "observed": round(realised, 4),
            "count": len(members),
            "error": round(realised - stated, 4),
        })
    return buckets


def score_records(records: Sequence[ThesisRecord]) -> CalibrationScore:
    """Compute the full scoreboard from resolved theses."""
    scoreable = [r for r in records if r.is_scoreable]
    score = CalibrationScore(n=len(scoreable))
    if not scoreable:
        return score

    probabilities: list[float] = []
    outcomes: list[int] = []

    for record in scoreable:
        # The thesis states P(direction is correct). A call that was stopped out
        # or ended the wrong way is outcome 0.
        probabilities.append(float(record.thesis.probability))
        outcomes.append(1 if record.outcome == Outcome.CORRECT else 0)

        if record.outcome == Outcome.CORRECT:
            score.correct += 1
        elif record.outcome == Outcome.INVALIDATED:
            score.invalidated += 1
        else:
            score.wrong += 1

    n = len(scoreable)
    score.hit_rate = sum(outcomes) / n
    score.hit_rate_low, score.hit_rate_high = wilson_interval(sum(outcomes), n)
    # The gate reads the sequential bound; the fixed one is for reporting only.
    (
        score.hit_rate_low_sequential,
        score.hit_rate_high_sequential,
    ) = sequential_wilson_interval(sum(outcomes), n)
    score.mean_probability = statistics.fmean(probabilities)
    score.overconfidence = score.mean_probability - score.hit_rate

    score.brier = statistics.fmean(
        (p - o) ** 2 for p, o in zip(probabilities, outcomes)
    )
    score.skill = 1 - (score.brier / BASELINE_BRIER)
    score.beats_baseline = score.brier < BASELINE_BRIER

    # Log loss, clipped so a single confident miss does not become infinite.
    score.log_loss = statistics.fmean(
        -math.log(max(1e-9, p if o == 1 else 1 - p)) for p, o in zip(probabilities, outcomes)
    )

    # ---- Murphy decomposition ----
    base_rate = score.hit_rate
    score.uncertainty = base_rate * (1 - base_rate)

    buckets = reliability_buckets(probabilities, outcomes)
    score.buckets = buckets

    reliability = 0.0
    resolution = 0.0
    for bucket in buckets:
        weight = bucket["count"] / n
        reliability += weight * (bucket["predicted"] - bucket["observed"]) ** 2
        resolution += weight * (bucket["observed"] - base_rate) ** 2
    score.reliability = reliability
    score.resolution = resolution

    # ---- significance ----
    # One-sided binomial test against a 0.5 null, normal-approximated. With the
    # small samples this system lives in, this is a rough guide rather than a
    # p-value worth quoting -- which is why `adequacy` is reported next to it.
    if n >= 10:
        standard_error = math.sqrt(0.25 / n)
        z = (score.hit_rate - 0.5) / standard_error if standard_error > 0 else 0.0
        score.significant = z > 1.645 and score.beats_baseline

    score.adequacy = _adequacy(n)

    score.by_asset = _group(scoreable, lambda r: _enum(r.thesis.asset))
    score.by_regime = _group(scoreable, lambda r: r.regime or "unknown")
    score.by_horizon = _group(scoreable, lambda r: _enum(r.thesis.horizon))
    return score


def _adequacy(n: int) -> str:
    """
    A blunt verdict on whether the numbers mean anything.

    Thresholds come from the width of the Wilson interval: at n=30 a 60% hit
    rate has a 95% CI of roughly [42%, 75%] -- wide enough that it cannot be
    distinguished from a coin flip. It takes around 100 resolved calls before a
    genuine 60% edge separates from 50% with confidence, and around 400 before
    the Brier decomposition's components are individually stable.
    """
    if n < 30:
        return "insufficient"
    if n < 100:
        return "preliminary"
    if n < 400:
        return "indicative"
    return "adequate"


def _group(records: Sequence[ThesisRecord], key_fn) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ThesisRecord]] = {}
    for record in records:
        grouped.setdefault(key_fn(record), []).append(record)

    out: dict[str, dict[str, Any]] = {}
    for key, members in sorted(grouped.items()):
        outcomes = [1 if m.outcome == Outcome.CORRECT else 0 for m in members]
        probabilities = [float(m.thesis.probability) for m in members]
        brier = statistics.fmean(
            (p - o) ** 2 for p, o in zip(probabilities, outcomes)
        )
        low, high = wilson_interval(sum(outcomes), len(members))
        out[key] = {
            "n": len(members),
            "hit_rate": round(statistics.fmean(outcomes), 4),
            "hit_rate_ci": [round(low, 4), round(high, 4)],
            "brier": round(brier, 5),
            "skill": round(1 - brier / BASELINE_BRIER, 4),
            "mean_probability": round(statistics.fmean(probabilities), 4),
        }
    return out


def _enum(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)
