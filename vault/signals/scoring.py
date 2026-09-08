"""
Per-signal scoring: every signal earns its own weight, or gets none.

## The rule, one level down

The capital gate refuses the agent size until the agent has demonstrated
calibration. This module applies the same rule to each signal individually:
a signal is weighted in the composite according to what it has been measured
to do, and a signal with no measured edge is weighted zero -- not "lightly",
zero. Fame is not evidence. The yield curve's recession record is a fact
about the last seventy years and not about whether this implementation of it,
on this data, at this horizon, predicts anything.

## What is measured

For each signal, over a historical window, and for a chosen target and
horizon:

    hit rate     fraction of times the signal's SIGN matched the sign of the
                 subsequent return. The blunt measure, and the one the
                 sequential bound is computed on.

    IC           the Spearman rank correlation between signal strength and
                 forward return. This is the measure that actually matters
                 for a continuous signal, because it uses the magnitude and
                 not just the sign. An IC of 0.05 is a genuinely useful
                 signal in liquid macro; 0.10 sustained is exceptional. Anyone
                 quoting 0.4 is either fitting in sample or has made an error.

    decay        IC at 1x, 2x and 4x the base horizon. A signal whose IC is
                 flat across horizons is probably picking up a slow-moving
                 state variable; one that decays fast is a timing signal and
                 needs to be traded quickly or not at all.

## Why the sequential bound again

Same reason as the capital gate, and worse here: there are twenty-three
signals, each tested at several horizons. Testing twenty-three signals at the
usual 95% level and keeping the ones that pass will hand you roughly one
"significant" signal from pure noise per twenty tested, every time, with a
straight face. The bound used is time-uniform AND the significance threshold
is Bonferroni-adjusted for the number of signals under test, which is
conservative and, for deciding where money goes, the right direction to err.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from vault.data.series import Series
from vault.resolve.scoring import sequential_wilson_interval, wilson_interval
from vault.signals.library import SIGNALS, SignalSpec
from vault.signals.types import SignalReading

logger = logging.getLogger(__name__)

__all__ = [
    "SignalScorecard", "SignalPerformance", "score_signal_history",
    "spearman", "information_coefficient",
]


# The minimum |IC| that counts as a signal rather than as noise. Set from the
# literature on macro factor returns, not from this sample: published,
# out-of-sample macro ICs cluster around 0.02-0.06, and anything demanding
# more than that as a floor will reject every real signal along with the fake
# ones.
MIN_MEANINGFUL_IC = 0.02


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """
    Spearman rank correlation, with average ranks for ties.

    Rank correlation rather than Pearson because signal strengths are squashed
    through tanh and forward returns are fat-tailed. Pearson on that pair is
    dominated by whichever three days had the largest moves, which is a
    measurement of the tail and not of the signal.
    """
    n = len(xs)
    if n < 8 or n != len(ys):
        return None

    def rank(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while (stop + 1 < len(order)
                   and values[order[stop + 1]] == values[order[index]]):
                stop += 1
            average = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                ranks[order[position]] = average
            index = stop + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mean_x, mean_y = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denom_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    denom_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if denom_x <= 0 or denom_y <= 0:
        return None
    return numerator / (denom_x * denom_y)


information_coefficient = spearman


@dataclass(slots=True)
class SignalPerformance:
    """One signal's measured record at one horizon."""

    key: str
    label: str
    family: str
    horizon_days: int

    n: int = 0
    hits: int = 0
    hit_rate: float = 0.0
    hit_rate_low: float = 0.0              # fixed-sample, for reporting
    hit_rate_low_sequential: float = 0.0   # always-valid, for deciding

    # The accuracy a signal that always says the same thing would achieve on
    # this sample. This, not 0.5, is the bar the hit rate is reported against.
    baseline_accuracy: float = 0.5
    n_positive: int = 0
    n_negative: int = 0

    # Independent observations remaining after deflating for window overlap,
    # and the alpha this signal is tested at once the number of signals under
    # test is accounted for.
    effective_n: int = 0
    alpha: float = 0.05

    ic: float | None = None
    ic_2x: float | None = None
    ic_4x: float | None = None

    mean_return_when_positive: float = 0.0
    mean_return_when_negative: float = 0.0

    # True when the signal reads the very series it is being scored against.
    # Such a signal is measuring the target's own autocorrelation, which is a
    # real statistical property and not a forecast.
    self_referential: bool = False
    inputs: tuple[str, ...] = ()

    weight: float = 0.0
    verdict: str = "untested"
    reason: str = ""

    @property
    def minimum_detectable_ic(self) -> float:
        """
        The smallest |IC| this sample could distinguish from zero.

        Worth surfacing rather than hiding, because it is usually the binding
        constraint and it is invisible otherwise. Inverting the Fisher z test:

            |r| > tanh( z_crit / sqrt(n_eff - 3) )

        With two years of history and a 21-day horizon there are roughly 34
        independent observations, and the minimum detectable IC is around
        0.50 -- far above any real macro signal, which live in the 0.02-0.06
        range. The honest conclusion is not "no signal works", it is "this
        sample cannot see a signal of realistic size", and the remedy is
        decades of history rather than a lower threshold.
        """
        if self.effective_n <= 4:
            return 1.0
        critical = NormalDist().inv_cdf(1 - self.alpha / 2)
        return math.tanh(critical / math.sqrt(self.effective_n - 3))

    @property
    def ic_significant(self) -> bool:
        if self.ic is None:
            return False
        return abs(self.ic) > self.minimum_detectable_ic

    @property
    def spread(self) -> float:
        """
        Mean forward return when the signal is positive minus when negative.

        The number a trader actually cares about, in the units of the target.
        A signal can have a fine hit rate and a negative spread if it is right
        often and small and wrong rarely and large.
        """
        return self.mean_return_when_positive - self.mean_return_when_negative

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "horizon_days": self.horizon_days,
            "n": self.n,
            "hit_rate": round(self.hit_rate, 4),
            "hit_rate_low": round(self.hit_rate_low, 4),
            "hit_rate_low_sequential": round(self.hit_rate_low_sequential, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "ic": round(self.ic, 4) if self.ic is not None else None,
            "ic_2x": round(self.ic_2x, 4) if self.ic_2x is not None else None,
            "ic_4x": round(self.ic_4x, 4) if self.ic_4x is not None else None,
            "spread": round(self.spread, 4),
            "self_referential": self.self_referential,
            "inputs": list(self.inputs),
            "effective_n": self.effective_n,
            "minimum_detectable_ic": round(self.minimum_detectable_ic, 4),
            "ic_significant": self.ic_significant,
            "weight": round(self.weight, 4),
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass(slots=True)
class SignalScorecard:
    """Every signal's record, and the weights that follow from it."""

    target: str
    horizon_days: int
    samples: int = 0
    performances: list[SignalPerformance] = field(default_factory=list)
    tests_run: int = 0
    alpha_per_test: float = 0.05

    # Fraction of forward windows in which the target rose. Equities drift up,
    # so this is typically well above 0.5, and it is the reason no signal is
    # ever tested against 0.5 -- see _assign_weight.
    base_rate: float = 0.5

    @property
    def weights(self) -> dict[str, float]:
        return {p.key: p.weight for p in self.performances}

    @property
    def earning(self) -> list[SignalPerformance]:
        return [p for p in self.performances if p.weight > 0]

    def weight_for(self, key: str) -> float:
        return self.weights.get(key, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "horizon_days": self.horizon_days,
            "samples": self.samples,
            "tests_run": self.tests_run,
            "alpha_per_test": round(self.alpha_per_test, 5),
            "base_rate": round(self.base_rate, 4),
            "baseline_accuracy": round(max(self.base_rate, 1 - self.base_rate), 4),
            "earning": len(self.earning),
            "total": len(self.performances),
            "signals": [p.to_dict() for p in self.performances],
        }

    def report(self) -> str:
        lines = [
            f"Signal scorecard vs {self.target} at {self.horizon_days}d "
            f"({self.samples} observations, {len(self.earning)} of "
            f"{len(self.performances)} signals earning weight)",
            f"  {self.target} rose in {self.base_rate:.1%} of windows, so the "
            f"bar every signal must clear is "
            f"{max(self.base_rate, 1 - self.base_rate):.1%}, not 50%.",
            "",
            f"  {'signal':24s} {'n':>5s} {'+/-':>9s} {'hit':>7s} {'anytime':>8s} "
            f"{'IC':>7s} {'spread':>8s} {'weight':>7s}  verdict",
            "  " + "-" * 100,
        ]
        for p in sorted(self.performances, key=lambda x: -x.weight):
            ic = f"{p.ic:+.3f}" if p.ic is not None else "   --"
            sides = f"{p.n_positive}/{p.n_negative}"
            lines.append(
                f"  {p.key:24s} {p.n:5d} {sides:>9s} {p.hit_rate:7.1%} "
                f"{p.hit_rate_low_sequential:8.1%} {ic:>7s} "
                f"{p.spread:+8.2f} {p.weight:7.3f}  {p.verdict}"
            )
        if not self.earning:
            lines += [
                "",
                "  NO signal has demonstrated an edge on this sample. Every",
                "  weight is zero, and the composite therefore contributes",
                "  nothing. That is the correct output when nothing has been",
                "  shown to work -- it is not a bug, and it is not a reason to",
                "  lower the threshold until something passes.",
            ]
        return "\n".join(lines)


def _forward_return(target: Series, when: date, horizon_days: int) -> float | None:
    """Percentage return from `when` to `when + horizon`, or None."""
    start = target.value_on(when, tolerance_days=5)
    end = target.value_on(when + timedelta(days=horizon_days), tolerance_days=5)
    if start is None or end is None or start <= 0:
        return None
    return (end / start - 1) * 100


def _history_as_of(series: Mapping[str, Series], when: date) -> dict[str, Series]:
    """
    The universe truncated to what was knowable on `when`.

    This is the whole ballgame for signal evaluation. Computing a signal from
    the full series and then correlating it with a "forward" return uses data
    from after the decision point, and produces beautiful ICs that vanish the
    moment the system is run live. Every signal in the backtest sees only its
    own past.
    """
    return {key: value.until(when) for key, value in series.items()}


def score_signal_history(
    series: Mapping[str, Series],
    *,
    target: str = "SP500",
    horizon_days: int = 21,
    step_days: int = 5,
    lookback_days: int = 720,
    min_samples: int = 40,
    specs: Sequence[SignalSpec] | None = None,
) -> SignalScorecard:
    """
    Walk history, recompute every signal at each step, and grade it.

    ``step_days`` of 5 with a 21-day horizon means consecutive observations
    OVERLAP, which inflates the effective sample: 100 overlapping 21-day
    windows are not 100 independent observations, they are closer to 24. The
    sequential bound is computed on the *effective* sample size for exactly
    this reason -- see the deflation below -- rather than on the raw count,
    which would otherwise let overlap manufacture significance.
    """
    specs = list(specs if specs is not None else SIGNALS.values())
    prices = series.get(target)
    if not prices:
        return SignalScorecard(target=target, horizon_days=horizon_days)

    end = prices.points[-1].on - timedelta(days=horizon_days)
    start = max(prices.points[0].on, end - timedelta(days=lookback_days))

    # Collect (strength, forward_return) pairs per signal.
    observations: dict[str, list[tuple[float, float]]] = {s.key: [] for s in specs}
    long_returns: dict[str, list[tuple[float, float, float]]] = {
        s.key: [] for s in specs
    }

    when = start
    samples = 0
    while when <= end:
        forward = _forward_return(prices, when, horizon_days)
        if forward is not None:
            history = _history_as_of(series, when)
            forward_2x = _forward_return(prices, when, horizon_days * 2)
            forward_4x = _forward_return(prices, when, horizon_days * 4)
            for spec in specs:
                reading = spec.evaluate(history)
                if not reading.usable:
                    continue
                observations[spec.key].append((reading.strength, forward))
                long_returns[spec.key].append((
                    reading.strength,
                    forward_2x if forward_2x is not None else float("nan"),
                    forward_4x if forward_4x is not None else float("nan"),
                ))
            samples += 1
        when += timedelta(days=step_days)

    # Overlap deflation. With a horizon of H days sampled every S days, each
    # observation shares (H - S)/H of its window with its neighbour. The
    # effective independent count is approximately n * S / H, floored at 1.
    overlap_factor = min(1.0, step_days / max(1, horizon_days))

    # The base rate of the TARGET, not of any signal: how often the thing
    # being predicted went up at all. Every hit rate below is judged against
    # this and not against a coin flip.
    all_forwards = [f for pairs in observations.values() for _, f in pairs]
    base_rate = (
        sum(1 for f in all_forwards if f > 0) / len(all_forwards)
        if all_forwards else 0.5
    )

    scorecard = SignalScorecard(
        target=target, horizon_days=horizon_days, samples=samples,
        tests_run=len(specs),
        alpha_per_test=0.05 / max(1, len(specs)),      # Bonferroni
        base_rate=base_rate,
    )

    for spec in specs:
        pairs = observations[spec.key]
        performance = SignalPerformance(
            key=spec.key, label=spec.label, family=spec.family,
            horizon_days=horizon_days, n=len(pairs),
            inputs=tuple(spec.requires),
            self_referential=target in spec.requires,
        )

        if len(pairs) < min_samples:
            performance.verdict = "insufficient"
            performance.reason = (
                f"{len(pairs)} usable observations against a {min_samples} "
                f"minimum; no weight is assigned on this little evidence"
            )
            scorecard.performances.append(performance)
            continue

        strengths = [s for s, _ in pairs]
        forwards = [f for _, f in pairs]

        hits = sum(
            1 for s, f in pairs
            if (s > 0 and f > 0) or (s < 0 and f < 0)
        )
        directional = sum(1 for s, _ in pairs if s != 0)
        performance.hits = hits
        performance.n_positive = sum(1 for s, _ in pairs if s > 0)
        performance.n_negative = sum(1 for s, _ in pairs if s < 0)
        performance.baseline_accuracy = max(base_rate, 1 - base_rate)
        performance.effective_n = max(1, int(len(pairs) * overlap_factor))
        performance.alpha = scorecard.alpha_per_test
        if directional:
            performance.hit_rate = hits / directional
            effective = max(1, int(directional * overlap_factor))
            effective_hits = int(round(performance.hit_rate * effective))
            performance.hit_rate_low = wilson_interval(hits, directional)[0]
            performance.hit_rate_low_sequential = sequential_wilson_interval(
                effective_hits, effective, scorecard.alpha_per_test,
            )[0]

        performance.ic = spearman(strengths, forwards)
        pairs_2x = [(s, a) for s, a, _ in long_returns[spec.key]
                    if not math.isnan(a)]
        pairs_4x = [(s, b) for s, _, b in long_returns[spec.key]
                    if not math.isnan(b)]
        if len(pairs_2x) >= 8:
            performance.ic_2x = spearman([s for s, _ in pairs_2x],
                                         [v for _, v in pairs_2x])
        if len(pairs_4x) >= 8:
            performance.ic_4x = spearman([s for s, _ in pairs_4x],
                                         [v for _, v in pairs_4x])

        positive = [f for s, f in pairs if s > 0]
        negative = [f for s, f in pairs if s < 0]
        performance.mean_return_when_positive = (
            statistics.fmean(positive) if positive else 0.0
        )
        performance.mean_return_when_negative = (
            statistics.fmean(negative) if negative else 0.0
        )

        _assign_weight(performance)
        scorecard.performances.append(performance)

    return scorecard


def _assign_weight(performance: SignalPerformance) -> None:
    """
    Turn a measured record into a weight, or into an explained zero.

    Four hurdles, all of which must clear:

    0. The signal must have taken BOTH sides. A signal observed positive 140
       times and negative 5 times has not been tested as a short signal, and
       crediting it for the 140 is crediting it for the market's own drift.

    1. The always-valid lower bound on the hit rate must exceed the BASELINE
       ACCURACY -- what a signal that always said the same thing would score
       on this sample -- at the Bonferroni-adjusted level.

       Testing against 0.5 is the mistake that makes signal libraries look
       good. Equities rise in most 21-day windows; on the sample this was
       first run against, the S&P rose in 84% of them. A permanently bullish
       signal therefore "predicts" direction 84% of the time while carrying
       no information whatsoever, and three such signals sailed through the
       first version of this function on exactly that basis. The bar is the
       naive strategy, not the coin flip -- the same principle as the 0.25
       Brier baseline in the capital gate.

    2. The IC must be meaningful AND agree in sign with the hit rate. A
       signal that is right more often than not while its magnitude
       anti-correlates with returns is right on small moves and wrong on big
       ones, which loses money.

    3. The spread must be positive. Same argument, in return units.

    The weight is then the IC itself, clipped -- not a rank, not a softmax.
    Using the measured effect size as the weight means a signal contributes
    in proportion to what it has been shown to do, and shrinks automatically
    when the measurement is weak.
    """
    ic = performance.ic

    # ---- hurdle -1: is it predicting a series from itself? ----
    #
    # A signal computed FROM the target series and scored AGAINST that same
    # series is measuring autocorrelation. That is a real statistical property
    # -- volatility genuinely is persistent and mean-reverting -- but it is not
    # a forecast, and it is not information the market has failed to price.
    #
    # This is not hypothetical. Scoring the 23 signals against VIXCLS on 40
    # years of real data produced exactly three "earning" signals, and all
    # three read VIX or realised equity vol: vol_regime at IC +0.200 is the
    # VIX's own mean reversion wearing a signal's clothes. Against NASDAQ100,
    # where no signal reads the target, nothing passed at all. A scanner that
    # reported the first result as an edge would be handing over the most
    # confident-sounding false positive available.
    if performance.self_referential:
        performance.verdict = "self-referential"
        performance.reason = (
            f"reads {', '.join(performance.inputs)}, which includes the target "
            f"itself. Any correlation here is the target's own "
            f"autocorrelation rather than a forecast, and it earns no weight"
        )
        return

    # ---- hurdle 0: has it taken both sides? ----
    #
    # A signal observed positive 145 times and negative 0 times has not been
    # tested as a short signal. Its hit rate is a restatement of the target's
    # own base rate, and four such signals were credited with an edge by the
    # first version of this function on exactly that basis.
    minority = min(performance.n_positive, performance.n_negative)
    required_minority = max(10, int(0.15 * performance.n))
    if minority < required_minority:
        performance.verdict = "one-sided"
        performance.reason = (
            f"read positive {performance.n_positive} times and negative "
            f"{performance.n_negative} times. With only {minority} "
            f"observations on its minority side (needs {required_minority}), "
            f"it has not been tested in both directions"
        )
        return

    # ---- hurdle 1: is the IC distinguishable from zero? ----
    #
    # This, and not the hit rate, is the gate. The hit rate is reported
    # because it is legible, but it is the wrong instrument: it is measured
    # against a base rate that drifts with the market, and it throws away
    # magnitude entirely. In a strong bull market a signal that correctly
    # says "smaller position here" adds real value while never beating the
    # directional base rate, and a hit-rate gate would discard it. The IC
    # keeps the magnitude and is invariant to the drift.
    #
    # Tested via the Fisher z-transform on the EFFECTIVE sample size, at the
    # Bonferroni-adjusted level. Fisher rather than the raw t because the
    # sampling distribution of a correlation is badly skewed at small n, and
    # small n is the whole regime here.
    if ic is None:
        performance.verdict = "no information"
        performance.reason = "too few observations to compute a correlation"
        return

    if not performance.ic_significant:
        performance.verdict = "not significant"
        performance.reason = (
            f"IC {ic:+.3f} against a minimum detectable "
            f"{performance.minimum_detectable_ic:.3f} at this effective sample "
            f"size ({performance.effective_n} independent observations after "
            f"deflating for overlap). The measurement cannot tell this apart "
            f"from zero -- which is a statement about how little data there "
            f"is, not proof the signal is worthless"
        )
        return

    if ic < 0:
        performance.verdict = "inverted"
        performance.reason = (
            f"IC is a significant {ic:+.3f}: the signal's magnitude "
            f"anti-correlates with the forward return. It is given no weight "
            f"rather than being flipped, because a sign error discovered by "
            f"searching twenty-three signals is usually overfitting, and "
            f"trading the inverse of an accident is still trading an accident"
        )
        return

    if performance.spread <= 0:
        performance.verdict = "negative spread"
        performance.reason = (
            f"mean forward return is {performance.mean_return_when_positive:+.2f}% "
            f"when positive against {performance.mean_return_when_negative:+.2f}% "
            f"when negative -- a spread of {performance.spread:+.2f}%"
        )
        return

    performance.weight = min(0.20, ic)
    performance.verdict = "earning"
    performance.reason = (
        f"IC {ic:+.3f} (significant against {performance.minimum_detectable_ic:.3f} "
        f"at n_eff={performance.effective_n}), hit rate "
        f"{performance.hit_rate:.1%} vs a {performance.baseline_accuracy:.1%} "
        f"baseline, spread {performance.spread:+.2f}%"
    )
