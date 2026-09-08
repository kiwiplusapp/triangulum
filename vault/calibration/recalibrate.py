"""
Recalibration: rescuing a forecaster that is right about ORDER but wrong about SCALE.

The Murphy decomposition splits a Brier score into reliability (are the stated
probabilities the real ones?) and resolution (does the forecaster distinguish
high-probability situations from low ones?). Those two failures have completely
different remedies, and conflating them throws away working systems.

    poor resolution   the forecaster cannot tell situations apart. Nothing to
                      fix downstream -- the model, features, or data are wrong.
    poor reliability  the ORDERING is informative but the NUMBERS are not. The
                      forecaster says 0.85 and is right 60% of the time.

The second case is fixable, and it is extremely common in LLM forecasters
specifically: language models are trained on confident prose and default to
overstating certainty. Measured on the simulation in ``vault.simulate``, an
agent with a genuine 58% edge that states 85% scores a Brier of 0.2955 -- worse
than the always-say-0.50 baseline -- and is blocked by the capital gate
forever, despite having something real to offer.

This module fits a monotone map from stated to realised probability and applies
it before the gate sees the number. The ordering is preserved exactly (that is
what "monotone" buys); only the scale is corrected.

Two fitters:

``PlattRecalibrator``     logistic in log-odds space. Two parameters, stable
                          from ~30 samples, corrects the systematic
                          over/under-confidence that dominates in practice.
                          The default.
``IsotonicRecalibrator``  a free-form monotone step function. Corrects any
                          monotone distortion, needs several hundred samples,
                          and overfits badly below that.

The recalibrated probability is what gets scored and what gets sized. The raw
stated probability is kept in the journal, because a system that silently
rewrites what the model said is not auditable.
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from vault.thesis.schema import Outcome, ThesisRecord

logger = logging.getLogger(__name__)

__all__ = [
    "PlattRecalibrator", "IsotonicRecalibrator", "RecalibrationReport",
    "fit_recalibrator", "MIN_SAMPLES_PLATT", "MIN_SAMPLES_ISOTONIC",
]

MIN_SAMPLES_PLATT = 30
MIN_SAMPLES_ISOTONIC = 200


@dataclass(slots=True)
class RecalibrationReport:
    method: str
    fitted: bool
    n: int
    brier_before: float = 0.0
    brier_after: float = 0.0
    improvement: float = 0.0
    parameters: dict[str, float] = field(default_factory=dict)
    examples: list[dict[str, float]] = field(default_factory=list)
    reason: str = ""

    @property
    def helped(self) -> bool:
        return self.fitted and self.brier_after < self.brier_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "fitted": self.fitted,
            "n": self.n,
            "brier_before": round(self.brier_before, 5),
            "brier_after": round(self.brier_after, 5),
            "improvement": round(self.improvement, 5),
            "helped": self.helped,
            "parameters": {k: round(v, 5) for k, v in self.parameters.items()},
            "examples": self.examples,
            "reason": self.reason,
        }

    def summary(self) -> str:
        if not self.fitted:
            return f"not fitted: {self.reason}"
        direction = "improved" if self.helped else "did not improve"
        return (
            f"{self.method} on {self.n} samples {direction} Brier "
            f"{self.brier_before:.4f} -> {self.brier_after:.4f} "
            f"({self.improvement:+.4f})"
        )


def _logit(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-min(x, 60)))
    z = math.exp(max(x, -60))
    return z / (1 + z)


class PlattRecalibrator:
    """
    Platt scaling: ``sigmoid(a * logit(p) + b)``.

    Fitted by batch gradient descent on log loss. Two parameters with clear
    meanings, which is why this is the default:

        a < 1  the forecaster is overconfident -- probabilities are pulled
               toward 0.5. This is the LLM case almost every time.
        a > 1  underconfident -- probabilities are pushed toward the extremes.
        b != 0 a directional bias: it says "up" too often, or too rarely.
    """

    def __init__(self, *, learning_rate: float = 0.08, iterations: int = 400) -> None:
        self.a = 1.0
        self.b = 0.0
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.n = 0
        self.fitted = False

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> "PlattRecalibrator":
        n = len(probabilities)
        if n < MIN_SAMPLES_PLATT:
            self.fitted = False
            return self

        logits = [_logit(p) for p in probabilities]
        self.a, self.b = 1.0, 0.0

        for _ in range(self.iterations):
            grad_a = 0.0
            grad_b = 0.0
            for logit, outcome in zip(logits, outcomes):
                predicted = _sigmoid(self.a * logit + self.b)
                error = predicted - outcome
                grad_a += error * logit
                grad_b += error
            self.a -= self.learning_rate * grad_a / n
            self.b -= self.learning_rate * grad_b / n

        # Clamp to a sane range. An `a` near zero would collapse every forecast
        # to a constant, which is technically well-calibrated and completely
        # useless -- and it happens when the fit sees almost no discrimination.
        self.a = max(0.05, min(5.0, self.a))
        self.b = max(-5.0, min(5.0, self.b))
        self.n = n
        self.fitted = True
        return self

    def apply(self, probability: float) -> float:
        if not self.fitted:
            return probability
        # Keep the result inside the schema's own bounds so a recalibrated
        # value stays a legal probability for the rest of the system.
        return min(0.95, max(0.05, _sigmoid(self.a * _logit(probability) + self.b)))

    @property
    def interpretation(self) -> str:
        if not self.fitted:
            return "not fitted"
        if self.a < 0.75:
            return (
                f"strongly overconfident (a={self.a:.2f}): stated probabilities are "
                f"pulled hard toward 0.5"
            )
        if self.a < 0.95:
            return f"mildly overconfident (a={self.a:.2f})"
        if self.a > 1.25:
            return f"underconfident (a={self.a:.2f}): probabilities pushed outward"
        return f"well scaled (a={self.a:.2f})"

    def to_dict(self) -> dict[str, Any]:
        return {"method": "platt", "a": self.a, "b": self.b, "n": self.n,
                "fitted": self.fitted}


class IsotonicRecalibrator:
    """
    Free-form monotone recalibration via pool-adjacent-violators.

    Strictly more flexible than Platt and correspondingly hungrier for data.
    Below a few hundred samples it fits the noise and reports a Brier
    improvement that does not survive out of sample, which is why the minimum
    is enforced rather than advisory.
    """

    def __init__(self) -> None:
        self._x: list[float] = []
        self._y: list[float] = []
        self.n = 0
        self.fitted = False

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> "IsotonicRecalibrator":
        n = len(probabilities)
        if n < MIN_SAMPLES_ISOTONIC:
            self.fitted = False
            return self

        # Collapse TIED x values first, into one weighted point per distinct
        # stated probability.
        #
        # This is not a micro-optimisation, it is a correctness requirement,
        # and getting it wrong is silent. Pool-adjacent-violators only merges
        # where the y-sequence DECREASES; run it on raw observations and a run
        # of tied x whose outcomes happen to arrive in non-decreasing order
        # (0, 0, 1, 1) is left as four separate blocks sitting at the same x.
        # `apply` then bisects into the first of them and returns that block's
        # value rather than the pooled mean.
        #
        # The case is not exotic -- it is the normal one. A model that states
        # 0.85 on every call produces exactly one distinct x, and the observed
        # bug was a 300-call record whose true frequency was 57.8% being
        # recalibrated to 0.500: a Brier of exactly 0.25, i.e. the correction
        # threw away the entire signal and reported the baseline.
        grouped: dict[float, list[float]] = {}
        for probability, outcome in zip(probabilities, outcomes):
            grouped.setdefault(float(probability), []).append(float(outcome))

        xs = sorted(grouped)
        values = [sum(grouped[x]) / len(grouped[x]) for x in xs]
        weights = [float(len(grouped[x])) for x in xs]

        index = 0
        while index < len(values) - 1:
            if values[index] <= values[index + 1]:
                index += 1
                continue
            total = weights[index] + weights[index + 1]
            pooled = (values[index] * weights[index]
                      + values[index + 1] * weights[index + 1]) / total
            values[index] = pooled
            weights[index] = total
            del values[index + 1]
            del weights[index + 1]
            del xs[index + 1]
            if index > 0:
                index -= 1

        self._x, self._y = xs, values
        self.n = n
        self.fitted = True
        return self

    def apply(self, probability: float) -> float:
        if not self.fitted or not self._x:
            return probability
        position = bisect.bisect_left(self._x, probability)
        if position == 0:
            return min(0.95, max(0.05, self._y[0]))
        if position >= len(self._x):
            return min(0.95, max(0.05, self._y[-1]))
        x0, x1 = self._x[position - 1], self._x[position]
        y0, y1 = self._y[position - 1], self._y[position]
        if x1 - x0 < 1e-12:
            return min(0.95, max(0.05, y0))
        interpolated = y0 + (y1 - y0) * (probability - x0) / (x1 - x0)
        return min(0.95, max(0.05, interpolated))

    def to_dict(self) -> dict[str, Any]:
        return {"method": "isotonic", "knots": len(self._x), "n": self.n,
                "fitted": self.fitted}


def fit_recalibrator(
    records: Sequence[ThesisRecord], *, method: str = "auto",
) -> tuple[Any, RecalibrationReport]:
    """
    Fit a recalibrator on resolved records and report whether it helped.

    Fitted IN SAMPLE, and the report says so. A proper out-of-sample estimate
    needs a train/test split that a system with 50 resolved calls cannot afford
    to make. The mitigation is the sample minimum plus the two-parameter form:
    Platt scaling on 50 points cannot overfit very much, because there is
    almost nothing to overfit with.
    """
    scoreable = [r for r in records if r.is_scoreable]
    probabilities = [float(r.thesis.probability) for r in scoreable]
    outcomes = [1 if r.outcome == Outcome.CORRECT else 0 for r in scoreable]
    n = len(scoreable)

    if n < MIN_SAMPLES_PLATT:
        return None, RecalibrationReport(
            method="none", fitted=False, n=n,
            reason=(
                f"{n} resolved calls; recalibration needs at least "
                f"{MIN_SAMPLES_PLATT}. Fitting a correction on fewer would be "
                f"fitting noise."
            ),
        )

    if method == "isotonic" or (method == "auto" and n >= MIN_SAMPLES_ISOTONIC):
        recalibrator = IsotonicRecalibrator().fit(probabilities, outcomes)
        name = "isotonic"
    else:
        recalibrator = PlattRecalibrator().fit(probabilities, outcomes)
        name = "platt"

    if not recalibrator.fitted:
        return None, RecalibrationReport(
            method=name, fitted=False, n=n, reason="fit did not converge",
        )

    before = sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / n
    adjusted = [recalibrator.apply(p) for p in probabilities]
    after = sum((p - o) ** 2 for p, o in zip(adjusted, outcomes)) / n

    examples = []
    for stated in (0.55, 0.65, 0.75, 0.85, 0.95):
        examples.append({
            "stated": stated,
            "recalibrated": round(recalibrator.apply(stated), 4),
        })

    report = RecalibrationReport(
        method=name, fitted=True, n=n,
        brier_before=before, brier_after=after, improvement=before - after,
        parameters=(
            {"a": recalibrator.a, "b": recalibrator.b}
            if name == "platt" else {"knots": float(len(recalibrator._x))}
        ),
        examples=examples,
        reason=(
            getattr(recalibrator, "interpretation", "") or "fitted"
        ),
    )
    logger.info("recalibration: %s", report.summary())
    return recalibrator, report
