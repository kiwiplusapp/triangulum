"""
Probability calibration.

A logistic regression's output is a number between 0 and 1, but it is not
automatically a *probability*. Under L1 regularisation and class imbalance the
raw scores are systematically shifted -- typically toward the base rate, so a
model that says 0.8 might be right only 65% of the time.

For a classifier that is used to rank things, this does not matter. For one
whose output is multiplied by a payoff and compared against a threshold, it
matters enormously: a 15-point calibration error on P(fill) flips the sign of
the EV calculation for every marginal cycle.

Two calibrators:

``PlattCalibrator``   fits a 1-D logistic ``sigmoid(a*x + b)`` mapping raw score
                      to calibrated probability. Cheap, online, robust with few
                      samples. The default.

``IsotonicCalibrator`` fits a monotone step function. Strictly more flexible --
                      it can correct any monotone distortion, not just a
                      logistic one -- but needs far more data and can overfit
                      badly below a few thousand samples.

Both report a reliability diagram, which is the honest way to look at a
probabilistic model: bucket the predictions, and check that things predicted at
0.7 happen about 70% of the time.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Sequence

from triangulum.learning.online_lr import sigmoid

__all__ = ["PlattCalibrator", "IsotonicCalibrator", "reliability_diagram", "brier_score"]


class PlattCalibrator:
    """Online Platt scaling: learns ``sigmoid(a*raw + b)`` by SGD."""

    def __init__(self, *, learning_rate: float = 0.01, min_samples: int = 50) -> None:
        self.a = 1.0
        self.b = 0.0
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.samples = 0
        self._buckets: list[list[float]] = [[0.0, 0.0] for _ in range(10)]
        self._brier_sum = 0.0

    def calibrate(self, raw_probability: float) -> float:
        """Map a raw model output to a calibrated probability."""
        if self.samples < self.min_samples:
            return raw_probability
        # Work in log-odds so the correction is a linear rescaling.
        p = min(1 - 1e-9, max(1e-9, raw_probability))
        logit = math.log(p / (1 - p))
        return sigmoid(self.a * logit + self.b)

    def observe(self, raw_probability: float, label: int) -> None:
        self.samples += 1
        p = min(1 - 1e-9, max(1e-9, raw_probability))
        logit = math.log(p / (1 - p))
        calibrated = sigmoid(self.a * logit + self.b)

        error = calibrated - label
        self.a -= self.learning_rate * error * logit
        self.b -= self.learning_rate * error

        index = min(9, int(raw_probability * 10))
        self._buckets[index][0] += 1
        self._buckets[index][1] += label
        self._brier_sum += (calibrated - label) ** 2

    @property
    def brier(self) -> float:
        """Mean squared error of the calibrated probabilities. Lower is better."""
        return self._brier_sum / self.samples if self.samples else 0.0

    def reliability(self) -> list[dict[str, float]]:
        out = []
        for index, (count, positives) in enumerate(self._buckets):
            if count == 0:
                continue
            out.append({
                "bucket": round(index / 10 + 0.05, 2),
                "predicted": round(index / 10 + 0.05, 3),
                "observed": round(positives / count, 3),
                "count": int(count),
                "error": round(positives / count - (index / 10 + 0.05), 3),
            })
        return out

    @property
    def calibration_error(self) -> float:
        """Expected calibration error: count-weighted |predicted - observed|."""
        total = sum(c for c, _ in self._buckets)
        if total == 0:
            return 0.0
        error = 0.0
        for index, (count, positives) in enumerate(self._buckets):
            if count == 0:
                continue
            predicted = index / 10 + 0.05
            error += count * abs(positives / count - predicted)
        return error / total

    def stats(self) -> dict[str, object]:
        return {
            "type": "platt",
            "a": round(self.a, 4),
            "b": round(self.b, 4),
            "samples": self.samples,
            "active": self.samples >= self.min_samples,
            "brier": round(self.brier, 5),
            "expected_calibration_error": round(self.calibration_error, 4),
            "reliability": self.reliability(),
        }

    def to_dict(self) -> dict:
        return {
            "type": "platt", "a": self.a, "b": self.b,
            "samples": self.samples,
            "buckets": self._buckets, "brier_sum": self._brier_sum,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlattCalibrator":
        c = cls()
        c.a = float(data.get("a", 1.0))
        c.b = float(data.get("b", 0.0))
        c.samples = int(data.get("samples", 0))
        c._buckets = [list(b) for b in data.get("buckets", c._buckets)]
        c._brier_sum = float(data.get("brier_sum", 0.0))
        return c


class IsotonicCalibrator:
    """
    Monotone step-function calibration via pool-adjacent-violators.

    Refits periodically on a bounded reservoir rather than incrementally: PAVA
    is O(n) and running it every sample would be wasteful, while running it
    never would leave the calibration stale.
    """

    def __init__(self, *, max_samples: int = 20_000, refit_every: int = 500,
                 min_samples: int = 500) -> None:
        self.max_samples = max_samples
        self.refit_every = refit_every
        self.min_samples = min_samples
        self._raw: list[float] = []
        self._labels: list[int] = []
        self._x: list[float] = []
        self._y: list[float] = []
        self.samples = 0

    def observe(self, raw_probability: float, label: int) -> None:
        self.samples += 1
        self._raw.append(raw_probability)
        self._labels.append(label)
        if len(self._raw) > self.max_samples:
            # Drop the oldest half: keeps memory bounded and biases the fit
            # toward recent behaviour, which is what we want under drift.
            half = self.max_samples // 2
            self._raw = self._raw[-half:]
            self._labels = self._labels[-half:]
        if self.samples % self.refit_every == 0:
            self.fit()

    def fit(self) -> None:
        if len(self._raw) < self.min_samples:
            return
        order = sorted(range(len(self._raw)), key=lambda i: self._raw[i])
        xs = [self._raw[i] for i in order]
        ys = [float(self._labels[i]) for i in order]

        # Pool adjacent violators.
        values = list(ys)
        weights = [1.0] * len(ys)
        index = 0
        while index < len(values) - 1:
            if values[index] <= values[index + 1]:
                index += 1
                continue
            total_weight = weights[index] + weights[index + 1]
            pooled = (values[index] * weights[index]
                      + values[index + 1] * weights[index + 1]) / total_weight
            values[index] = pooled
            weights[index] = total_weight
            del values[index + 1]
            del weights[index + 1]
            del xs[index + 1]
            if index > 0:
                index -= 1
        self._x = xs
        self._y = values

    def calibrate(self, raw_probability: float) -> float:
        if not self._x or self.samples < self.min_samples:
            return raw_probability
        position = bisect.bisect_left(self._x, raw_probability)
        if position == 0:
            return self._y[0]
        if position >= len(self._x):
            return self._y[-1]
        # Linear interpolation between the two nearest knots.
        x0, x1 = self._x[position - 1], self._x[position]
        y0, y1 = self._y[position - 1], self._y[position]
        if x1 - x0 < 1e-12:
            return y0
        return y0 + (y1 - y0) * (raw_probability - x0) / (x1 - x0)

    def stats(self) -> dict[str, object]:
        return {
            "type": "isotonic",
            "samples": self.samples,
            "knots": len(self._x),
            "active": bool(self._x) and self.samples >= self.min_samples,
        }


def reliability_diagram(
    predictions: Sequence[float], labels: Sequence[int], *, buckets: int = 10,
) -> list[dict[str, float]]:
    """Bucketed predicted-vs-observed frequencies. The honest model report."""
    counts = [0] * buckets
    positives = [0] * buckets
    for p, y in zip(predictions, labels):
        index = min(buckets - 1, int(p * buckets))
        counts[index] += 1
        positives[index] += y
    return [
        {
            "bucket": (i + 0.5) / buckets,
            "predicted": (i + 0.5) / buckets,
            "observed": positives[i] / counts[i],
            "count": counts[i],
        }
        for i in range(buckets) if counts[i] > 0
    ]


def brier_score(predictions: Sequence[float], labels: Sequence[int]) -> float:
    if not predictions:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(predictions, labels)) / len(predictions)
