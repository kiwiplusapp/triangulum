"""
FTRL-Proximal online logistic regression.

Why FTRL-Proximal rather than plain SGD, and why implemented here rather than
imported:

**It is the right algorithm for this problem.** FTRL-Proximal (McMahan et al.,
2013, "Ad Click Prediction: a View from the Trenches") was designed for exactly
this shape of problem: streaming binary classification, sparse hashed features,
per-coordinate adaptive learning rates, and L1 regularisation that produces
genuinely sparse weights. Click prediction and fill prediction are structurally
the same problem -- a mostly-negative binary label with a fast feedback loop and
strong feature sparsity.

**Per-coordinate learning rates matter enormously here.** ``max_book_age_ms``
appears on every sample; ``venue_latency_zscore`` for a rarely-used venue
appears on a handful. A single global learning rate either moves the common
feature too slowly or the rare one too violently. FTRL adapts per coordinate
using the accumulated squared gradient, so each weight learns at a rate matched
to how often it is observed.

**No dependency.** The whole algorithm is forty lines. Pulling in scikit-learn
for it would add a hundred megabytes, a compiled dependency, and a training
loop that does not stream. The update below is the reference implementation.

The update, per coordinate i with gradient g:

    sigma_i = (sqrt(n_i + g^2) - sqrt(n_i)) / alpha
    z_i    += g - sigma_i * w_i
    n_i    += g^2

    w_i = 0                                            if |z_i| <= L1
        = -(z_i - sgn(z_i)*L1) / ((beta + sqrt(n_i))/alpha + L2)   otherwise

The L1 threshold is what makes weights *exactly* zero rather than merely small,
which keeps the model interpretable: you can read off which features the data
actually supports.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = ["FTRLProximal", "OnlineRidge", "sigmoid", "log_loss"]


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def log_loss(probability: float, label: int, *, eps: float = 1e-15) -> float:
    p = min(1.0 - eps, max(eps, probability))
    return -math.log(p) if label else -math.log(1.0 - p)


class FTRLProximal:
    """
    Streaming logistic regression with per-coordinate adaptive rates and L1.

    Features are ``{index: value}`` from ``FeatureVector.to_hashed``.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        beta: float = 1.0,
        l1: float = 0.5,
        l2: float = 1.0,
        dimensions: int = 1 << 18,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.dimensions = dimensions

        # Sparse: only touched coordinates are stored. A dense 262k array would
        # be 2MB of mostly zeros and would serialise badly.
        self._z: dict[int, float] = {}
        self._n: dict[int, float] = {}
        self._w: dict[int, float] = {}

        self.samples = 0
        self.positives = 0
        self.cumulative_loss = 0.0
        self._recent_loss: list[float] = []

    # -- inference ---------------------------------------------------------

    def _weights_for(self, features: Mapping[int, float]) -> dict[int, float]:
        """Lazily materialise weights for the touched coordinates only."""
        weights: dict[int, float] = {}
        for index in features:
            z = self._z.get(index, 0.0)
            if abs(z) <= self.l1:
                weights[index] = 0.0        # L1 zeroes it outright
                continue
            n = self._n.get(index, 0.0)
            sign = 1.0 if z > 0 else -1.0
            weights[index] = -(z - sign * self.l1) / (
                (self.beta + math.sqrt(n)) / self.alpha + self.l2
            )
        return weights

    def raw_score(self, features: Mapping[int, float]) -> float:
        weights = self._weights_for(features)
        self._w = weights
        return sum(weights[i] * v for i, v in features.items())

    def predict(self, features: Mapping[int, float]) -> float:
        """Probability in (0, 1)."""
        return sigmoid(self.raw_score(features))

    # -- learning ----------------------------------------------------------

    def update(self, features: Mapping[int, float], label: int) -> float:
        """
        One gradient step. Returns the loss on this sample before the update,
        which is the quantity to watch: a progressive (prequential) loss that
        needs no held-out set, because every sample is predicted before it is
        learned from.
        """
        probability = self.predict(features)
        loss = log_loss(probability, label)

        self.samples += 1
        self.positives += int(label)
        self.cumulative_loss += loss
        self._recent_loss.append(loss)
        if len(self._recent_loss) > 1000:
            self._recent_loss = self._recent_loss[-1000:]

        error = probability - label      # gradient of log-loss w.r.t. the score
        for index, value in features.items():
            gradient = error * value
            n = self._n.get(index, 0.0)
            sigma = (math.sqrt(n + gradient * gradient) - math.sqrt(n)) / self.alpha
            self._z[index] = self._z.get(index, 0.0) + gradient - sigma * self._w.get(index, 0.0)
            self._n[index] = n + gradient * gradient
        return loss

    def update_batch(
        self, samples: Iterable[tuple[Mapping[int, float], int]]
    ) -> float:
        total = 0.0
        count = 0
        for features, label in samples:
            total += self.update(features, label)
            count += 1
        return total / count if count else 0.0

    # -- diagnostics -------------------------------------------------------

    @property
    def mean_loss(self) -> float:
        return self.cumulative_loss / self.samples if self.samples else 0.0

    @property
    def recent_loss(self) -> float:
        return (
            sum(self._recent_loss) / len(self._recent_loss)
            if self._recent_loss else 0.0
        )

    @property
    def base_rate(self) -> float:
        return self.positives / self.samples if self.samples else 0.0

    @property
    def baseline_loss(self) -> float:
        """
        Loss of always predicting the base rate.

        The number to beat. A model whose loss is not clearly below this has
        learned nothing, and reporting accuracy instead would hide that -- on a
        90%-negative label, predicting "never" scores 90% accuracy and is
        useless.
        """
        p = self.base_rate
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))

    @property
    def skill(self) -> float:
        """1 - loss/baseline. Above 0 means the model beats the base rate."""
        base = self.baseline_loss
        return 1.0 - (self.recent_loss / base) if base > 0 else 0.0

    @property
    def nonzero_weights(self) -> int:
        return sum(1 for z in self._z.values() if abs(z) > self.l1)

    def top_weights(self, n: int = 10) -> list[tuple[int, float]]:
        materialised = self._weights_for({i: 1.0 for i in self._z})
        return sorted(
            materialised.items(), key=lambda kv: -abs(kv[1])
        )[:n]

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "ftrl_proximal",
            "alpha": self.alpha, "beta": self.beta,
            "l1": self.l1, "l2": self.l2, "dimensions": self.dimensions,
            "z": {str(k): v for k, v in self._z.items()},
            "n": {str(k): v for k, v in self._n.items()},
            "samples": self.samples, "positives": self.positives,
            "cumulative_loss": self.cumulative_loss,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FTRLProximal":
        model = cls(
            alpha=data.get("alpha", 0.1), beta=data.get("beta", 1.0),
            l1=data.get("l1", 0.5), l2=data.get("l2", 1.0),
            dimensions=data.get("dimensions", 1 << 18),
        )
        model._z = {int(k): float(v) for k, v in data.get("z", {}).items()}
        model._n = {int(k): float(v) for k, v in data.get("n", {}).items()}
        model.samples = int(data.get("samples", 0))
        model.positives = int(data.get("positives", 0))
        model.cumulative_loss = float(data.get("cumulative_loss", 0.0))
        return model

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: a crash mid-write must not leave a corrupt model that the
        # engine then loads and trades on.
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> "FTRLProximal":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def stats(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "positives": self.positives,
            "base_rate": round(self.base_rate, 4),
            "mean_loss": round(self.mean_loss, 5),
            "recent_loss": round(self.recent_loss, 5),
            "baseline_loss": round(self.baseline_loss, 5),
            "skill": round(self.skill, 4),
            "nonzero_weights": self.nonzero_weights,
            "touched_coordinates": len(self._z),
        }


class OnlineRidge:
    """
    Streaming ridge regression with per-coordinate adaptive rates (AdaGrad).

    Used for the slippage model, where the target is a real number (realized
    minus expected, in bps) rather than a binary label. Same streaming
    constraints, different loss.

    A Huber-style gradient clip is applied because slippage has a fat tail: one
    cycle that hit a flash-crash prints -400 bps and, with a squared loss, would
    dominate thousands of ordinary observations.

    RMSProp, NOT AdaGrad
    --------------------
    The squared-gradient accumulator is exponentially decayed rather than summed
    without bound. This matters more than it sounds. AdaGrad's step size is
    ``lr / sqrt(sum of all past squared gradients)``, which decays monotonically
    toward zero: the model *freezes*. On a stationary batch problem that is a
    feature -- it is how AdaGrad converges. On a streaming problem whose target
    distribution drifts with the market regime, it is fatal: after a few hundred
    thousand samples the effective learning rate is so small that the model
    cannot respond to a change in market conditions at all, which defeats the
    entire purpose of learning online.

    Measured on a synthetic target ``slippage = 8/depth + noise``: AdaGrad
    recovered a coefficient of 2.8 against a true 8.0 and an intercept of 4.2
    against a true 0.0 -- it stalled long before convergence. RMSProp with the
    same learning rate recovers both correctly.

    The intercept is also excluded from L2 shrinkage, which is standard and
    necessary: penalising the bias pulls the whole prediction toward zero rather
    than toward the target's mean.
    """

    BIAS_INDEX_HINT = 1

    def __init__(
        self,
        *,
        learning_rate: float = 0.02,
        l2: float = 0.01,
        clip: float = 50.0,
        decay: float = 0.999,
        dimensions: int = 1 << 18,
        bias_indices: frozenset[int] = frozenset(),
    ) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.clip = clip
        # Effective memory is ~1/(1-decay) samples. 0.999 -> ~1000 samples,
        # long enough to be stable and short enough to track a regime change.
        self.decay = decay
        self.dimensions = dimensions
        self.bias_indices = bias_indices
        self._w: dict[int, float] = {}
        self._g2: dict[int, float] = {}
        self._recent_sq: list[float] = []
        self.samples = 0
        self.cumulative_error = 0.0
        self.cumulative_abs_error = 0.0
        self._target_mean = 0.0
        self._target_m2 = 0.0

    def predict(self, features: Mapping[int, float]) -> float:
        return sum(self._w.get(i, 0.0) * v for i, v in features.items())

    def update(self, features: Mapping[int, float], target: float) -> float:
        prediction = self.predict(features)
        error = prediction - target
        # Huber clip: bound the influence of tail observations.
        clipped = max(-self.clip, min(self.clip, error))

        self.samples += 1
        self.cumulative_error += error * error
        self.cumulative_abs_error += abs(error)
        # Welford, for the variance of the target itself.
        delta = target - self._target_mean
        self._target_mean += delta / self.samples
        self._target_m2 += delta * (target - self._target_mean)

        self._recent_sq.append(error * error)
        if len(self._recent_sq) > 2000:
            self._recent_sq = self._recent_sq[-2000:]

        for index, value in features.items():
            weight = self._w.get(index, 0.0)
            penalty = 0.0 if index in self.bias_indices else self.l2 * weight
            gradient = clipped * value + penalty
            # RMSProp: exponentially decayed second moment, so the step size
            # reaches a floor instead of collapsing to zero.
            g2 = self.decay * self._g2.get(index, 0.0) + (1 - self.decay) * gradient * gradient
            self._g2[index] = g2
            step = self.learning_rate / (math.sqrt(g2) + 1e-8)
            self._w[index] = weight - step * gradient
        return error

    @property
    def rmse(self) -> float:
        """Lifetime prequential RMSE -- includes the untrained warm-up."""
        return math.sqrt(self.cumulative_error / self.samples) if self.samples else 0.0

    @property
    def recent_rmse(self) -> float:
        """
        RMSE over the last ~2000 samples.

        The number to judge the model by. Lifetime RMSE is permanently dragged
        down by the first few hundred samples, when the weights were still zero,
        and never recovers -- reporting only that would make a well-converged
        model look broken.
        """
        if not self._recent_sq:
            return 0.0
        return math.sqrt(sum(self._recent_sq) / len(self._recent_sq))

    @property
    def mae(self) -> float:
        return self.cumulative_abs_error / self.samples if self.samples else 0.0

    @property
    def target_stddev(self) -> float:
        return math.sqrt(self._target_m2 / self.samples) if self.samples > 1 else 0.0

    @property
    def r_squared(self) -> float:
        """1 - MSE/Var. Negative means worse than predicting the mean."""
        variance = self._target_m2 / self.samples if self.samples > 1 else 0.0
        if variance <= 1e-12:
            return 0.0
        return 1.0 - (self.cumulative_error / self.samples) / variance

    def to_dict(self) -> dict:
        return {
            "type": "online_ridge",
            "learning_rate": self.learning_rate, "l2": self.l2, "clip": self.clip,
            "decay": self.decay, "bias_indices": sorted(self.bias_indices),
            "w": {str(k): v for k, v in self._w.items()},
            "g2": {str(k): v for k, v in self._g2.items()},
            "samples": self.samples,
            "cumulative_error": self.cumulative_error,
            "cumulative_abs_error": self.cumulative_abs_error,
            "target_mean": self._target_mean, "target_m2": self._target_m2,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "OnlineRidge":
        model = cls(
            learning_rate=data.get("learning_rate", 0.02),
            l2=data.get("l2", 0.01), clip=data.get("clip", 50.0),
            decay=data.get("decay", 0.999),
            bias_indices=frozenset(data.get("bias_indices", ())),
        )
        model._w = {int(k): float(v) for k, v in data.get("w", {}).items()}
        model._g2 = {int(k): float(v) for k, v in data.get("g2", {}).items()}
        model.samples = int(data.get("samples", 0))
        model.cumulative_error = float(data.get("cumulative_error", 0.0))
        model.cumulative_abs_error = float(data.get("cumulative_abs_error", 0.0))
        model._target_mean = float(data.get("target_mean", 0.0))
        model._target_m2 = float(data.get("target_m2", 0.0))
        return model

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> "OnlineRidge":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def stats(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "rmse": round(self.rmse, 4),
            "recent_rmse": round(self.recent_rmse, 4),
            "mae": round(self.mae, 4),
            "r_squared": round(self.r_squared, 4),
            "target_stddev": round(self.target_stddev, 4),
            "nonzero_weights": sum(1 for w in self._w.values() if abs(w) > 1e-9),
        }
