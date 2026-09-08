"""
Walk-forward training, and the comparison that decides whether the network
is used at all.

## The rule this module enforces

**The neural network does not get to be in the ensemble because it is a
neural network.** It is fitted alongside two baselines on identical purged
folds and judged on the same out-of-sample Brier score:

    base rate      always predict the historical frequency of an up move.
                   Carries no features and cannot be beaten by luck.
    logistic       one linear layer, same inputs, same folds.
    network        the MLP.

Whichever wins, wins. On a few hundred noisy macro samples the honest
expectation is that the logistic model wins or ties, and when it does, this
module says so in plain language and the ensemble weights follow. A system
that ships the deep model regardless has replaced measurement with taste.

## Why the base rate is in the comparison

Because it is very easy to beat a coin flip and very hard to beat "the market
usually goes up". A model scoring 0.24 Brier looks skilful until you notice
that predicting the base rate scores 0.23. Every report here carries that
number next to the others.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

from vault.nn.dataset import Dataset, Split, Standardiser, purged_walk_forward
from vault.nn.network import MLP, Activation, LayerSpec, sigmoid

logger = logging.getLogger(__name__)

__all__ = [
    "LogisticModel", "FoldResult", "TrainingReport",
    "walk_forward_train", "permutation_importance",
]


class LogisticModel:
    """
    Plain logistic regression with L2, trained by gradient descent.

    Present as the honest baseline. It has as many parameters as the network
    has inputs, it cannot represent an interaction, and on small noisy
    samples that is frequently an advantage rather than a limitation.
    """

    def __init__(self, n_features: int, *, learning_rate: float = 0.1,
                 l2: float = 1e-3, epochs: int = 400) -> None:
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.w = [0.0] * n_features
        self.b = 0.0
        self.fitted = False

    def predict_one(self, x: Sequence[float]) -> float:
        total = self.b
        for weight, value in zip(self.w, x):
            total += weight * value
        return sigmoid(total)

    def predict(self, xs: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_one(x) for x in xs]

    def fit(self, xs: Sequence[Sequence[float]],
            ys: Sequence[float]) -> "LogisticModel":
        n = len(xs)
        if n == 0:
            return self
        # Initialise the intercept at the log-odds of the base rate, so the
        # model starts from "always predict the base rate" and has to earn any
        # movement away from it.
        rate = min(1 - 1e-6, max(1e-6, sum(ys) / n))
        self.b = math.log(rate / (1 - rate))

        for _ in range(self.epochs):
            grad_w = [0.0] * self.n_features
            grad_b = 0.0
            for x, y in zip(xs, ys):
                error = self.predict_one(x) - y
                grad_b += error
                for index, value in enumerate(x):
                    grad_w[index] += error * value
            scale = self.learning_rate / n
            for index in range(self.n_features):
                self.w[index] -= scale * (grad_w[index] + self.l2 * self.w[index] * n)
            self.b -= scale * grad_b
        self.fitted = True
        return self

    def brier(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> float:
        if not xs:
            return 0.0
        return sum((self.predict_one(x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)


@dataclass(slots=True)
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    purged: int

    base_rate: float = 0.5
    brier_base: float = 0.25
    brier_logistic: float = 0.25
    brier_network: float = 0.25

    brier_network_in_sample: float = 0.25
    epochs_run: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "purged": self.purged,
            "base_rate": round(self.base_rate, 4),
            "brier_base": round(self.brier_base, 5),
            "brier_logistic": round(self.brier_logistic, 5),
            "brier_network": round(self.brier_network, 5),
            "brier_network_in_sample": round(self.brier_network_in_sample, 5),
            "overfit_gap": round(
                self.brier_network - self.brier_network_in_sample, 5),
            "epochs_run": self.epochs_run,
        }


@dataclass(slots=True)
class TrainingReport:
    """The result of walk-forward training, and the verdict that follows."""

    folds: list[FoldResult] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    n_samples: int = 0
    winner: str = "base_rate"
    usable: bool = False
    reason: str = ""
    importance: list[tuple[str, float]] = field(default_factory=list)

    def _mean(self, attribute: str) -> float:
        values = [getattr(f, attribute) for f in self.folds]
        return statistics.fmean(values) if values else 0.25

    @property
    def brier_base(self) -> float:
        return self._mean("brier_base")

    @property
    def brier_logistic(self) -> float:
        return self._mean("brier_logistic")

    @property
    def brier_network(self) -> float:
        return self._mean("brier_network")

    @property
    def overfit_gap(self) -> float:
        """Out-of-sample minus in-sample Brier for the network."""
        return self.brier_network - self._mean("brier_network_in_sample")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "folds": [f.to_dict() for f in self.folds],
            "brier_base": round(self.brier_base, 5),
            "brier_logistic": round(self.brier_logistic, 5),
            "brier_network": round(self.brier_network, 5),
            "overfit_gap": round(self.overfit_gap, 5),
            "winner": self.winner,
            "usable": self.usable,
            "reason": self.reason,
            "importance": [
                {"feature": name, "delta_brier": round(value, 5)}
                for name, value in self.importance
            ],
        }

    def report(self) -> str:
        lines = [
            f"Walk-forward training on {self.n_samples} samples, "
            f"{len(self.folds)} purged folds",
            "",
            f"  {'fold':>5s} {'train':>6s} {'test':>5s} {'purged':>7s} "
            f"{'base':>8s} {'logistic':>9s} {'network':>8s} {'in-samp':>8s}",
            "  " + "-" * 68,
        ]
        for fold in self.folds:
            lines.append(
                f"  {fold.fold:5d} {fold.n_train:6d} {fold.n_test:5d} "
                f"{fold.purged:7d} {fold.brier_base:8.4f} "
                f"{fold.brier_logistic:9.4f} {fold.brier_network:8.4f} "
                f"{fold.brier_network_in_sample:8.4f}"
            )
        lines += [
            "  " + "-" * 68,
            f"  {'mean':>5s} {'':>6s} {'':>5s} {'':>7s} "
            f"{self.brier_base:8.4f} {self.brier_logistic:9.4f} "
            f"{self.brier_network:8.4f}",
            "",
            f"  winner: {self.winner.upper()}",
            f"  {self.reason}",
        ]
        if self.importance:
            lines += ["", "  Permutation importance (out-of-sample Brier increase"
                          " when the feature is shuffled):"]
            for name, value in self.importance[:8]:
                bar = "#" * max(0, min(40, int(value * 2000)))
                lines.append(f"    {name:24s} {value:+.5f} {bar}")
            useless = [n for n, v in self.importance if v <= 0]
            if useless:
                lines.append(
                    f"    ...and {len(useless)} feature(s) whose removal would "
                    f"IMPROVE the score; they are noise to this model."
                )
        return "\n".join(lines)


def _brier_constant(prediction: float, ys: Sequence[float]) -> float:
    if not ys:
        return 0.25
    return sum((prediction - y) ** 2 for y in ys) / len(ys)


def walk_forward_train(
    dataset: Dataset,
    *,
    folds: int = 5,
    hidden: Sequence[LayerSpec] | None = None,
    epochs: int = 150,
    patience: int = 20,
    validation_fraction: float = 0.2,
    seed: int = 42,
    compute_importance: bool = True,
) -> TrainingReport:
    """
    Fit the network and both baselines on identical purged folds.

    Within each fold the training block is split again -- the last
    ``validation_fraction`` of it, chronologically, becomes the early-stopping
    validation set. Chronologically, not randomly: a random inner split leaks
    the same way the outer one would.
    """
    report = TrainingReport(
        feature_names=list(dataset.feature_names), n_samples=len(dataset),
    )
    splits = purged_walk_forward(dataset, folds=folds)
    if not splits:
        report.reason = (
            f"{len(dataset)} samples is not enough to build {folds} purged "
            f"folds with a usable training block. No model is fitted, and "
            f"nothing is used downstream."
        )
        return report

    n_features = len(dataset.feature_names)
    all_x, all_y = dataset.x, dataset.y

    for split in splits:
        train_x = [all_x[i] for i in split.train]
        train_y = [all_y[i] for i in split.train]
        test_x = [all_x[i] for i in split.test]
        test_y = [all_y[i] for i in split.test]

        # Scale on the training fold only.
        scaler = Standardiser().fit(train_x)
        train_scaled = scaler.transform(train_x)
        test_scaled = scaler.transform(test_x)

        # Inner chronological split for early stopping.
        cut = max(1, int(len(train_scaled) * (1 - validation_fraction)))
        inner_x, inner_y = train_scaled[:cut], train_y[:cut]
        val_x, val_y = train_scaled[cut:], train_y[cut:]

        base_rate = sum(train_y) / len(train_y) if train_y else 0.5

        logistic = LogisticModel(n_features).fit(train_scaled, train_y)
        network = MLP(n_features, hidden, learning_rate=0.01, l2=1e-3,
                      seed=seed + split.index)
        history = network.fit(inner_x, inner_y, x_val=val_x, y_val=val_y,
                              epochs=epochs, patience=patience)

        report.folds.append(FoldResult(
            fold=split.index,
            n_train=len(split.train), n_test=len(split.test),
            purged=split.purged,
            base_rate=base_rate,
            brier_base=_brier_constant(base_rate, test_y),
            brier_logistic=logistic.brier(test_scaled, test_y),
            brier_network=network.brier(test_scaled, test_y),
            brier_network_in_sample=network.brier(inner_x, inner_y),
            epochs_run=history.epochs_run,
        ))

    _decide(report)

    if compute_importance and report.folds:
        report.importance = permutation_importance(dataset, splits, seed=seed)

    return report


def _decide(report: TrainingReport) -> None:
    """
    Name the winner, and say plainly when the deep model is not it.

    A margin is required, not just a lower number: with five folds and a few
    hundred samples, a 0.001 Brier difference is noise, and declaring a winner
    on it would make the choice of model a coin flip dressed as a measurement.
    """
    margin = 0.005
    base = report.brier_base
    logistic = report.brier_logistic
    network = report.brier_network

    best = min(base, logistic, network)

    if base <= best + 1e-12 or (base < logistic - margin and base < network - margin):
        report.winner = "base_rate"
        report.usable = False
        report.reason = (
            f"Neither model beats simply predicting the base rate "
            f"({base:.4f} against {logistic:.4f} logistic and {network:.4f} "
            f"network). The features carry no usable information at this "
            f"horizon on this sample. Nothing is used downstream -- which is "
            f"the correct outcome, not a failure to be tuned away."
        )
        return

    if network < logistic - margin:
        report.winner = "network"
        report.usable = True
        report.reason = (
            f"The network beats logistic regression out of sample "
            f"({network:.4f} against {logistic:.4f}) and both beat the base "
            f"rate ({base:.4f}). The overfit gap is {report.overfit_gap:+.4f}."
        )
        return

    if logistic < base - margin:
        report.winner = "logistic"
        report.usable = True
        report.reason = (
            f"Logistic regression wins at {logistic:.4f} against {network:.4f} "
            f"for the network and {base:.4f} for the base rate. The extra "
            f"capacity did not pay for itself, which is the usual outcome on "
            f"a few hundred noisy macro samples. The linear model is used."
        )
        return

    report.winner = "base_rate"
    report.usable = False
    report.reason = (
        f"No model clears the base rate by the {margin:.3f} margin required "
        f"to call a winner (base {base:.4f}, logistic {logistic:.4f}, network "
        f"{network:.4f}). Differences this small are noise at this sample size."
    )


def permutation_importance(
    dataset: Dataset, splits: Sequence[Split], *, seed: int = 42,
    repeats: int = 3,
) -> list[tuple[str, float]]:
    """
    How much worse the out-of-sample Brier gets when one feature is shuffled.

    Measured on the TEST fold, after training, so it reports what the model
    actually relies on to generalise rather than what it happened to latch
    onto in training. Features with a negative score are worse than useless
    to the model: removing them would improve it.
    """
    import random

    rng = random.Random(seed)
    n_features = len(dataset.feature_names)
    all_x, all_y = dataset.x, dataset.y
    totals = [0.0] * n_features
    counted = 0

    for split in splits:
        train_x = [all_x[i] for i in split.train]
        train_y = [all_y[i] for i in split.train]
        test_x = [all_x[i] for i in split.test]
        test_y = [all_y[i] for i in split.test]
        if len(test_x) < 5:
            continue

        scaler = Standardiser().fit(train_x)
        train_scaled = scaler.transform(train_x)
        test_scaled = scaler.transform(test_x)

        model = LogisticModel(n_features).fit(train_scaled, train_y)
        baseline = model.brier(test_scaled, test_y)

        for feature in range(n_features):
            deltas = []
            for _ in range(repeats):
                shuffled = [row[:] for row in test_scaled]
                column = [row[feature] for row in shuffled]
                rng.shuffle(column)
                for row, value in zip(shuffled, column):
                    row[feature] = value
                deltas.append(model.brier(shuffled, test_y) - baseline)
            totals[feature] += statistics.fmean(deltas)
        counted += 1

    if not counted:
        return []
    scores = [(name, totals[i] / counted)
              for i, name in enumerate(dataset.feature_names)]
    return sorted(scores, key=lambda pair: -pair[1])
