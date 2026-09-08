"""
Building a learnable dataset out of a macro series universe.

## Purged, embargoed walk-forward

Ordinary k-fold cross-validation is wrong for this data, and wrong in a
direction that flatters the model. Two reasons, both fatal:

1. **Shuffling leaks the future.** A random split puts Tuesday in the test
   set and Wednesday in the training set. The model learns from tomorrow to
   predict today and reports a wonderful score it can never reproduce live.

2. **Overlapping labels leak across the boundary even without shuffling.**
   A 21-day forward return computed on day 100 covers days 100-121. A
   training sample from day 110 shares eleven days of outcome with it. Split
   at day 105 and the two sides of the split are still describing the same
   fortnight.

The fix, from Lopez de Prado: split chronologically, then PURGE from the
training set any sample whose label window overlaps the test window, and add
an EMBARGO of further samples immediately after the test set to account for
serial correlation the purge does not catch. The cost is training data. The
benefit is a validation number that means something.

## Standardisation

Fitted on the training fold only, then applied to the test fold. Fitting the
scaler on everything is the quietest leak of all -- it uses the test set's
mean and variance, which is information from the future, and it moves the
reported score by a surprising amount on small samples.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

from vault.data.series import Series
from vault.signals.library import SIGNALS, SignalSpec

logger = logging.getLogger(__name__)

__all__ = [
    "Dataset", "Sample", "Standardiser", "Split",
    "build_dataset", "purged_walk_forward",
]


@dataclass(slots=True)
class Sample:
    """One training example: what was knowable, and what happened next."""

    on: date
    features: list[float]
    label: float                 # 1.0 if the target rose over the horizon
    forward_return: float
    label_window_end: date       # for purging

    # Which signals were unusable at this point in time. Carried so the model
    # can be told "this feature is missing" rather than "this feature is zero".
    missing: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Dataset:
    feature_names: list[str]
    samples: list[Sample] = field(default_factory=list)
    target: str = ""
    horizon_days: int = 0

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def x(self) -> list[list[float]]:
        return [s.features for s in self.samples]

    @property
    def y(self) -> list[float]:
        return [s.label for s in self.samples]

    @property
    def base_rate(self) -> float:
        if not self.samples:
            return 0.5
        return sum(s.label for s in self.samples) / len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "horizon_days": self.horizon_days,
            "n": len(self.samples),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "base_rate": round(self.base_rate, 4),
            "span": [
                self.samples[0].on.isoformat(),
                self.samples[-1].on.isoformat(),
            ] if self.samples else None,
        }


class Standardiser:
    """
    Zero-mean unit-variance scaling, fitted on training data only.

    Constant features get a variance of 1 rather than 0 so they map to zero
    instead of producing a division by zero. A constant feature carries no
    information and should contribute nothing, which is exactly what a
    constant zero does.
    """

    __slots__ = ("means", "stdevs", "fitted")

    def __init__(self) -> None:
        self.means: list[float] = []
        self.stdevs: list[float] = []
        self.fitted = False

    def fit(self, rows: Sequence[Sequence[float]]) -> "Standardiser":
        if not rows:
            return self
        width = len(rows[0])
        self.means = []
        self.stdevs = []
        for column in range(width):
            values = [row[column] for row in rows]
            mean = statistics.fmean(values)
            if len(values) > 1:
                variance = statistics.pvariance(values, mu=mean)
            else:
                variance = 0.0
            stdev = math.sqrt(variance)
            self.means.append(mean)
            self.stdevs.append(stdev if stdev > 1e-9 else 1.0)
        self.fitted = True
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        if not self.fitted:
            return [list(row) for row in rows]
        return [
            [
                # Clip to +/-4 sigma. An unclipped outlier in a standardised
                # feature reaches the first layer as a value of 20 and, through
                # ReLU, dominates every other input for that sample.
                max(-4.0, min(4.0, (value - mean) / stdev))
                for value, mean, stdev in zip(row, self.means, self.stdevs)
            ]
            for row in rows
        ]

    def fit_transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return self.fit(rows).transform(rows)

    def to_dict(self) -> dict[str, Any]:
        return {"means": self.means, "stdevs": self.stdevs, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Standardiser":
        scaler = cls()
        scaler.means = list(raw.get("means", []))
        scaler.stdevs = list(raw.get("stdevs", []))
        scaler.fitted = bool(raw.get("fitted"))
        return scaler


@dataclass(slots=True)
class Split:
    """One purged walk-forward fold."""

    index: int
    train: list[int]
    test: list[int]
    purged: int = 0
    embargoed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.index,
            "train": len(self.train),
            "test": len(self.test),
            "purged": self.purged,
            "embargoed": self.embargoed,
        }


def build_dataset(
    series: Mapping[str, Series],
    *,
    target: str = "SP500",
    horizon_days: int = 21,
    step_days: int = 5,
    lookback_days: int = 2000,
    specs: Sequence[SignalSpec] | None = None,
) -> Dataset:
    """
    Walk history and emit one sample per step.

    Each sample's features are the signal vector as it would have been
    computed on that day -- point in time, via ``Series.until`` -- and its
    label is whether the target rose over the following ``horizon_days``.

    A signal that is unavailable or too stale on a given day contributes 0.0
    AND its index is recorded in ``Sample.missing``, so the caller can add
    missingness indicators rather than letting the model read "unavailable"
    as "neutral".
    """
    specs = list(specs if specs is not None else SIGNALS.values())
    names = [spec.key for spec in specs]
    prices = series.get(target)
    dataset = Dataset(feature_names=names, target=target, horizon_days=horizon_days)
    if not prices or len(prices) < 2:
        return dataset

    end = prices.points[-1].on - timedelta(days=horizon_days)
    start = max(prices.points[0].on, end - timedelta(days=lookback_days))

    when = start
    while when <= end:
        entry = prices.value_on(when, tolerance_days=5)
        exit_price = prices.value_on(when + timedelta(days=horizon_days),
                                     tolerance_days=5)
        if entry is None or exit_price is None or entry <= 0:
            when += timedelta(days=step_days)
            continue

        history = {key: value.until(when) for key, value in series.items()}
        features: list[float] = []
        missing: list[int] = []
        for index, spec in enumerate(specs):
            reading = spec.evaluate(history)
            if reading.usable:
                features.append(reading.weighted_strength)
            else:
                features.append(0.0)
                missing.append(index)

        forward = (exit_price / entry - 1) * 100
        dataset.samples.append(Sample(
            on=when,
            features=features,
            label=1.0 if forward > 0 else 0.0,
            forward_return=forward,
            label_window_end=when + timedelta(days=horizon_days),
            missing=missing,
        ))
        when += timedelta(days=step_days)

    return dataset


def purged_walk_forward(
    dataset: Dataset,
    *,
    folds: int = 5,
    embargo_fraction: float = 0.01,
    min_train: int = 40,
) -> list[Split]:
    """
    Chronological folds with purging and an embargo.

    Fold *i* tests on the *i*-th chronological block and trains on everything
    BEFORE it -- expanding window, not sliding, because throwing away the
    oldest data on a sample this small costs more than the regime drift it
    avoids. Then:

        purge     drop any training sample whose label window extends into
                  the test block's date range
        embargo   drop a further `embargo_fraction` of the sample immediately
                  following the test block

    Note what this does NOT do: it never trains on data after the test block.
    Some walk-forward implementations do, on the grounds that they are only
    "validating hyperparameters". That is still using the future, and the
    resulting number is still not achievable live.
    """
    n = len(dataset.samples)
    splits: list[Split] = []
    if n < min_train + folds:
        return splits

    block = n // (folds + 1)
    if block < 1:
        return splits

    embargo = max(1, int(n * embargo_fraction))

    for fold in range(folds):
        test_start = block * (fold + 1)
        test_end = min(n, test_start + block)
        if test_end - test_start < 1:
            continue

        test_indices = list(range(test_start, test_end))
        test_from = dataset.samples[test_start].on
        test_to = dataset.samples[test_end - 1].label_window_end

        train_indices: list[int] = []
        purged = 0
        for index in range(0, test_start):
            sample = dataset.samples[index]
            # Purge: this sample's outcome is still unfolding when the test
            # window opens, so it shares information with the test set.
            if sample.label_window_end >= test_from:
                purged += 1
                continue
            train_indices.append(index)

        embargoed = 0
        if test_end < n:
            embargoed = min(embargo, n - test_end)

        if len(train_indices) < min_train:
            continue

        splits.append(Split(
            index=fold, train=train_indices, test=test_indices,
            purged=purged, embargoed=embargoed,
        ))

    return splits
