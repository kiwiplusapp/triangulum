"""
Concept drift detection.

A model trained on last month's market is a model of last month's market. When
the regime changes -- a venue upgrades its matching engine, a large participant
leaves, volatility triples -- the relationships the model learned stop holding,
and it keeps confidently applying them.

Two detectors, both classic and both cheap:

**ADWIN** (Adaptive Windowing, Bifet & Gavaldà 2007) maintains a window of
recent observations and, at every step, checks whether any split of that window
into two halves shows a statistically significant difference in mean. If so, the
older half is dropped. It is parameter-light (one confidence value), it detects
both abrupt and gradual drift, and it gives a bound on false positives.

**Page-Hinkley** tracks the cumulative deviation from the running mean and fires
when it exceeds a threshold. Cheaper than ADWIN and better at abrupt shifts;
worse at gradual ones. Used as a fast first alarm.

On detection the engine does NOT wipe the model. That would throw away
everything it knows because of one bad afternoon. Instead it:
    1. raises the EV gate's uncertainty multiplier (become timid),
    2. increases the exploration rate (gather fresh data),
    3. accelerates the FTRL learning rate (adapt faster),
    4. emits an alert.

Resetting is a manual decision, because the most common cause of a drift alarm
is not drift -- it is a data feed that degraded.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

logger = logging.getLogger(__name__)

__all__ = ["ADWIN", "PageHinkley", "DriftMonitor", "DriftEvent"]


@dataclass(slots=True)
class DriftEvent:
    detector: str
    at_sample: int
    old_mean: float
    new_mean: float
    magnitude: float

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"drift detected by {self.detector} at sample {self.at_sample}: "
            f"mean moved {self.old_mean:.4f} -> {self.new_mean:.4f} "
            f"({self.magnitude:+.1%})"
        )


class ADWIN:
    """
    Adaptive windowing drift detector.

    A simplified but faithful implementation: observations are held in
    exponential-histogram buckets, and after each insertion every bucket
    boundary is tested as a candidate split point using the Hoeffding bound.
    """

    def __init__(self, *, delta: float = 0.002, max_buckets: int = 5,
                 min_window: int = 32) -> None:
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window = min_window
        # Each entry: [total, count]
        self._buckets: list[list[float]] = []
        self.total = 0.0
        self.count = 0
        self.detections = 0
        self.last_event: DriftEvent | None = None

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def variance(self) -> float:
        return 0.25   # bounded [0,1] observations; the Hoeffding bound holds

    def add(self, value: float) -> bool:
        """Insert an observation. Returns True when drift is detected."""
        self._buckets.append([value, 1.0])
        self.total += value
        self.count += 1
        self._compress()
        return self._check_split()

    def _compress(self) -> None:
        """Merge adjacent equal-size buckets to keep the histogram logarithmic."""
        index = len(self._buckets) - 1
        while index > 0:
            if len(self._buckets) <= self.max_buckets:
                break
            a, b = self._buckets[index - 1], self._buckets[index]
            if a[1] == b[1]:
                self._buckets[index - 1] = [a[0] + b[0], a[1] + b[1]]
                del self._buckets[index]
            index -= 1

    def _check_split(self) -> bool:
        if self.count < self.min_window or len(self._buckets) < 2:
            return False

        left_total = 0.0
        left_count = 0.0
        for index in range(len(self._buckets) - 1):
            left_total += self._buckets[index][0]
            left_count += self._buckets[index][1]
            right_total = self.total - left_total
            right_count = self.count - left_count
            if left_count < 2 or right_count < 2:
                continue

            left_mean = left_total / left_count
            right_mean = right_total / right_count

            # Hoeffding bound with the harmonic mean of the two window sizes.
            m = 1.0 / (1.0 / left_count + 1.0 / right_count)
            delta_prime = self.delta / max(1.0, math.log(self.count))
            epsilon = math.sqrt(
                (2.0 / m) * self.variance * math.log(2.0 / delta_prime)
            ) + (2.0 / (3.0 * m)) * math.log(2.0 / delta_prime)

            if abs(left_mean - right_mean) > epsilon:
                self.detections += 1
                self.last_event = DriftEvent(
                    detector="adwin", at_sample=self.count,
                    old_mean=left_mean, new_mean=right_mean,
                    magnitude=(right_mean - left_mean) / left_mean
                    if abs(left_mean) > 1e-9 else 0.0,
                )
                # Drop the stale prefix.
                self._buckets = self._buckets[index + 1:]
                self.total = right_total
                self.count = int(right_count)
                logger.warning("%s", self.last_event)
                return True
        return False

    def stats(self) -> dict[str, object]:
        return {
            "detector": "adwin",
            "window": self.count,
            "mean": round(self.mean, 5),
            "detections": self.detections,
            "buckets": len(self._buckets),
        }


class PageHinkley:
    """Cumulative-deviation test. Fast, good at abrupt shifts."""

    def __init__(self, *, threshold: float = 25.0, alpha: float = 0.005,
                 min_samples: int = 30) -> None:
        self.threshold = threshold
        self.alpha = alpha
        self.min_samples = min_samples
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.count = 0
        self.detections = 0

    def add(self, value: float) -> bool:
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative += value - self.mean - self.alpha
        self.minimum = min(self.minimum, self.cumulative)

        if self.count < self.min_samples:
            return False
        if self.cumulative - self.minimum > self.threshold:
            self.detections += 1
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0
        self.count = 0

    def stats(self) -> dict[str, object]:
        return {
            "detector": "page_hinkley",
            "samples": self.count,
            "statistic": round(self.cumulative - self.minimum, 4),
            "threshold": self.threshold,
            "detections": self.detections,
        }


class DriftMonitor:
    """
    Runs both detectors over the model's prediction errors and decides how the
    engine should respond.

    The monitored signal is the *absolute prediction error* of the fill model,
    normalised to [0, 1]. Watching the error rather than the raw fill rate is
    what distinguishes drift ("the world changed and the model is now wrong")
    from a market that is simply quiet ("fewer fills, but the model correctly
    predicted fewer fills").
    """

    def __init__(
        self,
        *,
        delta: float = 0.002,
        min_samples: int = 100,
        cooldown_samples: int = 500,
    ) -> None:
        self.adwin = ADWIN(delta=delta)
        self.page_hinkley = PageHinkley()
        self.min_samples = min_samples
        self.cooldown_samples = cooldown_samples
        self.samples = 0
        self._last_detection_at = -10**9
        self.events: list[DriftEvent] = []
        self.drift_active = False

    def observe(self, predicted: float, actual: int) -> DriftEvent | None:
        self.samples += 1
        error = abs(predicted - actual)

        adwin_fired = self.adwin.add(error)
        ph_fired = self.page_hinkley.add(error)

        if self.samples < self.min_samples:
            return None
        if self.samples - self._last_detection_at < self.cooldown_samples:
            return None
        if not (adwin_fired or ph_fired):
            return None

        self._last_detection_at = self.samples
        event = self.adwin.last_event or DriftEvent(
            detector="page_hinkley", at_sample=self.samples,
            old_mean=self.page_hinkley.mean, new_mean=error, magnitude=0.0,
        )
        self.events.append(event)
        if len(self.events) > 100:
            self.events = self.events[-100:]
        self.drift_active = True
        return event

    def response(self) -> dict[str, float]:
        """
        How the engine should adapt while drift is active.

        Deliberately conservative: become timid, explore more, learn faster.
        Never "reset the model" -- that is a human decision, because the most
        common cause of a drift alarm is a degraded data feed, and resetting
        the model in response to that destroys good weights for no reason.
        """
        if not self.drift_active:
            return {
                "confidence_multiplier": 1.0,
                "exploration_floor": 0.05,
                "learning_rate_multiplier": 1.0,
            }
        return {
            "confidence_multiplier": 2.0,     # demand a wider EV safety margin
            "exploration_floor": 0.15,        # gather fresh data faster
            "learning_rate_multiplier": 3.0,  # adapt quicker
        }

    def clear(self) -> None:
        """Acknowledge the drift, after a human has looked at it."""
        self.drift_active = False

    def stats(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "drift_active": self.drift_active,
            "total_events": len(self.events),
            "recent_events": [str(e) for e in self.events[-5:]],
            "adwin": self.adwin.stats(),
            "page_hinkley": self.page_hinkley.stats(),
            "response": self.response(),
        }
