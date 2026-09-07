"""
Time.

Two clocks, deliberately separated:

``wall``      -- ``time.time_ns()``. Comparable across machines and with venue
                 timestamps. Jumps when NTP steps it. Never use for durations.
``monotonic`` -- ``time.perf_counter_ns()``. Cannot go backwards. Meaningless as
                 an absolute value. Always use for durations and budgets.

Mixing them is the classic latency-measurement bug: an NTP step of 40ms in the
middle of a cycle silently turns into 40ms of phantom "latency" that poisons the
learner's features for as long as the model's memory lasts.

The ``Clock`` indirection also lets the backtester drive simulated time through
exactly the same code path as live, with no ``if backtest:`` branches sprinkled
through the engine.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "Clock",
    "SystemClock",
    "SimulatedClock",
    "LatencyBudget",
    "Stopwatch",
    "EwmaLatency",
    "NS_PER_US",
    "NS_PER_MS",
    "NS_PER_SEC",
]

NS_PER_US = 1_000
NS_PER_MS = 1_000_000
NS_PER_SEC = 1_000_000_000


class Clock(Protocol):
    def wall_ns(self) -> int: ...
    def mono_ns(self) -> int: ...
    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """Real time. The only clock used in paper and live modes."""

    __slots__ = ()

    def wall_ns(self) -> int:
        return time.time_ns()

    def mono_ns(self) -> int:
        return time.perf_counter_ns()

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)


class SimulatedClock:
    """
    Time under the backtester's control.

    ``sleep`` advances the clock instantly rather than yielding to the event
    loop, so a 30-day replay does not take 30 days. Anything that awaits a
    timeout therefore resolves deterministically -- which is exactly what you
    want when a backtest must be reproducible bit-for-bit.
    """

    __slots__ = ("_wall_ns", "_mono_ns", "_speed")

    def __init__(self, start_wall_ns: int = 0, speed: float = 0.0) -> None:
        self._wall_ns = start_wall_ns or time.time_ns()
        self._mono_ns = 0
        # speed == 0 -> instant. speed == 1.0 -> real time. 60.0 -> 60x.
        self._speed = speed

    def wall_ns(self) -> int:
        return self._wall_ns

    def mono_ns(self) -> int:
        return self._mono_ns

    def advance_ns(self, delta_ns: int) -> None:
        if delta_ns < 0:
            raise ValueError("simulated clock cannot go backwards")
        self._wall_ns += delta_ns
        self._mono_ns += delta_ns

    def set_wall_ns(self, wall_ns: int) -> None:
        """Jump to an absolute wall time, advancing monotonic by the same delta."""
        delta = wall_ns - self._wall_ns
        if delta < 0:
            raise ValueError(
                f"replay timestamps out of order: {wall_ns} < {self._wall_ns}"
            )
        self.advance_ns(delta)

    async def sleep(self, seconds: float) -> None:
        self.advance_ns(int(seconds * NS_PER_SEC))
        if self._speed > 0:
            await asyncio.sleep(seconds / self._speed)
        else:
            await asyncio.sleep(0)  # yield, so other tasks make progress


@dataclass(slots=True)
class Stopwatch:
    """Context manager measuring monotonic elapsed nanoseconds."""

    clock: Clock = field(default_factory=SystemClock)
    start_ns: int = 0
    end_ns: int = 0

    def __enter__(self) -> "Stopwatch":
        self.start_ns = self.clock.mono_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        self.end_ns = self.clock.mono_ns()

    @property
    def elapsed_ns(self) -> int:
        end = self.end_ns or self.clock.mono_ns()
        return end - self.start_ns

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / NS_PER_MS


@dataclass(slots=True)
class LatencyBudget:
    """
    A shrinking deadline carried through a cycle's execution.

    Arbitrage is a race against the opportunity's own decay. Rather than giving
    each leg a fixed timeout, the executor allocates a total budget at the top
    and each leg consumes from it. When the budget is exhausted mid-cycle the
    right move is to stop opening new legs and start unwinding -- a half-built
    cycle at t+300ms is a directional position nobody asked for.
    """

    total_ns: int
    clock: Clock = field(default_factory=SystemClock)
    start_ns: int = 0

    def __post_init__(self) -> None:
        if not self.start_ns:
            self.start_ns = self.clock.mono_ns()

    @classmethod
    def from_ms(cls, ms: float, clock: Clock | None = None) -> "LatencyBudget":
        return cls(total_ns=int(ms * NS_PER_MS), clock=clock or SystemClock())

    @property
    def elapsed_ns(self) -> int:
        return self.clock.mono_ns() - self.start_ns

    @property
    def remaining_ns(self) -> int:
        return max(0, self.total_ns - self.elapsed_ns)

    @property
    def remaining_sec(self) -> float:
        return self.remaining_ns / NS_PER_SEC

    @property
    def expired(self) -> bool:
        return self.remaining_ns <= 0

    @property
    def consumed_fraction(self) -> float:
        if self.total_ns <= 0:
            return 1.0
        return min(1.0, self.elapsed_ns / self.total_ns)

    def slice_for_leg(self, legs_remaining: int, *, safety: float = 0.85) -> float:
        """
        Seconds to allow the next leg.

        Divides what is left evenly among remaining legs and holds back a
        ``1 - safety`` reserve so there is always time to unwind. Never returns
        a negative number; callers check ``expired`` for the abort decision.
        """
        if legs_remaining <= 0:
            return 0.0
        return max(0.0, (self.remaining_sec * safety) / legs_remaining)

    def child(self, fraction: float) -> "LatencyBudget":
        return LatencyBudget(
            total_ns=int(self.remaining_ns * fraction),
            clock=self.clock,
            start_ns=self.clock.mono_ns(),
        )


@dataclass(slots=True)
class EwmaLatency:
    """
    Exponentially-weighted latency tracker with a jitter estimate.

    Feeds two consumers: the executor (to size the latency budget) and the
    learner (venue latency is a strong predictor of whether a leg fills). The
    variance estimate uses Welford-style EWMA so a single 2-second GC pause does
    not permanently inflate the mean.
    """

    alpha: float = 0.05
    mean_ns: float = 0.0
    var_ns2: float = 0.0
    count: int = 0
    max_ns: int = 0

    def observe(self, sample_ns: int) -> None:
        self.count += 1
        self.max_ns = max(self.max_ns, sample_ns)
        if self.count == 1:
            self.mean_ns = float(sample_ns)
            return
        delta = sample_ns - self.mean_ns
        self.mean_ns += self.alpha * delta
        self.var_ns2 = (1 - self.alpha) * (self.var_ns2 + self.alpha * delta * delta)

    @property
    def stddev_ns(self) -> float:
        return self.var_ns2 ** 0.5

    @property
    def mean_ms(self) -> float:
        return self.mean_ns / NS_PER_MS

    @property
    def p95_estimate_ns(self) -> float:
        """Gaussian approximation. Good enough for budgeting, not for SLOs."""
        return self.mean_ns + 1.645 * self.stddev_ns

    def reset(self) -> None:
        self.mean_ns = 0.0
        self.var_ns2 = 0.0
        self.count = 0
        self.max_ns = 0
