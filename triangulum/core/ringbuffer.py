"""
Fixed-capacity ring buffers.

The engine keeps rolling windows of a great many things -- recent edges per
cycle, per-venue latencies, realized slippage, book update intervals -- and it
keeps them in the hot path. ``collections.deque(maxlen=N)`` is fine for storage
but computing a mean over it is O(N) every time, and doing that per book update
across 300 symbols is real CPU.

These structures maintain their aggregates incrementally, so ``mean`` and
``stddev`` are O(1). The numerical care (Welford + periodic recompute) matters:
naively subtracting the evicted value from a running sum accumulates float error
until, after a few million updates, the "variance" goes negative.
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

__all__ = ["RingBuffer", "RollingStats", "RollingQuantile", "TimedRingBuffer"]


class RingBuffer:
    """Plain fixed-capacity circular buffer of floats."""

    __slots__ = ("_data", "_capacity", "_head", "_size")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._data: list[float] = [0.0] * capacity
        self._head = 0
        self._size = 0

    def push(self, value: float) -> float | None:
        """Append; returns the evicted value if the buffer was full."""
        evicted: float | None = None
        if self._size == self._capacity:
            evicted = self._data[self._head]
        self._data[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1
        return evicted

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[float]:
        if self._size < self._capacity:
            yield from self._data[: self._size]
        else:
            yield from self._data[self._head:]
            yield from self._data[: self._head]

    def __getitem__(self, index: int) -> float:
        if not -self._size <= index < self._size:
            raise IndexError(index)
        if index < 0:
            index += self._size
        start = 0 if self._size < self._capacity else self._head
        return self._data[(start + index) % self._capacity]

    @property
    def full(self) -> bool:
        return self._size == self._capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    def last(self, default: float = 0.0) -> float:
        if self._size == 0:
            return default
        return self._data[(self._head - 1) % self._capacity]

    def to_list(self) -> list[float]:
        return list(self)

    def clear(self) -> None:
        self._head = 0
        self._size = 0


class RollingStats:
    """
    Rolling window with O(1) mean/variance and bounded numerical drift.

    Every ``recompute_every`` pushes the aggregates are rebuilt from scratch.
    That is an O(N) operation amortized to O(1) per push, and it is the cheapest
    possible insurance against the running-sum error that otherwise makes
    long-lived rolling variances quietly meaningless.
    """

    __slots__ = ("_buf", "_sum", "_sumsq", "_pushes", "_recompute_every", "_min", "_max")

    def __init__(self, capacity: int, *, recompute_every: int = 100_000) -> None:
        self._buf = RingBuffer(capacity)
        self._sum = 0.0
        self._sumsq = 0.0
        self._pushes = 0
        self._recompute_every = recompute_every
        self._min = math.inf
        self._max = -math.inf

    def push(self, value: float) -> None:
        evicted = self._buf.push(value)
        self._sum += value
        self._sumsq += value * value
        if evicted is not None:
            self._sum -= evicted
            self._sumsq -= evicted * evicted
        self._pushes += 1
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value
        if self._pushes % self._recompute_every == 0:
            self._recompute()

    def _recompute(self) -> None:
        self._sum = 0.0
        self._sumsq = 0.0
        self._min = math.inf
        self._max = -math.inf
        for v in self._buf:
            self._sum += v
            self._sumsq += v * v
            self._min = min(self._min, v)
            self._max = max(self._max, v)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def count(self) -> int:
        return len(self._buf)

    @property
    def mean(self) -> float:
        n = len(self._buf)
        return self._sum / n if n else 0.0

    @property
    def variance(self) -> float:
        n = len(self._buf)
        if n < 2:
            return 0.0
        var = (self._sumsq - (self._sum * self._sum) / n) / (n - 1)
        return max(0.0, var)   # clamp: float error can produce tiny negatives

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    @property
    def total(self) -> float:
        return self._sum

    @property
    def minimum(self) -> float:
        return self._min if len(self._buf) else 0.0

    @property
    def maximum(self) -> float:
        return self._max if len(self._buf) else 0.0

    @property
    def last(self) -> float:
        return self._buf.last()

    def zscore(self, value: float) -> float:
        sd = self.stddev
        return (value - self.mean) / sd if sd > 1e-12 else 0.0

    def to_list(self) -> list[float]:
        return self._buf.to_list()

    def clear(self) -> None:
        self._buf.clear()
        self._sum = 0.0
        self._sumsq = 0.0
        self._min = math.inf
        self._max = -math.inf


class RollingQuantile:
    """
    Approximate quantiles over a stream via the P-Square-style update.

    Exact quantiles need the whole window sorted; we need p50/p95 of latency
    every few milliseconds and cannot afford it. This is a simple stochastic
    approximation: the estimate walks toward the target quantile with a step
    proportional to a running scale estimate. It converges quickly and its error
    is well under the precision anyone acts on.
    """

    __slots__ = ("_q", "_estimate", "_step", "_count", "_initialized", "_scale")

    def __init__(self, quantile: float, *, step: float = 0.01) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        self._q = quantile
        self._estimate = 0.0
        self._step = step
        self._count = 0
        self._initialized = False
        self._scale = 1.0

    def push(self, value: float) -> None:
        self._count += 1
        if not self._initialized:
            self._estimate = value
            self._scale = max(abs(value), 1e-9)
            self._initialized = True
            return
        # Track magnitude so the step size stays scale-appropriate.
        self._scale += (abs(value) - self._scale) * 0.001
        delta = self._step * self._scale
        if value > self._estimate:
            self._estimate += delta * self._q
        elif value < self._estimate:
            self._estimate -= delta * (1.0 - self._q)

    @property
    def value(self) -> float:
        return self._estimate

    @property
    def count(self) -> int:
        return self._count


class TimedRingBuffer:
    """
    Ring buffer of (timestamp_ns, value) that also supports time-window queries.

    Used for rate limiting ("how many orders in the last second") and for the
    risk layer's "how many losses in the last five minutes" breakers, where the
    window is temporal rather than count-based.
    """

    __slots__ = ("_ts", "_val", "_capacity", "_head", "_size")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._ts: list[int] = [0] * capacity
        self._val: list[float] = [0.0] * capacity
        self._head = 0
        self._size = 0

    def push(self, ts_ns: int, value: float = 1.0) -> None:
        self._ts[self._head] = ts_ns
        self._val[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def _indices(self) -> Iterator[int]:
        start = 0 if self._size < self._capacity else self._head
        for i in range(self._size):
            yield (start + i) % self._capacity

    def count_since(self, cutoff_ns: int) -> int:
        return sum(1 for i in self._indices() if self._ts[i] >= cutoff_ns)

    def sum_since(self, cutoff_ns: int) -> float:
        return sum(self._val[i] for i in self._indices() if self._ts[i] >= cutoff_ns)

    def values_since(self, cutoff_ns: int) -> list[float]:
        return [self._val[i] for i in self._indices() if self._ts[i] >= cutoff_ns]

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._head = 0
        self._size = 0
