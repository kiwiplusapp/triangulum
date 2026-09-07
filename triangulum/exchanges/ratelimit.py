"""
Rate limiting.

Venues punish bursts harshly -- Binance escalates from a 429 to an IP ban, and
Kraken's counter decays so slowly that a moment's carelessness costs minutes of
lockout. Being rate-limited mid-cycle is worse than being slow: it strands you
holding leg one of a three-leg position.

Two layers:

``TokenBucket``   classic bucket. Smooths bursts, allows a configurable amount
                  of accumulated credit.

``WeightedLimiter`` models the venue's own accounting, where different
                  endpoints cost different weights against a shared budget
                  (Binance: 1200 weight/minute, a depth-20 snapshot costs 1, a
                  depth-5000 snapshot costs 50).

Both are async and fair: waiters are served in arrival order, so a burst of
order submissions cannot starve the book resync that would let us trade again.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = ["TokenBucket", "WeightedLimiter", "AdaptiveThrottle"]


class TokenBucket:
    """Refills continuously at ``rate`` tokens/sec, capped at ``capacity``."""

    __slots__ = ("rate", "capacity", "_tokens", "_last", "_lock", "_waits", "_wait_time")

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()
        self._waits = 0
        self._wait_time = 0.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    async def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available. Returns seconds waited."""
        if tokens > self.capacity:
            raise ValueError(
                f"request of {tokens} exceeds bucket capacity {self.capacity}"
            )
        start = time.monotonic()
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    waited = time.monotonic() - start
                    if waited > 0.001:
                        self._waits += 1
                        self._wait_time += waited
                    return waited
                deficit = tokens - self._tokens
                await asyncio.sleep(deficit / self.rate)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking variant, for paths that must never wait."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    def stats(self) -> dict[str, float]:
        return {
            "rate": self.rate,
            "capacity": self.capacity,
            "available": self.available,
            "waits": self._waits,
            "total_wait_sec": self._wait_time,
        }


@dataclass
class _Window:
    limit: float
    period_sec: float
    events: deque = field(default_factory=deque)

    def prune(self, now: float) -> None:
        cutoff = now - self.period_sec
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def used(self, now: float) -> float:
        self.prune(now)
        return sum(w for _, w in self.events)

    def time_until_available(self, weight: float, now: float) -> float:
        self.prune(now)
        used = sum(w for _, w in self.events)
        if used + weight <= self.limit:
            return 0.0
        # Wait until enough of the oldest events age out.
        needed = used + weight - self.limit
        freed = 0.0
        for ts, w in self.events:
            freed += w
            if freed >= needed:
                return max(0.0, (ts + self.period_sec) - now)
        return self.period_sec

    def record(self, weight: float, now: float) -> None:
        self.events.append((now, weight))


class WeightedLimiter:
    """
    Multi-window weighted limiter matching how venues actually account.

    Binance, for example, enforces simultaneously:
        1200 request-weight per minute
        50 orders per 10 seconds
        160,000 orders per day
    A limiter that models only the first will get you banned by the second.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._windows: dict[str, _Window] = {}
        self._lock = asyncio.Lock()
        self._blocked_until = 0.0
        self.throttle_events = 0

    def add_window(self, key: str, limit: float, period_sec: float) -> "WeightedLimiter":
        self._windows[key] = _Window(limit=limit, period_sec=period_sec)
        return self

    async def acquire(self, weights: dict[str, float] | None = None,
                      default_weight: float = 1.0) -> float:
        """Wait until every applicable window admits the request."""
        start = time.monotonic()
        async with self._lock:
            while True:
                now = time.monotonic()
                if now < self._blocked_until:
                    await asyncio.sleep(self._blocked_until - now)
                    continue
                delays = []
                for key, window in self._windows.items():
                    w = (weights or {}).get(key, default_weight)
                    delays.append(window.time_until_available(w, now))
                delay = max(delays) if delays else 0.0
                if delay <= 0:
                    for key, window in self._windows.items():
                        window.record((weights or {}).get(key, default_weight), now)
                    return time.monotonic() - start
                self.throttle_events += 1
                await asyncio.sleep(min(delay, 5.0))

    def penalize(self, seconds: float) -> None:
        """
        Hard stop after a 429.

        Called with the venue's ``Retry-After``. Respecting it exactly is not
        politeness -- ignoring it is how a 429 becomes an IP ban, and an IP ban
        mid-cycle means an unhedged position you cannot close.
        """
        self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)
        logger.warning("limiter %s penalized for %.1fs", self.name, seconds)

    @property
    def blocked(self) -> bool:
        return time.monotonic() < self._blocked_until

    def stats(self) -> dict[str, object]:
        now = time.monotonic()
        return {
            "name": self.name,
            "throttle_events": self.throttle_events,
            "blocked": self.blocked,
            "windows": {
                k: {"used": w.used(now), "limit": w.limit, "period": w.period_sec}
                for k, w in self._windows.items()
            },
        }


class AdaptiveThrottle:
    """
    Backs off automatically when the venue signals stress.

    Additive-increase / multiplicative-decrease on the request rate, driven by
    observed 429s and latency. The same control law TCP uses, for the same
    reason: it finds the venue's actual limit without needing to be told, and it
    converges rather than oscillating.
    """

    def __init__(
        self,
        base_rate: float,
        *,
        min_rate: float = 0.5,
        max_rate: float | None = None,
        increase_per_sec: float = 0.1,
        decrease_factor: float = 0.5,
    ) -> None:
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate or base_rate * 2
        self.increase_per_sec = increase_per_sec
        self.decrease_factor = decrease_factor
        self._rate = base_rate
        self._last_adjust = time.monotonic()
        self._bucket = TokenBucket(base_rate)
        self.backoffs = 0

    async def acquire(self, tokens: float = 1.0) -> float:
        self._maybe_increase()
        return await self._bucket.acquire(tokens)

    def _maybe_increase(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_adjust
        if elapsed < 1.0:
            return
        self._last_adjust = now
        if self._rate < self.max_rate:
            self._rate = min(self.max_rate, self._rate + self.increase_per_sec * elapsed)
            self._bucket.rate = self._rate

    def on_rate_limited(self) -> None:
        self.backoffs += 1
        self._rate = max(self.min_rate, self._rate * self.decrease_factor)
        self._bucket.rate = self._rate
        self._bucket.capacity = max(1.0, self._rate)
        logger.warning("adaptive throttle: rate reduced to %.2f/s", self._rate)

    def on_success(self) -> None:
        self._maybe_increase()

    @property
    def rate(self) -> float:
        return self._rate
