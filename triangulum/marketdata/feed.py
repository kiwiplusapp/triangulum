"""
Market-data feed abstraction.

A feed is anything that pushes book updates into a :class:`BookManager`. Live
venue WebSockets are feeds; so is the historical replayer. Keeping them behind
one interface is what lets the backtester exercise the *actual* strategy and
execution code rather than a parallel implementation that drifts out of sync --
the classic reason backtests flatter reality.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import random
from typing import AsyncIterator, Sequence

from triangulum.core.constants import (
    MAX_RECONNECT_BACKOFF_SEC,
    RECONNECT_BACKOFF_BASE_SEC,
)
from triangulum.core.types import Symbol
from triangulum.marketdata.book_manager import BookManager

logger = logging.getLogger(__name__)

__all__ = ["Feed", "FeedState", "reconnect_backoff"]


class FeedState:
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    SYNCING = "syncing"       # connected, replaying snapshots
    LIVE = "live"
    FAILED = "failed"


def reconnect_backoff(attempt: int, *, jitter: bool = True) -> float:
    """
    Exponential backoff with full jitter.

    Full jitter rather than the more common "exponential plus a small random
    term" because when a venue drops every client at once, correlated retries
    are what keeps it down. Randomizing across the whole interval decorrelates
    the herd.
    """
    ceiling = min(MAX_RECONNECT_BACKOFF_SEC, RECONNECT_BACKOFF_BASE_SEC * (2 ** attempt))
    return random.uniform(0, ceiling) if jitter else ceiling


class Feed(abc.ABC):
    """Base class for market-data sources."""

    def __init__(self, name: str, books: BookManager) -> None:
        self.name = name
        self.books = books
        self.state = FeedState.DISCONNECTED
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._reconnects = 0
        self._messages = 0

    @abc.abstractmethod
    async def subscribe(self, symbols: Sequence[Symbol]) -> None:
        """Begin streaming the given symbols."""

    @abc.abstractmethod
    async def run(self) -> None:
        """Main loop. Should return only when stopped or unrecoverably failed."""

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._supervise(), name=f"feed-{self.name}")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.state = FeedState.DISCONNECTED
        self.books.mark_disconnected(self.name)

    async def _supervise(self) -> None:
        """Restart ``run`` forever with backoff. Only cancellation stops us."""
        attempt = 0
        while not self._stop.is_set():
            try:
                self.state = FeedState.CONNECTING
                await self.run()
                attempt = 0          # a clean return resets the backoff
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                self._reconnects += 1
                delay = reconnect_backoff(attempt)
                logger.warning(
                    "feed %s failed (attempt %d): %s -- reconnecting in %.1fs",
                    self.name, attempt, exc, delay,
                )
                self.state = FeedState.DISCONNECTED
                self.books.mark_disconnected(self.name)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    continue
            if self._stop.is_set():
                return
            await asyncio.sleep(reconnect_backoff(1))

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def stats(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state,
            "reconnects": self._reconnects,
            "messages": self._messages,
            "running": self.running,
        }
