"""
Asynchronous in-process event bus.

Design constraints that shaped this
-----------------------------------
1. **A slow subscriber must never stall the market-data path.** The dashboard
   pushing 40 KB of JSON over a congested WebSocket cannot be allowed to delay
   an order-book update by even a millisecond. So every subscriber owns a
   bounded queue and the bus drops on overflow rather than applying
   backpressure upstream.

2. **Dropping must be visible.** A silent drop is a bug that manifests three
   weeks later as "the equity curve on the dashboard doesn't match the ledger".
   Every drop is counted per-topic and per-subscriber and surfaced in
   ``/health``.

3. **Topics are hierarchical.** ``book.binance.BTC/USDT`` is matched by
   subscribers to ``book.binance.*`` and ``book.**``. Cheap wildcard matching
   keeps the dashboard's "subscribe to everything" from needing N subscriptions.

The bus is intentionally *not* durable, ordered across topics, or
cross-process. Anything that needs those properties goes through the SQLite
storage layer instead.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)

__all__ = ["Event", "EventBus", "Subscription", "Topics"]


class Topics:
    """Canonical topic names. String constants, so typos fail at import."""

    BOOK_UPDATE = "book.update"
    QUOTE_UPDATE = "quote.update"
    TRADE_PRINT = "trade.print"

    OPPORTUNITY_DETECTED = "opportunity.detected"
    OPPORTUNITY_REJECTED = "opportunity.rejected"
    CYCLE_PLANNED = "cycle.planned"
    CYCLE_STARTED = "cycle.started"
    CYCLE_LEG_FILLED = "cycle.leg_filled"
    CYCLE_COMPLETED = "cycle.completed"
    CYCLE_FAILED = "cycle.failed"
    CYCLE_UNWOUND = "cycle.unwound"

    ORDER_SUBMITTED = "order.submitted"
    ORDER_UPDATED = "order.updated"
    ORDER_FILLED = "order.filled"
    ORDER_REJECTED = "order.rejected"

    BALANCE_UPDATED = "balance.updated"
    PNL_UPDATED = "pnl.updated"
    EQUITY_SNAPSHOT = "equity.snapshot"

    RISK_LIMIT_BREACHED = "risk.limit_breached"
    RISK_CIRCUIT_OPENED = "risk.circuit_opened"
    RISK_CIRCUIT_CLOSED = "risk.circuit_closed"
    KILL_SWITCH = "risk.kill_switch"

    MODEL_UPDATED = "learning.model_updated"
    MODEL_DRIFT = "learning.drift_detected"
    BANDIT_ARM_SELECTED = "learning.arm_selected"

    VENUE_CONNECTED = "venue.connected"
    VENUE_DISCONNECTED = "venue.disconnected"
    VENUE_ERROR = "venue.error"

    ENGINE_STARTED = "engine.started"
    ENGINE_STOPPING = "engine.stopping"
    ENGINE_HEARTBEAT = "engine.heartbeat"
    LOG_RECORD = "engine.log"


@dataclass(slots=True)
class Event:
    topic: str
    payload: Any
    ts_ns: int = 0
    source: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Event({self.topic!r}, source={self.source!r})"


@dataclass(slots=True)
class Subscription:
    pattern: str
    queue: asyncio.Queue
    name: str
    dropped: int = 0
    delivered: int = 0
    _closed: bool = False

    def matches(self, topic: str) -> bool:
        if self.pattern == "**" or self.pattern == topic:
            return True
        # '**' matches across dots, '*' matches within one segment.
        if "**" in self.pattern:
            return fnmatch.fnmatchcase(topic, self.pattern.replace("**", "*"))
        if "*" in self.pattern:
            pat_parts = self.pattern.split(".")
            top_parts = topic.split(".")
            if len(pat_parts) != len(top_parts):
                return False
            return all(
                p == "*" or p == t for p, t in zip(pat_parts, top_parts)
            )
        return False

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class EventBus:
    """
    Fan-out bus with bounded per-subscriber queues.

    ``publish`` is synchronous and non-blocking by design: it is called from the
    order-book decode loop where an ``await`` would mean a context switch per
    price level.
    """

    def __init__(self, *, default_maxsize: int = 2048) -> None:
        self._subs: list[Subscription] = []
        self._default_maxsize = default_maxsize
        self._published: defaultdict[str, int] = defaultdict(int)
        self._dropped_by_topic: defaultdict[str, int] = defaultdict(int)
        self._match_cache: dict[str, tuple[Subscription, ...]] = {}
        self._closed = False

    # -- subscription ------------------------------------------------------

    def subscribe(
        self,
        pattern: str,
        *,
        name: str = "",
        maxsize: int | None = None,
    ) -> Subscription:
        sub = Subscription(
            pattern=pattern,
            queue=asyncio.Queue(maxsize=maxsize or self._default_maxsize),
            name=name or pattern,
        )
        self._subs.append(sub)
        self._match_cache.clear()
        logger.debug("bus: subscribed %s to %s", sub.name, pattern)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        sub.close()
        try:
            self._subs.remove(sub)
        except ValueError:
            pass
        self._match_cache.clear()

    def _matching(self, topic: str) -> tuple[Subscription, ...]:
        cached = self._match_cache.get(topic)
        if cached is not None:
            return cached
        matched = tuple(s for s in self._subs if s.matches(topic))
        # Bound the cache: topics include symbol names, so the key space is
        # large but finite. 4096 is far above any realistic pair count.
        if len(self._match_cache) < 4096:
            self._match_cache[topic] = matched
        return matched

    # -- publishing --------------------------------------------------------

    def publish(self, topic: str, payload: Any, *, ts_ns: int = 0, source: str = "") -> int:
        """
        Deliver to every matching subscriber. Returns the number delivered.

        Never raises, never blocks, never awaits. A full queue drops the *oldest*
        item and enqueues the new one, because for market data the freshest
        message is always the most valuable.
        """
        if self._closed:
            return 0
        self._published[topic] += 1
        event = Event(topic=topic, payload=payload, ts_ns=ts_ns, source=source)
        delivered = 0
        for sub in self._matching(topic):
            if sub.closed:
                continue
            try:
                sub.queue.put_nowait(event)
                sub.delivered += 1
                delivered += 1
            except asyncio.QueueFull:
                sub.dropped += 1
                self._dropped_by_topic[topic] += 1
                try:
                    sub.queue.get_nowait()          # evict oldest
                    sub.queue.put_nowait(event)
                    sub.delivered += 1
                    delivered += 1
                except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
                    pass
        return delivered

    async def publish_and_drain(self, topic: str, payload: Any) -> None:
        """Publish then yield, letting subscribers run. Used in tests."""
        self.publish(topic, payload)
        await asyncio.sleep(0)

    # -- consumption -------------------------------------------------------

    async def listen(self, sub: Subscription):
        """Async iterator over a subscription's events."""
        while not sub.closed:
            event = await sub.queue.get()
            yield event

    async def run_handler(
        self,
        pattern: str,
        handler: Callable[[Event], Awaitable[None]],
        *,
        name: str = "",
        maxsize: int | None = None,
    ) -> None:
        """
        Long-running task: pump matching events into ``handler``.

        Handler exceptions are logged and swallowed. A subscriber that crashes
        must not take down the bus or its siblings -- in a trading engine, the
        dashboard dying is an inconvenience and the executor dying is an
        incident, and they share this bus.
        """
        sub = self.subscribe(pattern, name=name or getattr(handler, "__name__", pattern),
                             maxsize=maxsize)
        try:
            async for event in self.listen(sub):
                try:
                    await handler(event)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("bus: handler %s failed on %s", sub.name, event.topic)
        except asyncio.CancelledError:
            pass
        finally:
            self.unsubscribe(sub)

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "subscribers": len(self._subs),
            "published_by_topic": dict(self._published),
            "dropped_by_topic": dict(self._dropped_by_topic),
            "total_published": sum(self._published.values()),
            "total_dropped": sum(self._dropped_by_topic.values()),
            "queues": [
                {
                    "name": s.name,
                    "pattern": s.pattern,
                    "depth": s.queue.qsize(),
                    "delivered": s.delivered,
                    "dropped": s.dropped,
                }
                for s in self._subs
            ],
        }

    @property
    def healthy(self) -> bool:
        """Any subscriber above 80% queue occupancy means we are falling behind."""
        for s in self._subs:
            if s.queue.maxsize and s.queue.qsize() / s.queue.maxsize > 0.8:
                return False
        return True

    def close(self) -> None:
        self._closed = True
        for sub in list(self._subs):
            sub.close()
        self._subs.clear()
        self._match_cache.clear()
