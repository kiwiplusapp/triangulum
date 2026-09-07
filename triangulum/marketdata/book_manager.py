"""
The market-data plane: owns every order book, tracks freshness, and tells the
strategy layer what it is allowed to look at.

Its most important job is *saying no*. A book is tradeable only when:

    - it has been initialized from a snapshot,
    - it is not crossed,
    - its most recent update is inside the freshness budget,
    - its venue is connected,
    - and it has not been quarantined by a checksum or sequence failure.

Every one of those conditions has, at some point, been the thing standing
between a working strategy and a strategy that spent an afternoon "arbitraging"
a frozen WebSocket.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from decimal import Decimal
from typing import Callable, Iterable, Iterator

from triangulum.core.clock import Clock, SystemClock
from triangulum.core.decimal_math import ZERO
from triangulum.core.errors import BookChecksumMismatch, SequenceGap
from triangulum.core.eventbus import EventBus, Topics
from triangulum.core.ringbuffer import RollingStats
from triangulum.core.types import Asset, BookSnapshot, Quote, Symbol
from triangulum.marketdata.orderbook import OrderBook

logger = logging.getLogger(__name__)

__all__ = ["BookManager", "BookHealth"]


class BookHealth:
    """Per-symbol data-quality tracking. Feeds both the risk layer and the HUD."""

    __slots__ = (
        "symbol", "update_intervals_ms", "gaps", "checksum_failures",
        "resnapshots", "last_update_ns", "quarantined_until_ns", "updates",
    )

    def __init__(self, symbol: Symbol) -> None:
        self.symbol = symbol
        self.update_intervals_ms = RollingStats(256)
        self.gaps = 0
        self.checksum_failures = 0
        self.resnapshots = 0
        self.last_update_ns = 0
        self.quarantined_until_ns = 0
        self.updates = 0

    def record_update(self, now_ns: int) -> None:
        if self.last_update_ns:
            self.update_intervals_ms.push((now_ns - self.last_update_ns) / 1e6)
        self.last_update_ns = now_ns
        self.updates += 1

    def quarantine(self, now_ns: int, duration_ms: float = 2000.0) -> None:
        self.quarantined_until_ns = now_ns + int(duration_ms * 1e6)

    def is_quarantined(self, now_ns: int) -> bool:
        return now_ns < self.quarantined_until_ns

    @property
    def mean_interval_ms(self) -> float:
        return self.update_intervals_ms.mean

    @property
    def tick_rate_hz(self) -> float:
        m = self.mean_interval_ms
        return 1000.0 / m if m > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return (self.gaps + self.checksum_failures) / max(1, self.updates)


class BookManager:
    """Registry of live order books keyed by ``(venue, canonical pair)``."""

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        max_depth: int = 20,
        max_age_ns: int = 250_000_000,
    ) -> None:
        self._books: dict[str, OrderBook] = {}
        self._health: dict[str, BookHealth] = {}
        self._by_venue: defaultdict[str, list[str]] = defaultdict(list)
        self._by_asset: defaultdict[str, list[str]] = defaultdict(list)
        self._connected_venues: set[str] = set()
        self._bus = bus
        self._clock = clock or SystemClock()
        self._max_depth = max_depth
        self._max_age_ns = max_age_ns
        self._resnapshot_hooks: list[Callable[[Symbol], None]] = []

    # -- registration ------------------------------------------------------

    def register(self, symbol: Symbol) -> OrderBook:
        key = symbol.key
        existing = self._books.get(key)
        if existing is not None:
            return existing
        book = OrderBook(symbol, max_depth=self._max_depth)
        self._books[key] = book
        self._health[key] = BookHealth(symbol)
        self._by_venue[symbol.venue].append(key)
        self._by_asset[symbol.base.code].append(key)
        self._by_asset[symbol.quote.code].append(key)
        return book

    def on_resnapshot_needed(self, hook: Callable[[Symbol], None]) -> None:
        """Register a callback invoked when a book must be re-snapshotted."""
        self._resnapshot_hooks.append(hook)

    # -- venue connection state -------------------------------------------

    def mark_connected(self, venue: str) -> None:
        self._connected_venues.add(venue)
        if self._bus:
            self._bus.publish(Topics.VENUE_CONNECTED, {"venue": venue})

    def mark_disconnected(self, venue: str) -> None:
        self._connected_venues.discard(venue)
        # Invalidate every book on the venue. A book from before a disconnect is
        # worthless: we do not know what we missed while the socket was down.
        for key in self._by_venue.get(venue, ()):
            self._books[key].reset()
        if self._bus:
            self._bus.publish(Topics.VENUE_DISCONNECTED, {"venue": venue})

    def is_connected(self, venue: str) -> bool:
        return venue in self._connected_venues

    # -- access ------------------------------------------------------------

    def get(self, symbol: Symbol) -> OrderBook | None:
        return self._books.get(symbol.key)

    def get_by_key(self, key: str) -> OrderBook | None:
        return self._books.get(key)

    def require(self, symbol: Symbol) -> OrderBook:
        book = self._books.get(symbol.key)
        if book is None:
            book = self.register(symbol)
        return book

    def health(self, symbol: Symbol) -> BookHealth:
        return self._health[symbol.key]

    def __iter__(self) -> Iterator[OrderBook]:
        return iter(self._books.values())

    def __len__(self) -> int:
        return len(self._books)

    def books_for_venue(self, venue: str) -> list[OrderBook]:
        return [self._books[k] for k in self._by_venue.get(venue, ())]

    def books_touching(self, asset: Asset) -> list[OrderBook]:
        return [self._books[k] for k in self._by_asset.get(asset.code, ())]

    # -- the gate ----------------------------------------------------------

    def is_tradeable(self, symbol: Symbol, now_ns: int | None = None) -> bool:
        """The single authority on whether a book may inform a trading decision."""
        now = now_ns if now_ns is not None else self._clock.wall_ns()
        book = self._books.get(symbol.key)
        if book is None or not book.initialized or book.crossed:
            return False
        if symbol.venue not in self._connected_venues:
            return False
        health = self._health[symbol.key]
        if health.is_quarantined(now):
            return False
        return book.age_ns(now) <= self._max_age_ns

    def tradeable_books(self, now_ns: int | None = None) -> list[OrderBook]:
        now = now_ns if now_ns is not None else self._clock.wall_ns()
        return [b for b in self._books.values() if self.is_tradeable(b.symbol, now)]

    def stale_symbols(self, now_ns: int | None = None) -> list[tuple[str, float]]:
        now = now_ns if now_ns is not None else self._clock.wall_ns()
        out: list[tuple[str, float]] = []
        for key, book in self._books.items():
            if book.initialized and book.age_ns(now) > self._max_age_ns:
                out.append((key, book.age_ns(now) / 1e6))
        return sorted(out, key=lambda x: -x[1])

    # -- ingestion ---------------------------------------------------------

    def apply_snapshot(
        self,
        symbol: Symbol,
        bids: list[tuple[Decimal, Decimal]],
        asks: list[tuple[Decimal, Decimal]],
        *,
        sequence: int = 0,
        ts_venue_ns: int = 0,
    ) -> OrderBook:
        now = self._clock.wall_ns()
        book = self.require(symbol)
        book.apply_snapshot(
            bids, asks, sequence=sequence, ts_venue_ns=ts_venue_ns, ts_local_ns=now
        )
        health = self._health[symbol.key]
        health.record_update(now)
        health.resnapshots += 1
        self._publish(book, now)
        return book

    def apply_delta(
        self,
        symbol: Symbol,
        bids: list[tuple[Decimal, Decimal]],
        asks: list[tuple[Decimal, Decimal]],
        *,
        sequence: int = 0,
        prev_sequence: int | None = None,
        ts_venue_ns: int = 0,
        checksum: int | None = None,
        checksum_algorithm: str = "kraken",
    ) -> OrderBook | None:
        """
        Apply an incremental update, handling integrity failures by quarantining
        the symbol and requesting a fresh snapshot.

        Returns None when the update was rejected -- callers should not read the
        book in that case.
        """
        now = self._clock.wall_ns()
        book = self.require(symbol)
        health = self._health[symbol.key]

        try:
            book.apply_delta(
                bids, asks,
                sequence=sequence,
                prev_sequence=prev_sequence,
                ts_venue_ns=ts_venue_ns,
                ts_local_ns=now,
            )
        except SequenceGap as exc:
            health.gaps += 1
            health.quarantine(now)
            logger.warning("book gap on %s: %s", symbol.key, exc)
            self._request_resnapshot(symbol)
            return None

        if checksum is not None:
            try:
                book.verify_checksum(checksum, algorithm=checksum_algorithm)
            except BookChecksumMismatch as exc:
                health.checksum_failures += 1
                health.quarantine(now)
                logger.warning("book checksum failure on %s: %s", symbol.key, exc)
                self._request_resnapshot(symbol)
                return None

        health.record_update(now)
        self._publish(book, now)
        return book

    def _request_resnapshot(self, symbol: Symbol) -> None:
        for hook in self._resnapshot_hooks:
            try:
                hook(symbol)
            except Exception:  # pragma: no cover - hook is venue code
                logger.exception("resnapshot hook failed for %s", symbol.key)

    def _publish(self, book: OrderBook, now_ns: int) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            f"{Topics.BOOK_UPDATE}.{book.symbol.venue}",
            book.symbol,
            ts_ns=now_ns,
            source=book.symbol.venue,
        )

    # -- aggregate views ---------------------------------------------------

    def quotes(self, now_ns: int | None = None) -> dict[str, Quote]:
        now = now_ns if now_ns is not None else self._clock.wall_ns()
        return {
            b.symbol.key: b.quote()
            for b in self._books.values()
            if self.is_tradeable(b.symbol, now)
        }

    def snapshots(self, depth: int = 10) -> dict[str, BookSnapshot]:
        return {k: b.snapshot(depth) for k, b in self._books.items() if b.initialized}

    def stats(self) -> dict[str, object]:
        now = self._clock.wall_ns()
        tradeable = sum(1 for b in self._books.values() if self.is_tradeable(b.symbol, now))
        total_gaps = sum(h.gaps for h in self._health.values())
        total_cs = sum(h.checksum_failures for h in self._health.values())
        rates = [h.tick_rate_hz for h in self._health.values() if h.updates > 8]
        return {
            "books": len(self._books),
            "tradeable": tradeable,
            "connected_venues": sorted(self._connected_venues),
            "sequence_gaps": total_gaps,
            "checksum_failures": total_cs,
            "quarantined": sum(1 for h in self._health.values() if h.is_quarantined(now)),
            "mean_tick_rate_hz": sum(rates) / len(rates) if rates else 0.0,
            "total_updates": sum(h.updates for h in self._health.values()),
        }
