"""
Historical replay feed.

Drives recorded market data through the identical :class:`BookManager` the live
engine uses, at a controllable speed, under a :class:`SimulatedClock`. The
strategy, the sizing, the fee engine, the risk guardrails and the learner all
run unmodified; only the clock and the exchange adapter change.

This is the property that makes a backtest worth reading. The most common way
backtests lie is that they run a *different* code path from production -- a
vectorised pandas pipeline that computes edges in a way the live event-driven
engine never could, on data with no gaps, no staleness and no partial fills.
Here, a bug in the sizing code produces the same wrong answer in both.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Sequence

from triangulum.core.clock import SimulatedClock
from triangulum.core.decimal_math import D
from triangulum.core.types import Symbol
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.feed import Feed, FeedState
from triangulum.marketdata.normalizer import SymbolNormalizer
from triangulum.marketdata.recorder import RecordType, iter_recordings

logger = logging.getLogger(__name__)

__all__ = ["ReplayFeed"]


class ReplayFeed(Feed):
    """
    Replays a recording directory.

    ``speed`` semantics:
        0.0  -- as fast as the CPU allows (default; a month replays in minutes)
        1.0  -- wall-clock real time
        60.0 -- one minute of market data per second

    The simulated clock is advanced to each record's own timestamp before the
    record is applied, so book ages, latency budgets and cooldowns all behave
    exactly as they would live.
    """

    def __init__(
        self,
        directory: str | Path,
        books: BookManager,
        normalizer: SymbolNormalizer,
        clock: SimulatedClock,
        *,
        speed: float = 0.0,
        venues: Sequence[str] = (),
        start_ns: int = 0,
        end_ns: int = 0,
        progress_every: int = 250_000,
    ) -> None:
        super().__init__(name="replay", books=books)
        self.directory = Path(directory)
        self.normalizer = normalizer
        self.clock = clock
        self.speed = speed
        self.venues = tuple(venues)
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.progress_every = progress_every
        self.records_applied = 0
        self.records_skipped = 0
        self._finished = asyncio.Event()

    async def subscribe(self, symbols: Sequence[Symbol]) -> None:
        # Replay streams whatever was recorded; the subscription set is implied
        # by the contents of the files.
        return None

    def _records(self) -> Iterator[dict[str, Any]]:
        if self.venues:
            for venue in self.venues:
                yield from iter_recordings(self.directory, stream=venue)
        else:
            yield from iter_recordings(self.directory)

    async def run(self) -> None:
        self.state = FeedState.SYNCING
        for venue in (self.venues or self._discover_venues()):
            self.books.mark_connected(venue)

        last_yield = 0
        for record in self._records():
            if self._stop.is_set():
                break
            rtype = record.get("t")
            if rtype not in (RecordType.BOOK, RecordType.QUOTE):
                continue

            ts = int(record.get("tl") or record.get("ts") or 0)
            if not ts:
                self.records_skipped += 1
                continue
            if self.start_ns and ts < self.start_ns:
                continue
            if self.end_ns and ts > self.end_ns:
                break

            # Advance simulated time to this record. Out-of-order records across
            # files are tolerated by clamping rather than raising: recordings
            # from different venues interleave imperfectly.
            if ts > self.clock.wall_ns():
                if self.speed > 0:
                    await self.clock.sleep((ts - self.clock.wall_ns()) / 1e9)
                else:
                    self.clock.set_wall_ns(ts)

            self._apply(record, rtype)
            self.records_applied += 1
            self._messages += 1

            # Yield to the event loop periodically so the strategy tasks run.
            if self.records_applied - last_yield >= 32:
                last_yield = self.records_applied
                await asyncio.sleep(0)

            if self.progress_every and self.records_applied % self.progress_every == 0:
                logger.info("replay: %d records applied", self.records_applied)

        self.state = FeedState.LIVE
        self._finished.set()
        logger.info(
            "replay complete: %d applied, %d skipped",
            self.records_applied, self.records_skipped,
        )

    def _discover_venues(self) -> list[str]:
        venues: set[str] = set()
        if self.directory.exists():
            for path in self.directory.iterdir():
                name = path.name.split("-")[0]
                if name and name != "meta":
                    venues.add(name)
        return sorted(venues)

    def _apply(self, record: dict[str, Any], rtype: str) -> None:
        pair = record.get("s", "")
        if not pair or "/" not in pair:
            self.records_skipped += 1
            return
        venue = record.get("v") or self._venue_of(record)
        base_code, _, quote_code = pair.partition("/")
        symbol = self.normalizer.canonical(venue, pair)
        if symbol is None:
            symbol = self.normalizer.register(
                venue, f"{base_code}{quote_code}", base_code, quote_code
            )

        if rtype == RecordType.BOOK:
            bids = [(D(p), D(s)) for p, s in record.get("b", [])]
            asks = [(D(p), D(s)) for p, s in record.get("a", [])]
            self.books.apply_snapshot(
                symbol, bids, asks,
                sequence=int(record.get("sq", 0)),
                ts_venue_ns=int(record.get("tv", 0)),
            )
        else:
            bid, ask = D(record.get("b", "0")), D(record.get("a", "0"))
            bs, asz = D(record.get("bs", "0")), D(record.get("as", "0"))
            if bid > 0 and ask > 0:
                self.books.apply_snapshot(symbol, [(bid, bs)], [(ask, asz)])

    def _venue_of(self, record: dict[str, Any]) -> str:
        legs = record.get("legs")
        if legs and isinstance(legs, list):
            return legs[0].get("venue", "replay")
        return self.venues[0] if self.venues else "replay"

    async def wait_finished(self) -> None:
        await self._finished.wait()

    @property
    def finished(self) -> bool:
        return self._finished.is_set()
