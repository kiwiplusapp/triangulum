"""
Mutable L2 order book with incremental updates.

This is the single most correctness-critical data structure in the engine.
Everything downstream -- edge calculation, sizing, slippage estimation -- is a
function of the book, and a book that is silently wrong produces a strategy that
confidently loses money. The defensive machinery here is not paranoia; every
guard corresponds to a failure mode that real venues exhibit:

**Sequence gaps.** Venues number their diffs. A missing number means we lost a
message and the book is now a lie. The only correct response is to discard and
resnapshot -- patching over the gap is how you end up quoting off a level that
was removed 200ms ago.

**Checksums.** Kraken and OKX publish a CRC32 over the top levels. When it
disagrees with ours, we have a bug or a lost message, and we cannot tell which.
Resnapshot either way.

**Crossed books.** ``best_bid >= best_ask`` on a single venue is almost always a
decoding bug (wrong side, wrong scale, stale level). It is *also* exactly what a
naive triangular detector sees as a giant free-money opportunity, which is why
this class refuses to serve a crossed book to the strategy layer.

**Zero-size deletes.** Every venue signals "remove this level" as a size-zero
update, but they disagree about whether the price then disappears from the
snapshot. Handled uniformly here.

Implementation note: levels live in a dict keyed by price plus a lazily-sorted
cache of the keys. Rebuilding a sorted list on every delta would dominate the
CPU profile at 5k updates/sec; instead the sort is invalidated on write and
recomputed at most once per read burst.
"""

from __future__ import annotations

import zlib
from decimal import Decimal
from typing import Iterable, Iterator, Sequence

from triangulum.core.decimal_math import D, ZERO, bps, safe_div
from triangulum.core.errors import BookChecksumMismatch, CrossedBook, SequenceGap
from triangulum.core.types import BookSnapshot, PriceLevel, Quote, Side, Symbol

__all__ = ["OrderBook", "BookSide", "WalkResult", "walk_book"]


class BookSide:
    """
    One side of the book.

    ``descending`` is True for bids (best = highest) and False for asks
    (best = lowest). Keeping the flag rather than subclassing keeps the hot
    methods monomorphic, which matters more than it looks like it should.
    """

    __slots__ = ("_levels", "_sorted", "_dirty", "descending")

    def __init__(self, descending: bool) -> None:
        self._levels: dict[Decimal, Decimal] = {}
        self._sorted: list[Decimal] = []
        self._dirty = True
        self.descending = descending

    def set(self, price: Decimal, size: Decimal) -> None:
        if size <= 0:
            if self._levels.pop(price, None) is not None:
                self._dirty = True
            return
        prev = self._levels.get(price)
        if prev is None:
            self._dirty = True
        self._levels[price] = size

    def clear(self) -> None:
        self._levels.clear()
        self._sorted.clear()
        self._dirty = False

    def replace(self, levels: Iterable[tuple[Decimal, Decimal]]) -> None:
        self._levels = {p: s for p, s in levels if s > 0}
        self._dirty = True

    def _ensure_sorted(self) -> None:
        if self._dirty:
            self._sorted = sorted(self._levels.keys(), reverse=self.descending)
            self._dirty = False

    @property
    def prices(self) -> list[Decimal]:
        self._ensure_sorted()
        return self._sorted

    def size_at(self, price: Decimal) -> Decimal:
        return self._levels.get(price, ZERO)

    def best(self) -> Decimal:
        self._ensure_sorted()
        return self._sorted[0] if self._sorted else ZERO

    def best_size(self) -> Decimal:
        b = self.best()
        return self._levels.get(b, ZERO) if b else ZERO

    def top(self, n: int) -> list[PriceLevel]:
        self._ensure_sorted()
        return [PriceLevel(p, self._levels[p]) for p in self._sorted[:n]]

    def truncate(self, max_levels: int) -> None:
        """
        Drop levels beyond ``max_levels``.

        Venues that stream unbounded depth will otherwise grow this dict without
        limit across a long session. We only ever walk 8 levels deep, so keeping
        20 is generous.
        """
        self._ensure_sorted()
        if len(self._sorted) <= max_levels:
            return
        for price in self._sorted[max_levels:]:
            self._levels.pop(price, None)
        self._sorted = self._sorted[:max_levels]

    def __len__(self) -> int:
        return len(self._levels)

    def __iter__(self) -> Iterator[PriceLevel]:
        self._ensure_sorted()
        for p in self._sorted:
            yield PriceLevel(p, self._levels[p])

    def total_size(self, levels: int = 0) -> Decimal:
        if levels <= 0:
            return sum(self._levels.values(), ZERO)
        return sum((lvl.size for lvl in self.top(levels)), ZERO)

    def total_notional(self, levels: int = 0) -> Decimal:
        src = list(self) if levels <= 0 else self.top(levels)
        return sum((lvl.price * lvl.size for lvl in src), ZERO)


class OrderBook:
    """
    Live L2 book for one symbol on one venue.

    Thread-confined: one book is owned by exactly one asyncio task (the venue's
    decode loop). Readers take immutable snapshots.
    """

    __slots__ = (
        "symbol", "bids", "asks", "sequence", "ts_venue_ns", "ts_local_ns",
        "max_depth", "_snapshot_count", "_update_count", "_gap_count",
        "_checksum_failures", "_crossed_count", "_initialized", "_last_gap_seq",
    )

    def __init__(self, symbol: Symbol, *, max_depth: int = 20) -> None:
        self.symbol = symbol
        self.bids = BookSide(descending=True)
        self.asks = BookSide(descending=False)
        self.sequence = 0
        self.ts_venue_ns = 0
        self.ts_local_ns = 0
        self.max_depth = max_depth
        self._snapshot_count = 0
        self._update_count = 0
        self._gap_count = 0
        self._checksum_failures = 0
        self._crossed_count = 0
        self._initialized = False
        self._last_gap_seq = 0

    # -- state -------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized and bool(self.bids) and bool(self.asks)

    @property
    def best_bid(self) -> Decimal:
        return self.bids.best()

    @property
    def best_ask(self) -> Decimal:
        return self.asks.best()

    @property
    def mid(self) -> Decimal:
        b, a = self.best_bid, self.best_ask
        if b <= 0 or a <= 0:
            return ZERO
        return (b + a) / D(2)

    @property
    def spread_bps(self) -> Decimal:
        m = self.mid
        if m <= 0:
            return ZERO
        return bps(safe_div(self.best_ask - self.best_bid, m))

    @property
    def crossed(self) -> bool:
        b, a = self.best_bid, self.best_ask
        return b > 0 and a > 0 and b >= a

    def age_ns(self, now_ns: int) -> int:
        return max(0, now_ns - self.ts_local_ns)

    def is_fresh(self, now_ns: int, max_age_ns: int) -> bool:
        return self.initialized and self.age_ns(now_ns) <= max_age_ns

    # -- mutation ----------------------------------------------------------

    def apply_snapshot(
        self,
        bids: Sequence[tuple[Decimal, Decimal]],
        asks: Sequence[tuple[Decimal, Decimal]],
        *,
        sequence: int = 0,
        ts_venue_ns: int = 0,
        ts_local_ns: int = 0,
    ) -> None:
        self.bids.replace(bids)
        self.asks.replace(asks)
        self.bids.truncate(self.max_depth)
        self.asks.truncate(self.max_depth)
        self.sequence = sequence
        self.ts_venue_ns = ts_venue_ns
        self.ts_local_ns = ts_local_ns
        self._snapshot_count += 1
        self._initialized = True

    def apply_delta(
        self,
        bids: Sequence[tuple[Decimal, Decimal]],
        asks: Sequence[tuple[Decimal, Decimal]],
        *,
        sequence: int = 0,
        prev_sequence: int | None = None,
        ts_venue_ns: int = 0,
        ts_local_ns: int = 0,
        strict_sequence: bool = True,
    ) -> None:
        """
        Apply an incremental update.

        ``prev_sequence`` is the venue's "this diff applies on top of sequence
        N" field where available (OKX, Bybit). When absent we fall back to
        requiring monotonic increase, which catches reordering but not loss.
        """
        if strict_sequence and self._initialized and sequence:
            expected = prev_sequence if prev_sequence is not None else self.sequence
            if prev_sequence is not None:
                if expected != self.sequence:
                    self._gap_count += 1
                    self._last_gap_seq = sequence
                    self._initialized = False
                    raise SequenceGap(
                        "order book sequence gap",
                        symbol=self.symbol.key,
                        have=self.sequence,
                        expected=expected,
                    )
            elif sequence <= self.sequence:
                # Duplicate or out-of-order replay: ignore, do not corrupt.
                return

        for price, size in bids:
            self.bids.set(price, size)
        for price, size in asks:
            self.asks.set(price, size)

        # Trim only occasionally; truncation forces a sort.
        self._update_count += 1
        if self._update_count % 64 == 0:
            self.bids.truncate(self.max_depth)
            self.asks.truncate(self.max_depth)

        if sequence:
            self.sequence = sequence
        self.ts_venue_ns = ts_venue_ns or self.ts_venue_ns
        self.ts_local_ns = ts_local_ns or self.ts_local_ns
        self._initialized = True

        if self.crossed:
            self._crossed_count += 1
            self._resolve_cross()

    def _resolve_cross(self) -> None:
        """
        Remove levels that cross.

        A genuine cross cannot persist; it is either a stale level we failed to
        delete or a message we applied out of order. Peeling the crossed levels
        off the top restores a consistent book. If the cross is deep (more than
        a handful of levels) the book is too damaged to trust and we
        de-initialize it, forcing a resnapshot.
        """
        peeled = 0
        while self.crossed and peeled < 5:
            bid, ask = self.best_bid, self.best_ask
            bid_ts_size = self.bids.size_at(bid)
            ask_ts_size = self.asks.size_at(ask)
            # Drop whichever side has less resting size: the smaller level is
            # more likely to be the stale remnant.
            if bid_ts_size <= ask_ts_size:
                self.bids.set(bid, ZERO)
            else:
                self.asks.set(ask, ZERO)
            peeled += 1
        if self.crossed:
            self._initialized = False

    def verify_checksum(self, expected: int, *, algorithm: str = "kraken") -> None:
        actual = self.compute_checksum(algorithm)
        if actual != expected:
            self._checksum_failures += 1
            self._initialized = False
            raise BookChecksumMismatch(
                "order book checksum mismatch",
                symbol=self.symbol.key,
                expected=expected,
                actual=actual,
                algorithm=algorithm,
            )

    def compute_checksum(self, algorithm: str = "kraken") -> int:
        """
        CRC32 over the top-of-book, in the venue's own encoding.

        Kraken: top 10 of each side, price and size with the decimal point and
        leading zeros stripped, asks first, concatenated.
        OKX: top 25, alternating bid/ask, ``price:size`` joined by ``:``.
        """
        if algorithm == "kraken":
            parts: list[str] = []
            for lvl in self.asks.top(10):
                parts.append(_kraken_num(lvl.price))
                parts.append(_kraken_num(lvl.size))
            for lvl in self.bids.top(10):
                parts.append(_kraken_num(lvl.price))
                parts.append(_kraken_num(lvl.size))
            return zlib.crc32("".join(parts).encode("ascii"))
        if algorithm == "okx":
            bids = self.bids.top(25)
            asks = self.asks.top(25)
            parts = []
            for i in range(25):
                if i < len(bids):
                    parts.append(f"{_plain(bids[i].price)}:{_plain(bids[i].size)}")
                if i < len(asks):
                    parts.append(f"{_plain(asks[i].price)}:{_plain(asks[i].size)}")
            crc = zlib.crc32(":".join(parts).encode("ascii"))
            # OKX publishes a signed 32-bit integer.
            return crc - 2**32 if crc >= 2**31 else crc
        raise ValueError(f"unknown checksum algorithm {algorithm!r}")

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.sequence = 0
        self._initialized = False

    # -- reading -----------------------------------------------------------

    def snapshot(self, depth: int | None = None) -> BookSnapshot:
        n = depth or self.max_depth
        return BookSnapshot(
            symbol=self.symbol,
            bids=tuple(self.bids.top(n)),
            asks=tuple(self.asks.top(n)),
            sequence=self.sequence,
            ts_venue_ns=self.ts_venue_ns,
            ts_local_ns=self.ts_local_ns,
        )

    def quote(self) -> Quote:
        return Quote(
            symbol=self.symbol,
            bid=self.best_bid,
            ask=self.best_ask,
            bid_size=self.bids.best_size(),
            ask_size=self.asks.best_size(),
            ts_venue_ns=self.ts_venue_ns,
            ts_local_ns=self.ts_local_ns,
        )

    def require_sane(self) -> None:
        """Raise if the book must not be traded on. Called before every plan."""
        if not self.initialized:
            raise CrossedBook("book not initialized", symbol=self.symbol.key)
        if self.crossed:
            raise CrossedBook(
                "book is crossed",
                symbol=self.symbol.key,
                bid=str(self.best_bid),
                ask=str(self.best_ask),
            )

    def stats(self) -> dict[str, object]:
        return {
            "symbol": self.symbol.key,
            "sequence": self.sequence,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "snapshots": self._snapshot_count,
            "updates": self._update_count,
            "gaps": self._gap_count,
            "checksum_failures": self._checksum_failures,
            "crossed_events": self._crossed_count,
            "initialized": self._initialized,
        }


def _kraken_num(value: Decimal) -> str:
    """Kraken checksum encoding: strip the decimal point and leading zeros."""
    s = format(value.normalize(), "f")
    s = s.replace(".", "").lstrip("0")
    return s or "0"


def _plain(value: Decimal) -> str:
    return format(value.normalize(), "f")


# --------------------------------------------------------------------------
# Book walking -- the core of realistic sizing
# --------------------------------------------------------------------------


class WalkResult:
    """
    Outcome of consuming liquidity across price levels.

    ``filled_quantity`` may be less than requested when the book is thinner than
    the order. That partial-fill case is not an edge case in arbitrage: it is
    the normal outcome of asking for size at the touch, and a sizing model that
    assumes full fill at the best price systematically overstates every edge it
    reports.
    """

    __slots__ = (
        "filled_quantity", "filled_notional", "average_price", "levels_consumed",
        "exhausted", "worst_price", "requested_quantity",
    )

    def __init__(
        self,
        filled_quantity: Decimal,
        filled_notional: Decimal,
        average_price: Decimal,
        levels_consumed: int,
        exhausted: bool,
        worst_price: Decimal,
        requested_quantity: Decimal,
    ) -> None:
        self.filled_quantity = filled_quantity
        self.filled_notional = filled_notional
        self.average_price = average_price
        self.levels_consumed = levels_consumed
        self.exhausted = exhausted
        self.worst_price = worst_price
        self.requested_quantity = requested_quantity

    @property
    def complete(self) -> bool:
        return self.filled_quantity >= self.requested_quantity

    @property
    def fill_ratio(self) -> Decimal:
        return safe_div(self.filled_quantity, self.requested_quantity)

    def slippage_bps(self, reference_price: Decimal) -> Decimal:
        """
        Cost of walking the book versus filling everything at ``reference_price``
        (normally the touch). Always reported as a positive cost.
        """
        if reference_price <= 0 or self.filled_quantity <= 0:
            return ZERO
        return abs(bps(safe_div(self.average_price - reference_price, reference_price)))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"WalkResult(qty={self.filled_quantity}, avg={self.average_price}, "
            f"levels={self.levels_consumed}, exhausted={self.exhausted})"
        )


def walk_book(
    levels: Sequence[PriceLevel],
    target_quantity: Decimal,
    *,
    max_levels: int = 8,
    consumption_cap: Decimal | None = None,
) -> WalkResult:
    """
    Consume ``target_quantity`` of base units across ``levels``.

    ``consumption_cap`` limits how much of each level's resting size we are
    willing to take (default: all of it). Capping at, say, 35% models the fact
    that we are not alone in the queue and that sweeping a whole level is both
    visible and adversely selected.
    """
    if target_quantity <= 0 or not levels:
        return WalkResult(ZERO, ZERO, ZERO, 0, True, ZERO, target_quantity)

    remaining = target_quantity
    filled = ZERO
    notional = ZERO
    consumed = 0
    worst = ZERO

    for level in levels[:max_levels]:
        if remaining <= 0:
            break
        available = level.size
        if consumption_cap is not None:
            available = available * consumption_cap
        take = min(remaining, available)
        if take <= 0:
            continue
        filled += take
        notional += take * level.price
        remaining -= take
        worst = level.price
        consumed += 1

    avg = safe_div(notional, filled) if filled > 0 else ZERO
    return WalkResult(
        filled_quantity=filled,
        filled_notional=notional,
        average_price=avg,
        levels_consumed=consumed,
        exhausted=remaining > 0,
        worst_price=worst,
        requested_quantity=target_quantity,
    )


def walk_book_for_notional(
    levels: Sequence[PriceLevel],
    target_notional: Decimal,
    *,
    max_levels: int = 8,
    consumption_cap: Decimal | None = None,
) -> WalkResult:
    """
    Spend ``target_notional`` of quote currency across ``levels``.

    The mirror of :func:`walk_book`, needed because a BUY leg is naturally
    specified in quote units ("spend 33 USDT") while a SELL leg is specified in
    base units ("sell 0.0005 BTC"). Getting this asymmetry wrong is the second
    most common triangular-arbitrage bug after fee direction.
    """
    if target_notional <= 0 or not levels:
        return WalkResult(ZERO, ZERO, ZERO, 0, True, ZERO, ZERO)

    remaining_notional = target_notional
    filled = ZERO
    notional = ZERO
    consumed = 0
    worst = ZERO

    for level in levels[:max_levels]:
        if remaining_notional <= 0:
            break
        available = level.size
        if consumption_cap is not None:
            available = available * consumption_cap
        level_notional = available * level.price
        if level_notional <= remaining_notional:
            take = available
            spend = level_notional
        else:
            spend = remaining_notional
            take = safe_div(spend, level.price)
        if take <= 0:
            continue
        filled += take
        notional += spend
        remaining_notional -= spend
        worst = level.price
        consumed += 1

    avg = safe_div(notional, filled) if filled > 0 else ZERO
    return WalkResult(
        filled_quantity=filled,
        filled_notional=notional,
        average_price=avg,
        levels_consumed=consumed,
        exhausted=remaining_notional > 0,
        worst_price=worst,
        requested_quantity=safe_div(target_notional, avg) if avg > 0 else ZERO,
    )
