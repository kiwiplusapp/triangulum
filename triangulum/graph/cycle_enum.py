"""
Precomputed cycle enumeration.

Bellman-Ford is the right general tool, but for the specific case of 3- and
4-cycles on a graph whose *topology* barely changes -- pairs are listed and
delisted on a timescale of weeks, while prices change thousands of times a
second -- enumerating the cycles once and re-pricing them on every scan is
dramatically faster and, more importantly, *complete*.

Bellman-Ford finds negative cycles but not necessarily all of them, and which
ones it finds depends on relaxation order. When the whole point is to rank every
opportunity by expected value and pick the best, "some negative cycles" is not
the same as "all of them". So the engine runs both:

    enumerate once  ->  price all triangles every scan   (complete, ranked)
    Bellman-Ford    ->  catch longer cycles + validate    (general, unbounded)

The enumeration is O(V * d^2) where d is the mean out-degree, done once at
startup and on instrument-list changes. Pricing is then a flat loop over a
precomputed list with no graph traversal at all, which vectorises well and has
completely predictable cost -- valuable in a latency-sensitive loop where a
tail-latency spike costs an opportunity.

Deduplication matters more than it looks. The cycle A->B->C->A is the same trade
as B->C->A->B, and its *reverse* C->B->A->C is a genuinely different trade (it
uses the other side of every book, and its profitability is uncorrelated). So
rotations are deduplicated and reversals are not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterator, Sequence

from triangulum.core.decimal_math import D, ONE, ZERO, bps, geometric_product
from triangulum.core.types import Asset, Leg, Symbol
from triangulum.graph.currency_graph import CurrencyGraph, Edge

logger = logging.getLogger(__name__)

__all__ = ["CycleTemplate", "CycleEnumerator", "PricedCycle"]


@dataclass(frozen=True, slots=True)
class CycleTemplate:
    """
    A cycle's *topology*: which assets, in which order, on which venues.

    Prices are absent by design. The template is stable across thousands of
    price updates; only its valuation changes.
    """

    assets: tuple[str, ...]          # ("USDT", "BTC", "ETH") -- closes implicitly
    venues: tuple[str, ...]          # venue per leg
    key: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            object.__setattr__(
                self, "key",
                "|".join(f"{a}@{v}" for a, v in zip(self.assets, self.venues)),
            )

    @property
    def length(self) -> int:
        return len(self.assets)

    @property
    def is_cross_venue(self) -> bool:
        return len(set(self.venues)) > 1

    def legs(self) -> Iterator[tuple[str, str, str]]:
        """Yield ``(from_code, to_code, venue)`` for each leg."""
        n = len(self.assets)
        for i in range(n):
            yield self.assets[i], self.assets[(i + 1) % n], self.venues[i]

    @property
    def path(self) -> str:
        return " -> ".join(self.assets + (self.assets[0],))


@dataclass(slots=True)
class PricedCycle:
    """A template valued against the current graph."""

    template: CycleTemplate
    edges: tuple[Edge, ...]
    gross_return: Decimal
    gross_edge_bps: Decimal
    max_book_age_ns: int
    min_depth_notional: Decimal
    total_fee_bps: Decimal
    total_slippage_bps: Decimal

    @property
    def profitable(self) -> bool:
        return self.gross_return > ONE

    @property
    def path(self) -> str:
        return self.template.path

    @property
    def venues(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.template.venues))

    def to_legs(self) -> tuple[Leg, ...]:
        return tuple(e.to_leg() for e in self.edges)


class CycleEnumerator:
    """
    Enumerates and re-prices cycle templates.

    Call :meth:`enumerate` when the instrument universe changes, then
    :meth:`price_all` on every scan.
    """

    def __init__(
        self,
        graph: CurrencyGraph,
        *,
        min_length: int = 3,
        max_length: int = 4,
        start_assets: Sequence[str] = (),
        allow_cross_venue: bool = False,
        max_templates: int = 20_000,
    ) -> None:
        self.graph = graph
        self.min_length = min_length
        self.max_length = max_length
        self.start_assets = tuple(start_assets)
        self.allow_cross_venue = allow_cross_venue
        self.max_templates = max_templates
        self._templates: list[CycleTemplate] = []
        self._by_asset: dict[str, list[int]] = {}
        self.enumerations = 0
        self.truncated = False

    # -- enumeration -------------------------------------------------------

    def enumerate(self) -> list[CycleTemplate]:
        """
        Depth-first walk from each permitted start asset.

        Complexity is bounded by ``max_templates``. On Binance's full universe
        the raw 3-cycle count from USDT alone runs to several thousand, which is
        entirely manageable; allowing 4-cycles across the full universe explodes
        into the millions, which is why the cap exists and why the default
        start-asset list is short.
        """
        self.graph.reindex_if_dirty()
        templates: list[CycleTemplate] = []
        seen: set[frozenset] = set()

        starts = self.start_assets or tuple(self.graph.nodes)

        for start in starts:
            if start not in set(self.graph.nodes):
                continue
            self._walk(start, start, [], [], templates, seen)
            if len(templates) >= self.max_templates:
                self.truncated = True
                break

        self._templates = templates
        self._reindex()
        self.enumerations += 1
        logger.info(
            "enumerated %d cycle templates (lengths %d-%d, cross_venue=%s)%s",
            len(templates), self.min_length, self.max_length,
            self.allow_cross_venue, " [TRUNCATED]" if self.truncated else "",
        )
        return templates

    def _walk(
        self,
        start: str,
        current: str,
        asset_path: list[str],
        venue_path: list[str],
        out: list[CycleTemplate],
        seen: set[frozenset],
    ) -> None:
        if len(out) >= self.max_templates:
            return

        path = asset_path + [current]

        if len(path) > self.max_length:
            return

        for edge in self.graph.edges_from(current):
            target = edge.to.code

            if not self.allow_cross_venue and venue_path and edge.venue != venue_path[0]:
                continue

            if target == start:
                if len(path) < self.min_length:
                    continue
                venues = tuple(venue_path + [edge.venue])
                assets = tuple(path)
                # Deduplicate rotations: canonicalise on the sorted leg set.
                signature = frozenset(
                    (assets[i], assets[(i + 1) % len(assets)], venues[i])
                    for i in range(len(assets))
                )
                if signature in seen:
                    continue
                seen.add(signature)
                out.append(CycleTemplate(assets=assets, venues=venues))
                if len(out) >= self.max_templates:
                    return
                continue

            if target in path:
                continue        # no revisiting: that would be two cycles glued
            if len(path) >= self.max_length:
                continue

            self._walk(start, target, path, venue_path + [edge.venue], out, seen)

    def _reindex(self) -> None:
        self._by_asset = {}
        for index, template in enumerate(self._templates):
            for asset in template.assets:
                self._by_asset.setdefault(asset, []).append(index)

    # -- pricing -----------------------------------------------------------

    def price(self, template: CycleTemplate) -> PricedCycle | None:
        """Value one template. Returns None if any leg is unusable."""
        edges: list[Edge] = []
        for frm, to, venue in template.legs():
            edge = self.graph.edge(frm, to, venue)
            if edge is None or not edge.usable:
                return None
            edges.append(edge)

        gross = geometric_product([e.rate for e in edges])
        return PricedCycle(
            template=template,
            edges=tuple(edges),
            gross_return=gross,
            gross_edge_bps=bps(gross - ONE),
            max_book_age_ns=max(e.book_age_ns for e in edges),
            min_depth_notional=min(e.depth_notional for e in edges),
            total_fee_bps=sum((e.fee_bps for e in edges), ZERO),
            total_slippage_bps=sum((e.slippage_bps for e in edges), ZERO),
        )

    def price_all(
        self,
        *,
        min_edge_bps: Decimal = ZERO,
        max_book_age_ns: int = 0,
        limit: int = 0,
    ) -> list[PricedCycle]:
        """
        Value every template, keeping those that clear the screens.

        Returned sorted by gross edge descending. Note that this is a *screen*,
        not a decision: gross edge ignores the size we can actually get, the
        probability of filling, and the cost of a failed cycle. The EV gate in
        ``triangulum.learning`` makes the decision.
        """
        results: list[PricedCycle] = []
        for template in self._templates:
            priced = self.price(template)
            if priced is None:
                continue
            if priced.gross_edge_bps < min_edge_bps:
                continue
            if max_book_age_ns and priced.max_book_age_ns > max_book_age_ns:
                continue
            results.append(priced)

        results.sort(key=lambda c: -c.gross_edge_bps)
        return results[:limit] if limit else results

    def templates_touching(self, asset: str) -> list[CycleTemplate]:
        return [self._templates[i] for i in self._by_asset.get(asset, ())]

    # -- introspection -----------------------------------------------------

    @property
    def templates(self) -> Sequence[CycleTemplate]:
        return self._templates

    def __len__(self) -> int:
        return len(self._templates)

    def stats(self) -> dict[str, object]:
        by_length: dict[int, int] = {}
        cross = 0
        for t in self._templates:
            by_length[t.length] = by_length.get(t.length, 0) + 1
            if t.is_cross_venue:
                cross += 1
        return {
            "templates": len(self._templates),
            "by_length": by_length,
            "cross_venue": cross,
            "truncated": self.truncated,
            "enumerations": self.enumerations,
            "start_assets": list(self.start_assets),
        }
