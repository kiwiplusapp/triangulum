"""
Single-venue triangular (and polygonal) arbitrage.

The workhorse. Prices every enumerated cycle template on the venue, screens on
gross edge and freshness, and emits the survivors as opportunities.

Two design choices worth defending:

**Enumerate once, price every scan.** The set of *possible* cycles changes when
instruments are listed or delisted -- weekly. The *value* of those cycles changes
thousands of times a second. Recomputing the topology on every scan would be
pure waste, and worse, it would make scan latency depend on graph size in a way
that spikes exactly when the market is busiest.

**Screen on gross edge, decide on expected value.** ``min_edge_bps`` here is a
cheap first filter whose only job is to keep the expensive machinery -- depth
walks, feature extraction, model inference -- off the 99.9% of cycles that are
obviously not worth it. It is deliberately *loose*. Making it tight would be
throwing away the exploration data the learner needs, and the learner is what
actually decides.

A note on what "profitable" means at this stage: the edge reported here already
includes fees and the depth walk at the reference notional. It does not include
the probability that all three legs actually fill, which is where most of the
apparent profit goes.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Sequence

from triangulum.core.clock import Clock
from triangulum.core.decimal_math import D, ZERO
from triangulum.core.types import Asset, Leg, Opportunity, new_id
from triangulum.graph.bellman_ford import find_negative_cycles
from triangulum.graph.currency_graph import CurrencyGraph
from triangulum.graph.cycle_enum import CycleEnumerator, PricedCycle
from triangulum.strategy.base import Strategy

logger = logging.getLogger(__name__)

__all__ = ["TriangularStrategy"]


class TriangularStrategy(Strategy):
    def __init__(
        self,
        graph: CurrencyGraph,
        *,
        venue: str = "",
        min_length: int = 3,
        max_length: int = 4,
        start_assets: Sequence[str] = ("USDT", "USDC", "BTC"),
        allow_cross_venue: bool = False,
        use_bellman_ford: bool = True,
        max_templates: int = 20_000,
        **kwargs,
    ) -> None:
        super().__init__(name=f"triangular:{venue or 'all'}", graph=graph, **kwargs)
        self.venue = venue
        self.min_length = min_length
        self.max_length = max_length
        self.start_assets = tuple(start_assets)
        self.use_bellman_ford = use_bellman_ford
        self.enumerator = CycleEnumerator(
            graph,
            min_length=min_length,
            max_length=max_length,
            start_assets=start_assets,
            allow_cross_venue=allow_cross_venue,
            max_templates=max_templates,
        )
        self._enumerated = False
        self.bellman_ford_extras = 0

    def enumerate(self) -> int:
        """Rebuild cycle templates. Call after the instrument universe changes."""
        templates = self.enumerator.enumerate()
        self._enumerated = True
        return len(templates)

    def scan(self, now_ns: int) -> list[Opportunity]:
        if not self.enabled:
            return []
        if not self._enumerated:
            self.enumerate()

        start = self.clock.mono_ns()
        opportunities: list[Opportunity] = []
        seen_paths: set[str] = set()

        priced = self.enumerator.price_all(
            min_edge_bps=self.min_edge_bps,
            max_book_age_ns=self.max_book_age_ns,
            limit=self.max_per_scan * 4,
        )

        for cycle in priced:
            if len(opportunities) >= self.max_per_scan:
                break
            opportunity = self._to_opportunity(cycle, now_ns)
            if opportunity is None:
                continue
            seen_paths.add(cycle.path)
            opportunities.append(opportunity)

        # Bellman-Ford as a second opinion. It searches unbounded cycle lengths
        # and different relaxation orders, so it occasionally surfaces a cycle
        # the template enumeration missed -- most often one that crosses a pair
        # listed after the last enumeration.
        if self.use_bellman_ford and len(opportunities) < self.max_per_scan:
            for cycle in find_negative_cycles(
                self.graph,
                min_length=self.min_length,
                max_length=self.max_length,
                start_assets=self.start_assets,
                min_edge_bps=float(self.min_edge_bps),
                max_cycles=self.max_per_scan,
            ):
                if cycle.path in seen_paths:
                    continue
                if len(opportunities) >= self.max_per_scan:
                    break
                if cycle.max_book_age_ns > self.max_book_age_ns:
                    self.stats.rejected_stale += 1
                    continue
                key = _cycle_key(cycle.path, cycle.venues)
                if self._on_cooldown(key, now_ns):
                    continue
                legs = tuple(e.to_leg() for e in cycle.edges)
                start_asset = cycle.edges[0].frm
                self._mark_fired(key, now_ns)
                self.bellman_ford_extras += 1
                opportunities.append(Opportunity(
                    opportunity_id=new_id("opp-"),
                    legs=legs,
                    start_asset=start_asset,
                    gross_edge_bps=D(str(round(cycle.edge_bps, 6))),
                    reference_notional=self.graph.reference_notional,
                    ts_detected_ns=now_ns,
                    book_ages_ns=tuple(e.book_age_ns for e in cycle.edges),
                    venues=cycle.venues,
                ))

        self._record(opportunities, (self.clock.mono_ns() - start) / 1000.0)
        return opportunities

    def _to_opportunity(self, cycle: PricedCycle, now_ns: int) -> Opportunity | None:
        if cycle.gross_edge_bps < self.min_edge_bps:
            self.stats.rejected_edge += 1
            return None
        if cycle.max_book_age_ns > self.max_book_age_ns:
            self.stats.rejected_stale += 1
            return None

        key = _cycle_key(cycle.path, cycle.venues)
        if self._on_cooldown(key, now_ns):
            return None

        start_asset = cycle.edges[0].frm
        if self.start_assets and start_asset.code not in self.start_assets:
            return None

        self._mark_fired(key, now_ns)
        return Opportunity(
            opportunity_id=new_id("opp-"),
            legs=cycle.to_legs(),
            start_asset=start_asset,
            gross_edge_bps=cycle.gross_edge_bps,
            reference_notional=self.graph.reference_notional,
            ts_detected_ns=now_ns,
            book_ages_ns=tuple(e.book_age_ns for e in cycle.edges),
            venues=cycle.venues,
        )

    def stats_dict(self) -> dict[str, object]:
        base = self.stats.to_dict()
        base.update({
            "templates": len(self.enumerator),
            "enumerator": self.enumerator.stats(),
            "bellman_ford_extras": self.bellman_ford_extras,
        })
        return base


def _cycle_key(path: str, venues: Sequence[str]) -> str:
    return f"{path}@{','.join(venues)}"
