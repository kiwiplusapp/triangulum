"""
The currency graph.

THE CENTRAL IDEA
================

Model every asset as a node and every tradable conversion as a directed edge.
An edge A -> B carries the rate ``r(A->B)``: how many units of B you receive per
unit of A, *after fees and after actually walking the book*.

A profitable cycle is one where the product of its rates exceeds 1:

    r(A->B) * r(B->C) * r(C->A) > 1

Take logarithms and this becomes a sum:

    log r(A->B) + log r(B->C) + log r(C->A) > 0

Now define the edge weight as ``w = -log r``. A profitable cycle is exactly a
cycle whose weights sum to a *negative* number -- a negative-weight cycle. And
finding negative-weight cycles is a solved problem with a textbook algorithm:
Bellman-Ford, in O(V*E).

This transformation is the whole trick. It turns "search all triangles" -- which
is O(V^3) and misses 4-cycles and 5-cycles entirely -- into a single
shortest-path relaxation that finds profitable cycles of *any* length,
simultaneously, in one pass.

WHERE THE TEXTBOOK STOPS AND REALITY BEGINS
===========================================

Every published treatment of this stops at the paragraph above. Four things it
omits are the difference between a demo and a system that does not lose money:

**1. The rate is a function of size.**
A single scalar rate is a linearization at one notional. Ask for twice the size
and you walk deeper into the book and get a worse rate. So every edge here is
built at an explicit ``reference_notional``, and a cycle that survives the
search is *re-validated* with a full depth walk at its actual size before a
single order is sent. The search uses linearized rates because it must be fast;
the decision uses exact ones because it must be right.

**2. Fees live inside the rate, not outside it.**
It is tempting to compute a gross cycle return and subtract 30 bps at the end.
That is wrong, because fees are charged in different assets at different legs
and they compound. Fee-inclusive edge rates are the only formulation that
composes correctly under multiplication.

**3. Staleness is not a detail.**
An edge computed from a book that last ticked 400ms ago is a memory. Cycles
built from stale edges are the single largest source of phantom opportunities,
and they are systematically biased toward looking profitable -- because the
reason the book stopped ticking is often that the price moved and our feed
lagged. Every edge carries its book's age, and the freshness gate is applied
during construction, not after.

**4. The graph must be cheap to rebuild.**
Books tick thousands of times per second. Rebuilding a 400-node graph from
scratch on every tick is not viable. Edges are therefore updated in place and
the search runs on a schedule, with a dirty-set optimisation that reconsiders
only the parts of the graph that actually changed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Iterator, Mapping, Sequence

from triangulum.core.decimal_math import (
    D, ONE, ZERO, bps, geometric_product, ln, safe_div,
)
from triangulum.core.types import Asset, Leg, Side, Symbol
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.orderbook import OrderBook, walk_book, walk_book_for_notional

logger = logging.getLogger(__name__)

__all__ = ["Edge", "CurrencyGraph", "EdgeQuality"]


class EdgeQuality:
    """Why an edge was excluded. Surfaced in the HUD to explain a sparse graph."""

    OK = "ok"
    STALE = "stale"
    CROSSED = "crossed"
    NO_BOOK = "no_book"
    THIN = "thin"                 # not enough depth for the reference notional
    LOT_VALUE = "lot_value"       # quantization drag exceeds budget at this size
    MIN_NOTIONAL = "min_notional"
    DISABLED = "disabled"


@dataclass(slots=True)
class Edge:
    """
    A directed conversion, priced at a reference notional.

    ``rate`` is net of fees and net of the book walk. ``weight`` is ``-ln(rate)``
    and is what the cycle search operates on.
    """

    frm: Asset
    to: Asset
    symbol: Symbol
    side: Side

    rate: Decimal = ZERO
    weight: float = math.inf

    # Diagnostics carried alongside, because a cycle's quality is the worst of
    # its edges and the planner needs to know which edge is the weak one.
    fee_bps: Decimal = ZERO
    slippage_bps: Decimal = ZERO
    spread_bps: Decimal = ZERO
    depth_notional: Decimal = ZERO
    levels_consumed: int = 0
    book_age_ns: int = 0
    book_sequence: int = 0
    quality: str = EdgeQuality.NO_BOOK
    reference_notional: Decimal = ZERO
    lot_value: Decimal = ZERO
    updated_ns: int = 0

    @property
    def venue(self) -> str:
        return self.symbol.venue

    @property
    def usable(self) -> bool:
        return self.quality == EdgeQuality.OK and self.rate > 0 and math.isfinite(self.weight)

    @property
    def edge_bps(self) -> Decimal:
        """
        How far this single conversion deviates from its own mid price.

        Always negative in a normal market: you pay the half-spread and the fee.
        Useful as a sanity check -- a positive value means a crossed book or a
        sign error.
        """
        return bps(self.rate - ONE)

    def to_leg(self) -> Leg:
        return Leg(symbol=self.symbol, side=self.side, from_asset=self.frm, to_asset=self.to)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Edge({self.frm}->{self.to} @{self.symbol.venue} "
            f"rate={self.rate:.8f} w={self.weight:+.8f} {self.quality})"
        )


class CurrencyGraph:
    """
    Directed multigraph of assets connected by tradable conversions.

    Nodes are ``Asset``. Between the same pair of assets there may be several
    edges (different venues), and the graph keeps the best one per (from, to,
    venue) triple. Cross-venue cycles are only meaningful when the strategy
    explicitly allows them, since they require holding inventory on both venues.
    """

    def __init__(
        self,
        books: BookManager,
        *,
        base_currency: str = "USDT",
        reference_notional: Decimal = D("100"),
        max_book_age_ns: int = 250_000_000,
        maker_fee_bps: Decimal = D("10"),
        taker_fee_bps: Decimal = D("10"),
        max_levels: int = 8,
        consumption_cap: Decimal = D("0.35"),
        drag_budget_bps: Decimal = D("5"),
        enforce_lot_value: bool = True,
        cycle_legs_for_budget: int = 3,
    ) -> None:
        self.books = books
        # The reference notional is denominated in ``base_currency`` -- NOT in
        # each pair's own quote asset. This distinction is the difference
        # between a working graph and one that silently rejects everything:
        # "100" means 100 USDT on BTC/USDT and 100 USDT *converted to BTC*
        # (~0.0017 BTC) on ETH/BTC. Sizing every pair at 100 units of its own
        # quote asks for $6,000,000 of depth on a BTC-quoted pair, exhausts the
        # book, and marks every such edge THIN.
        self.base_currency = base_currency
        self.reference_notional = reference_notional
        self.max_book_age_ns = max_book_age_ns
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.max_levels = max_levels
        self.consumption_cap = consumption_cap
        self.drag_budget_bps = drag_budget_bps
        self.enforce_lot_value = enforce_lot_value
        self.cycle_legs_for_budget = cycle_legs_for_budget

        self._edges: dict[tuple[str, str, str], Edge] = {}
        self._adjacency: dict[str, list[Edge]] = {}
        self._assets: dict[str, Asset] = {}
        self._fee_overrides: dict[str, tuple[Decimal, Decimal]] = {}
        self._lot_steps: dict[str, Decimal] = {}
        self._min_notionals: dict[str, Decimal] = {}
        # Value of one unit of each asset, in base_currency. Rebuilt every scan.
        self._valuation: dict[str, Decimal] = {base_currency: ONE}
        self._unvalued: set[str] = set()
        self._dirty = True

        self.rebuilds = 0
        self.excluded: dict[str, int] = {}

    # -- configuration -----------------------------------------------------

    def set_venue_fees(self, venue: str, maker_bps: Decimal, taker_bps: Decimal) -> None:
        self._fee_overrides[venue] = (maker_bps, taker_bps)

    def set_symbol_rules(
        self, symbol: Symbol, lot_step: Decimal, min_notional: Decimal
    ) -> None:
        """
        Feed the venue's real trading rules in.

        Without these the graph will happily propose a cycle through an
        instrument whose lot grid eats the entire edge -- which, at $100, is
        most of them.
        """
        self._lot_steps[symbol.key] = lot_step
        self._min_notionals[symbol.key] = min_notional

    def _fees_for(self, venue: str) -> tuple[Decimal, Decimal]:
        return self._fee_overrides.get(venue, (self.maker_fee_bps, self.taker_fee_bps))

    # -- construction ------------------------------------------------------

    def build(self, symbols: Iterable[Symbol], now_ns: int) -> None:
        """(Re)build every edge from the current books."""
        self._edges.clear()
        self._adjacency.clear()
        self.excluded = {}
        symbols = list(symbols)
        self._build_valuation(symbols)
        for symbol in symbols:
            self._build_pair(symbol, now_ns)
        self._reindex()
        self.rebuilds += 1
        self._dirty = False

    # -- valuation ---------------------------------------------------------

    def _build_valuation(self, symbols: Sequence[Symbol]) -> None:
        """
        Value every asset in ``base_currency``, so a single reference notional
        can be expressed in any pair's quote asset.

        Two passes. The first takes direct quotes against the base currency
        (BTC/USDT gives BTC). The second propagates through already-valued
        assets (ETH/BTC gives ETH once BTC is known). Two passes is enough for
        any realistic universe, where every asset is at most one hop from a
        major quote; assets still unvalued after that are recorded in
        ``_unvalued`` and their edges are excluded rather than guessed at.

        Mid prices are used deliberately. This is a *sizing* input, not a
        pricing one -- the exact executable rate comes from the book walk. Using
        the mid here keeps the reference notional stable rather than making it
        jitter with the spread.
        """
        self._valuation = {self.base_currency: ONE}

        for _ in range(2):
            for symbol in symbols:
                book = self.books.get(symbol)
                if book is None or not book.initialized or book.crossed:
                    continue
                mid = book.mid
                if mid <= 0:
                    continue
                base_code, quote_code = symbol.base.code, symbol.quote.code
                # price is quote-per-base
                if quote_code in self._valuation and base_code not in self._valuation:
                    self._valuation[base_code] = mid * self._valuation[quote_code]
                elif base_code in self._valuation and quote_code not in self._valuation:
                    self._valuation[quote_code] = safe_div(
                        self._valuation[base_code], mid
                    )

        self._unvalued = set()
        for symbol in symbols:
            for code in (symbol.base.code, symbol.quote.code):
                if code not in self._valuation:
                    self._unvalued.add(code)

    def value_of(self, asset_code: str) -> Decimal:
        """Units of ``base_currency`` per unit of ``asset_code``. 0 if unknown."""
        return self._valuation.get(asset_code, ZERO)

    def notional_in(self, quote_code: str) -> Decimal:
        """
        The reference notional, expressed in ``quote_code``.

        ``reference_notional`` is in base currency; this converts it. Returns
        ZERO when the quote asset cannot be valued, which the edge builder
        treats as "do not price this edge" rather than falling back to a wrong
        number.
        """
        if quote_code == self.base_currency:
            return self.reference_notional
        value = self._valuation.get(quote_code, ZERO)
        if value <= 0:
            return ZERO
        return safe_div(self.reference_notional, value)

    def update_symbol(self, symbol: Symbol, now_ns: int) -> None:
        """Refresh just the two edges belonging to one instrument."""
        self._build_pair(symbol, now_ns)
        self._dirty = True

    def _build_pair(self, symbol: Symbol, now_ns: int) -> None:
        """Create or refresh the forward and reverse edges for one instrument."""
        book = self.books.get(symbol)
        base, quote = symbol.base, symbol.quote
        self._assets.setdefault(base.code, base)
        self._assets.setdefault(quote.code, quote)

        # quote -> base is a BUY (spend quote, receive base)
        # base -> quote is a SELL (spend base, receive quote)
        forward = self._price_edge(quote, base, symbol, Side.BUY, book, now_ns)
        reverse = self._price_edge(base, quote, symbol, Side.SELL, book, now_ns)
        self._edges[(quote.code, base.code, symbol.venue)] = forward
        self._edges[(base.code, quote.code, symbol.venue)] = reverse

    def _price_edge(
        self,
        frm: Asset,
        to: Asset,
        symbol: Symbol,
        side: Side,
        book: OrderBook | None,
        now_ns: int,
    ) -> Edge:
        edge = Edge(frm=frm, to=to, symbol=symbol, side=side,
                    reference_notional=self.reference_notional, updated_ns=now_ns)

        if book is None or not book.initialized:
            edge.quality = EdgeQuality.NO_BOOK
            self._count_exclusion(edge.quality)
            return edge

        edge.book_age_ns = book.age_ns(now_ns)
        edge.book_sequence = book.sequence
        edge.spread_bps = book.spread_bps

        if book.crossed:
            edge.quality = EdgeQuality.CROSSED
            self._count_exclusion(edge.quality)
            return edge
        if edge.book_age_ns > self.max_book_age_ns:
            edge.quality = EdgeQuality.STALE
            self._count_exclusion(edge.quality)
            return edge

        snapshot = book.snapshot(self.max_levels)
        levels = snapshot.side(side)
        if not levels:
            edge.quality = EdgeQuality.THIN
            self._count_exclusion(edge.quality)
            return edge

        # Reference notional converted into THIS pair's quote asset.
        notional_quote = self.notional_in(symbol.quote.code)
        if notional_quote <= 0:
            edge.quality = EdgeQuality.NO_BOOK
            self._count_exclusion(edge.quality)
            return edge
        edge.reference_notional = notional_quote

        _maker_bps, taker_bps = self._fees_for(symbol.venue)
        fee_fraction = taker_bps / D(10_000)
        edge.fee_bps = taker_bps

        touch = levels[0].price
        lot_step = self._lot_steps.get(symbol.key, ZERO)
        min_notional = self._min_notionals.get(symbol.key, ZERO)

        if side is Side.BUY:
            # Spend ``reference_notional`` units of quote to receive base.
            walk = walk_book_for_notional(
                levels, notional_quote,
                max_levels=self.max_levels, consumption_cap=self.consumption_cap,
            )
            if walk.exhausted or walk.filled_quantity <= 0:
                edge.quality = EdgeQuality.THIN
                self._count_exclusion(edge.quality)
                return edge
            # The fee on a BUY is charged in the base asset received.
            received = walk.filled_quantity * (ONE - fee_fraction)
            spent = notional_quote
            edge.depth_notional = walk.filled_notional
            notional_for_rules = notional_quote
        else:
            # Spend base to receive quote. The reference notional is expressed
            # in quote, so convert it to a base quantity at the touch first.
            target_base = safe_div(notional_quote, touch)
            walk = walk_book(
                levels, target_base,
                max_levels=self.max_levels, consumption_cap=self.consumption_cap,
            )
            if walk.exhausted or walk.filled_quantity <= 0:
                edge.quality = EdgeQuality.THIN
                self._count_exclusion(edge.quality)
                return edge
            # The fee on a SELL is charged in the quote asset received.
            received = walk.filled_notional * (ONE - fee_fraction)
            spent = walk.filled_quantity
            edge.depth_notional = walk.filled_notional
            notional_for_rules = walk.filled_notional

        edge.levels_consumed = walk.levels_consumed
        edge.slippage_bps = walk.slippage_bps(touch)

        if min_notional > 0 and notional_for_rules < min_notional:
            edge.quality = EdgeQuality.MIN_NOTIONAL
            self._count_exclusion(edge.quality)
            return edge

        if lot_step > 0:
            edge.lot_value = lot_step * touch
            if self.enforce_lot_value:
                # Budget derived in exchanges.spec.max_lot_value_for_budget.
                budget = (
                    D(2) * notional_for_rules
                    * (self.drag_budget_bps / D(10_000))
                    / D(self.cycle_legs_for_budget)
                )
                if edge.lot_value > budget:
                    edge.quality = EdgeQuality.LOT_VALUE
                    self._count_exclusion(edge.quality)
                    return edge

        rate = safe_div(received, spent)
        if rate <= 0:
            edge.quality = EdgeQuality.THIN
            self._count_exclusion(edge.quality)
            return edge

        edge.rate = rate
        edge.weight = -ln(rate)
        edge.quality = EdgeQuality.OK
        return edge

    def _count_exclusion(self, reason: str) -> None:
        self.excluded[reason] = self.excluded.get(reason, 0) + 1

    def _reindex(self) -> None:
        self._adjacency = {}
        for edge in self._edges.values():
            if edge.usable:
                self._adjacency.setdefault(edge.frm.code, []).append(edge)

    def reindex_if_dirty(self) -> None:
        if self._dirty:
            self._reindex()
            self._dirty = False

    # -- access ------------------------------------------------------------

    def edges_from(self, asset: Asset | str) -> Sequence[Edge]:
        code = asset if isinstance(asset, str) else asset.code
        return self._adjacency.get(code, ())

    def edge(self, frm: str, to: str, venue: str) -> Edge | None:
        return self._edges.get((frm, to, venue))

    def best_edge(self, frm: Asset, to: Asset) -> Edge | None:
        """Highest-rate edge between two assets across all venues."""
        best: Edge | None = None
        for edge in self._adjacency.get(frm.code, ()):
            if edge.to.code != to.code:
                continue
            if best is None or edge.rate > best.rate:
                best = edge
        return best

    @property
    def nodes(self) -> Sequence[str]:
        return list(self._adjacency.keys())

    @property
    def usable_edges(self) -> list[Edge]:
        return [e for e in self._edges.values() if e.usable]

    def asset(self, code: str) -> Asset | None:
        return self._assets.get(code)

    def __len__(self) -> int:
        return sum(len(v) for v in self._adjacency.values())

    # -- validation --------------------------------------------------------

    def exact_cycle_return(self, edges: Sequence[Edge]) -> Decimal:
        """
        Exact product of a cycle's rates, in full decimal precision.

        The search works in floating-point logs for speed. Before any capital
        moves, the candidate is re-multiplied here. The two answers agree to
        many digits in normal conditions -- but "normal conditions" is doing a
        lot of work in a system that only acts on 2 bps discrepancies, and the
        exact computation costs microseconds.
        """
        return geometric_product([e.rate for e in edges])

    def cycle_edge_bps(self, edges: Sequence[Edge]) -> Decimal:
        return bps(self.exact_cycle_return(edges) - ONE)

    def verify_cycle_closes(self, edges: Sequence[Edge]) -> bool:
        if not edges:
            return False
        for a, b in zip(edges, edges[1:]):
            if a.to.code != b.frm.code:
                return False
        return edges[-1].to.code == edges[0].frm.code

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, object]:
        by_venue: dict[str, int] = {}
        for edge in self._edges.values():
            if edge.usable:
                by_venue[edge.venue] = by_venue.get(edge.venue, 0) + 1
        ages = [e.book_age_ns / 1e6 for e in self._edges.values() if e.usable]
        return {
            "nodes": len(self._adjacency),
            "edges_total": len(self._edges),
            "edges_usable": sum(1 for e in self._edges.values() if e.usable),
            "edges_by_venue": by_venue,
            "excluded": dict(self.excluded),
            "rebuilds": self.rebuilds,
            "mean_book_age_ms": round(sum(ages) / len(ages), 2) if ages else 0.0,
            "reference_notional": str(self.reference_notional),
            "base_currency": self.base_currency,
            "valued_assets": len(self._valuation),
            "unvalued_assets": sorted(self._unvalued)[:10],
        }

    def connectivity_report(self) -> dict[str, object]:
        """
        Degree distribution -- the diagnostic that tells you whether cycles can
        exist at all.

        An asset with in-degree or out-degree below 2 cannot be an interior node
        of any cycle. A graph where most nodes have degree 1 is a star, not a
        mesh, and will yield no cycles no matter how long you search. This is
        exactly the shape an equities universe has, and reporting it is more
        useful than silently returning zero opportunities forever.
        """
        out_degree = {node: len(edges) for node, edges in self._adjacency.items()}
        in_degree: dict[str, int] = {}
        for edge in self._edges.values():
            if edge.usable:
                in_degree[edge.to.code] = in_degree.get(edge.to.code, 0) + 1

        cyclable = [
            node for node in self._adjacency
            if out_degree.get(node, 0) >= 2 and in_degree.get(node, 0) >= 2
        ]
        return {
            "nodes": len(self._adjacency),
            "nodes_that_can_be_in_a_cycle": len(cyclable),
            "max_out_degree": max(out_degree.values()) if out_degree else 0,
            "mean_out_degree": (
                sum(out_degree.values()) / len(out_degree) if out_degree else 0.0
            ),
            "hubs": sorted(out_degree.items(), key=lambda kv: -kv[1])[:10],
            "is_star_topology": len(cyclable) <= 1 and len(self._adjacency) > 3,
        }
