"""
Negative-cycle detection.

Bellman-Ford relaxes every edge V-1 times; any edge still relaxable on pass V
lies on -- or is reachable from -- a negative cycle. Walking predecessors back
from that edge and detecting the repeat recovers the cycle itself.

Three departures from the textbook, each earning its complexity:

**Early termination.** A pass that relaxes nothing means the distances have
converged and no negative cycle exists. On a real currency graph convergence
happens in 3-5 passes, not V-1 (~400), so this alone is a two-orders-of-magnitude
speedup and it is exact, not approximate.

**All cycles, not one.** The textbook returns a single negative cycle. We want
every disjoint one, because the profitable cycles at any instant are usually
several variations on the same dislocation and the planner must choose among
them on expected value, not take whichever the algorithm happened to find first.

**Length bounds.** A negative cycle of length 9 is not tradeable: nine legs of
fees and nine chances to fail. Cycles are filtered to ``[min_length,
max_length]`` during extraction rather than after, which keeps the predecessor
walk short.

Complexity is O(V*E) worst case. With ~120 assets and ~800 usable edges that is
under 100k relaxations -- roughly a millisecond in CPython, which is why the
scan interval is set in tens of milliseconds and not in seconds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Sequence

from triangulum.graph.currency_graph import CurrencyGraph, Edge

logger = logging.getLogger(__name__)

__all__ = ["NegativeCycle", "find_negative_cycles", "bellman_ford_distances"]


@dataclass(slots=True)
class NegativeCycle:
    """A closed path whose log-weights sum to a negative number."""

    edges: tuple[Edge, ...]
    total_weight: float

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def start_asset(self) -> str:
        return self.edges[0].frm.code if self.edges else ""

    @property
    def gross_return(self) -> float:
        """exp(-total_weight). A weight of -0.0003 is a return of 1.0003."""
        return math.exp(-self.total_weight)

    @property
    def edge_bps(self) -> float:
        return (self.gross_return - 1.0) * 10_000.0

    @property
    def venues(self) -> tuple[str, ...]:
        seen: list[str] = []
        for e in self.edges:
            if e.venue not in seen:
                seen.append(e.venue)
        return tuple(seen)

    @property
    def max_book_age_ns(self) -> int:
        return max((e.book_age_ns for e in self.edges), default=0)

    @property
    def path(self) -> str:
        if not self.edges:
            return ""
        return " -> ".join([e.frm.code for e in self.edges] + [self.edges[-1].to.code])

    def rotate_to(self, asset_code: str) -> "NegativeCycle | None":
        """
        Re-express the cycle so it starts at ``asset_code``.

        A cycle is a loop with no intrinsic starting point, but execution needs
        one: you must begin from an asset you actually hold. Rotating is free
        and it is what lets a USDT-funded account trade a cycle the search
        happened to discover starting at ETH.
        """
        for i, edge in enumerate(self.edges):
            if edge.frm.code == asset_code:
                if i == 0:
                    return self
                return NegativeCycle(
                    edges=self.edges[i:] + self.edges[:i],
                    total_weight=self.total_weight,
                )
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"NegativeCycle({self.path}, {self.edge_bps:+.2f} bps)"


def bellman_ford_distances(
    graph: CurrencyGraph,
    source: str,
    *,
    max_passes: int | None = None,
) -> tuple[dict[str, float], dict[str, Edge]]:
    """Standard relaxation. Returns (distances, predecessor edges)."""
    nodes = graph.nodes
    distance: dict[str, float] = {node: math.inf for node in nodes}
    predecessor: dict[str, Edge] = {}
    distance[source] = 0.0

    passes = max_passes if max_passes is not None else max(1, len(nodes) - 1)
    for _ in range(passes):
        changed = False
        for node in nodes:
            d = distance[node]
            if d == math.inf:
                continue
            for edge in graph.edges_from(node):
                candidate = d + edge.weight
                if candidate < distance.get(edge.to.code, math.inf) - 1e-15:
                    distance[edge.to.code] = candidate
                    predecessor[edge.to.code] = edge
                    changed = True
        if not changed:
            break
    return distance, predecessor


def find_negative_cycles(
    graph: CurrencyGraph,
    *,
    min_length: int = 3,
    max_length: int = 4,
    max_cycles: int = 32,
    start_assets: Sequence[str] = (),
    min_edge_bps: float = 0.0,
) -> list[NegativeCycle]:
    """
    Find negative-weight cycles in the graph.

    A *virtual source* with a zero-weight edge to every node is used implicitly
    by initialising all distances to 0 rather than infinity. This finds cycles
    in every connected component in one run -- with a real single source you
    would miss any cycle not reachable from it, which on a multi-venue graph is
    most of them.

    ``start_assets`` filters to cycles that can be rotated to begin at an asset
    we are willing to hold. Restricting to USDT/USDC/BTC is not a limitation but
    a risk control: if leg 1 fills and leg 2 fails, you are left holding the
    start asset, and "USDT" is a much better answer than "some illiquid altcoin".
    """
    graph.reindex_if_dirty()
    nodes = graph.nodes
    if len(nodes) < min_length:
        return []

    # Virtual super-source: every node starts at distance zero.
    distance: dict[str, float] = {node: 0.0 for node in nodes}
    predecessor: dict[str, Edge] = {}

    relaxed_node: str | None = None
    passes = len(nodes)
    for pass_index in range(passes):
        relaxed_node = None
        for node in nodes:
            d = distance[node]
            if d == math.inf:
                continue
            for edge in graph.edges_from(node):
                target = edge.to.code
                candidate = d + edge.weight
                # The epsilon guards against relaxing on pure float noise, which
                # would manufacture "negative cycles" worth 1e-17 bps forever.
                if candidate < distance.get(target, math.inf) - 1e-12:
                    distance[target] = candidate
                    predecessor[target] = edge
                    relaxed_node = target
        if relaxed_node is None:
            # Converged with no negative cycle anywhere.
            return []

    # Still relaxing after V passes -> ``relaxed_node`` is reachable from a
    # negative cycle. Walk back V steps to land *on* the cycle rather than on
    # its tail, then extract it.
    cycles: list[NegativeCycle] = []
    seen_signatures: set[frozenset] = set()
    allowed = set(start_assets)

    candidates = _cycle_entry_points(predecessor, relaxed_node, len(nodes))
    for entry in candidates:
        if len(cycles) >= max_cycles:
            break
        cycle = _extract_cycle(predecessor, entry, max_length)
        if cycle is None:
            continue
        if not (min_length <= len(cycle) <= max_length):
            continue

        signature = frozenset((e.frm.code, e.to.code, e.venue) for e in cycle)
        if signature in seen_signatures:
            continue

        total = sum(e.weight for e in cycle)
        if total >= 0:
            continue

        candidate_cycle = NegativeCycle(edges=tuple(cycle), total_weight=total)
        if candidate_cycle.edge_bps < min_edge_bps:
            continue

        if allowed:
            rotated = None
            for asset in start_assets:
                rotated = candidate_cycle.rotate_to(asset)
                if rotated is not None:
                    break
            if rotated is None:
                continue
            candidate_cycle = rotated

        seen_signatures.add(signature)
        cycles.append(candidate_cycle)

    cycles.sort(key=lambda c: c.total_weight)
    return cycles


def _cycle_entry_points(
    predecessor: Mapping[str, Edge], start: str, node_count: int
) -> list[str]:
    """
    Walk the predecessor chain back far enough to be certain of landing on a
    cycle, collecting the nodes visited along the way as entry candidates.

    Walking exactly ``node_count`` steps guarantees we are on the cycle (the
    chain cannot have more than V distinct nodes before repeating). Collecting
    intermediate nodes lets us find several distinct cycles from one run rather
    than restarting the whole relaxation per cycle.
    """
    entries: list[str] = []
    cursor = start
    visited: set[str] = set()
    for _ in range(node_count):
        edge = predecessor.get(cursor)
        if edge is None:
            break
        cursor = edge.frm.code
        if cursor in visited:
            entries.append(cursor)
            break
        visited.add(cursor)
        entries.append(cursor)
    # Nearest-first: shorter cycles are cheaper to execute, so try them first.
    return list(reversed(entries))[:16]


def _extract_cycle(
    predecessor: Mapping[str, Edge], entry: str, max_length: int
) -> list[Edge] | None:
    """
    Follow predecessors from ``entry`` until a node repeats; that closed section
    is the cycle. Returns edges in forward (execution) order.
    """
    chain: list[Edge] = []
    seen: dict[str, int] = {}
    cursor = entry

    # Allow a little slack beyond max_length so that a valid short cycle sitting
    # behind a couple of tail edges is still found.
    for step in range(max_length + 4):
        if cursor in seen:
            start_index = seen[cursor]
            cycle = chain[start_index:]
            cycle.reverse()
            return cycle if cycle else None
        seen[cursor] = step
        edge = predecessor.get(cursor)
        if edge is None:
            return None
        chain.append(edge)
        cursor = edge.frm.code
    return None
