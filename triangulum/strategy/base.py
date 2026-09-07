"""
Strategy interface.

A strategy converts market state into :class:`Opportunity` objects. It does not
size them, does not decide whether to trade them, and never sends an order.
That separation is deliberate: the strategy answers "what is mispriced?", the
planner answers "how much can I actually get?", the EV gate answers "is it worth
it?", and the executor answers "how do I work the legs?". Collapsing any two of
those into one class is how strategies end up with hardcoded risk assumptions
that nobody can audit.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from triangulum.core.clock import Clock, SystemClock
from triangulum.core.decimal_math import D, ZERO
from triangulum.core.ringbuffer import RollingStats
from triangulum.core.types import Opportunity, new_id
from triangulum.graph.currency_graph import CurrencyGraph

logger = logging.getLogger(__name__)

__all__ = ["Strategy", "StrategyStats"]


@dataclass(slots=True)
class StrategyStats:
    name: str
    scans: int = 0
    opportunities: int = 0
    rejected_edge: int = 0
    rejected_stale: int = 0
    rejected_cooldown: int = 0
    scan_duration_us: RollingStats = field(default_factory=lambda: RollingStats(512))
    edge_bps: RollingStats = field(default_factory=lambda: RollingStats(1024))

    @property
    def hit_rate(self) -> float:
        return self.opportunities / self.scans if self.scans else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scans": self.scans,
            "opportunities": self.opportunities,
            "opportunities_per_scan": round(self.hit_rate, 4),
            "rejected_edge": self.rejected_edge,
            "rejected_stale": self.rejected_stale,
            "rejected_cooldown": self.rejected_cooldown,
            "mean_scan_us": round(self.scan_duration_us.mean, 1),
            "p95_scan_us": round(
                self.scan_duration_us.mean + 1.645 * self.scan_duration_us.stddev, 1
            ),
            "mean_edge_bps": round(self.edge_bps.mean, 3),
            "max_edge_bps": round(self.edge_bps.maximum, 3),
        }


class Strategy(abc.ABC):
    """Base class for opportunity detectors."""

    def __init__(
        self,
        name: str,
        graph: CurrencyGraph,
        *,
        clock: Clock | None = None,
        min_edge_bps: Decimal = D("3"),
        max_book_age_ns: int = 250_000_000,
        cooldown_ns: int = 750_000_000,
        max_per_scan: int = 16,
    ) -> None:
        self.name = name
        self.graph = graph
        self.clock = clock or SystemClock()
        self.min_edge_bps = min_edge_bps
        self.max_book_age_ns = max_book_age_ns
        self.cooldown_ns = cooldown_ns
        self.max_per_scan = max_per_scan
        self.stats = StrategyStats(name=name)
        self.enabled = True
        self._last_fired: dict[str, int] = {}

    @abc.abstractmethod
    def scan(self, now_ns: int) -> list[Opportunity]:
        """Return opportunities visible in the current market state."""

    # -- shared helpers ----------------------------------------------------

    def _on_cooldown(self, cycle_key: str, now_ns: int) -> bool:
        """
        Suppress a path we traded very recently.

        Repeated immediate firing on the same cycle is far more often a symptom
        of a stale book than of a persistent dislocation: a real edge is taken
        by someone within milliseconds, so if it is *still* there on the next
        scan, the more likely explanation is that our data has not updated.
        """
        last = self._last_fired.get(cycle_key)
        if last is None:
            return False
        if now_ns - last < self.cooldown_ns:
            self.stats.rejected_cooldown += 1
            return True
        return False

    def _mark_fired(self, cycle_key: str, now_ns: int) -> None:
        self._last_fired[cycle_key] = now_ns
        # Bound the memory: a long session touches many distinct paths.
        if len(self._last_fired) > 8192:
            cutoff = now_ns - self.cooldown_ns * 4
            self._last_fired = {
                k: v for k, v in self._last_fired.items() if v > cutoff
            }

    def _record(self, opportunities: Sequence[Opportunity], scan_us: float) -> None:
        self.stats.scans += 1
        self.stats.scan_duration_us.push(scan_us)
        self.stats.opportunities += len(opportunities)
        for opp in opportunities:
            self.stats.edge_bps.push(float(opp.gross_edge_bps))

    def reset(self) -> None:
        self._last_fired.clear()
        self.stats = StrategyStats(name=self.name)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name} enabled={self.enabled}>"
