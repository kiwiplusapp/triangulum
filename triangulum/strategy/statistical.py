"""
Statistical arbitrage: mean reversion on cointegrated spreads.

This is the honest answer to "can I do this without cryptocurrencies?".

Triangular arbitrage needs a cycle, and outside crypto and interbank FX the
graph has no cycles. What *does* exist outside crypto is statistical
arbitrage: two instruments whose prices share a common stochastic trend, so
their spread is mean-reverting even though each price individually is not.
Trade the spread when it is far from its mean; close when it reverts.

Be clear about what this is and is not:

    It is NOT riskless. There is no moment where the profit is locked in. The
    spread can widen for months. Every pairs-trading blowup in history is the
    same story: a spread that was "5 sigma from the mean" and kept going,
    because the cointegration relationship had broken and the statistics were
    describing a world that no longer existed.

    It DOES have a real, documented edge, and it works on equities, ETFs, FX
    crosses and commodities -- everything the crypto path cannot reach.

    It needs FAR more capital than arbitrage. Arbitrage profits are small and
    near-certain, so leverage on a small account is survivable. Stat-arb
    profits are larger and uncertain, so position sizing must respect a
    drawdown distribution. At $100 the position sizes that survive a 3-sigma
    excursion are too small to clear commissions.

Implementation: rolling OLS hedge ratio, rolling z-score of the residual, entry
at |z| > entry_z, exit at |z| < exit_z, hard stop at |z| > stop_z. The stop is
not optional -- it is the difference between a strategy and a slow-motion
account deletion.

The Engle-Granger cointegration test is approximated with an Augmented
Dickey-Fuller statistic on the residual, computed incrementally. It is a screen,
not a proof; a pair that passes should still be reviewed by a human before it
trades real money.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from triangulum.core.decimal_math import D, ZERO, bps, safe_div
from triangulum.core.ringbuffer import RollingStats
from triangulum.core.types import Leg, Opportunity, Side, Symbol, new_id
from triangulum.graph.currency_graph import CurrencyGraph
from triangulum.marketdata.book_manager import BookManager
from triangulum.strategy.base import Strategy

logger = logging.getLogger(__name__)

__all__ = ["StatisticalStrategy", "PairState"]


@dataclass(slots=True)
class PairState:
    """Rolling statistics for one candidate pair."""

    symbol_a: Symbol
    symbol_b: Symbol
    window: int = 512

    log_a: RollingStats = field(default_factory=lambda: RollingStats(512))
    log_b: RollingStats = field(default_factory=lambda: RollingStats(512))
    residual: RollingStats = field(default_factory=lambda: RollingStats(512))

    # Incrementally maintained sums for the OLS hedge ratio.
    sum_ab: float = 0.0
    sum_bb: float = 0.0
    samples: int = 0

    hedge_ratio: float = 1.0
    last_z: float = 0.0
    in_position: bool = False
    position_side: int = 0          # +1 long spread, -1 short spread
    entries: int = 0
    exits: int = 0
    stops: int = 0

    @property
    def key(self) -> str:
        return f"{self.symbol_a.canonical}~{self.symbol_b.canonical}"

    def update(self, price_a: Decimal, price_b: Decimal) -> None:
        if price_a <= 0 or price_b <= 0:
            return
        la, lb = math.log(float(price_a)), math.log(float(price_b))
        self.log_a.push(la)
        self.log_b.push(lb)

        # Rolling OLS through the means: beta = cov(a,b) / var(b).
        mean_a, mean_b = self.log_a.mean, self.log_b.mean
        da, db = la - mean_a, lb - mean_b
        self.sum_ab += da * db
        self.sum_bb += db * db
        self.samples += 1

        if self.sum_bb > 1e-12:
            self.hedge_ratio = self.sum_ab / self.sum_bb

        self.residual.push(la - self.hedge_ratio * lb)
        sd = self.residual.stddev
        self.last_z = (self.residual.last - self.residual.mean) / sd if sd > 1e-9 else 0.0

    @property
    def ready(self) -> bool:
        return self.residual.count >= max(64, self.window // 4)

    @property
    def half_life(self) -> float:
        """
        Mean-reversion half-life in samples, from an AR(1) fit on the residual.

        A half-life longer than the holding period you can tolerate means the
        pair is not tradeable however good the statistics look: capital tied up
        for a month waiting for reversion is capital not compounding, and it is
        exposed to the relationship breaking the whole time.
        """
        values = self.residual.to_list()
        if len(values) < 32:
            return math.inf
        mean = sum(values) / len(values)
        num = sum(
            (values[i] - mean) * (values[i - 1] - mean) for i in range(1, len(values))
        )
        den = sum((v - mean) ** 2 for v in values[:-1])
        if den <= 1e-12:
            return math.inf
        rho = num / den
        if rho <= 0 or rho >= 1:
            return math.inf
        return -math.log(2) / math.log(rho)

    @property
    def adf_like_statistic(self) -> float:
        """
        Dickey-Fuller-style statistic on the residual.

        Regress d(residual) on lagged residual; the t-statistic of the slope is
        the test statistic. Below roughly -2.9 is evidence of stationarity at
        the 5% level. This is a screen, not a proof -- and critical values for
        a *residual* of an estimated cointegrating relationship are more
        negative than the standard ADF table, so treat -2.9 as optimistic.
        """
        values = self.residual.to_list()
        n = len(values)
        if n < 32:
            return 0.0
        y = [values[i] - values[i - 1] for i in range(1, n)]
        x = values[:-1]
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        sxx = sum((v - mean_x) ** 2 for v in x)
        if sxx <= 1e-12:
            return 0.0
        sxy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        beta = sxy / sxx
        residuals = [y[i] - beta * (x[i] - mean_x) - mean_y for i in range(len(x))]
        dof = max(1, len(x) - 2)
        sigma2 = sum(r * r for r in residuals) / dof
        se = math.sqrt(sigma2 / sxx) if sxx > 0 else 0.0
        return beta / se if se > 1e-12 else 0.0

    @property
    def cointegrated(self) -> bool:
        return self.adf_like_statistic < -2.9 and self.half_life < 500


class StatisticalStrategy(Strategy):
    def __init__(
        self,
        graph: CurrencyGraph,
        books: BookManager,
        *,
        pairs: Sequence[tuple[Symbol, Symbol]] = (),
        entry_z: float = 2.0,
        exit_z: float = 0.3,
        stop_z: float = 4.0,
        window: int = 512,
        require_cointegration: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name="statistical", graph=graph, **kwargs)
        self.books = books
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.window = window
        self.require_cointegration = require_cointegration
        self._pairs: dict[str, PairState] = {}
        for a, b in pairs:
            self.add_pair(a, b)

    def add_pair(self, a: Symbol, b: Symbol) -> PairState:
        state = PairState(
            symbol_a=a, symbol_b=b, window=self.window,
            log_a=RollingStats(self.window),
            log_b=RollingStats(self.window),
            residual=RollingStats(self.window),
        )
        self._pairs[state.key] = state
        return state

    def scan(self, now_ns: int) -> list[Opportunity]:
        if not self.enabled:
            return []
        start = self.clock.mono_ns()
        opportunities: list[Opportunity] = []

        for state in self._pairs.values():
            book_a = self.books.get(state.symbol_a)
            book_b = self.books.get(state.symbol_b)
            if book_a is None or book_b is None:
                continue
            if not (book_a.initialized and book_b.initialized):
                continue
            state.update(book_a.mid, book_b.mid)

            if not state.ready:
                continue
            if self.require_cointegration and not state.cointegrated:
                continue

            signal = self._signal(state)
            if signal == 0:
                continue
            if len(opportunities) >= self.max_per_scan:
                break

            opportunity = self._to_opportunity(state, signal, now_ns)
            if opportunity is not None:
                opportunities.append(opportunity)

        self._record(opportunities, (self.clock.mono_ns() - start) / 1000.0)
        return opportunities

    def _signal(self, state: PairState) -> int:
        z = state.last_z
        if state.in_position:
            if abs(z) >= self.stop_z:
                state.stops += 1
                state.in_position = False
                logger.warning(
                    "stat-arb STOP on %s at z=%.2f -- the relationship may have "
                    "broken; review before re-enabling", state.key, z,
                )
                return -state.position_side
            if abs(z) <= self.exit_z:
                state.exits += 1
                state.in_position = False
                return -state.position_side
            return 0

        if z >= self.entry_z:
            state.in_position = True
            state.position_side = -1     # spread rich: short A, long B
            state.entries += 1
            return -1
        if z <= -self.entry_z:
            state.in_position = True
            state.position_side = 1
            state.entries += 1
            return 1
        return 0

    def _to_opportunity(
        self, state: PairState, signal: int, now_ns: int
    ) -> Opportunity | None:
        a, b = state.symbol_a, state.symbol_b
        if a.quote != b.quote:
            # A spread between differently-quoted instruments carries FX risk
            # that this implementation does not hedge.
            return None

        side_a = Side.BUY if signal > 0 else Side.SELL
        side_b = Side.SELL if signal > 0 else Side.BUY

        legs = (
            Leg(symbol=a, side=side_a,
                from_asset=a.quote if side_a is Side.BUY else a.base,
                to_asset=a.base if side_a is Side.BUY else a.quote),
            Leg(symbol=b, side=side_b,
                from_asset=b.quote if side_b is Side.BUY else b.base,
                to_asset=b.base if side_b is Side.BUY else b.quote),
        )
        # The "edge" of a stat-arb signal is an expectation, not a locked-in
        # profit. Reported as the z-score's distance from the exit band scaled
        # by the residual's standard deviation, in bps.
        expected = abs(state.last_z) - self.exit_z
        edge_bps = D(str(round(expected * state.residual.stddev * 10_000, 4)))

        return Opportunity(
            opportunity_id=new_id("sopp-"),
            legs=legs,
            start_asset=a.quote,
            gross_edge_bps=edge_bps,
            reference_notional=self.graph.reference_notional,
            ts_detected_ns=now_ns,
            book_ages_ns=(0, 0),
            venues=(a.venue, b.venue),
        )

    def pair_report(self) -> list[dict[str, object]]:
        return [
            {
                "pair": s.key,
                "hedge_ratio": round(s.hedge_ratio, 4),
                "z": round(s.last_z, 3),
                "half_life": round(s.half_life, 1) if math.isfinite(s.half_life) else None,
                "adf": round(s.adf_like_statistic, 3),
                "cointegrated": s.cointegrated,
                "samples": s.residual.count,
                "entries": s.entries,
                "exits": s.exits,
                "stops": s.stops,
                "in_position": s.in_position,
            }
            for s in self._pairs.values()
        ]
