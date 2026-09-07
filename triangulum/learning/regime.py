"""
Market regime detection.

The same cycle behaves differently depending on market conditions. Maker legs
fill reliably in a quiet book and never fill in a fast one; spreads that are
normally 2 bps go to 20 during a liquidation cascade. A model that ignores the
regime averages over all of them and is right in none.

Rather than a hidden Markov model -- which needs a lot of data and a lot of
tuning to earn its keep -- this classifies into a small number of interpretable
buckets from two observables:

    realized volatility (EWMA of squared mid returns, annualised)
    mean spread across the tracked universe

    QUIET       low vol, tight spreads     -> maker legs viable, edges small
    NORMAL      the usual state
    ACTIVE      elevated vol, wider spreads -> more edges, harder fills
    STRESSED    high vol, wide spreads      -> mostly phantoms; trade less
    ILLIQUID    low vol but wide spreads    -> thin book, high slippage risk

The bucket label feeds the bandit as context, so it can learn a different
best-arm per regime. Five labels is deliberate: enough to be useful, few enough
that each accumulates samples in reasonable time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping

from triangulum.core.decimal_math import D, ZERO
from triangulum.core.ringbuffer import RollingStats

__all__ = ["Regime", "RegimeDetector"]


class Regime:
    QUIET = "quiet"
    NORMAL = "normal"
    ACTIVE = "active"
    STRESSED = "stressed"
    ILLIQUID = "illiquid"
    UNKNOWN = "unknown"

    ALL = (QUIET, NORMAL, ACTIVE, STRESSED, ILLIQUID)


@dataclass(slots=True)
class _SymbolState:
    last_mid: float = 0.0
    ewma_variance: float = 0.0
    spread_bps: RollingStats = field(default_factory=lambda: RollingStats(256))
    updates: int = 0


class RegimeDetector:
    """Classifies the current market regime from volatility and spreads."""

    def __init__(
        self,
        *,
        volatility_alpha: float = 0.02,
        quiet_vol_percentile: float = 0.25,
        stressed_vol_percentile: float = 0.85,
        min_updates: int = 64,
    ) -> None:
        self.volatility_alpha = volatility_alpha
        self.quiet_vol_percentile = quiet_vol_percentile
        self.stressed_vol_percentile = stressed_vol_percentile
        self.min_updates = min_updates

        self._symbols: dict[str, _SymbolState] = {}
        self._market_vol = RollingStats(2048)
        self._market_spread = RollingStats(2048)
        self._current = Regime.UNKNOWN
        self._transitions = 0
        self._history: list[tuple[int, str]] = []

    def observe(self, symbol_key: str, mid: Decimal, spread_bps: Decimal,
                ts_ns: int = 0) -> None:
        state = self._symbols.setdefault(symbol_key, _SymbolState())
        mid_f = float(mid)
        if mid_f <= 0:
            return

        if state.last_mid > 0:
            log_return = math.log(mid_f / state.last_mid)
            # EWMA variance of log returns.
            state.ewma_variance = (
                (1 - self.volatility_alpha) * state.ewma_variance
                + self.volatility_alpha * log_return * log_return
            )
        state.last_mid = mid_f
        state.spread_bps.push(float(spread_bps))
        state.updates += 1

        if state.updates >= self.min_updates:
            self._market_vol.push(math.sqrt(state.ewma_variance) * 10_000)  # in bps
            self._market_spread.push(float(spread_bps))

    def classify(self, ts_ns: int = 0) -> str:
        """Current regime label."""
        if self._market_vol.count < self.min_updates:
            return Regime.UNKNOWN

        vol = self._market_vol.last
        spread = self._market_spread.mean
        vol_z = self._market_vol.zscore(vol)
        spread_z = self._market_spread.zscore(self._market_spread.last)

        if vol_z > 2.0 and spread_z > 1.5:
            regime = Regime.STRESSED
        elif vol_z > 1.0:
            regime = Regime.ACTIVE
        elif spread_z > 1.5 and vol_z < 0.0:
            # Wide spreads without volatility means the book has thinned out --
            # the most dangerous state for a taker, because the *apparent* edge
            # is large and entirely uncapturable.
            regime = Regime.ILLIQUID
        elif vol_z < -0.5 and spread_z < 0.0:
            regime = Regime.QUIET
        else:
            regime = Regime.NORMAL

        if regime != self._current:
            self._transitions += 1
            self._history.append((ts_ns, regime))
            if len(self._history) > 500:
                self._history = self._history[-500:]
            self._current = regime
        return regime

    def symbol_volatility_z(self, symbol_key: str) -> float:
        state = self._symbols.get(symbol_key)
        if state is None or state.updates < self.min_updates:
            return 0.0
        vol_bps = math.sqrt(state.ewma_variance) * 10_000
        return self._market_vol.zscore(vol_bps)

    @property
    def current(self) -> str:
        return self._current

    def stats(self) -> dict[str, object]:
        return {
            "regime": self._current,
            "transitions": self._transitions,
            "tracked_symbols": len(self._symbols),
            "market_volatility_bps": round(self._market_vol.mean, 3),
            "market_spread_bps": round(self._market_spread.mean, 3),
            "volatility_stddev": round(self._market_vol.stddev, 3),
            "recent_history": [r for _, r in self._history[-10:]],
        }
