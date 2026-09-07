"""
Synthetic market generator.

Used by the demo mode and by tests. Generates order books for a small universe
with realistic microstructure, and injects triangular dislocations at a
controllable rate and size.

The point is not to model markets faithfully -- it is to produce a stream with
the *statistical properties that matter* for testing this engine:

    · dislocations are rare, small, and short-lived
    · most apparent edges are inside the spread and uncapturable
    · book updates arrive at irregular intervals, so staleness varies
    · occasionally a book goes quiet, producing exactly the stale-but-attractive
      opportunity the EV gate exists to reject

That last property is deliberate. A generator that only produces clean data
would let a broken engine look healthy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterator, Sequence

from triangulum.core.decimal_math import D, ZERO
from triangulum.core.types import Symbol
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.normalizer import SymbolNormalizer

__all__ = ["SyntheticMarket", "MarketParams", "DEMO_UNIVERSE"]


# (base, quote, initial price, lot step, tick size, min notional)
# Prices and grids are the real ones from Binance spot, because the whole point
# of the lot-value screen is that these specific numbers decide feasibility.
DEMO_UNIVERSE: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("BTC",  "USDT", "62000",   "0.00001", "0.01",     "5"),
    ("ETH",  "USDT", "3050",    "0.0001",  "0.01",     "5"),
    ("SOL",  "USDT", "148",     "0.001",   "0.01",     "5"),
    ("XRP",  "USDT", "0.542",   "0.1",     "0.0001",   "5"),
    ("ADA",  "USDT", "0.447",   "0.1",     "0.0001",   "5"),
    ("TRX",  "USDT", "0.1187",  "0.1",     "0.00001",  "5"),
    ("ETH",  "BTC",  "0.049193","0.0001",  "0.000001", "0.0001"),
    ("SOL",  "BTC",  "0.002387","0.001",   "0.0000001","0.0001"),
    ("XRP",  "BTC",  "0.0000087","0.1",    "0.00000001","0.0001"),
    ("ADA",  "BTC",  "0.0000072","0.1",    "0.00000001","0.0001"),
    ("TRX",  "BTC",  "0.0000019","0.1",    "0.00000001","0.0001"),
    ("SOL",  "ETH",  "0.04852", "0.001",   "0.00001",  "0.001"),
    ("XRP",  "ETH",  "0.000178","0.1",     "0.0000001","0.001"),
    ("ADA",  "ETH",  "0.000147","0.1",     "0.0000001","0.001"),
    ("TRX",  "ETH",  "0.0000389","0.1",    "0.0000001","0.001"),
)


@dataclass(slots=True)
class MarketParams:
    """Knobs on the synthetic market's realism."""

    # Per-tick log-return volatility of the underlying mid.
    volatility: float = 0.00035
    # Half-spread in basis points, drawn per update.
    spread_bps_mean: float = 3.0
    spread_bps_std: float = 1.2
    # Depth at the touch, as a multiple of a $100 order.
    depth_multiple_mean: float = 40.0

    # Dislocation injection. A "dislocation" pushes one cross pair away from
    # the value implied by its two legs, creating a genuine triangular edge.
    dislocation_probability: float = 0.012
    dislocation_bps_mean: float = 9.0
    dislocation_bps_std: float = 5.0
    dislocation_decay: float = 0.55      # fraction remaining per tick

    # Probability a book skips an update, going stale. This is what produces
    # the attractive-looking phantom opportunities.
    stale_probability: float = 0.06
    stale_ticks_mean: float = 6.0

    seed: int | None = 7


class SyntheticMarket:
    """Generates a stream of order-book updates for a fixed universe."""

    def __init__(
        self,
        books: BookManager,
        normalizer: SymbolNormalizer,
        *,
        venue: str = "binance",
        universe: Sequence[tuple[str, str, str, str, str, str]] = DEMO_UNIVERSE,
        params: MarketParams | None = None,
    ) -> None:
        self.books = books
        self.normalizer = normalizer
        self.venue = venue
        self.params = params or MarketParams()
        self._rng = random.Random(self.params.seed)

        self.symbols: list[Symbol] = []
        self.specs: dict[str, tuple[Decimal, Decimal, Decimal]] = {}
        self._mid: dict[str, float] = {}
        self._dislocation: dict[str, float] = {}
        self._stale_until: dict[str, int] = {}
        self.ticks = 0
        self.dislocations_injected = 0

        for base, quote, price, lot, tick, min_notional in universe:
            symbol = self.normalizer.register(venue, f"{base}{quote}", base, quote)
            self.symbols.append(symbol)
            self.specs[symbol.key] = (D(lot), D(tick), D(min_notional))
            self._mid[symbol.key] = float(price)
            self._dislocation[symbol.key] = 0.0
        self.books.mark_connected(venue)

    # -- generation --------------------------------------------------------

    def step(self) -> int:
        """Advance one tick. Returns how many books were updated."""
        self.ticks += 1
        p = self.params
        updated = 0

        # 1. Random-walk the base USDT prices; cross rates are then implied.
        for symbol in self.symbols:
            if symbol.quote.code != "USDT":
                continue
            key = symbol.key
            self._mid[key] *= math.exp(self._rng.gauss(0, p.volatility))

        # 2. Derive cross rates from the USDT legs, so the graph is internally
        #    consistent -- and therefore has NO arbitrage until one is injected.
        for symbol in self.symbols:
            if symbol.quote.code == "USDT":
                continue
            base_usdt = self._usdt_price(symbol.base.code)
            quote_usdt = self._usdt_price(symbol.quote.code)
            if base_usdt and quote_usdt:
                self._mid[symbol.key] = base_usdt / quote_usdt

        # 3. Inject and decay dislocations on cross pairs only.
        for symbol in self.symbols:
            key = symbol.key
            if symbol.quote.code == "USDT":
                continue
            if self._rng.random() < p.dislocation_probability:
                magnitude = abs(self._rng.gauss(p.dislocation_bps_mean, p.dislocation_bps_std))
                self._dislocation[key] = magnitude * self._rng.choice((1, -1))
                self.dislocations_injected += 1
            else:
                self._dislocation[key] *= p.dislocation_decay
                if abs(self._dislocation[key]) < 0.05:
                    self._dislocation[key] = 0.0

        # 4. Publish books.
        for symbol in self.symbols:
            key = symbol.key
            if self.ticks < self._stale_until.get(key, 0):
                continue          # deliberately stale
            if self._rng.random() < p.stale_probability:
                self._stale_until[key] = self.ticks + max(
                    1, int(self._rng.expovariate(1 / p.stale_ticks_mean))
                )
                continue

            mid = self._mid[key] * (1 + self._dislocation.get(key, 0.0) / 10_000)
            half_spread = max(
                0.2, self._rng.gauss(p.spread_bps_mean, p.spread_bps_std)
            ) / 2 / 10_000
            bid = mid * (1 - half_spread)
            ask = mid * (1 + half_spread)

            lot, tick, _min_notional = self.specs[key]
            depth_notional = p.depth_multiple_mean * (0.4 + self._rng.random() * 1.6) * 100

            bids, asks = [], []
            for level in range(5):
                step = tick * D(level * 3 + 1)
                size = D(str(depth_notional / max(mid, 1e-12))) * D(str(1 + level * 0.6))
                bids.append((_q(D(str(bid)) - step, tick), _round_size(size, lot)))
                asks.append((_q(D(str(ask)) + step, tick), _round_size(size, lot)))

            bids = [(p_, s) for p_, s in bids if p_ > 0 and s > 0]
            asks = [(p_, s) for p_, s in asks if p_ > 0 and s > 0]
            if bids and asks and bids[0][0] < asks[0][0]:
                self.books.apply_snapshot(symbol, bids, asks, sequence=self.ticks)
                updated += 1

        return updated

    def _usdt_price(self, asset: str) -> float:
        if asset == "USDT":
            return 1.0
        for symbol in self.symbols:
            if symbol.base.code == asset and symbol.quote.code == "USDT":
                return self._mid[symbol.key]
        return 0.0

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def stats(self) -> dict[str, object]:
        return {
            "ticks": self.ticks,
            "symbols": len(self.symbols),
            "dislocations_injected": self.dislocations_injected,
            "active_dislocations": sum(
                1 for v in self._dislocation.values() if abs(v) > 0.1
            ),
            "stale_books": sum(
                1 for k, until in self._stale_until.items() if until > self.ticks
            ),
        }


def _q(value: Decimal, tick: Decimal) -> Decimal:
    from triangulum.core.decimal_math import floor_to_step
    return floor_to_step(value, tick) if tick > 0 else value


def _round_size(value: Decimal, lot: Decimal) -> Decimal:
    from triangulum.core.decimal_math import floor_to_step
    return floor_to_step(value, lot) if lot > 0 else value
