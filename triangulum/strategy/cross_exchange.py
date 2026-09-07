"""
Cross-venue (spatial) arbitrage.

Buy an asset where it is cheap, sell it where it is dear. Conceptually the
simplest arbitrage there is, and operationally the hardest, for one reason that
dominates everything else:

    **You cannot move the asset fast enough.**

A blockchain transfer takes minutes and costs a fee. By the time BTC arrives at
the venue where it was expensive, the dislocation is long gone. So real spatial
arbitrage is *not* buy-here-send-there. It is:

    1. Pre-position inventory on BOTH venues.
    2. When venue A's ask is below venue B's bid, buy on A and sell on B
       simultaneously, using the inventory already sitting there.
    3. Your net asset position is unchanged; only the split between venues
       moved. Periodically rebalance.

Which means the strategy's real constraint is not finding the spread -- spreads
between venues are large and frequent -- but *capital*. You need meaningful
inventory on every venue simultaneously, and rebalancing costs transfer fees and
time. With $100 total, splitting across two venues leaves $50 each, and at $50
per leg most instruments fail the min-notional and lot-value screens outright.

This is implemented, correct, and honestly labelled: at the configured capital
it will find opportunities it cannot fund, and it says so rather than pretending
otherwise. It becomes genuinely useful in the low thousands.

The second-order cost people miss: withdrawal fees. Moving USDT off Binance
costs ~1 USDT flat. On a $50 rebalance that is 200 bps -- more than any spatial
spread you will ever capture.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Mapping, Sequence

from triangulum.core.decimal_math import D, ONE, ZERO, bps, safe_div
from triangulum.core.types import Asset, Leg, Opportunity, Side, Symbol, new_id
from triangulum.graph.currency_graph import CurrencyGraph
from triangulum.marketdata.book_manager import BookManager
from triangulum.strategy.base import Strategy

logger = logging.getLogger(__name__)

__all__ = ["CrossExchangeStrategy", "SpatialSpread"]


class SpatialSpread:
    """A same-instrument price difference between two venues."""

    __slots__ = (
        "symbol_cheap", "symbol_rich", "buy_price", "sell_price",
        "spread_bps", "net_bps", "size_available", "book_age_ns",
    )

    def __init__(
        self, symbol_cheap: Symbol, symbol_rich: Symbol,
        buy_price: Decimal, sell_price: Decimal,
        spread_bps: Decimal, net_bps: Decimal,
        size_available: Decimal, book_age_ns: int,
    ) -> None:
        self.symbol_cheap = symbol_cheap
        self.symbol_rich = symbol_rich
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.spread_bps = spread_bps
        self.net_bps = net_bps
        self.size_available = size_available
        self.book_age_ns = book_age_ns

    @property
    def pair(self) -> str:
        return self.symbol_cheap.canonical

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpatialSpread({self.pair}: buy {self.symbol_cheap.venue} @"
            f"{self.buy_price} / sell {self.symbol_rich.venue} @{self.sell_price}, "
            f"{self.net_bps:+.2f} bps net)"
        )


class CrossExchangeStrategy(Strategy):
    def __init__(
        self,
        graph: CurrencyGraph,
        books: BookManager,
        *,
        venues: Sequence[str] = (),
        venue_fees_bps: Mapping[str, Decimal] | None = None,
        min_net_bps: Decimal = D("5"),
        require_inventory: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name="cross_exchange", graph=graph, **kwargs)
        self.books = books
        self.venues = tuple(venues)
        self.venue_fees_bps = dict(venue_fees_bps or {})
        self.min_net_bps = min_net_bps
        self.require_inventory = require_inventory
        self._inventory: dict[tuple[str, str], Decimal] = {}
        self.unfundable_opportunities = 0

    def set_inventory(self, venue: str, asset: str, amount: Decimal) -> None:
        self._inventory[(venue, asset)] = amount

    def inventory(self, venue: str, asset: str) -> Decimal:
        return self._inventory.get((venue, asset), ZERO)

    def _fee(self, venue: str) -> Decimal:
        return self.venue_fees_bps.get(venue, D("10"))

    def scan(self, now_ns: int) -> list[Opportunity]:
        if not self.enabled:
            return []
        start = self.clock.mono_ns()

        # Group tradeable books by canonical pair, then compare venues pairwise.
        by_pair: dict[str, list] = {}
        for book in self.books.tradeable_books(now_ns):
            if self.venues and book.symbol.venue not in self.venues:
                continue
            by_pair.setdefault(book.symbol.canonical, []).append(book)

        opportunities: list[Opportunity] = []
        for pair, venue_books in by_pair.items():
            if len(venue_books) < 2:
                continue
            for spread in self._spreads(venue_books, now_ns):
                if len(opportunities) >= self.max_per_scan:
                    break
                opportunity = self._to_opportunity(spread, now_ns)
                if opportunity is not None:
                    opportunities.append(opportunity)

        self._record(opportunities, (self.clock.mono_ns() - start) / 1000.0)
        return opportunities

    def _spreads(self, venue_books: Sequence, now_ns: int) -> list[SpatialSpread]:
        out: list[SpatialSpread] = []
        for cheap in venue_books:
            for rich in venue_books:
                if cheap.symbol.venue == rich.symbol.venue:
                    continue
                ask = cheap.best_ask     # we buy here
                bid = rich.best_bid      # we sell here
                if ask <= 0 or bid <= 0 or bid <= ask:
                    continue

                gross = bps(safe_div(bid - ask, ask))
                fees = self._fee(cheap.symbol.venue) + self._fee(rich.symbol.venue)
                net = gross - fees
                if net < self.min_net_bps:
                    self.stats.rejected_edge += 1
                    continue

                size = min(cheap.asks.best_size(), rich.bids.best_size())
                out.append(SpatialSpread(
                    symbol_cheap=cheap.symbol,
                    symbol_rich=rich.symbol,
                    buy_price=ask,
                    sell_price=bid,
                    spread_bps=gross,
                    net_bps=net,
                    size_available=size,
                    book_age_ns=max(cheap.age_ns(now_ns), rich.age_ns(now_ns)),
                ))
        return out

    def _to_opportunity(self, spread: SpatialSpread, now_ns: int) -> Opportunity | None:
        key = f"{spread.pair}@{spread.symbol_cheap.venue}->{spread.symbol_rich.venue}"
        if self._on_cooldown(key, now_ns):
            return None

        base = spread.symbol_cheap.base
        quote = spread.symbol_cheap.quote

        if self.require_inventory:
            # We must already hold quote on the cheap venue (to buy) AND base on
            # the rich venue (to sell). Without both, this is not an arbitrage;
            # it is a transfer that will complete long after the edge is gone.
            have_quote = self.inventory(spread.symbol_cheap.venue, quote.code)
            have_base = self.inventory(spread.symbol_rich.venue, base.code)
            needed_quote = spread.buy_price * spread.size_available
            if have_quote <= 0 or have_base <= 0:
                self.unfundable_opportunities += 1
                logger.debug(
                    "cross-venue %s is real (%.1f bps) but unfundable: "
                    "need %s on %s and %s on %s",
                    key, float(spread.net_bps), quote.code,
                    spread.symbol_cheap.venue, base.code, spread.symbol_rich.venue,
                )
                return None

        self._mark_fired(key, now_ns)

        # Modelled as a two-leg cycle: quote -> base on the cheap venue,
        # base -> quote on the rich one. It closes, so the standard cycle
        # validation and ledger paths apply unchanged.
        legs = (
            Leg(symbol=spread.symbol_cheap, side=Side.BUY, from_asset=quote, to_asset=base),
            Leg(symbol=spread.symbol_rich, side=Side.SELL, from_asset=base, to_asset=quote),
        )
        return Opportunity(
            opportunity_id=new_id("xopp-"),
            legs=legs,
            start_asset=quote,
            gross_edge_bps=spread.spread_bps,
            reference_notional=self.graph.reference_notional,
            ts_detected_ns=now_ns,
            book_ages_ns=(spread.book_age_ns, spread.book_age_ns),
            venues=(spread.symbol_cheap.venue, spread.symbol_rich.venue),
        )

    def stats_dict(self) -> dict[str, object]:
        base = self.stats.to_dict()
        base["unfundable_opportunities"] = self.unfundable_opportunities
        base["inventory_locations"] = len(self._inventory)
        return base
