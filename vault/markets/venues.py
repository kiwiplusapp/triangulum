"""
Kalshi and Polymarket adapters.

Both are read-only and keyless: public market data only, no credentials, no
order placement. The scanner finds and reports; a human places the orders.
That boundary is deliberate at this stage -- an unattended bot with live
credentials on an unverified fee model is how $200 becomes $0.

## The structural difference that drives the whole design

**Kalshi**: YES and NO are two views of ONE book. Empirically, on live data,
``no_ask == 1 - yes_bid`` and ``no_bid == 1 - yes_ask``, exactly. The API even
reflects this -- there are ``yes_bid_size_fp`` and ``yes_ask_size_fp`` fields
and no NO-side equivalents.

The consequence: ``yes_ask + no_ask == 1 + spread >= 1`` identically. **The
textbook "buy YES and NO together for less than $1" arbitrage cannot exist
within a single Kalshi market.** Any scanner reporting one there has a bug.

**Polymarket**: YES and NO are separate ERC-1155 tokens with separate CLOB
books. They can and do disagree. So the single-market check IS meaningful
there, and it is the cheapest real opportunity on offer.

That asymmetry is recorded per contract as ``shares_book`` and is what the
arbitrage layer keys off, rather than being hardcoded per venue at the call
site where it would drift out of sync.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from vault.data.http import http_get
from vault.markets.types import (
    Book,
    BookLevel,
    Contract,
    ExhaustiveEvidence,
    MarketGroup,
    Venue,
    infer_exhaustiveness,
)

logger = logging.getLogger(__name__)

__all__ = ["KalshiAdapter", "PolymarketAdapter", "load_groups"]

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"


def _get_json(url: str, *, timeout: float = 30.0) -> Any:
    return json.loads(http_get(url, timeout=timeout).decode("utf-8", "replace"))


def _parse_time(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------


class KalshiAdapter:
    """Public Kalshi market data, grouped by event."""

    venue = Venue.KALSHI

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def fetch_groups(self, *, limit: int = 200, min_volume: float = 0.0,
                     pages: int = 1) -> list[MarketGroup]:
        """
        Events with their nested markets.

        The events endpoint is used rather than /markets because it is the only
        place the ``mutually_exclusive`` flag appears, and that flag is a
        precondition for every multi-outcome check downstream.
        """
        groups: list[MarketGroup] = []
        cursor = ""
        for _ in range(max(1, pages)):
            url = (
                f"{KALSHI_BASE}/events?limit={limit}&status=open"
                f"&with_nested_markets=true"
            )
            if cursor:
                url += f"&cursor={cursor}"
            try:
                payload = _get_json(url, timeout=self.timeout)
            except Exception as exc:
                logger.warning("kalshi events fetch failed: %s", exc)
                break

            for raw in payload.get("events", []):
                group = self._to_group(raw)
                if group is None:
                    continue
                if min_volume and group.total_volume_24h < min_volume:
                    continue
                groups.append(group)

            cursor = payload.get("cursor") or ""
            if not cursor:
                break
        return groups

    def _to_group(self, raw: dict[str, Any]) -> MarketGroup | None:
        markets = raw.get("markets") or []
        if not markets:
            return None

        contracts = [self._to_contract(m, raw) for m in markets]
        contracts = [c for c in contracts if c is not None]
        if not contracts:
            return None

        group = MarketGroup(
            venue=self.venue,
            group_id=raw.get("event_ticker", ""),
            title=raw.get("title", ""),
            contracts=contracts,
            mutually_exclusive=bool(raw.get("mutually_exclusive")),
            close_time=min(
                (c.close_time for c in contracts if c.close_time), default=None,
            ),
            raw={k: v for k, v in raw.items() if k != "markets"},
        )
        group.exhaustive_evidence, group.exhaustive_note = infer_exhaustiveness(group)
        return group

    def _to_contract(self, raw: dict[str, Any],
                     event: dict[str, Any]) -> Contract | None:
        ticker = raw.get("ticker")
        if not ticker:
            return None

        yes_bid = _f(raw.get("yes_bid_dollars"))
        yes_ask = _f(raw.get("yes_ask_dollars"))
        bid_size = _f(raw.get("yes_bid_size_fp"))
        ask_size = _f(raw.get("yes_ask_size_fp"))

        yes = Book()
        if yes_bid > 0 and bid_size > 0:
            yes.bids.append(BookLevel(yes_bid, bid_size))
        if yes_ask > 0 and ask_size > 0:
            yes.asks.append(BookLevel(yes_ask, ask_size))

        # The NO book is the YES book reflected. Built explicitly rather than
        # read from the payload so the identity is visible in the code: the
        # size available to buy NO at (1 - yes_bid) IS the size resting on the
        # YES bid.
        no = Book()
        if yes_ask > 0 and ask_size > 0:
            no.bids.append(BookLevel(1.0 - yes_ask, ask_size))
        if yes_bid > 0 and bid_size > 0:
            no.asks.append(BookLevel(1.0 - yes_bid, bid_size))

        return Contract(
            venue=self.venue,
            market_id=ticker,
            group_id=raw.get("event_ticker", event.get("event_ticker", "")),
            title=raw.get("title", ""),
            outcome_label=(
                raw.get("yes_sub_title") or raw.get("title") or ticker
            ),
            yes=yes,
            no=no,
            close_time=_parse_time(raw.get("close_time")),
            volume_24h=_f(raw.get("volume_24h_fp")),
            liquidity=_f(raw.get("liquidity_dollars")),
            min_order_size=1.0,
            tick_size=0.01,
            shares_book=True,          # the defining Kalshi property
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------


class PolymarketAdapter:
    """
    Public Polymarket data: gamma for structure, CLOB for depth.

    Gamma's ``outcomePrices`` are indicative, not executable. Every price used
    for an arbitrage decision comes from the CLOB book endpoint, because a
    midpoint that cannot be traded at is not a price.
    """

    venue = Venue.POLYMARKET

    def __init__(self, *, timeout: float = 30.0, fetch_books: bool = True,
                 max_books: int = 60) -> None:
        self.timeout = timeout
        self.fetch_books = fetch_books
        self.max_books = max_books
        self._books_fetched = 0

    def fetch_groups(self, *, limit: int = 40, min_volume: float = 0.0,
                     order: str = "volume24hr") -> list[MarketGroup]:
        url = (
            f"{POLY_GAMMA}/events?closed=false&limit={limit}"
            f"&order={order}&ascending=false"
        )
        try:
            payload = _get_json(url, timeout=self.timeout)
        except Exception as exc:
            logger.warning("polymarket events fetch failed: %s", exc)
            return []

        events = payload if isinstance(payload, list) else payload.get("data", [])
        groups: list[MarketGroup] = []
        for raw in events:
            group = self._to_group(raw)
            if group is None:
                continue
            if min_volume and group.total_volume_24h < min_volume:
                continue
            groups.append(group)
        return groups

    def _to_group(self, raw: dict[str, Any]) -> MarketGroup | None:
        markets = raw.get("markets") or []
        if not markets:
            return None

        contracts = []
        for market in markets:
            contract = self._to_contract(market, raw)
            if contract is not None:
                contracts.append(contract)
        if not contracts:
            return None

        # negRisk is Polymarket's own flag for a mutually exclusive outcome
        # set -- their "negative risk" markets, where holding NO on every
        # outcome is a bounded position.
        exclusive = any(bool(m.get("negRisk")) for m in markets)

        group = MarketGroup(
            venue=self.venue,
            group_id=str(raw.get("id", "")),
            title=raw.get("title", ""),
            contracts=contracts,
            mutually_exclusive=exclusive,
            close_time=_parse_time(raw.get("endDate")),
            raw={k: v for k, v in raw.items() if k != "markets"},
        )
        group.exhaustive_evidence, group.exhaustive_note = infer_exhaustiveness(group)
        return group

    def _to_contract(self, raw: dict[str, Any],
                     event: dict[str, Any]) -> Contract | None:
        token_ids = raw.get("clobTokenIds")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except json.JSONDecodeError:
                token_ids = None
        if not token_ids or len(token_ids) < 2:
            return None

        contract = Contract(
            venue=self.venue,
            market_id=str(raw.get("conditionId") or raw.get("id") or ""),
            group_id=str(event.get("id", "")),
            title=raw.get("question", ""),
            outcome_label=raw.get("groupItemTitle") or raw.get("question", ""),
            close_time=_parse_time(raw.get("endDate")),
            volume_24h=_f(raw.get("volume24hr")),
            liquidity=_f(raw.get("liquidityNum") or raw.get("liquidity")),
            min_order_size=_f(raw.get("orderMinSize"), 5.0),
            tick_size=_f(raw.get("orderPriceMinTickSize"), 0.01),
            shares_book=False,         # separate token books; arb is possible
            raw=raw,
        )

        if self.fetch_books and self._books_fetched < self.max_books:
            contract.yes = self._book(str(token_ids[0]))
            contract.no = self._book(str(token_ids[1]))
            self._books_fetched += 2
        else:
            # Fall back to the indicative quotes, clearly marked by having a
            # single level with unknown size. The arbitrage layer requires
            # real sizes, so these cannot produce a tradeable opportunity.
            best_bid, best_ask = _f(raw.get("bestBid")), _f(raw.get("bestAsk"))
            if best_bid > 0:
                contract.yes.bids.append(BookLevel(best_bid, 0.0))
            if best_ask > 0:
                contract.yes.asks.append(BookLevel(best_ask, 0.0))

        return contract

    def _book(self, token_id: str) -> Book:
        try:
            payload = _get_json(
                f"{POLY_CLOB}/book?token_id={token_id}", timeout=self.timeout,
            )
        except Exception as exc:
            logger.debug("polymarket book fetch failed for %s: %s", token_id, exc)
            return Book()

        book = Book()
        for level in payload.get("bids") or []:
            price, size = _f(level.get("price")), _f(level.get("size"))
            if price > 0 and size > 0:
                book.bids.append(BookLevel(price, size))
        for level in payload.get("asks") or []:
            price, size = _f(level.get("price")), _f(level.get("size"))
            if price > 0 and size > 0:
                book.asks.append(BookLevel(price, size))

        # The CLOB returns levels in ascending price order on both sides.
        # Bids must descend for `best_bid` and the walk in `cost_to_fill` to
        # mean what they say.
        book.bids.sort(key=lambda level: -level.price)
        book.asks.sort(key=lambda level: level.price)
        return book


# ---------------------------------------------------------------------------


def load_groups(
    *, venues: Sequence[str] = (Venue.KALSHI, Venue.POLYMARKET),
    kalshi_pages: int = 2, polymarket_limit: int = 25,
    min_volume: float = 0.0, fetch_books: bool = True,
) -> list[MarketGroup]:
    """Fetch from every requested venue, tolerating one of them being down."""
    groups: list[MarketGroup] = []
    if Venue.KALSHI in venues:
        groups.extend(
            KalshiAdapter().fetch_groups(pages=kalshi_pages, min_volume=min_volume)
        )
    if Venue.POLYMARKET in venues:
        groups.extend(
            PolymarketAdapter(fetch_books=fetch_books).fetch_groups(
                limit=polymarket_limit, min_volume=min_volume,
            )
        )
    return groups
