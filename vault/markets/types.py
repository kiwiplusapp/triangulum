"""
Prediction-market primitives.

## Why this is a separate world from Triangulum's order books

A prediction market contract pays exactly $1 if an event happens and $0 if it
does not. That single fact changes the arithmetic completely:

    - A price IS a probability. 0.27 means the market says 27%.
    - Complementary contracts have a hard identity: YES + NO = $1 at
      settlement, always, with no basis risk and no funding.
    - Arbitrage is therefore *checkable in closed form* rather than estimated.
      There is no model risk in "I paid $0.96 for something that pays $1".

That last property is why this is the right instrument for $200. The edge does
not depend on a forecast being right. It depends on arithmetic being right,
and on the trade actually filling.

## The distinction that decides everything: exclusive vs exhaustive

Two different structural facts, constantly conflated, with opposite
consequences:

    MUTUALLY EXCLUSIVE   at most one outcome resolves YES
    EXHAUSTIVE           at least one outcome resolves YES

Kalshi's `mutually_exclusive` flag and Polymarket's `negRisk` flag both assert
the first. **Neither asserts the second**, and the difference is the single
easiest way to report fake arbitrage with total confidence.

Live example from Kalshi at the time of writing: "Who will the next Pope be?"
listed 7 candidates whose YES asks summed to $0.288. Buying all seven looks
like paying 29c for a guaranteed $1. It is not: there are more than seven
possible popes, so the real outcome may be none of the listed ones, and the
basket pays zero. The sum being far below $1 is *evidence of
non-exhaustiveness*, not evidence of free money.

See :mod:`vault.markets.arbitrage` for how the two sides of the trade are
gated differently as a result -- the sell side survives without
exhaustiveness, the buy side does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

__all__ = [
    "Venue", "Side", "BookLevel", "Book", "Contract", "MarketGroup",
    "ExhaustiveEvidence",
]


class Venue:
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class Side:
    YES = "yes"
    NO = "no"


@dataclass(slots=True, frozen=True)
class BookLevel:
    price: float          # dollars per contract, in [0, 1]
    size: float           # contracts available at this price

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass(slots=True)
class Book:
    """One side's ladder. Bids descend, asks ascend."""

    bids: list[BookLevel] = field(default_factory=list)
    asks: list[BookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def depth_to(self, limit_price: float, *, buying: bool) -> float:
        """
        Contracts fillable without crossing past ``limit_price``.

        This is the number that decides whether an opportunity is real or a
        screenshot. A 3-cent edge on 4 contracts is $0.12 before fees, which
        does not cover the effort of placing two orders, and reporting it
        alongside a genuinely fillable trade trains you to ignore the list.
        """
        levels = self.asks if buying else self.bids
        total = 0.0
        for level in levels:
            if buying and level.price > limit_price:
                break
            if not buying and level.price < limit_price:
                break
            total += level.size
        return total

    def cost_to_fill(self, contracts: float, *, buying: bool) -> tuple[float, float]:
        """
        Walk the book. Returns ``(total_cost, contracts_actually_filled)``.

        Walks the ladder rather than assuming the touch price holds, because
        prediction-market books are thin and the second level is often several
        cents away. Sizing at the touch and reporting that as the edge is how
        a scanner produces opportunities that evaporate on execution.
        """
        levels = self.asks if buying else self.bids
        remaining = contracts
        cost = 0.0
        for level in levels:
            if remaining <= 0:
                break
            take = min(remaining, level.size)
            cost += take * level.price
            remaining -= take
        return cost, contracts - remaining


@dataclass(slots=True)
class Contract:
    """One binary contract, with whatever depth the venue exposed."""

    venue: str
    market_id: str            # venue-native identifier for placing an order
    group_id: str             # the event/series this belongs to
    title: str
    outcome_label: str        # "Kaja Kallas", "Yes", "Over 12.5"

    yes: Book = field(default_factory=Book)
    no: Book = field(default_factory=Book)

    close_time: datetime | None = None
    volume_24h: float = 0.0
    liquidity: float = 0.0
    min_order_size: float = 1.0
    tick_size: float = 0.01

    # Kalshi derives the NO book from the YES book (no_ask = 1 - yes_bid), so
    # the two are the same liquidity seen from two directions. Polymarket
    # tokenises YES and NO separately, so they are genuinely independent books
    # that can disagree. `shares_book` records which, because the single-market
    # arbitrage check is only meaningful when it is False.
    shares_book: bool = True

    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_quotes(self) -> bool:
        return bool(self.yes.asks or self.yes.bids)

    @property
    def implied_probability(self) -> float | None:
        """Mid of the YES book, which is the market's probability estimate."""
        bid, ask = self.yes.best_bid, self.yes.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "market_id": self.market_id,
            "group_id": self.group_id,
            "title": self.title,
            "outcome_label": self.outcome_label,
            "yes_bid": self.yes.best_bid,
            "yes_ask": self.yes.best_ask,
            "no_bid": self.no.best_bid,
            "no_ask": self.no.best_ask,
            "implied_probability": self.implied_probability,
            "volume_24h": round(self.volume_24h, 2),
            "liquidity": round(self.liquidity, 2),
            "min_order_size": self.min_order_size,
            "shares_book": self.shares_book,
            "close_time": self.close_time.isoformat() if self.close_time else None,
        }


class ExhaustiveEvidence:
    """
    How confident we are that a group's listed outcomes cover every case.

    Deliberately an explicit enum rather than a boolean, because the honest
    answer is usually "nobody told us" and a boolean forces that into a lie in
    one direction or the other.
    """

    # The venue explicitly models the group as covering the full space --
    # Kalshi binary YES/NO on a single question, for instance.
    ASSERTED = "asserted"

    # The structure strongly implies it: a two-outcome group whose labels are
    # complementary, or a group whose prices sum close to 1 with tight spreads.
    INFERRED = "inferred"

    # Mutually exclusive per the venue, but nothing says the list is complete.
    # This is the common case for "who will be the next X" markets and it is
    # the one that must never be traded on the buy side.
    UNKNOWN = "unknown"

    # Positively known not to be exhaustive -- an explicit "Other"/"Field"
    # outcome is absent while the candidate list is obviously partial.
    DENIED = "denied"


@dataclass(slots=True)
class MarketGroup:
    """
    A set of contracts that resolve together -- one event.

    ``mutually_exclusive`` comes from the venue. ``exhaustive_evidence`` is
    ours, and is deliberately pessimistic by default.
    """

    venue: str
    group_id: str
    title: str
    contracts: list[Contract] = field(default_factory=list)
    mutually_exclusive: bool = False
    exhaustive_evidence: str = ExhaustiveEvidence.UNKNOWN
    exhaustive_note: str = ""
    close_time: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.contracts)

    @property
    def quoted(self) -> list[Contract]:
        return [c for c in self.contracts if c.has_quotes]

    @property
    def is_exhaustive(self) -> bool:
        """Only the two positive verdicts count. UNKNOWN is treated as no."""
        return self.exhaustive_evidence in (
            ExhaustiveEvidence.ASSERTED, ExhaustiveEvidence.INFERRED,
        )

    @property
    def sum_yes_ask(self) -> float | None:
        prices = [c.yes.best_ask for c in self.contracts]
        return sum(p for p in prices if p is not None) if all(
            p is not None for p in prices
        ) and prices else None

    @property
    def sum_yes_bid(self) -> float | None:
        prices = [c.yes.best_bid for c in self.contracts]
        return sum(p for p in prices if p is not None) if all(
            p is not None for p in prices
        ) and prices else None

    @property
    def total_volume_24h(self) -> float:
        return sum(c.volume_24h for c in self.contracts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "group_id": self.group_id,
            "title": self.title,
            "outcomes": len(self.contracts),
            "quoted": len(self.quoted),
            "mutually_exclusive": self.mutually_exclusive,
            "exhaustive_evidence": self.exhaustive_evidence,
            "exhaustive_note": self.exhaustive_note,
            "sum_yes_ask": (
                round(self.sum_yes_ask, 4) if self.sum_yes_ask is not None else None
            ),
            "sum_yes_bid": (
                round(self.sum_yes_bid, 4) if self.sum_yes_bid is not None else None
            ),
            "volume_24h": round(self.total_volume_24h, 2),
            "close_time": self.close_time.isoformat() if self.close_time else None,
        }


_LOWER_TAIL = re.compile(
    r"\b(or\s+(below|less|lower|under|fewer)|below|under|less\s+than|"
    r"at\s+most|no\s+more\s+than|<=?)\b", re.I)
_UPPER_TAIL = re.compile(
    r"\b(or\s+(above|more|higher|over|greater)|above|over|more\s+than|"
    r"at\s+least|greater\s+than|>=?)\b", re.I)
_CATCH_ALL = re.compile(
    r"\b(other|another|someone\s+else|any\s+other|none\s+of|field|"
    r"no\s+one|nobody|neither)\b", re.I)


def _covers_the_line(labels: Sequence[str]) -> bool:
    """
    True when the outcome labels tile a numeric range with both tails open.

    A set like "0.0% or Below", "0.1% to 0.5%", ..., "6.1% or Above" covers
    every real number by construction: there is an unbounded bucket at each
    end and the middle is partitioned. That is a STRUCTURAL fact about the
    labels, which is exactly the kind of evidence worth acting on -- unlike a
    guess from prices.

    This matters because the price-based heuristic gets these badly wrong. A
    Kalshi market on 2034 GDP growth had 14 buckets tiling the line from
    "0.0% or Below" to "6.1% or Above" -- genuinely exhaustive -- while its
    YES bids summed to 0.52, because a market nine years out is quoted wide
    and thin. Reading that as "outcomes are missing" confuses ILLIQUID with
    INCOMPLETE and throws away a real opportunity.
    """
    if len(labels) < 3:
        return False
    has_lower = any(_LOWER_TAIL.search(label or "") for label in labels)
    has_upper = any(_UPPER_TAIL.search(label or "") for label in labels)
    if not (has_lower and has_upper):
        return False
    # At least half the labels should look like ranges or numbers, so a
    # candidate list that happens to contain the word "over" does not qualify.
    numeric = sum(
        1 for label in labels if re.search(r"\d", label or "")
    )
    return numeric >= max(3, len(labels) // 2)


def infer_exhaustiveness(group: MarketGroup) -> tuple[str, str]:
    """
    Decide, conservatively, whether a group's outcomes cover the whole space.

    The rules, in order:

    1. A single binary contract with a real YES and NO side is exhaustive by
       construction -- the event happens or it does not.

    2. A two-outcome mutually exclusive group is exhaustive if its asks bracket
       $1 sensibly. Two outcomes that are mutually exclusive and priced to sum
       near 1 are almost certainly complementary.

    3. Everything else with many outcomes is UNKNOWN, unless the prices give
       positive evidence AGAINST it. A mutually exclusive group whose YES asks
       sum well below $1 is not a bargain -- it is a group missing outcomes.
       That inference is the important one and it is why this returns DENIED
       rather than getting excited.

    The asymmetry is deliberate. Being wrong about exhaustiveness costs money
    only on the buy side, so the buy side is the side that has to prove it.
    """
    contracts = group.contracts
    if len(contracts) == 1:
        contract = contracts[0]
        if contract.yes.asks and contract.no.asks:
            return (
                ExhaustiveEvidence.ASSERTED,
                "single binary contract: the event resolves yes or no, so the "
                "two sides cover the space by construction",
            )
        return (
            ExhaustiveEvidence.UNKNOWN,
            "single contract without both sides quoted",
        )

    if not group.mutually_exclusive:
        return (
            ExhaustiveEvidence.UNKNOWN,
            "the venue does not mark this group mutually exclusive, so the "
            "outcomes may overlap as well as under-cover",
        )

    labels = [c.outcome_label for c in contracts]

    # Structural evidence beats price evidence and is checked first.
    if _covers_the_line(labels):
        return (
            ExhaustiveEvidence.INFERRED,
            f"the {len(contracts)} outcome labels tile a numeric range with "
            f"an open bucket at each end, so they cover every possible value "
            f"by construction",
        )

    if any(_CATCH_ALL.search(label or "") for label in labels):
        return (
            ExhaustiveEvidence.INFERRED,
            "the outcome set includes an explicit catch-all bucket, which "
            "closes the space",
        )

    total_ask = group.sum_yes_ask
    total_bid = group.sum_yes_bid

    if len(contracts) == 2 and total_ask is not None and 0.95 <= total_ask <= 1.15:
        return (
            ExhaustiveEvidence.INFERRED,
            f"two mutually exclusive outcomes whose asks sum to "
            f"{total_ask:.3f}, consistent with a complementary pair",
        )

    if total_bid is not None and total_bid < 0.80:
        return (
            ExhaustiveEvidence.DENIED,
            f"YES bids across {len(contracts)} mutually exclusive outcomes sum "
            f"to only {total_bid:.3f}. A complete set cannot be worth less "
            f"than $1 in aggregate, so outcomes are missing from this list -- "
            f"this is evidence AGAINST exhaustiveness, not a cheap basket",
        )

    return (
        ExhaustiveEvidence.UNKNOWN,
        f"{len(contracts)} mutually exclusive outcomes, but nothing states "
        f"that they are exhaustive; treated as incomplete",
    )
