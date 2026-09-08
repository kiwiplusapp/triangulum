"""
Arbitrage detection across prediction markets.

## The three checks

**1. Complementary (single market).** Buy YES and NO on the same question. One
of them pays $1. If the two asks sum to less than $1 after fees, that is a
locked profit with no forecast in it.

Only meaningful where YES and NO have *separate books*. On Kalshi they do not:
``no_ask == 1 - yes_bid`` identically, so ``yes_ask + no_ask == 1 + spread``,
which is never below 1. The check is skipped there rather than run and always
failing, and ``Contract.shares_book`` is what decides.

**2. Mutually exclusive basket.** A group where at most one outcome resolves
YES. Two directions, and they are **not symmetric**:

    SELL side -- sell YES on every outcome (equivalently buy NO on every one).
        Payout is n-1 if exactly one resolves YES, n if none does. Minimum
        payout n-1, cost n - sum(bids). Profit >= sum(bids) - 1.
        => Guaranteed whenever sum(yes_bid) > 1. Needs mutual exclusivity ONLY.

    BUY side -- buy YES on every outcome for sum(asks).
        Payout is $1 if one resolves YES, and ZERO if none does.
        => Guaranteed only if the outcome set is EXHAUSTIVE.

That asymmetry is the whole game. Live Kalshi data at the time of writing had
"Who will the next Pope be?" with seven candidates whose YES asks summed to
$0.288. Buying all seven looks like 29 cents for a certain dollar. It is not:
there are more than seven possible popes. A sum far below $1 is *evidence the
list is incomplete*, not evidence of free money -- and the sell side, which
needs no such assumption, is where the durable edge lives.

**3. Cross-venue.** The same question priced on both venues. Buy the cheap
YES, buy the cheap NO on the other venue; the pair pays $1 exactly once.
Sound in principle and dangerous in practice, because two questions that read
alike can resolve differently -- different sources, different cutoffs,
different treatment of an ambiguous outcome. These are reported as
**candidates requiring manual confirmation of the resolution criteria**, never
as locked arbitrage, and the resolution text of both sides travels with the
opportunity so it can actually be checked.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from vault.markets.fees import FeeModel, fee_model_for
from vault.markets.types import Contract, ExhaustiveEvidence, MarketGroup, Venue

logger = logging.getLogger(__name__)

__all__ = [
    "Opportunity", "Leg", "find_complementary", "find_basket",
    "find_cross_venue", "scan_groups", "MIN_EDGE_DOLLARS",
]

# An opportunity worth less than this in total is noise: two orders, two fills,
# settlement risk and attention for less than the price of a coffee. Reporting
# them trains you to skim the list, which is how the real one gets missed.
MIN_EDGE_DOLLARS = 0.50

# Below this many contracts the fee rounding on Kalshi (up to the cent) eats
# most of a small edge, and the fill risk on a thin book is high.
MIN_CONTRACTS = 5.0


@dataclass(slots=True)
class Leg:
    """One order that would need to be placed."""

    venue: str
    market_id: str
    label: str
    side: str                 # "yes" or "no"
    action: str               # "buy" or "sell"
    price: float              # average price after walking the book
    contracts: float

    @property
    def notional(self) -> float:
        return self.price * self.contracts

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "market_id": self.market_id,
            "label": self.label,
            "side": self.side,
            "action": self.action,
            "price": round(self.price, 4),
            "contracts": round(self.contracts, 2),
            "notional": round(self.notional, 2),
        }


@dataclass(slots=True)
class Opportunity:
    """A candidate trade, with everything needed to judge and place it."""

    kind: str                     # complementary | basket_sell | basket_buy | cross_venue
    title: str
    venue: str
    group_id: str
    legs: list[Leg] = field(default_factory=list)

    contracts: float = 0.0
    cost: float = 0.0             # total outlay
    guaranteed_return: float = 0.0
    fees: float = 0.0

    # Whether the payoff is locked by arithmetic, or depends on an assumption
    # that has not been verified. Anything False is a research lead, not a trade.
    locked: bool = False
    assumptions: list[str] = field(default_factory=list)
    fee_models: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    @property
    def gross_profit(self) -> float:
        return self.guaranteed_return - self.cost

    @property
    def net_profit(self) -> float:
        return self.gross_profit - self.fees

    @property
    def return_on_cost(self) -> float:
        return self.net_profit / self.cost if self.cost > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "venue": self.venue,
            "group_id": self.group_id,
            "contracts": round(self.contracts, 2),
            "cost": round(self.cost, 2),
            "guaranteed_return": round(self.guaranteed_return, 2),
            "gross_profit": round(self.gross_profit, 4),
            "fees": round(self.fees, 4),
            "net_profit": round(self.net_profit, 4),
            "return_on_cost": round(self.return_on_cost, 5),
            "locked": self.locked,
            "assumptions": self.assumptions,
            "legs": [leg.to_dict() for leg in self.legs],
            "fee_models": self.fee_models,
            "notes": self.notes,
        }

    def explain(self) -> str:
        head = (
            f"[{self.kind}] {self.title[:66]}\n"
            f"  {self.contracts:.0f} contracts, cost ${self.cost:.2f} -> "
            f"guaranteed ${self.guaranteed_return:.2f}  "
            f"gross ${self.gross_profit:+.2f}  fees ${self.fees:.2f}  "
            f"NET ${self.net_profit:+.2f} ({self.return_on_cost:+.2%})"
        )
        lines = [head]
        for leg in self.legs:
            lines.append(
                f"    {leg.action:4s} {leg.side:3s} @ {leg.price:.3f} x "
                f"{leg.contracts:6.0f}  {leg.venue:11s} {leg.label[:38]}"
            )
        if not self.locked:
            lines.append("  NOT LOCKED -- depends on:")
            for assumption in self.assumptions:
                lines.append(f"    - {assumption}")
        return "\n".join(lines)


def _capped_contracts(available: Sequence[float], budget: float,
                      unit_cost: float) -> float:
    """Contracts limited by the thinnest leg and by the money on hand."""
    if not available or unit_cost <= 0:
        return 0.0
    by_depth = min(available)
    by_budget = budget / unit_cost
    return max(0.0, min(by_depth, by_budget))


# ---------------------------------------------------------------------------
# 1. complementary
# ---------------------------------------------------------------------------


def find_complementary(contract: Contract, *, budget: float = 200.0,
                       fee_model: FeeModel | None = None) -> Opportunity | None:
    """
    Buy YES and NO on the same question for less than $1 combined.

    Skipped entirely when the two sides share a book, because there the sum is
    ``1 + spread`` by construction and any "opportunity" found would be an
    artefact of the adapter rather than a fact about the market.
    """
    if contract.shares_book:
        return None

    yes_ask, no_ask = contract.yes.best_ask, contract.no.best_ask
    if yes_ask is None or no_ask is None:
        return None
    unit = yes_ask + no_ask
    if unit >= 1.0:
        return None

    fees = fee_model or fee_model_for(contract.venue)
    depth = [
        contract.yes.depth_to(yes_ask, buying=True),
        contract.no.depth_to(no_ask, buying=True),
    ]
    contracts = _capped_contracts(depth, budget, unit)
    contracts = min(contracts, budget / unit)
    if contracts < max(MIN_CONTRACTS, contract.min_order_size):
        return None

    yes_cost, yes_filled = contract.yes.cost_to_fill(contracts, buying=True)
    no_cost, no_filled = contract.no.cost_to_fill(contracts, buying=True)
    contracts = min(yes_filled, no_filled)
    if contracts < max(MIN_CONTRACTS, contract.min_order_size):
        return None

    # Re-cost at the achievable size so the reported prices are the ones that
    # would actually be paid, not the touch.
    yes_cost, _ = contract.yes.cost_to_fill(contracts, buying=True)
    no_cost, _ = contract.no.cost_to_fill(contracts, buying=True)
    cost = yes_cost + no_cost

    total_fees = (
        fees.cost(contracts, yes_cost / contracts)
        + fees.cost(contracts, no_cost / contracts)
    )

    opportunity = Opportunity(
        kind="complementary",
        title=contract.title,
        venue=contract.venue,
        group_id=contract.group_id,
        contracts=contracts,
        cost=cost,
        guaranteed_return=contracts * 1.0,
        fees=total_fees,
        locked=True,
        legs=[
            Leg(contract.venue, contract.market_id, contract.outcome_label,
                "yes", "buy", yes_cost / contracts, contracts),
            Leg(contract.venue, contract.market_id, contract.outcome_label,
                "no", "buy", no_cost / contracts, contracts),
        ],
        fee_models=[fees.to_dict()],
        notes=(
            "YES and NO on the same question, bought together. Exactly one "
            "settles at $1, so the payoff is arithmetic rather than a forecast."
        ),
    )
    if not fees.verified:
        opportunity.locked = False
        opportunity.assumptions.append(
            f"the {contract.venue} fee model is UNVERIFIED: {fees.formula}. "
            f"Gross edge is ${opportunity.gross_profit:.2f}; if real fees "
            f"exceed that, this is a loss."
        )
    return opportunity if opportunity.net_profit >= MIN_EDGE_DOLLARS else None


# ---------------------------------------------------------------------------
# 2. mutually exclusive basket
# ---------------------------------------------------------------------------


def find_basket(group: MarketGroup, *, budget: float = 200.0,
                fee_model: FeeModel | None = None) -> list[Opportunity]:
    """
    The two basket trades. See the module docstring for why they differ.

    Returns both when both qualify. The sell side is the one that survives an
    incomplete outcome list; the buy side is gated on exhaustiveness and is
    reported unlocked when that cannot be established.
    """
    if not group.mutually_exclusive or len(group.contracts) < 2:
        return []
    quoted = [c for c in group.contracts if c.has_quotes]
    if len(quoted) != len(group.contracts):
        # A basket is only a basket if every outcome can be traded. A missing
        # leg means an unhedged hole exactly where the payoff assumption lives.
        return []

    fees = fee_model or fee_model_for(group.venue)
    found: list[Opportunity] = []

    # ---- sell side: sum(yes_bid) > 1 ----
    bids = [c.yes.best_bid for c in quoted]
    if all(b is not None for b in bids):
        total_bid = sum(bids)
        if total_bid > 1.0:
            n = len(quoted)
            # Selling YES on Kalshi means buying NO at (1 - yes_bid); the cost
            # per basket is n - sum(bids) and the worst-case payout is n - 1.
            unit_cost = n - total_bid
            depth = [c.yes.depth_to(c.yes.best_bid, buying=False) for c in quoted]
            contracts = _capped_contracts(depth, budget, max(unit_cost, 1e-9))
            if contracts >= MIN_CONTRACTS:
                cost = contracts * unit_cost
                total_fees = sum(
                    fees.cost(contracts, 1.0 - (c.yes.best_bid or 0.0))
                    for c in quoted
                )
                opportunity = Opportunity(
                    kind="basket_sell",
                    title=group.title,
                    venue=group.venue,
                    group_id=group.group_id,
                    contracts=contracts,
                    cost=cost,
                    guaranteed_return=contracts * (n - 1),
                    fees=total_fees,
                    locked=True,
                    legs=[
                        Leg(c.venue, c.market_id, c.outcome_label, "no", "buy",
                            1.0 - (c.yes.best_bid or 0.0), contracts)
                        for c in quoted
                    ],
                    fee_models=[fees.to_dict()],
                    notes=(
                        f"YES bids across {n} mutually exclusive outcomes sum to "
                        f"{total_bid:.4f}. Selling all of them (buying every NO) "
                        f"pays at least {n - 1} per basket whether or not the "
                        f"outcome list is complete -- this direction does NOT "
                        f"require exhaustiveness."
                    ),
                )
                if not fees.verified:
                    opportunity.locked = False
                    opportunity.assumptions.append(
                        f"the {group.venue} fee model is UNVERIFIED: {fees.formula}"
                    )
                if opportunity.net_profit >= MIN_EDGE_DOLLARS:
                    found.append(opportunity)

    # ---- buy side: sum(yes_ask) < 1, and ONLY if exhaustive ----
    asks = [c.yes.best_ask for c in quoted]
    if all(a is not None for a in asks):
        total_ask = sum(asks)
        if total_ask < 1.0:
            depth = [c.yes.depth_to(c.yes.best_ask, buying=True) for c in quoted]
            contracts = _capped_contracts(depth, budget, total_ask)
            if contracts >= MIN_CONTRACTS:
                cost = contracts * total_ask
                total_fees = sum(
                    fees.cost(contracts, c.yes.best_ask or 0.0) for c in quoted
                )
                opportunity = Opportunity(
                    kind="basket_buy",
                    title=group.title,
                    venue=group.venue,
                    group_id=group.group_id,
                    contracts=contracts,
                    cost=cost,
                    guaranteed_return=contracts * 1.0,
                    fees=total_fees,
                    locked=group.is_exhaustive,
                    legs=[
                        Leg(c.venue, c.market_id, c.outcome_label, "yes", "buy",
                            c.yes.best_ask or 0.0, contracts)
                        for c in quoted
                    ],
                    fee_models=[fees.to_dict()],
                    notes=(
                        f"YES asks across {len(quoted)} outcomes sum to "
                        f"{total_ask:.4f}."
                    ),
                )
                if not group.is_exhaustive:
                    opportunity.assumptions.append(
                        f"EXHAUSTIVENESS IS NOT ESTABLISHED "
                        f"({group.exhaustive_evidence}): {group.exhaustive_note}. "
                        f"If the real outcome is not among these "
                        f"{len(quoted)}, the basket pays ZERO and the whole "
                        f"${cost:.2f} is lost. A sum well below $1 is usually "
                        f"evidence the list is incomplete, not a bargain."
                    )
                if not fees.verified:
                    opportunity.assumptions.append(
                        f"the {group.venue} fee model is UNVERIFIED: {fees.formula}"
                    )
                    opportunity.locked = False
                if opportunity.net_profit >= MIN_EDGE_DOLLARS:
                    found.append(opportunity)

    return found


# ---------------------------------------------------------------------------
# 3. cross venue
# ---------------------------------------------------------------------------

_STOP = frozenset({
    "will", "the", "be", "a", "an", "of", "in", "on", "to", "for", "by", "at",
    "is", "are", "and", "or", "who", "what", "when", "before", "after", "next",
    "this", "that", "than", "with", "from", "any", "how", "many", "much",
})


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap on content words. Crude on purpose -- see below."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def find_cross_venue(
    contracts_a: Sequence[Contract], contracts_b: Sequence[Contract],
    *, budget: float = 200.0, min_similarity: float = 0.55,
) -> list[Opportunity]:
    """
    Same question, two venues, prices that disagree enough to lock a profit.

    Buy YES on the venue where YES is cheap and NO on the venue where NO is
    cheap. If the two prices sum below $1, exactly one pays out and the
    difference is profit.

    **Every result is unlocked.** Matching is done on word overlap, which is
    a heuristic and nothing more: two questions can share every keyword and
    still resolve differently because they name different sources, different
    cutoff times, or different handling of an ambiguous case. The resolution
    text of both sides is attached so the criteria can actually be compared
    before any money moves. Treating a fuzzy title match as a hedge is the
    fastest way to turn "arbitrage" into two uncorrelated directional bets.
    """
    found: list[Opportunity] = []
    for left in contracts_a:
        if not left.has_quotes:
            continue
        for right in contracts_b:
            if not right.has_quotes:
                continue
            score = similarity(left.title, right.title)
            if score < min_similarity:
                continue

            for buy_yes, buy_no in ((left, right), (right, left)):
                yes_ask = buy_yes.yes.best_ask
                no_ask = buy_no.no.best_ask
                if yes_ask is None or no_ask is None:
                    continue
                unit = yes_ask + no_ask
                if unit >= 1.0:
                    continue

                depth = [
                    buy_yes.yes.depth_to(yes_ask, buying=True),
                    buy_no.no.depth_to(no_ask, buying=True),
                ]
                contracts = _capped_contracts(depth, budget, unit)
                if contracts < MIN_CONTRACTS:
                    continue

                fees_yes = fee_model_for(buy_yes.venue)
                fees_no = fee_model_for(buy_no.venue)
                total_fees = (
                    fees_yes.cost(contracts, yes_ask)
                    + fees_no.cost(contracts, no_ask)
                )
                opportunity = Opportunity(
                    kind="cross_venue",
                    title=f"{buy_yes.title[:44]} | {buy_no.title[:44]}",
                    venue=f"{buy_yes.venue}+{buy_no.venue}",
                    group_id=f"{buy_yes.market_id}~{buy_no.market_id}",
                    contracts=contracts,
                    cost=contracts * unit,
                    guaranteed_return=contracts * 1.0,
                    fees=total_fees,
                    locked=False,
                    legs=[
                        Leg(buy_yes.venue, buy_yes.market_id,
                            buy_yes.outcome_label, "yes", "buy", yes_ask, contracts),
                        Leg(buy_no.venue, buy_no.market_id,
                            buy_no.outcome_label, "no", "buy", no_ask, contracts),
                    ],
                    fee_models=[fees_yes.to_dict(), fees_no.to_dict()],
                    assumptions=[
                        f"the two questions were matched on word overlap "
                        f"({score:.0%}), NOT on resolution criteria. Confirm by "
                        f"hand that both resolve on the same event, the same "
                        f"source and the same cutoff before trading.",
                        f"'{buy_yes.title[:70]}' ({buy_yes.venue})",
                        f"'{buy_no.title[:70]}' ({buy_no.venue})",
                    ],
                    notes=(
                        f"YES at {yes_ask:.3f} on {buy_yes.venue} plus NO at "
                        f"{no_ask:.3f} on {buy_no.venue} = {unit:.3f}."
                    ),
                )
                if opportunity.net_profit >= MIN_EDGE_DOLLARS:
                    found.append(opportunity)
    return found


# ---------------------------------------------------------------------------


def scan_groups(groups: Sequence[MarketGroup], *, budget: float = 200.0,
                cross_venue: bool = True) -> list[Opportunity]:
    """Run every check over every group and return what survived."""
    found: list[Opportunity] = []

    for group in groups:
        for contract in group.contracts:
            opportunity = find_complementary(contract, budget=budget)
            if opportunity:
                found.append(opportunity)
        found.extend(find_basket(group, budget=budget))

    if cross_venue:
        by_venue: dict[str, list[Contract]] = {}
        for group in groups:
            for contract in group.contracts:
                by_venue.setdefault(contract.venue, []).append(contract)
        venues = sorted(by_venue)
        for i, left in enumerate(venues):
            for right in venues[i + 1:]:
                found.extend(
                    find_cross_venue(by_venue[left], by_venue[right], budget=budget)
                )

    # Locked first, then by net profit. A guaranteed $2 outranks a speculative
    # $50, because the speculative one is not an arbitrage.
    found.sort(key=lambda o: (not o.locked, -o.net_profit))
    return found
