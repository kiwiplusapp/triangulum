"""
Prediction-market arbitrage tests.

The load-bearing property here is a refusal, not a discovery:

    **The scanner must not report a basket as locked arbitrage unless the
    outcome set provably covers every case.**

Scanning 4,953 live contracts produced eight candidate baskets, the largest
showing a "+706% guaranteed return" on "What will be the 51st state in Trump's
term?" -- four outcomes bought for $0.054 against a $1 payoff. It is not
arbitrage: the most likely outcome by far is that there is no 51st state at
all, and the basket then pays zero. Every one of the eight was correctly
reported unlocked.

The second property is structural and venue-specific: on Kalshi, YES and NO
are one book (``no_ask == 1 - yes_bid``), so the classic "buy both sides under
$1" trade is arithmetically impossible there. A scanner reporting one has a
bug, and ``test_kalshi_complementary_arbitrage_is_impossible`` is what catches
it.
"""

from __future__ import annotations

import pytest

from vault.markets.arbitrage import (
    MIN_CONTRACTS,
    find_basket,
    find_complementary,
    find_cross_venue,
    scan_groups,
    similarity,
)
from vault.markets.fees import KALSHI_FEES, POLYMARKET_FEES, fee_model_for
from vault.markets.types import (
    Book,
    BookLevel,
    Contract,
    ExhaustiveEvidence,
    MarketGroup,
    Venue,
    infer_exhaustiveness,
)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _book(bid: float | None, ask: float | None, size: float = 500.0) -> Book:
    book = Book()
    if bid is not None:
        book.bids.append(BookLevel(bid, size))
    if ask is not None:
        book.asks.append(BookLevel(ask, size))
    return book


def kalshi_contract(label: str, yes_bid: float, yes_ask: float,
                    size: float = 500.0, group: str = "EV") -> Contract:
    """A Kalshi contract, with the NO book derived exactly as the venue does."""
    return Contract(
        venue=Venue.KALSHI, market_id=f"K-{label}", group_id=group,
        title=label, outcome_label=label,
        yes=_book(yes_bid, yes_ask, size),
        no=_book(1 - yes_ask, 1 - yes_bid, size),
        shares_book=True, min_order_size=1.0,
    )


def poly_contract(label: str, *, yes_ask: float, no_ask: float,
                  yes_bid: float = 0.0, no_bid: float = 0.0,
                  size: float = 500.0) -> Contract:
    """A Polymarket contract, whose two sides are independent books."""
    return Contract(
        venue=Venue.POLYMARKET, market_id=f"P-{label}", group_id="PG",
        title=label, outcome_label=label,
        yes=_book(yes_bid or None, yes_ask, size),
        no=_book(no_bid or None, no_ask, size),
        shares_book=False, min_order_size=5.0,
    )


def group_of(contracts, *, exclusive: bool = True,
             venue: str = Venue.KALSHI) -> MarketGroup:
    group = MarketGroup(
        venue=venue, group_id="EV", title="Test event",
        contracts=list(contracts), mutually_exclusive=exclusive,
    )
    group.exhaustive_evidence, group.exhaustive_note = infer_exhaustiveness(group)
    return group


# ---------------------------------------------------------------------------
# the Kalshi structural identity
# ---------------------------------------------------------------------------


def test_kalshi_yes_and_no_are_one_book():
    """no_ask == 1 - yes_bid, exactly. Verified against live data."""
    contract = kalshi_contract("A", 0.40, 0.44)
    assert contract.no.best_ask == pytest.approx(1 - 0.40)
    assert contract.no.best_bid == pytest.approx(1 - 0.44)


def test_kalshi_complementary_arbitrage_is_impossible():
    """
    yes_ask + no_ask == 1 + spread, which is never below 1. The check must be
    skipped rather than run and coincidentally failing.
    """
    for bid, ask in ((0.40, 0.44), (0.01, 0.02), (0.95, 0.99), (0.50, 0.50)):
        contract = kalshi_contract("A", bid, ask)
        assert contract.yes.best_ask + contract.no.best_ask >= 1.0
        assert find_complementary(contract, budget=200.0) is None


def test_polymarket_complementary_arbitrage_is_found():
    """
    Separate token books CAN disagree, and that is the cheapest real trade on
    offer. 0.46 + 0.50 = 0.96 for something that pays $1.
    """
    contract = poly_contract("Q", yes_ask=0.46, no_ask=0.50)
    opportunity = find_complementary(contract, budget=200.0)
    assert opportunity is not None
    assert opportunity.kind == "complementary"
    assert opportunity.gross_profit > 0
    assert len(opportunity.legs) == 2
    assert {leg.side for leg in opportunity.legs} == {"yes", "no"}


def test_a_fair_polymarket_market_yields_nothing():
    assert find_complementary(
        poly_contract("Q", yes_ask=0.52, no_ask=0.50), budget=200.0
    ) is None


# ---------------------------------------------------------------------------
# exclusive vs exhaustive -- the core distinction
# ---------------------------------------------------------------------------


def test_a_candidate_list_is_not_exhaustive():
    """The Pope trap: seven names out of an unbounded field."""
    group = group_of([
        kalshi_contract(name, 0.02, 0.05)
        for name in ("Pizzaballa", "Parolin", "Ambongo", "Tagle")
    ])
    assert not group.is_exhaustive
    assert group.exhaustive_evidence in (
        ExhaustiveEvidence.UNKNOWN, ExhaustiveEvidence.DENIED,
    )


def test_a_cheap_basket_is_evidence_of_missing_outcomes():
    """
    A complete set cannot be worth less than $1 in aggregate. A sum far below
    it means outcomes are absent, which is the opposite of a bargain.
    """
    group = group_of([
        kalshi_contract(name, 0.01, 0.02) for name in ("A", "B", "C", "D")
    ])
    assert group.exhaustive_evidence == ExhaustiveEvidence.DENIED
    assert "missing" in group.exhaustive_note


def test_numeric_ranges_that_tile_the_line_are_exhaustive():
    """
    The false negative the price heuristic produced on real data: a 2034 GDP
    market whose 14 buckets genuinely covered the number line, while its bids
    summed to 0.52 because a market nine years out is quoted thin.
    """
    labels = ["0.0% or Below", "0.1% to 0.5%", "0.6% to 1.0%",
              "1.1% to 1.5%", "1.6% to 2.0%", "6.1% or Above"]
    group = group_of([kalshi_contract(l, 0.02, 0.12) for l in labels])
    assert group.is_exhaustive
    assert group.exhaustive_evidence == ExhaustiveEvidence.INFERRED
    assert "tile" in group.exhaustive_note


def test_an_explicit_catch_all_closes_the_space():
    group = group_of([
        kalshi_contract("Alice", 0.30, 0.34),
        kalshi_contract("Bob", 0.25, 0.29),
        kalshi_contract("Someone else", 0.35, 0.40),
    ])
    assert group.is_exhaustive


def test_a_single_binary_contract_is_exhaustive_by_construction():
    group = group_of([kalshi_contract("Yes", 0.40, 0.44)], exclusive=False)
    assert group.exhaustive_evidence == ExhaustiveEvidence.ASSERTED


def test_a_non_exclusive_group_is_never_assumed_exhaustive():
    group = group_of(
        [kalshi_contract(x, 0.30, 0.34) for x in ("A", "B", "C")],
        exclusive=False,
    )
    assert not group.is_exhaustive


# ---------------------------------------------------------------------------
# the basket asymmetry
# ---------------------------------------------------------------------------


def test_the_sell_side_does_not_require_exhaustiveness():
    """
    Selling every outcome pays at least n-1 whether or not the list is
    complete: if none resolves YES, every short wins. So sum(bid) > 1 is
    enough on its own.
    """
    group = group_of([
        kalshi_contract("A", 0.60, 0.64),
        kalshi_contract("B", 0.55, 0.59),
    ])
    assert group.sum_yes_bid > 1.0
    found = find_basket(group, budget=200.0)
    sells = [o for o in found if o.kind == "basket_sell"]
    assert sells, "an overpriced exclusive basket must be sellable"
    assert sells[0].gross_profit > 0
    assert not group.is_exhaustive       # and it did not need to be
    assert "does NOT require exhaustiveness" in sells[0].notes


def test_the_buy_side_is_unlocked_without_exhaustiveness():
    """The 51st-state trade: enormous apparent return, not arbitrage."""
    group = group_of([
        kalshi_contract(name, 0.004, 0.006)
        for name in ("District of Columbia", "Puerto Rico", "Canada", "Greenland")
    ])
    found = find_basket(group, budget=200.0)
    buys = [o for o in found if o.kind == "basket_buy"]
    assert buys, "the arithmetic should still be surfaced"
    opportunity = buys[0]
    assert not opportunity.locked
    assert any("EXHAUSTIVENESS" in a for a in opportunity.assumptions)
    assert any("pays ZERO" in a for a in opportunity.assumptions)


def test_the_buy_side_locks_when_the_set_provably_covers_everything():
    labels = ["0.0% or Below", "0.1% to 2.0%", "2.1% to 4.0%", "4.1% or Above"]
    group = group_of([kalshi_contract(l, 0.15, 0.20) for l in labels])
    assert group.is_exhaustive
    found = find_basket(group, budget=200.0)
    buys = [o for o in found if o.kind == "basket_buy"]
    assert buys
    # Still unlocked, but now only because the FEE model is unverified --
    # a different and much smaller assumption than the exhaustiveness one.
    assert all("EXHAUSTIVENESS" not in a for a in buys[0].assumptions)
    assert any("fee model is UNVERIFIED" in a for a in buys[0].assumptions)


def test_a_basket_with_an_unquoted_leg_is_skipped():
    """A missing leg is an unhedged hole exactly where the payoff lives."""
    contracts = [kalshi_contract("A", 0.30, 0.34), kalshi_contract("B", 0.30, 0.34)]
    contracts.append(Contract(
        venue=Venue.KALSHI, market_id="K-C", group_id="EV",
        title="C", outcome_label="C",
    ))
    assert find_basket(group_of(contracts), budget=200.0) == []


def test_a_non_exclusive_group_yields_no_basket():
    group = group_of(
        [kalshi_contract("A", 0.60, 0.64), kalshi_contract("B", 0.55, 0.59)],
        exclusive=False,
    )
    assert find_basket(group, budget=200.0) == []


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------


def test_size_is_capped_by_the_thinnest_leg():
    """A basket is only as large as its least liquid outcome."""
    group = group_of([
        kalshi_contract("A", 0.60, 0.64, size=1000),
        kalshi_contract("B", 0.55, 0.59, size=12),
    ])
    found = find_basket(group, budget=10_000.0)
    assert found
    assert found[0].contracts <= 12


def test_size_is_capped_by_the_budget():
    contract = poly_contract("Q", yes_ask=0.40, no_ask=0.50, size=1_000_000)
    opportunity = find_complementary(contract, budget=90.0)
    assert opportunity is not None
    assert opportunity.cost <= 90.0 + 1e-6


def test_walking_the_book_beats_assuming_the_touch_holds():
    """The second level is often several cents away on these venues."""
    contract = poly_contract("Q", yes_ask=0.30, no_ask=0.40)
    contract.yes.asks.append(BookLevel(0.60, 5000))
    cost, filled = contract.yes.cost_to_fill(1000, buying=True)
    assert filled == 1000
    # 500 at 0.30 plus 500 at 0.60, not 1000 at the touch.
    assert cost == pytest.approx(500 * 0.30 + 500 * 0.60)
    assert cost > 1000 * 0.30


def test_a_trade_too_small_to_matter_is_not_reported():
    contract = poly_contract("Q", yes_ask=0.46, no_ask=0.50, size=2)
    assert find_complementary(contract, budget=200.0) is None


# ---------------------------------------------------------------------------
# fees
# ---------------------------------------------------------------------------


def test_kalshi_fees_peak_in_the_middle_and_vanish_at_the_extremes():
    """The p(1-p) shape: most arbitrage legs are cheap, so fees are small."""
    middle = KALSHI_FEES.cost(100, 0.50)
    edge = KALSHI_FEES.cost(100, 0.02)
    assert middle > edge
    assert KALSHI_FEES.cost(100, 0.50) == pytest.approx(0.07 * 100 * 0.25, abs=0.01)


def test_kalshi_fee_rounding_bites_on_tiny_orders():
    """One contract at 50c costs 2c in fees -- 4% of notional."""
    assert KALSHI_FEES.cost(1, 0.50) == pytest.approx(0.02)


def test_every_fee_model_declares_itself_unverified():
    """
    None of these has been checked against a real fill. Saying so in the data
    is what stops an assumption becoming a fact by repetition.
    """
    for model in (KALSHI_FEES, POLYMARKET_FEES):
        assert not model.verified
        assert model.confidence
        assert model.source


def test_an_unknown_venue_gets_a_punitive_fee_not_a_free_one():
    model = fee_model_for("some-new-exchange")
    assert model.proportional_rate >= 0.02
    assert not model.verified


def test_an_unverified_fee_model_unlocks_the_opportunity():
    opportunity = find_complementary(
        poly_contract("Q", yes_ask=0.46, no_ask=0.50), budget=200.0
    )
    assert opportunity is not None
    assert not opportunity.locked
    assert any("UNVERIFIED" in a for a in opportunity.assumptions)


# ---------------------------------------------------------------------------
# cross venue
# ---------------------------------------------------------------------------


def test_cross_venue_matches_are_never_locked():
    """
    Two questions can share every keyword and still resolve differently.
    Treating a title match as a hedge turns arbitrage into two uncorrelated
    directional bets.
    """
    left = [kalshi_contract("Will Bitcoin close above 100000 in December", 0.30, 0.34)]
    right = [poly_contract("Will Bitcoin close above 100000 in December",
                           yes_ask=0.80, no_ask=0.55)]
    found = find_cross_venue(left, right, budget=200.0)
    for opportunity in found:
        assert not opportunity.locked
        assert any("resolution criteria" in a or "resolve on the same event" in a
                   for a in opportunity.assumptions)


def test_dissimilar_questions_are_not_matched():
    left = [kalshi_contract("Will it rain in Seattle tomorrow", 0.30, 0.34)]
    right = [poly_contract("Who wins the French presidential election",
                           yes_ask=0.20, no_ask=0.20)]
    assert find_cross_venue(left, right, budget=200.0) == []


def test_similarity_ignores_filler_words():
    high = similarity("Will Bitcoin close above 100000 in December",
                      "Bitcoin closes above 100000 in December")
    low = similarity("Will Bitcoin close above 100000",
                     "Will the Democrats win the House")
    assert high > 0.5 > low


# ---------------------------------------------------------------------------
# the whole scan
# ---------------------------------------------------------------------------


def test_locked_opportunities_sort_ahead_of_larger_unlocked_ones():
    """A guaranteed $2 outranks a speculative $150, because the second is not
    an arbitrage."""
    exhaustive = group_of(
        [kalshi_contract(l, 0.60, 0.62)
         for l in ("0.0% or Below", "0.1% to 2.0%", "2.1% or Above")],
    )
    speculative = group_of(
        [kalshi_contract(n, 0.004, 0.006)
         for n in ("Canada", "Greenland", "Mexico", "Cuba")],
    )
    found = scan_groups([exhaustive, speculative], budget=200.0,
                        cross_venue=False)
    assert found
    locked_positions = [i for i, o in enumerate(found) if o.locked]
    unlocked_positions = [i for i, o in enumerate(found) if not o.locked]
    if locked_positions and unlocked_positions:
        assert max(locked_positions) < min(unlocked_positions)


def test_an_empty_universe_produces_nothing_rather_than_raising():
    assert scan_groups([], budget=200.0) == []


def test_every_opportunity_serialises():
    import json

    found = scan_groups(
        [group_of([kalshi_contract("A", 0.60, 0.64),
                   kalshi_contract("B", 0.55, 0.59)])],
        budget=200.0, cross_venue=False,
    )
    for opportunity in found:
        json.dumps(opportunity.to_dict())
        assert opportunity.explain()
