"""
Resolver tests.

The rule that matters here: **the stop is checked across the whole window, not
at the endpoint.** A thesis stopped out on day three and recovered by day seven
is INVALIDATED, because that is what would have happened to the position.
Grading only the endpoint systematically flatters any strategy with wide stops,
and it is the single easiest way to build a track record that looks good and
means nothing.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from vault.data.series import Series
from vault.resolve.resolver import resolve_one
from vault.thesis.schema import Outcome

from tests.unit.vault_factories import make_record


ENTRY = 5600.0


def _series(values: list[float], *, start: date, key: str = "SP500") -> Series:
    return Series.from_pairs(
        key, [(start + timedelta(days=i), v) for i, v in enumerate(values)],
        label=key, units="index",
    )


def _record(**kwargs):
    created = datetime.now(timezone.utc) - timedelta(days=30)
    return make_record(created_at=created, entry_price=ENTRY, **kwargs)


# ---------------------------------------------------------------------------
# The endpoint cases
# ---------------------------------------------------------------------------


def test_an_up_call_that_rises_is_correct():
    record = _record()
    prices = _series([ENTRY * 1.005 * (1 + 0.003 * i) for i in range(8)],
                     start=record.created_at.date() + timedelta(days=1))
    outcome, exit_price, realized, _, _ = resolve_one(record, prices)
    assert outcome == Outcome.CORRECT
    assert realized > 0
    assert exit_price > ENTRY


def test_an_up_call_that_drifts_down_within_the_stop_is_wrong():
    """Wrong, not invalidated: it lost, but it never hit the stop."""
    record = _record(invalidation_pct=5.0)
    prices = _series([ENTRY * (1 - 0.002 * i) for i in range(8)],
                     start=record.created_at.date() + timedelta(days=1))
    outcome, _, realized, invalidated_on, _ = resolve_one(record, prices)
    assert outcome == Outcome.WRONG
    assert realized < 0
    assert invalidated_on is None


def test_a_down_call_that_falls_is_correct():
    record = _record(direction="down")
    prices = _series([ENTRY * (1 - 0.004 * i) for i in range(8)],
                     start=record.created_at.date() + timedelta(days=1))
    outcome, _, realized, _, _ = resolve_one(record, prices)
    assert outcome == Outcome.CORRECT
    assert realized < 0


def test_a_flat_close_is_wrong_not_a_tie():
    """A directional call that did not happen is wrong."""
    record = _record()
    prices = _series([ENTRY] * 8, start=record.created_at.date() + timedelta(days=1))
    outcome, _, realized, _, note = resolve_one(record, prices)
    assert outcome == Outcome.WRONG
    assert realized == 0.0
    assert "wrong, not a tie" in note


# ---------------------------------------------------------------------------
# The window scan -- the rule that makes the record honest
# ---------------------------------------------------------------------------


def test_a_stop_hit_mid_window_invalidates_even_if_it_recovers():
    """
    Down through the stop on day three, back above entry by day seven. The
    endpoint says CORRECT. The position was closed on day three.
    """
    record = _record(invalidation_pct=1.8)
    stop = record.thesis.invalidation_price(ENTRY)
    assert stop < ENTRY

    start = record.created_at.date() + timedelta(days=1)
    prices = _series(
        [ENTRY, ENTRY * 0.995, stop - 5, ENTRY * 0.99, ENTRY * 1.01,
         ENTRY * 1.02, ENTRY * 1.03],
        start=start,
    )

    outcome, exit_price, realized, invalidated_on, note = resolve_one(record, prices)
    assert outcome == Outcome.INVALIDATED, (
        "the resolver graded the endpoint and missed a stop breach mid-window"
    )
    assert invalidated_on == start + timedelta(days=2)
    assert exit_price == stop - 5
    assert realized < 0
    assert "breached" in note


def test_a_down_call_is_stopped_out_by_a_rise():
    record = _record(direction="down", invalidation_pct=1.8)
    stop = record.thesis.invalidation_price(ENTRY)
    assert stop > ENTRY

    start = record.created_at.date() + timedelta(days=1)
    prices = _series([ENTRY, stop + 5, ENTRY * 0.95, ENTRY * 0.90],
                     start=start)
    outcome, _, _, invalidated_on, _ = resolve_one(record, prices)
    assert outcome == Outcome.INVALIDATED
    assert invalidated_on == start + timedelta(days=1)


def test_the_entry_day_itself_is_not_scanned_for_the_stop():
    """
    The scan window opens strictly after commit. A price printed on the entry
    day is the entry, not an adverse excursion away from it.
    """
    record = _record(invalidation_pct=1.8)
    stop = record.thesis.invalidation_price(ENTRY)
    prices = _series([stop - 100] + [ENTRY * 1.03] * 7,
                     start=record.created_at.date())
    outcome, _, _, _, _ = resolve_one(record, prices)
    assert outcome == Outcome.CORRECT


def test_prices_after_the_horizon_are_ignored():
    """A call is graded on its own horizon, not on what happened next."""
    record = _record(invalidation_pct=1.8)
    stop = record.thesis.invalidation_price(ENTRY)
    start = record.created_at.date() + timedelta(days=1)
    prices = _series([ENTRY * 1.02] * 8 + [stop - 50] * 5, start=start)
    outcome, _, _, _, _ = resolve_one(record, prices)
    assert outcome == Outcome.CORRECT


# ---------------------------------------------------------------------------
# Pending and unresolvable
# ---------------------------------------------------------------------------


def test_a_call_whose_horizon_has_not_elapsed_is_pending():
    created = datetime.now(timezone.utc) - timedelta(days=2)
    record = make_record(created_at=created, entry_price=ENTRY)
    prices = _series([ENTRY, ENTRY * 1.001], start=created.date() + timedelta(days=1))
    outcome, _, _, _, note = resolve_one(record, prices)
    assert outcome == Outcome.PENDING
    assert "horizon has not elapsed" in note


def test_a_horizon_with_no_nearby_price_is_unresolvable_not_wrong():
    """
    Missing data is missing data. Grading it as WRONG would quietly punish the
    forecaster for a gap in the feed.
    """
    record = _record()
    prices = _series([ENTRY] * 3, start=record.created_at.date() - timedelta(days=40))
    outcome, exit_price, _, _, note = resolve_one(record, prices)
    assert outcome == Outcome.UNRESOLVABLE
    assert exit_price is None
    assert "no price" in note


def test_a_weekend_horizon_resolves_against_the_prior_close():
    """Six days of tolerance, so a Saturday horizon is not unresolvable."""
    record = _record()
    start = record.created_at.date() + timedelta(days=1)
    # Only one observation, three days before the horizon.
    prices = _series([ENTRY * 1.02], start=record.resolve_on - timedelta(days=3))
    assert prices.points[0].on < record.resolve_on
    outcome, _, _, _, _ = resolve_one(record, prices)
    assert outcome == Outcome.CORRECT


def test_a_nonpositive_entry_price_is_unresolvable():
    record = make_record(entry_price=1.0)
    record.entry_price = 0.0
    outcome, _, _, _, note = resolve_one(record, _series([100.0], start=date.today()))
    assert outcome == Outcome.UNRESOLVABLE
    assert "entry price" in note
