"""
Scoring tests, centred on the one that found a real hole in the capital gate.

The gate used to test a FIXED-SAMPLE Wilson bound while being consulted after
every single resolved call. That is optional stopping, and it is not a subtle
effect: a forecaster with no edge at all crossed the fixed bound in roughly
one run in seven. ``test_fixed_bound_leaks_under_repeated_looks`` measures
that leak directly and ``test_sequential_bound_holds_its_advertised_rate``
shows the always-valid bound holding to what it advertises.

The rest cover the Brier machinery the gate reads: the 0.25 baseline, Murphy's
decomposition, and the two failure modes a single number conflates.
"""

from __future__ import annotations

import math
import random

import pytest

from vault.resolve.scoring import (
    BASELINE_BRIER,
    reliability_buckets,
    score_records,
    sequential_wilson_interval,
    sequential_z,
    wilson_interval,
)

from tests.unit.vault_factories import make_record, resolve_record


# ---------------------------------------------------------------------------
# Wilson, fixed and sequential
# ---------------------------------------------------------------------------


def test_wilson_stays_inside_the_unit_interval():
    """The reason Wilson is used at all: the normal approximation does not."""
    low, high = wilson_interval(8, 8)
    assert 0.0 <= low <= high <= 1.0
    # Normal approximation would give 1.0 +/- 0 here and 1.28 at 9/10.
    assert high == 1.0
    assert low < 1.0


def test_wilson_of_nothing_is_not_a_crash():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert sequential_wilson_interval(0, 0) == (0.0, 0.0)


def test_a_streak_of_seven_from_ten_does_not_clear_a_coin_flip():
    """The motivating example from the gate's docstring."""
    low, _ = wilson_interval(7, 10)
    assert low == pytest.approx(0.40, abs=0.02)
    assert low < 0.5


def test_sequential_z_grows_with_sample_size():
    """
    The whole mechanism: the critical value must widen as looks accumulate,
    or repeated testing silently eats the error budget.
    """
    assert sequential_z(50) > 1.96
    assert sequential_z(1000) > sequential_z(50) > sequential_z(10)
    # sqrt(log n) growth -- slow enough to stay usable.
    assert sequential_z(50) == pytest.approx(3.15, abs=0.02)
    assert sequential_z(1000) == pytest.approx(3.59, abs=0.02)


def test_sequential_bound_is_strictly_more_conservative():
    for successes, n in ((32, 50), (65, 100), (170, 300)):
        fixed_low, _ = wilson_interval(successes, n)
        seq_low, _ = sequential_wilson_interval(successes, n)
        assert seq_low < fixed_low, (successes, n)


# ---------------------------------------------------------------------------
# The measurement that justifies the change
# ---------------------------------------------------------------------------


def _ever_crosses(true_p: float, calls: int, *, sequential: bool,
                  trials: int, seed: int, min_n: int = 50) -> float:
    """Fraction of runs in which a lower bound EVER exceeds 0.5."""
    rng = random.Random(seed)
    crossings = 0
    for _ in range(trials):
        successes = 0
        for n in range(1, calls + 1):
            if rng.random() < true_p:
                successes += 1
            if n < min_n:
                continue
            low = (
                sequential_wilson_interval(successes, n)[0] if sequential
                else wilson_interval(successes, n)[0]
            )
            if low > 0.5:
                crossings += 1
                break
    return crossings / trials


def test_fixed_bound_leaks_under_repeated_looks():
    """
    A 95% bound tested after every one of 300 calls is not a 95% bound.

    This is the defect that let the no-edge "lucky" profile take $356 of
    simulated capital at call 50 while the summary line printed HELD.
    """
    leak = _ever_crosses(0.50, 300, sequential=False, trials=600, seed=7)
    assert leak > 0.08, (
        f"expected the fixed bound to leak badly under repeated looks, "
        f"measured {leak:.1%} -- if this has become small the test has "
        f"stopped measuring what it claims to"
    )


def test_sequential_bound_holds_its_advertised_rate():
    """The same experiment against the always-valid bound."""
    leak = _ever_crosses(0.50, 300, sequential=True, trials=600, seed=7)
    assert leak <= 0.05, f"always-valid bound leaked {leak:.1%}, budget is 5%"


def test_sequential_bound_still_admits_a_genuine_edge():
    """
    Conservatism is only a virtue if a real edge still gets through. A true
    65% forecaster must clear the bound, not be locked out forever.
    """
    admitted = _ever_crosses(0.65, 400, sequential=True, trials=100, seed=11)
    assert admitted > 0.90, (
        f"a genuine 65% edge cleared the gate in only {admitted:.0%} of runs; "
        f"the bound has been made too conservative to be useful"
    )


# ---------------------------------------------------------------------------
# Brier and Murphy
# ---------------------------------------------------------------------------


def test_always_saying_half_scores_exactly_the_baseline():
    """0.25 is not an arbitrary threshold; it is what indecision scores."""
    records = [
        resolve_record(make_record(probability=0.5, index=i), correct=(i % 2 == 0))
        for i in range(40)
    ]
    score = score_records(records)
    assert score.brier == pytest.approx(BASELINE_BRIER)
    assert score.skill == pytest.approx(0.0)
    assert not score.beats_baseline


def test_a_perfect_confident_forecaster_scores_near_zero():
    records = [
        resolve_record(make_record(probability=0.95, index=i), correct=True)
        for i in range(30)
    ]
    score = score_records(records)
    assert score.brier == pytest.approx(0.0025)
    assert score.beats_baseline
    assert score.hit_rate == 1.0


def test_confidence_is_not_free_when_you_are_wrong():
    """
    Hit rate cannot tell these two apart; Brier must. Both are right 60% of
    the time, one says 0.55 and one says 0.95.
    """
    modest = [
        resolve_record(make_record(probability=0.55, index=i), correct=(i % 10 < 6))
        for i in range(50)
    ]
    strident = [
        resolve_record(make_record(probability=0.95, index=i), correct=(i % 10 < 6))
        for i in range(50)
    ]
    a, b = score_records(modest), score_records(strident)
    assert a.hit_rate == b.hit_rate == pytest.approx(0.60)
    assert a.brier < b.brier
    assert a.beats_baseline and not b.beats_baseline
    assert b.overconfidence > a.overconfidence


def test_murphy_decomposition_reconstructs_the_brier_score():
    """BS = reliability - resolution + uncertainty, to bucketing error."""
    rng = random.Random(3)
    records = []
    for i in range(400):
        p = rng.choice([0.55, 0.6, 0.7, 0.8, 0.9])
        records.append(
            resolve_record(make_record(probability=p, index=i), correct=rng.random() < p)
        )
    score = score_records(records)
    rebuilt = score.reliability - score.resolution + score.uncertainty
    assert rebuilt == pytest.approx(score.brier, abs=0.01)


def test_calibrated_but_useless_has_good_reliability_and_no_resolution():
    """
    The failure a single number hides: always predicting the base rate is
    perfectly calibrated and carries no information at all.
    """
    rng = random.Random(5)
    records = [
        resolve_record(make_record(probability=0.6, index=i), correct=rng.random() < 0.6)
        for i in range(300)
    ]
    score = score_records(records)
    assert score.reliability < 0.005      # well calibrated
    assert score.resolution < 0.005       # and tells you nothing


def test_pending_records_are_not_scored():
    records = [make_record(index=i) for i in range(10)]
    records.append(resolve_record(make_record(index=99), correct=True))
    score = score_records(records)
    assert score.n == 1


def test_invalidated_counts_as_wrong_not_as_absent():
    """
    A thesis stopped out is a thesis that lost money. Dropping it from the
    denominator would flatter every wide stop in the system.
    """
    records = [
        resolve_record(make_record(index=0), correct=True),
        resolve_record(make_record(index=1), invalidated=True),
    ]
    score = score_records(records)
    assert score.n == 2
    assert score.invalidated == 1
    assert score.hit_rate == pytest.approx(0.5)


def test_reliability_buckets_report_observed_against_stated():
    probabilities = [0.6] * 10 + [0.9] * 10
    outcomes = [1] * 6 + [0] * 4 + [1] * 9 + [0]
    buckets = reliability_buckets(probabilities, outcomes)
    by_predicted = {round(b["predicted"], 2): b for b in buckets}
    assert by_predicted[0.6]["observed"] == pytest.approx(0.6)
    assert by_predicted[0.9]["observed"] == pytest.approx(0.9)


def test_log_loss_is_finite_even_for_a_confident_miss():
    """Unclipped, one 0.95 call that misses makes the mean infinite."""
    records = [resolve_record(make_record(probability=0.95, index=0), correct=False)]
    score = score_records(records)
    assert math.isfinite(score.log_loss)


def test_adequacy_is_honest_about_small_samples():
    small = score_records([
        resolve_record(make_record(index=i), correct=True) for i in range(12)
    ])
    assert small.adequacy == "insufficient"
    bigger = score_records([
        resolve_record(make_record(index=i), correct=(i % 3 > 0)) for i in range(120)
    ])
    assert bigger.adequacy in {"indicative", "adequate"}
