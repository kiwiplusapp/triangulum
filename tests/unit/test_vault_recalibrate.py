"""
Recalibration tests.

Recalibration is the mechanism that rescues a forecaster whose *ordering* is
good and whose *numbers* are inflated -- a real and common failure, and one
the capital gate blocks on, correctly, because a stated probability that
cannot be trusted cannot be used for sizing.

The regression here is subtle and was silent: pool-adjacent-violators only
merges where the outcome sequence decreases, so running it on raw observations
left several blocks sitting at the SAME stated probability whenever a run of
tied inputs happened to arrive in non-decreasing order. ``apply`` then bisected
into the first such block and returned its value instead of the pooled mean.

The case is the normal one, not an exotic one: a model that states 0.85 on
every call produces exactly one distinct input. A 300-call record whose true
frequency was 57.8% came back recalibrated to 0.500 -- a Brier of exactly
0.25, meaning the "correction" had discarded the entire signal and returned
the baseline while reporting success.
"""

from __future__ import annotations

import random

import pytest

from vault.calibration.recalibrate import (
    IsotonicRecalibrator,
    PlattRecalibrator,
    fit_recalibrator,
)

from tests.unit.vault_factories import make_record, resolve_record


# ---------------------------------------------------------------------------
# The tied-input regression
# ---------------------------------------------------------------------------


def test_isotonic_recovers_the_observed_rate_from_a_single_stated_value():
    """
    Every call stated at 0.85, true frequency well below it. The correction
    must land on the observed frequency, not on 0.5.
    """
    rng = random.Random(3)
    outcomes = [1 if rng.random() < 0.578 else 0 for _ in range(300)]
    observed = sum(outcomes) / len(outcomes)

    recalibrator = IsotonicRecalibrator().fit([0.85] * 300, outcomes)
    assert recalibrator.fitted
    assert recalibrator.apply(0.85) == pytest.approx(observed, abs=1e-9)
    assert recalibrator.apply(0.85) != pytest.approx(0.5, abs=0.01)


def test_isotonic_collapses_ties_to_one_knot_per_distinct_input():
    rng = random.Random(5)
    probabilities = [0.6] * 150 + [0.8] * 150
    outcomes = [1 if rng.random() < 0.55 else 0 for _ in range(150)]
    outcomes += [1 if rng.random() < 0.70 else 0 for _ in range(150)]

    recalibrator = IsotonicRecalibrator().fit(probabilities, outcomes)
    assert recalibrator._x == sorted(set(probabilities))
    assert len(recalibrator._x) == 2


def test_the_ordering_of_tied_outcomes_does_not_change_the_fit():
    """
    The bug was order-dependent: the same 300 calls in a different sequence
    produced a different correction. A calibration curve that depends on the
    order the wins arrived in is not a calibration curve.
    """
    rng = random.Random(11)
    outcomes = [1 if rng.random() < 0.6 else 0 for _ in range(300)]
    shuffled = sorted(outcomes)             # worst case: all losses first

    a = IsotonicRecalibrator().fit([0.85] * 300, outcomes).apply(0.85)
    b = IsotonicRecalibrator().fit([0.85] * 300, shuffled).apply(0.85)
    assert a == pytest.approx(b, abs=1e-9)


def test_isotonic_stays_monotone():
    """The one guarantee the method exists to provide."""
    rng = random.Random(7)
    probabilities, outcomes = [], []
    for stated in (0.55, 0.65, 0.75, 0.85, 0.95):
        for _ in range(80):
            probabilities.append(stated)
            outcomes.append(1 if rng.random() < stated - 0.1 else 0)

    recalibrator = IsotonicRecalibrator().fit(probabilities, outcomes)
    corrected = [recalibrator.apply(p) for p in (0.55, 0.65, 0.75, 0.85, 0.95)]
    assert corrected == sorted(corrected), corrected


def test_isotonic_output_stays_inside_the_schema_bounds():
    recalibrator = IsotonicRecalibrator().fit([0.9] * 300, [1] * 300)
    assert 0.05 <= recalibrator.apply(0.9) <= 0.95


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------


def test_platt_pulls_an_inflated_forecaster_down():
    rng = random.Random(13)
    probabilities = [0.85] * 200
    outcomes = [1 if rng.random() < 0.58 else 0 for _ in range(200)]

    recalibrator = PlattRecalibrator().fit(probabilities, outcomes)
    assert recalibrator.apply(0.85) < 0.85
    assert recalibrator.apply(0.85) == pytest.approx(0.58, abs=0.08)


def test_platt_leaves_an_already_calibrated_forecaster_roughly_alone():
    rng = random.Random(17)
    probabilities, outcomes = [], []
    for stated in (0.55, 0.65, 0.75, 0.85):
        for _ in range(120):
            probabilities.append(stated)
            outcomes.append(1 if rng.random() < stated else 0)

    recalibrator = PlattRecalibrator().fit(probabilities, outcomes)
    for stated in (0.55, 0.65, 0.75, 0.85):
        assert recalibrator.apply(stated) == pytest.approx(stated, abs=0.10)


# ---------------------------------------------------------------------------
# The chooser
# ---------------------------------------------------------------------------


def _records(n: int, *, stated: float, true_rate: float, seed: int = 21):
    rng = random.Random(seed)
    return [
        resolve_record(
            make_record(index=i, probability=stated), correct=rng.random() < true_rate
        )
        for i in range(n)
    ]


def test_recalibration_is_refused_on_too_few_samples():
    """Fitting a correction on twenty calls is fitting noise."""
    recalibrator, report = fit_recalibrator(_records(20, stated=0.85, true_rate=0.55))
    assert recalibrator is None
    assert not report.fitted
    assert "at least" in report.reason


def test_recalibration_improves_the_brier_of_an_overconfident_record():
    records = _records(300, stated=0.85, true_rate=0.58)
    recalibrator, report = fit_recalibrator(records)
    assert recalibrator is not None
    assert report.fitted
    assert report.brier_after < report.brier_before
    assert report.helped
    # The whole point: the corrected numbers clear the baseline the raw ones failed.
    assert report.brier_before > 0.25 > report.brier_after


def test_the_report_says_it_was_fitted_in_sample():
    """An in-sample improvement that does not say so is a lie by omission."""
    _, report = fit_recalibrator(_records(300, stated=0.85, true_rate=0.58))
    assert "sample" in report.summary().lower()
