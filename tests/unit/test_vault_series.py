"""
Series and schema tests.

The series case worth its own test is ``year_over_year``: macro series publish
irregularly, so counting back twelve observations silently compares the wrong
months. It has to be a calendar lookup.

The schema tests cover the two validators that exist to stop the model from
producing an unfalsifiable call -- a "risk" that names no observable event, and
a high confidence paired with a stop too tight to be consistent with it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from vault.data.series import Series, correlation
from vault.thesis.schema import Direction, Thesis

from tests.unit.vault_factories import make_thesis


def _daily(values, *, end: date | None = None, key: str = "X") -> Series:
    end = end or date.today()
    start = end - timedelta(days=len(values) - 1)
    return Series.from_pairs(
        key, [(start + timedelta(days=i), v) for i, v in enumerate(values)],
        label=key, units="index",
    )


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


def test_year_over_year_uses_the_calendar_not_the_observation_count():
    """
    A monthly series with a publication gap. Counting back twelve points lands
    on the wrong month; a calendar lookup does not.
    """
    today = date.today().replace(day=1)
    points = []
    value = 100.0
    for months_ago in range(26, -1, -1):
        # Skip one month to create the gap that breaks index-based lookback.
        if months_ago == 5:
            continue
        when = today - timedelta(days=30 * months_ago)
        points.append((when, value))
        value *= 1.01

    series = Series.from_pairs("CPI", points, label="CPI", units="index")
    yoy = series.year_over_year()
    assert yoy is not None
    # Twelve months of 1% compounding, allowing for the 30-day month drift.
    assert yoy == pytest.approx(12.68, abs=2.0)


def test_year_over_year_is_none_without_a_year_of_history():
    assert _daily([100.0, 101.0, 102.0]).year_over_year() is None


def test_staleness_is_measured_in_days():
    fresh = _daily([1.0, 2.0], end=date.today())
    stale = _daily([1.0, 2.0], end=date.today() - timedelta(days=45))
    assert fresh.staleness_days == 0
    assert stale.staleness_days == 45


def test_value_on_respects_its_tolerance():
    series = _daily([100.0, 101.0, 102.0], end=date.today())
    target = date.today() - timedelta(days=30)
    assert series.value_on(target, tolerance_days=2) is None
    assert series.value_on(date.today(), tolerance_days=2) == 102.0


def test_an_empty_series_answers_rather_than_raises():
    empty = Series.from_pairs("E", [], label="E", units="x")
    assert not empty
    assert empty.latest is None
    assert empty.year_over_year() is None
    assert empty.pct_change() is None
    assert empty.zscore() is None


def test_correlation_on_returns_not_levels():
    """
    Two independent random walks correlate strongly in levels and not at all
    in returns. Reading the level correlation is how spurious relationships
    get into a brief.
    """
    import random

    rng = random.Random(4)
    a_vals, b_vals = [100.0], [100.0]
    for _ in range(400):
        a_vals.append(a_vals[-1] * (1 + rng.gauss(0, 0.01)))
        b_vals.append(b_vals[-1] * (1 + rng.gauss(0, 0.01)))

    a, b = _daily(a_vals, key="A"), _daily(b_vals, key="B")
    on_returns = correlation(a, b, window=300, on_returns=True)
    assert on_returns is not None
    assert abs(on_returns) < 0.25


def test_correlation_of_a_series_with_itself_is_one():
    series = _daily([100.0 + i * 0.7 + (i % 5) for i in range(200)])
    assert correlation(series, series, window=150, on_returns=True) == pytest.approx(
        1.0, abs=1e-6
    )


# ---------------------------------------------------------------------------
# Thesis schema
# ---------------------------------------------------------------------------


def test_a_well_formed_thesis_is_accepted():
    thesis = make_thesis()
    assert thesis.direction == Direction.UP
    assert thesis.horizon_days == 7
    assert thesis.sign == 1


def test_the_stop_and_target_sit_on_the_right_sides_of_entry():
    up = make_thesis(direction="up", magnitude_pct=2.5, invalidation_pct=1.8)
    down = make_thesis(direction="down", magnitude_pct=2.5, invalidation_pct=1.8)
    assert up.invalidation_price(100.0) == pytest.approx(98.2)
    assert up.target_price(100.0) == pytest.approx(102.5)
    assert down.invalidation_price(100.0) == pytest.approx(101.8)
    assert down.target_price(100.0) == pytest.approx(97.5)


def test_a_vague_key_risk_is_rejected():
    """
    "Market conditions could change" is not a falsifiable risk, and a thesis
    whose risk cannot be observed cannot be graded.
    """
    pydantic = pytest.importorskip("pydantic")
    with pytest.raises(pydantic.ValidationError):
        Thesis(
            asset="SP500", direction="up", horizon="1w", probability=0.65,
            magnitude_pct=2.5, invalidation_pct=1.8,
            regime_dependency="late cycle",
            key_risk="Market conditions change",
            reasoning="...", primary_evidence=["T10Y3M"],
        )


def test_high_confidence_with_a_tight_stop_is_rejected():
    """"Very likely right" and "kill it on a small adverse move" disagree."""
    pydantic = pytest.importorskip("pydantic")
    with pytest.raises(pydantic.ValidationError):
        Thesis(
            asset="SP500", direction="up", horizon="1w", probability=0.85,
            magnitude_pct=5.0, invalidation_pct=0.5,
            regime_dependency="late cycle",
            key_risk="A core PCE print above 0.35% m/m would invalidate this",
            reasoning="...", primary_evidence=["T10Y3M"],
        )


@pytest.mark.parametrize("probability", [0.0, 0.04, 0.96, 1.0])
def test_certainty_is_not_expressible(probability: float):
    """
    The schema caps probabilities at [0.05, 0.95]. A forecaster that can say
    1.0 will eventually say it and be wrong, and the Brier score of that is
    unbounded in practice.
    """
    pydantic = pytest.importorskip("pydantic")
    with pytest.raises(pydantic.ValidationError):
        make_thesis(probability=probability)


def test_a_zero_invalidation_is_rejected():
    """No stop means no falsifiable exit."""
    pydantic = pytest.importorskip("pydantic")
    with pytest.raises(pydantic.ValidationError):
        make_thesis(invalidation_pct=0.0)
