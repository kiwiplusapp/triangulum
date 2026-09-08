"""
Ensemble tests.

The property under test is a negative one, and it is the important one:

    **The predictor cannot produce a confident number out of components that
    have not earned it.**

The failure mode this is built against is the ensemble that averages three
unvalidated models and reports the average as though the averaging conferred
reliability. Here, every source's weight is proportional to how much it beat
the base rate by, out of sample; a source that did not beat the base rate gets
zero; and when nothing beat it, the prediction IS the base rate, at full
weight, with the provenance string saying exactly that.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from vault.data.fixtures import build_fixture_universe
from vault.data.series import Series
from vault.nn.ensemble import MIN_EDGE, Predictor
from vault.nn.pipeline import learn


def _planted(seed: int = 3, strength: float = 0.0004) -> dict[str, Series]:
    """A universe where the VIX genuinely drives forward equity returns."""
    universe = dict(build_fixture_universe(seed=7))
    rng = random.Random(seed)
    end = universe["VIXCLS"].points[-1].on
    n = 900
    start = end - timedelta(days=n - 1)

    vix, level = [], 18.0
    for _ in range(n):
        level = max(9.0, min(45.0, level * (1 + rng.gauss(0, 0.05))
                             + rng.gauss(0, 0.3)))
        vix.append(level)
    prices, price = [], 4000.0
    for i in range(n):
        price *= 1 + (20.0 - vix[i]) * strength + rng.gauss(0, 0.006)
        prices.append(price)

    universe["VIXCLS"] = Series.from_pairs(
        "VIXCLS", [(start + timedelta(days=i), v) for i, v in enumerate(vix)],
        label="VIX", units="index")
    universe["SP500"] = Series.from_pairs(
        "SP500", [(start + timedelta(days=i), v) for i, v in enumerate(prices)],
        label="S&P 500", units="index")
    return universe


# ---------------------------------------------------------------------------
# an untrained predictor
# ---------------------------------------------------------------------------


def test_an_untrained_predictor_returns_the_base_rate():
    predictor = Predictor()
    prediction = predictor.predict(build_fixture_universe())
    assert prediction.probability == pytest.approx(0.5)
    assert prediction.weight_base == 1.0
    assert prediction.edge == pytest.approx(0.0)
    assert prediction.confidence == "none"


def test_an_untrained_predictor_says_it_knows_nothing():
    prediction = Predictor().predict(build_fixture_universe())
    assert "Nothing has demonstrated an edge" in prediction.provenance


# ---------------------------------------------------------------------------
# trained on noise
# ---------------------------------------------------------------------------


def test_training_on_noise_produces_exactly_the_base_rate():
    """
    The load-bearing assertion. Random walks in, base rate out -- not a
    slightly-off-base-rate number that would look like a weak signal.
    """
    universe = build_fixture_universe(seed=5)
    result = learn(universe, target="SP500", horizon_days=10, step_days=5,
                   lookback_days=880, epochs=60)
    prediction = result.predictor.predict(universe)

    assert result.scorecard.earning == []
    assert not result.training.usable
    assert prediction.weight_signals == 0.0
    assert prediction.weight_model == 0.0
    assert prediction.weight_base == 1.0
    assert prediction.probability == pytest.approx(prediction.base_rate, abs=1e-9)
    assert prediction.edge == pytest.approx(0.0, abs=1e-9)


def test_a_failed_model_is_not_quietly_averaged_in():
    universe = build_fixture_universe(seed=5)
    result = learn(universe, target="SP500", horizon_days=10, step_days=5,
                   lookback_days=880, epochs=60)
    assert result.predictor.model_kind == "none"
    assert result.predictor.predict(universe).from_model is None


# ---------------------------------------------------------------------------
# trained on a real relationship
# ---------------------------------------------------------------------------


def test_a_demonstrated_signal_moves_the_prediction_off_the_base_rate():
    universe = _planted()
    result = learn(universe, target="SP500", horizon_days=10, step_days=5,
                   lookback_days=880, epochs=60)
    prediction = result.predictor.predict(universe)

    assert result.scorecard.earning, "the planted relationship was not found"
    assert prediction.weight_signals > 0
    assert prediction.weight_base < 1.0
    assert abs(prediction.edge) > 0.005


def test_every_contribution_is_traceable_to_a_named_signal():
    universe = _planted()
    result = learn(universe, target="SP500", horizon_days=10, step_days=5,
                   lookback_days=880, epochs=60)
    prediction = result.predictor.predict(universe)

    assert prediction.contributions
    for item in prediction.contributions:
        assert item["signal"]
        assert item["weight"] > 0
        assert item["contribution"] == pytest.approx(
            item["strength"] * item["weight"], abs=1e-4
        )


def test_the_weights_always_sum_to_one():
    for universe in (build_fixture_universe(seed=5), _planted()):
        result = learn(universe, target="SP500", horizon_days=10, step_days=5,
                       lookback_days=880, epochs=60)
        prediction = result.predictor.predict(universe)
        total = (prediction.weight_signals + prediction.weight_model
                 + prediction.weight_base)
        assert total == pytest.approx(1.0, abs=1e-9)


def test_the_prediction_never_fully_escapes_the_base_rate():
    """
    Even a demonstrably skilful blend is capped. On a sample this size, a
    model that could override the base rate entirely would be overfitting
    with extra steps.
    """
    universe = _planted()
    result = learn(universe, target="SP500", horizon_days=10, step_days=5,
                   lookback_days=880, epochs=60)
    prediction = result.predictor.predict(universe)
    assert prediction.weight_base >= 0.40
    assert 0.05 <= prediction.probability <= 0.95


def test_a_signal_below_the_edge_floor_earns_nothing():
    predictor = Predictor()
    predictor.adopt(
        scorecard=None, report=None, model=None, scaler=None,
        base_rate=0.55, feature_names=[],
    )
    assert predictor._weight_model == 0.0
    assert predictor._weight_signals == 0.0
    assert MIN_EDGE > 0


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def test_the_predictor_serialises_with_its_evidence(tmp_path):
    """
    The saved artefact carries the scorecard and the training report, not
    just the weights. A model file that cannot say what justified it is a
    model file nobody can audit later.
    """
    import json

    universe = _planted()
    result = learn(universe, target="SP500", horizon_days=10, step_days=5,
                   lookback_days=880, epochs=60)
    path = tmp_path / "predictor.json"
    result.predictor.save(path)

    raw = json.loads(path.read_text())
    assert raw["scorecard"] is not None
    assert raw["training"] is not None
    assert raw["weights"]["signals"] + raw["weights"]["model"] + \
        raw["weights"]["base_rate"] == pytest.approx(1.0, abs=1e-3)
    assert raw["feature_names"]


def test_the_learning_report_names_every_rejection():
    universe = build_fixture_universe(seed=5)
    result = learn(universe, target="SP500", horizon_days=10, step_days=5,
                   lookback_days=880, epochs=60)
    text = result.report()
    assert "earning weight" in text
    for performance in result.scorecard.performances:
        if performance.n:
            assert performance.verdict in text
            assert performance.reason, performance.key
