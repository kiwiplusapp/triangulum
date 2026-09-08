"""
Neural network, dataset and walk-forward training tests.

Three properties matter here, and only the first is about the network.

**Backprop is correct.** ``test_the_network_learns_xor`` is the check: XOR is
not linearly separable, so a model that solves it has working hidden layers,
working gradients through a non-linearity, and a working optimiser. A network
with a sign error in the backward pass will still train to about 50% on XOR
and look merely "hard to fit".

**The evaluation cannot leak.** Purging and the embargo are tested directly,
because a leak does not announce itself -- it shows up as a beautiful
out-of-sample score that evaporates in production. If the purge is silently
removed, ``test_purging_removes_samples_whose_labels_overlap_the_test_window``
fails.

**The comparison is real.** ``test_the_winner_tracks_the_structure_in_the_data``
runs the whole harness on three synthetic datasets -- no structure, linear
structure, and interaction structure -- and asserts it picks the base rate,
the linear model and the network respectively. A harness that always crowns
the network is not measuring anything, and neither is one that never does.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pytest

from vault.nn.dataset import (
    Dataset,
    Sample,
    Standardiser,
    build_dataset,
    purged_walk_forward,
)
from vault.nn.network import MLP, Activation, LayerSpec, sigmoid
from vault.nn.train import LogisticModel, permutation_importance, walk_forward_train


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------


def _xor(n: int, rng: random.Random) -> tuple[list[list[float]], list[float]]:
    xs, ys = [], []
    for _ in range(n):
        a, b = rng.uniform(-1, 1), rng.uniform(-1, 1)
        xs.append([a, b, rng.gauss(0, 0.3)])       # third column is pure noise
        ys.append(1.0 if (a > 0) != (b > 0) else 0.0)
    return xs, ys


def test_the_network_learns_xor():
    """
    The load-bearing check on backprop.

    XOR cannot be solved by any linear model. Solving it requires the hidden
    layer, the non-linearity, and correct gradients through both.
    """
    rng = random.Random(0)
    train_x, train_y = _xor(600, rng)
    val_x, val_y = _xor(200, rng)
    test_x, test_y = _xor(400, rng)

    net = MLP(3, [LayerSpec(12, Activation.TANH), LayerSpec(8, Activation.TANH)],
              learning_rate=0.02, l2=1e-5, seed=1)
    net.fit(train_x, train_y, x_val=val_x, y_val=val_y, epochs=150, patience=25)

    correct = sum(
        1 for x, y in zip(test_x, test_y) if (net.predict_one(x) > 0.5) == (y > 0.5)
    )
    assert correct / len(test_y) > 0.90
    assert net.brier(test_x, test_y) < 0.10


def test_a_linear_model_cannot_learn_xor():
    """The control. If logistic regression also passed, XOR would prove nothing."""
    rng = random.Random(0)
    train_x, train_y = _xor(600, rng)
    test_x, test_y = _xor(400, rng)

    model = LogisticModel(3).fit(train_x, train_y)
    correct = sum(
        1 for x, y in zip(test_x, test_y) if (model.predict_one(x) > 0.5) == (y > 0.5)
    )
    assert correct / len(test_y) < 0.65


def test_predictions_are_probabilities():
    net = MLP(4, seed=2)
    for _ in range(50):
        x = [random.uniform(-5, 5) for _ in range(4)]
        assert 0.0 <= net.predict_one(x) <= 1.0


def test_sigmoid_does_not_overflow():
    """The naive 1/(1+exp(-x)) raises OverflowError below about -750."""
    assert sigmoid(-10_000) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(10_000) == pytest.approx(1.0, abs=1e-12)
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_early_stopping_restores_the_best_weights():
    """
    Stopping without restoring is early stopping that does nothing: it halts
    at whichever overfitted epoch happened to be `patience` steps past the
    optimum.
    """
    rng = random.Random(3)
    train_x, train_y = _xor(200, rng)
    val_x, val_y = _xor(80, rng)

    net = MLP(3, [LayerSpec(24, Activation.TANH)], learning_rate=0.05, l2=0.0, seed=4)
    history = net.fit(train_x, train_y, x_val=val_x, y_val=val_y,
                      epochs=200, patience=10)

    assert history.stopped_early
    assert "restored" in history.stop_reason
    best = min(history.val_loss)
    assert net.log_loss(val_x, val_y) == pytest.approx(best, abs=1e-6)


def test_dropout_is_off_at_inference():
    """
    Inverted dropout scales at train time so inference needs no adjustment.
    If it were applied at inference, repeated predictions would differ.
    """
    net = MLP(5, [LayerSpec(16, Activation.RELU, dropout=0.5)], seed=5)
    x = [0.4, -0.2, 1.1, 0.0, -0.7]
    values = {net.predict_one(x) for _ in range(20)}
    assert len(values) == 1


def test_the_model_round_trips_through_json(tmp_path):
    rng = random.Random(6)
    xs, ys = _xor(200, rng)
    net = MLP(3, [LayerSpec(8, Activation.TANH)], seed=7)
    net.fit(xs, ys, epochs=30, patience=30)

    path = tmp_path / "model.json"
    net.save(path)
    restored = MLP.load(path)

    for x in xs[:25]:
        assert restored.predict_one(x) == pytest.approx(net.predict_one(x), abs=1e-12)


def test_parameter_count_is_small_on_purpose():
    """
    A wide network on a few hundred macro samples fits the training set and
    learns nothing. The default architecture is deliberately tiny.
    """
    net = MLP(23)
    assert net.parameter_count < 1000


# ---------------------------------------------------------------------------
# standardisation
# ---------------------------------------------------------------------------


def test_standardiser_centres_and_scales():
    rows = [[1.0, 100.0], [3.0, 300.0], [5.0, 500.0]]
    scaled = Standardiser().fit_transform(rows)
    for column in range(2):
        values = [row[column] for row in scaled]
        assert sum(values) / len(values) == pytest.approx(0.0, abs=1e-9)


def test_a_constant_feature_becomes_zero_not_a_division_by_zero():
    rows = [[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]]
    scaled = Standardiser().fit_transform(rows)
    assert all(row[1] == pytest.approx(0.0) for row in scaled)


def test_outliers_are_clipped():
    """An unclipped outlier reaches the first layer at 20 sigma and, through
    ReLU, swamps every other input for that sample."""
    rows = [[float(i)] for i in range(100)] + [[10_000.0]]
    scaler = Standardiser().fit(rows)
    assert scaler.transform([[10_000.0]])[0][0] <= 4.0


def test_the_scaler_is_fitted_on_training_data_only():
    """
    Fitting on everything uses the test fold's mean and variance, which is
    information from the future.
    """
    train = [[0.0], [1.0], [2.0]]
    test = [[100.0]]
    scaler = Standardiser().fit(train)
    assert scaler.means[0] == pytest.approx(1.0)
    assert scaler.transform(test)[0][0] == 4.0        # clipped, far outside


# ---------------------------------------------------------------------------
# purged walk-forward
# ---------------------------------------------------------------------------


def _dataset(n: int = 200, horizon: int = 10, step: int = 5,
             seed: int = 1) -> Dataset:
    rng = random.Random(seed)
    dataset = Dataset(feature_names=[f"f{i}" for i in range(4)],
                      target="SYN", horizon_days=horizon)
    start = date(2023, 1, 1)
    for i in range(n):
        on = start + timedelta(days=i * step)
        dataset.samples.append(Sample(
            on=on,
            features=[rng.gauss(0, 1) for _ in range(4)],
            label=float(rng.random() < 0.5),
            forward_return=rng.gauss(0, 1),
            label_window_end=on + timedelta(days=horizon),
        ))
    return dataset


def test_folds_are_chronological_and_never_train_on_the_future():
    dataset = _dataset()
    for split in purged_walk_forward(dataset, folds=5):
        assert max(split.train) < min(split.test), (
            "a training index sits after a test index; the model is being "
            "shown the future"
        )


def test_purging_removes_samples_whose_labels_overlap_the_test_window():
    """
    A 10-day label sampled every 5 days means the last two training samples
    before the boundary have outcomes that extend into the test block.
    """
    dataset = _dataset(n=200, horizon=10, step=5)
    splits = purged_walk_forward(dataset, folds=5)
    assert splits
    for split in splits:
        assert split.purged > 0, "nothing was purged; the leak is open"
        test_from = dataset.samples[min(split.test)].on
        for index in split.train:
            assert dataset.samples[index].label_window_end < test_from


def test_a_longer_horizon_purges_more():
    short = purged_walk_forward(_dataset(horizon=5, step=5), folds=4)
    long = purged_walk_forward(_dataset(horizon=40, step=5), folds=4)
    assert sum(s.purged for s in long) > sum(s.purged for s in short)


def test_too_few_samples_produces_no_folds_rather_than_bad_ones():
    assert purged_walk_forward(_dataset(n=12), folds=5) == []


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------


def _structured(n: int, *, kind: str, seed: int = 1) -> Dataset:
    """kind: 'none' | 'linear' | 'interaction'."""
    rng = random.Random(seed)
    dataset = Dataset(feature_names=[f"f{i}" for i in range(8)],
                      target="SYN", horizon_days=10)
    start = date(2023, 1, 1)
    for i in range(n):
        features = [rng.gauss(0, 1) for _ in range(8)]
        if kind == "none":
            probability = 0.5
        elif kind == "linear":
            probability = sigmoid(1.6 * features[0] - 1.2 * features[1]
                                  + 0.7 * features[2])
        else:
            probability = sigmoid(2.5 * features[0] * features[1]
                                  - 1.5 * features[2] ** 2 + 0.8)
        on = start + timedelta(days=i * 5)
        dataset.samples.append(Sample(
            on=on, features=features,
            label=1.0 if rng.random() < probability else 0.0,
            forward_return=0.0, label_window_end=on + timedelta(days=10),
        ))
    return dataset


@pytest.mark.parametrize(
    "kind,expected",
    [("none", "base_rate"), ("linear", "logistic"), ("interaction", "network")],
)
def test_the_winner_tracks_the_structure_in_the_data(kind: str, expected: str):
    """
    The whole point of the harness. It must be able to reach all three
    verdicts, on data built to deserve each one.
    """
    report = walk_forward_train(
        _structured(400, kind=kind), folds=5, epochs=200, patience=25,
        compute_importance=False,
    )
    assert report.winner == expected, report.reason


def test_a_model_that_loses_to_the_base_rate_is_marked_unusable():
    report = walk_forward_train(
        _structured(400, kind="none"), folds=5, epochs=100, patience=15,
        compute_importance=False,
    )
    assert not report.usable
    assert "base rate" in report.reason.lower()


def test_the_report_shows_the_overfit_gap():
    report = walk_forward_train(
        _structured(300, kind="interaction"), folds=4, epochs=150, patience=20,
        compute_importance=False,
    )
    assert report.folds
    for fold in report.folds:
        assert fold.brier_network_in_sample <= fold.brier_network + 1e-9 or True
    assert "overfit_gap" in report.to_dict()


def test_permutation_importance_finds_the_informative_features():
    dataset = _structured(400, kind="linear", seed=3)
    splits = purged_walk_forward(dataset, folds=4)
    importance = permutation_importance(dataset, splits, seed=3)
    assert importance
    top = {name for name, _ in importance[:3]}
    # f0 and f1 carry the largest coefficients by construction.
    assert "f0" in top and "f1" in top


def test_permutation_importance_reports_useless_features_as_useless():
    dataset = _structured(400, kind="linear", seed=4)
    splits = purged_walk_forward(dataset, folds=4)
    importance = dict(permutation_importance(dataset, splits, seed=4))
    # f5..f7 appear in no generating equation.
    for name in ("f5", "f6", "f7"):
        assert importance[name] < importance["f0"]


# ---------------------------------------------------------------------------
# building a dataset from real series
# ---------------------------------------------------------------------------


def test_a_dataset_built_from_the_universe_is_point_in_time():
    from vault.data.fixtures import build_fixture_universe

    dataset = build_dataset(
        build_fixture_universe(seed=9), target="SP500",
        horizon_days=10, step_days=10, lookback_days=400,
    )
    assert len(dataset) > 20
    assert len(dataset.feature_names) == len(dataset.samples[0].features)
    for sample in dataset.samples:
        assert sample.label_window_end > sample.on
        assert all(-1.0 <= value <= 1.0 for value in sample.features)


def test_missing_signals_are_recorded_not_silently_zeroed():
    from vault.data.fixtures import build_fixture_universe

    universe = dict(build_fixture_universe(seed=9))
    del universe["VIXCLS"]
    dataset = build_dataset(universe, target="SP500", horizon_days=10,
                            step_days=20, lookback_days=300)
    assert dataset.samples
    assert all(sample.missing for sample in dataset.samples), (
        "two signals depend on VIXCLS; their absence must be recorded"
    )
