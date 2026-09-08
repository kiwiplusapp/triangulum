"""
The learning pipeline: score the signals, fit the models, build the predictor.

One entry point, ``learn``, so the whole thing is reproducible from a series
universe and a handful of parameters. It returns a fitted :class:`Predictor`
and a report that states, in order:

    1. which signals earned weight, and what every rejected one was rejected
       for
    2. how the network and the linear model did against the base rate on
       purged walk-forward folds
    3. what the resulting blend weights are, and why

The design constraint throughout: **every number that ends up influencing a
position has to be traceable to a measurement, and every measurement has to
be reported alongside what it was measured against.**
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from vault.data.series import Series
from vault.nn.dataset import Dataset, Standardiser, build_dataset, purged_walk_forward
from vault.nn.ensemble import Predictor
from vault.nn.network import MLP
from vault.nn.train import LogisticModel, TrainingReport, walk_forward_train
from vault.signals.scoring import SignalScorecard, score_signal_history

logger = logging.getLogger(__name__)

__all__ = ["LearningResult", "learn"]


@dataclass(slots=True)
class LearningResult:
    predictor: Predictor
    scorecard: SignalScorecard
    training: TrainingReport
    dataset: Dataset

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "scorecard": self.scorecard.to_dict(),
            "training": self.training.to_dict(),
            "predictor": {
                "model_kind": self.predictor.model_kind,
                "summary": self.predictor.summary(),
            },
        }

    def report(self) -> str:
        return "\n\n".join([
            self.scorecard.report(),
            self.training.report(),
            "  " + self.predictor.summary(),
        ])


def learn(
    series: Mapping[str, Series],
    *,
    target: str = "SP500",
    horizon_days: int = 21,
    step_days: int = 5,
    lookback_days: int = 2000,
    folds: int = 5,
    epochs: int = 150,
    seed: int = 42,
) -> LearningResult:
    """
    Score signals, fit models, and assemble a predictor from what survived.

    ``step_days`` defaults to 5 rather than 1 deliberately. Daily sampling of
    a 21-day horizon produces observations that share 95% of their window; it
    multiplies the row count without adding information, and every statistic
    computed on it then has to be deflated by the same factor anyway.
    """
    scorecard = score_signal_history(
        series, target=target, horizon_days=horizon_days,
        step_days=step_days, lookback_days=lookback_days,
    )

    dataset = build_dataset(
        series, target=target, horizon_days=horizon_days,
        step_days=step_days, lookback_days=lookback_days,
    )

    training = walk_forward_train(
        dataset, folds=folds, epochs=epochs, seed=seed,
    )

    predictor = Predictor()

    # Refit the winning model on ALL the data, once the comparison has
    # decided which one to use. The folds decided the question; the final
    # model should see everything available.
    #
    # Note what this costs: the reported Brier belongs to the fold models, not
    # to this one. That is the standard trade and it is stated rather than
    # glossed -- the alternative, reporting a score for the refitted model, is
    # an in-sample number wearing an out-of-sample label.
    model: MLP | LogisticModel | None = None
    scaler = Standardiser()
    if training.usable and dataset.samples:
        rows, labels = dataset.x, dataset.y
        scaled = scaler.fit_transform(rows)
        n_features = len(dataset.feature_names)
        if training.winner == "network":
            model = MLP(n_features, learning_rate=0.01, l2=1e-3, seed=seed)
            cut = max(1, int(len(scaled) * 0.8))
            model.fit(
                scaled[:cut], labels[:cut],
                x_val=scaled[cut:], y_val=labels[cut:],
                epochs=epochs, patience=20,
            )
        elif training.winner == "logistic":
            model = LogisticModel(n_features).fit(scaled, labels)

    predictor.adopt(
        scorecard=scorecard,
        report=training,
        model=model,
        scaler=scaler,
        base_rate=dataset.base_rate if dataset.samples else 0.5,
        feature_names=list(dataset.feature_names),
    )

    logger.info(
        "learned: %d/%d signals earning, model=%s, %d samples",
        len(scorecard.earning), len(scorecard.performances),
        predictor.model_kind, len(dataset),
    )
    return LearningResult(
        predictor=predictor, scorecard=scorecard,
        training=training, dataset=dataset,
    )
