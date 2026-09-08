"""
A feed-forward neural network, written from scratch in pure Python.

## On the choice of framework

There is no neural-network library called Obsidian -- Obsidian is a Markdown
note-taking application. The nearest things by name are ONNX (a model
interchange format, not a training framework) and a handful of unrelated
projects. Rather than guess at which was meant, this is implemented directly:
forward pass, backpropagation and Adam, about four hundred lines, no
dependencies.

That is not a consolation prize, it is the right call for this codebase.
NumPy is not installed in this environment and PyTorch would be a 700MB
dependency to train a network with roughly six hundred parameters on a few
hundred samples. Everything else here -- Bellman-Ford over log prices, FTRL,
RMSProp, Thompson sampling -- is implemented the same way, and the arithmetic
is legible rather than hidden behind a framework. If the model ever needs to
be larger than this, the honest move is to swap in PyTorch wholesale, not to
grow this file.

## What it predicts

P(the macro thesis resolves correct), from the signal vector. A probability,
not a return, because a probability is the thing the capital gate consumes
and the thing the Brier score can grade.

## Why it is small on purpose

The architecture defaults to 23 -> 16 -> 8 -> 1: about 600 parameters. With
a few hundred training samples, that is already generous. A wider network
would fit the training set perfectly and learn nothing, and the sample size
here is set by how many macro calls the agent has actually resolved, which
grows at one per day at best. Every defence against overfitting that fits in
a small model is on by default -- L2, dropout, early stopping on a held-out
split, and a hard cap on epochs -- and the training report states the
in-sample and out-of-sample scores side by side so the gap is impossible to
miss.

## The honest expectation

A network on a few hundred noisy macro samples will usually NOT beat logistic
regression out of sample. ``vault/nn/train.py`` fits both and reports which
won. When the linear model wins, that is the answer, and the ensemble weights
follow the measurement rather than the ambition.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

__all__ = ["MLP", "LayerSpec", "TrainingHistory", "Activation"]

Vector = list[float]
Matrix = list[list[float]]


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------


class Activation:
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LINEAR = "linear"


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _relu_grad(y: float) -> float:
    # Derivative expressed in terms of the OUTPUT, which is what the backward
    # pass has to hand. For ReLU the output is enough: y > 0 iff x > 0.
    return 1.0 if y > 0.0 else 0.0


def _tanh(x: float) -> float:
    return math.tanh(max(-30.0, min(30.0, x)))


def _tanh_grad(y: float) -> float:
    return 1.0 - y * y


def sigmoid(x: float) -> float:
    """Numerically stable logistic. The naive form overflows below about -700."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    exponent = math.exp(max(x, -60.0))
    return exponent / (1.0 + exponent)


def _sigmoid_grad(y: float) -> float:
    return y * (1.0 - y)


_FORWARD: dict[str, Callable[[float], float]] = {
    Activation.RELU: _relu,
    Activation.TANH: _tanh,
    Activation.SIGMOID: sigmoid,
    Activation.LINEAR: lambda x: x,
}

_BACKWARD: dict[str, Callable[[float], float]] = {
    Activation.RELU: _relu_grad,
    Activation.TANH: _tanh_grad,
    Activation.SIGMOID: _sigmoid_grad,
    Activation.LINEAR: lambda y: 1.0,
}


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


@dataclass
class LayerSpec:
    units: int
    activation: str = Activation.RELU
    dropout: float = 0.0


class Dense:
    """One fully-connected layer, with its own Adam moments."""

    __slots__ = (
        "n_in", "n_out", "activation", "dropout",
        "w", "b", "mw", "vw", "mb", "vb",
        "_last_input", "_last_output", "_mask",
    )

    def __init__(self, n_in: int, n_out: int, activation: str,
                 dropout: float, rng: random.Random) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.dropout = dropout

        # He initialisation for ReLU, Xavier otherwise.
        #
        # This is not a detail. With the naive "small random uniform" the
        # activation variance shrinks layer by layer, the gradients that reach
        # the first layer are numerically negligible, and the network trains
        # its last layer only -- which looks like convergence to a mediocre
        # score and is indistinguishable from "the features are weak" unless
        # you go looking.
        if activation == Activation.RELU:
            scale = math.sqrt(2.0 / n_in)
        else:
            scale = math.sqrt(1.0 / n_in)

        self.w: Matrix = [
            [rng.gauss(0.0, scale) for _ in range(n_in)] for _ in range(n_out)
        ]
        self.b: Vector = [0.0] * n_out

        self.mw: Matrix = [[0.0] * n_in for _ in range(n_out)]
        self.vw: Matrix = [[0.0] * n_in for _ in range(n_out)]
        self.mb: Vector = [0.0] * n_out
        self.vb: Vector = [0.0] * n_out

        self._last_input: Vector = []
        self._last_output: Vector = []
        self._mask: Vector | None = None

    def forward(self, x: Vector, *, training: bool, rng: random.Random) -> Vector:
        self._last_input = x
        activate = _FORWARD[self.activation]
        out: Vector = []
        for row, bias in zip(self.w, self.b):
            total = bias
            for weight, value in zip(row, x):
                total += weight * value
            out.append(activate(total))

        if training and self.dropout > 0.0:
            # Inverted dropout: scale at TRAIN time so inference needs no
            # adjustment. Forgetting the scaling is the classic silent bug --
            # the model trains fine and then systematically under-predicts in
            # production, because every activation is (1 - p) times smaller
            # than it was during training.
            keep = 1.0 - self.dropout
            self._mask = [
                (1.0 / keep) if rng.random() < keep else 0.0 for _ in out
            ]
            out = [value * m for value, m in zip(out, self._mask)]
        else:
            self._mask = None

        self._last_output = out
        return out

    def backward(self, grad_out: Vector) -> Vector:
        """
        Returns the gradient with respect to this layer's INPUT, and
        accumulates the weight/bias gradients into ``_grad_w`` / ``_grad_b``
        via the returned tuple in :meth:`MLP._backprop`.
        """
        raise NotImplementedError      # handled inline in MLP for speed


@dataclass
class TrainingHistory:
    """What happened during a fit. Reported, not hidden."""

    epochs_run: int = 0
    best_epoch: int = 0
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str = ""

    @property
    def best_val_loss(self) -> float:
        return min(self.val_loss) if self.val_loss else float("inf")

    @property
    def final_gap(self) -> float:
        """
        Validation loss minus training loss at the best epoch.

        The overfitting tell. A gap near zero on a few hundred samples usually
        means the model is too small to fit anything; a large one means it has
        memorised the training set. Both are reported rather than smoothed
        over.
        """
        if not self.val_loss or not self.train_loss:
            return 0.0
        index = min(self.best_epoch, len(self.train_loss) - 1)
        return self.val_loss[index] - self.train_loss[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_val_loss": round(self.best_val_loss, 5),
            "final_gap": round(self.final_gap, 5),
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "train_loss": [round(v, 5) for v in self.train_loss],
            "val_loss": [round(v, 5) for v in self.val_loss],
        }


class MLP:
    """
    A small multi-layer perceptron trained with Adam on binary cross-entropy.

    Adam rather than plain SGD because the input features here have very
    different effective scales even after standardisation -- a signal that is
    usually saturated at +/-1 and one that hovers near zero need different
    step sizes, and a single global learning rate serves neither. Adam's
    per-parameter second-moment estimate handles that without hand-tuning,
    which matters when the model is retrained automatically on a schedule and
    nobody is watching.
    """

    def __init__(
        self,
        n_features: int,
        hidden: Sequence[LayerSpec] | None = None,
        *,
        learning_rate: float = 0.01,
        l2: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2 = l2
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.seed = seed
        self._rng = random.Random(seed)
        self._step = 0

        specs = list(hidden) if hidden is not None else [
            LayerSpec(16, Activation.RELU, dropout=0.10),
            LayerSpec(8, Activation.RELU, dropout=0.10),
        ]
        self.spec = specs

        self.layers: list[Dense] = []
        size = n_features
        for layer_spec in specs:
            self.layers.append(
                Dense(size, layer_spec.units, layer_spec.activation,
                      layer_spec.dropout, self._rng)
            )
            size = layer_spec.units
        # Output layer: one unit, sigmoid, never dropped out.
        self.layers.append(Dense(size, 1, Activation.SIGMOID, 0.0, self._rng))

        self.history = TrainingHistory()
        self.fitted = False

    # -- inference ---------------------------------------------------------

    def predict_one(self, x: Sequence[float], *, training: bool = False) -> float:
        values = list(x)
        for layer in self.layers:
            values = layer.forward(values, training=training, rng=self._rng)
        return values[0]

    def predict(self, xs: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_one(x) for x in xs]

    # -- training ----------------------------------------------------------

    def _backprop(self, x: Sequence[float], target: float) -> float:
        """
        One forward/backward pass. Returns this sample's loss.

        The output layer is sigmoid and the loss is binary cross-entropy, so
        the gradient of the loss with respect to the output layer's PRE-
        activation collapses to (prediction - target). The sigmoid derivative
        and the cross-entropy derivative cancel exactly; computing both and
        multiplying them is a common way to introduce a subtle factor of
        y(1-y) that stalls training whenever the model is confident.
        """
        prediction = self.predict_one(x, training=True)
        clipped = min(1 - 1e-9, max(1e-9, prediction))
        loss = -(target * math.log(clipped) + (1 - target) * math.log(1 - clipped))

        # Gradient wrt the output layer's pre-activation.
        delta: Vector = [prediction - target]

        for index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[index]
            inputs = layer._last_input

            grad_w = [[d * value for value in inputs] for d in delta]
            grad_b = list(delta)

            if index > 0:
                previous = self.layers[index - 1]
                grad_input = [0.0] * layer.n_in
                for out_index, d in enumerate(delta):
                    row = layer.w[out_index]
                    for in_index, weight in enumerate(row):
                        grad_input[in_index] += d * weight

                # Chain through the previous layer's activation, and through
                # its dropout mask -- a dropped unit received no gradient
                # because it contributed nothing to the output.
                grad_fn = _BACKWARD[previous.activation]
                new_delta: Vector = []
                for unit_index, gradient in enumerate(grad_input):
                    output = previous._last_output[unit_index]
                    if previous._mask is not None:
                        if previous._mask[unit_index] == 0.0:
                            new_delta.append(0.0)
                            continue
                        # Undo the inverted-dropout scaling to recover the
                        # pre-scaling activation the derivative expects.
                        output = output / previous._mask[unit_index]
                    new_delta.append(gradient * grad_fn(output))
                delta = new_delta

            self._apply_adam(layer, grad_w, grad_b)

        return loss

    def _apply_adam(self, layer: Dense, grad_w: Matrix, grad_b: Vector) -> None:
        self._step += 1
        bias_correction_1 = 1 - self.beta1 ** self._step
        bias_correction_2 = 1 - self.beta2 ** self._step

        for i in range(layer.n_out):
            row_w, row_m, row_v, row_g = (
                layer.w[i], layer.mw[i], layer.vw[i], grad_w[i],
            )
            for j in range(layer.n_in):
                # L2 as a gradient term rather than as decoupled weight decay:
                # with Adam these differ, and the coupled form is the one the
                # loss reported below actually corresponds to.
                gradient = row_g[j] + self.l2 * row_w[j]
                row_m[j] = self.beta1 * row_m[j] + (1 - self.beta1) * gradient
                row_v[j] = self.beta2 * row_v[j] + (1 - self.beta2) * gradient * gradient
                m_hat = row_m[j] / bias_correction_1
                v_hat = row_v[j] / bias_correction_2
                row_w[j] -= self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)

            gradient = grad_b[i]
            layer.mb[i] = self.beta1 * layer.mb[i] + (1 - self.beta1) * gradient
            layer.vb[i] = self.beta2 * layer.vb[i] + (1 - self.beta2) * gradient * gradient
            m_hat = layer.mb[i] / bias_correction_1
            v_hat = layer.vb[i] / bias_correction_2
            layer.b[i] -= self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)

    def fit(
        self,
        x_train: Sequence[Sequence[float]],
        y_train: Sequence[float],
        *,
        x_val: Sequence[Sequence[float]] | None = None,
        y_val: Sequence[float] | None = None,
        epochs: int = 200,
        patience: int = 20,
        shuffle: bool = True,
        verbose: bool = False,
    ) -> TrainingHistory:
        """
        Train, with early stopping on the validation split.

        ``patience`` epochs without improvement stops the run and RESTORES the
        best weights. Stopping without restoring is the version of early
        stopping that does nothing: it halts at whichever overfitted epoch
        happened to be `patience` steps past the optimum.
        """
        history = TrainingHistory()
        has_validation = bool(x_val) and y_val is not None and len(x_val) > 0

        best_loss = float("inf")
        best_weights: list[tuple[Matrix, Vector]] | None = None
        since_improvement = 0
        order = list(range(len(x_train)))

        for epoch in range(epochs):
            if shuffle:
                self._rng.shuffle(order)

            total = 0.0
            for index in order:
                total += self._backprop(x_train[index], y_train[index])
            train_loss = total / max(1, len(order))
            history.train_loss.append(train_loss)

            if has_validation:
                validation_loss = self.log_loss(x_val, y_val)
            else:
                validation_loss = train_loss
            history.val_loss.append(validation_loss)
            history.epochs_run = epoch + 1

            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                history.best_epoch = epoch
                best_weights = [
                    ([row[:] for row in layer.w], layer.b[:])
                    for layer in self.layers
                ]
                since_improvement = 0
            else:
                since_improvement += 1

            if verbose and epoch % 20 == 0:
                logger.info("epoch %d: train %.5f val %.5f",
                            epoch, train_loss, validation_loss)

            if since_improvement >= patience:
                history.stopped_early = True
                history.stop_reason = (
                    f"no validation improvement for {patience} epochs; "
                    f"restored the weights from epoch {history.best_epoch}"
                )
                break

        if best_weights is not None:
            for layer, (weights, biases) in zip(self.layers, best_weights):
                layer.w = weights
                layer.b = biases

        if not history.stopped_early:
            history.stop_reason = f"ran the full {epochs} epochs"

        self.history = history
        self.fitted = True
        return history

    # -- metrics -----------------------------------------------------------

    def log_loss(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> float:
        if not xs:
            return 0.0
        total = 0.0
        for x, y in zip(xs, ys):
            p = min(1 - 1e-9, max(1e-9, self.predict_one(x)))
            total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return total / len(xs)

    def brier(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> float:
        if not xs:
            return 0.0
        return sum((self.predict_one(x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "spec": [
                {"units": s.units, "activation": s.activation, "dropout": s.dropout}
                for s in self.spec
            ],
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "seed": self.seed,
            "fitted": self.fitted,
            "layers": [{"w": layer.w, "b": layer.b} for layer in self.layers],
            "history": self.history.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MLP":
        model = cls(
            raw["n_features"],
            [LayerSpec(s["units"], s["activation"], s["dropout"])
             for s in raw["spec"]],
            learning_rate=raw.get("learning_rate", 0.01),
            l2=raw.get("l2", 1e-4),
            seed=raw.get("seed", 42),
        )
        for layer, stored in zip(model.layers, raw["layers"]):
            layer.w = [list(row) for row in stored["w"]]
            layer.b = list(stored["b"])
        model.fitted = raw.get("fitted", False)
        return model

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __repr__(self) -> str:      # pragma: no cover
        shape = " -> ".join(
            [str(self.n_features)] + [str(s.units) for s in self.spec] + ["1"]
        )
        return f"MLP({shape}, params={self.parameter_count})"

    @property
    def parameter_count(self) -> int:
        return sum(layer.n_in * layer.n_out + layer.n_out for layer in self.layers)
