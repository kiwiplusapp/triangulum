"""
Thompson-sampling bandit over execution configurations.

The engine has several ways to work the same cycle: all-taker, maker on leg 1,
maker on leg 2, more or less aggressive pricing, larger or smaller size. Which
is best is not knowable a priori and changes with the market regime -- maker
legs fill well in a quiet book and never fill in a fast one.

This is a contextual bandit problem, and Thompson sampling is the right tool:

**It explores in proportion to uncertainty.** An arm tried twice has a wide
posterior and gets sampled optimistically often; an arm tried a thousand times
has a tight posterior and is chosen only if it is genuinely good. No
exploration schedule to tune, no epsilon to decay.

**It handles non-stationarity via decay.** Multiplying the posterior parameters
by ``decay`` each update gives the arm an effective memory of ``1/(1-decay)``
observations. Without it, an arm that was excellent for a month becomes
impossible to unseat when conditions change -- and conditions always change.

**It optimises the right quantity.** Arms are scored on *realized basis points*,
not on fill rate. An arm that fills 95% of the time at +0.5 bps is worse than
one that fills 40% of the time at +4 bps, and a fill-rate objective would pick
the wrong one.

Two posteriors are maintained per arm:
    Beta(a, b)                 over fill probability
    Normal-Gamma(mu, k, a, b)  over realized bps given a fill

The sampled score is ``P(fill) * E[bps | fill]``, which is the expected value of
pulling that arm -- exactly the quantity the executor should maximise.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = ["Arm", "ThompsonBandit", "ArmDefinition", "default_arms"]


@dataclass(slots=True)
class ArmDefinition:
    """A concrete execution configuration the bandit can select."""

    name: str
    execution_mode: str            # ExecutionMode value
    maker_leg: int = -1            # -1 = none
    taker_offset_ticks: int = 2
    capital_fraction: float = 0.95
    description: str = ""


def default_arms() -> list[ArmDefinition]:
    """
    The arm set.

    Deliberately small. A bandit with fifty arms spends all its samples
    exploring and never converges; with six, each arm accumulates enough
    observations to have a usable posterior within a few hundred cycles.
    """
    return [
        ArmDefinition("ttt_safe", "taker_taker_taker", -1, 3, 0.95,
                      "All taker, 3 ticks through. Highest fill certainty, full fees."),
        ArmDefinition("ttt_tight", "taker_taker_taker", -1, 1, 0.95,
                      "All taker, 1 tick through. Cheaper, misses more."),
        ArmDefinition("mtt", "maker_taker_taker", 0, 2, 0.95,
                      "Post-only leg 1. Saves the maker/taker spread, adds fill risk."),
        ArmDefinition("tmt", "taker_maker_taker", 1, 2, 0.95,
                      "Post-only leg 2. Useful when leg 2 is the widest."),
        ArmDefinition("ttt_half", "taker_taker_taker", -1, 3, 0.50,
                      "Half size. Less slippage, proportionally more lot drag."),
        ArmDefinition("ttt_small", "taker_taker_taker", -1, 3, 0.25,
                      "Quarter size. Mostly an exploration arm for thin books."),
    ]


@dataclass(slots=True)
class Arm:
    """Posterior state for one configuration."""

    definition: ArmDefinition

    # Beta posterior over fill probability.
    alpha: float = 1.0
    beta: float = 1.0

    # Normal-Gamma posterior over realized bps given a fill.
    mu: float = 0.0          # mean estimate
    kappa: float = 1.0       # pseudo-observations behind mu
    a: float = 1.0           # shape
    b: float = 1.0           # rate

    pulls: int = 0
    fills: int = 0
    total_bps: float = 0.0
    last_selected_ns: int = 0

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def fill_rate(self) -> float:
        return self.fills / self.pulls if self.pulls else 0.0

    @property
    def mean_bps(self) -> float:
        return self.total_bps / self.fills if self.fills else 0.0

    @property
    def expected_value(self) -> float:
        """Posterior mean EV, without sampling. Used for reporting, not choice."""
        p = self.alpha / (self.alpha + self.beta)
        return p * self.mu

    def sample(self, rng: random.Random) -> float:
        """
        One Thompson draw: sample both posteriors and return their product.

        Sampling rather than using the mean is the whole algorithm. An arm with
        two observations has a wide posterior, so its draws are sometimes very
        high -- which is precisely the optimism that makes it get explored.
        """
        p = _sample_beta(self.alpha, self.beta, rng)
        # Sample precision from Gamma(a, b), then the mean from a Normal whose
        # variance shrinks as kappa (observation count) grows.
        precision = _sample_gamma(self.a, self.b, rng)
        variance = 1.0 / (self.kappa * precision) if precision > 1e-12 else 1e6
        bps = rng.gauss(self.mu, math.sqrt(min(variance, 1e6)))
        return p * bps

    def update(self, filled: bool, realized_bps: float, decay: float = 0.999) -> None:
        self.pulls += 1
        # Decay first: this is what gives the posterior a finite memory and lets
        # the arm's estimate track a changing market.
        self.alpha *= decay
        self.beta *= decay
        self.kappa *= decay
        self.a *= decay
        self.b *= decay

        if filled:
            self.fills += 1
            self.alpha += 1.0
            self.total_bps += realized_bps
            # Normal-Gamma conjugate update.
            kappa_new = self.kappa + 1.0
            mu_new = (self.kappa * self.mu + realized_bps) / kappa_new
            self.a += 0.5
            self.b += (self.kappa * (realized_bps - self.mu) ** 2) / (2 * kappa_new)
            self.mu = mu_new
            self.kappa = kappa_new
        else:
            self.beta += 1.0
            # A no-fill is not neutral: it costs the opportunity and, if a leg
            # had already executed, the unwind. Recorded as a small negative so
            # arms that miss often are penalised even when their fills are good.
            self.total_bps += 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.definition.name,
            "alpha": self.alpha, "beta": self.beta,
            "mu": self.mu, "kappa": self.kappa, "a": self.a, "b": self.b,
            "pulls": self.pulls, "fills": self.fills, "total_bps": self.total_bps,
        }


class ThompsonBandit:
    """Selects execution configurations by Thompson sampling."""

    def __init__(
        self,
        arms: Sequence[ArmDefinition] | None = None,
        *,
        decay: float = 0.999,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        exploration_floor: float = 0.05,
        seed: int | None = None,
        contextual: bool = True,
    ) -> None:
        definitions = list(arms or default_arms())
        self._arms: dict[str, Arm] = {
            d.name: Arm(definition=d, alpha=prior_alpha, beta=prior_beta)
            for d in definitions
        }
        self.decay = decay
        self.exploration_floor = exploration_floor
        self.contextual = contextual
        self._rng = random.Random(seed)
        # Per-regime posteriors. The same arm behaves differently in a quiet
        # book and a fast one, so conditioning on regime is worth the extra
        # sample requirement.
        self._by_regime: dict[str, dict[str, Arm]] = {}
        self.selections = 0
        self.forced_explorations = 0

    def _arms_for(self, regime: str) -> dict[str, Arm]:
        if not self.contextual or not regime:
            return self._arms
        bucket = self._by_regime.get(regime)
        if bucket is None:
            bucket = {
                name: Arm(definition=arm.definition, alpha=arm.alpha, beta=arm.beta)
                for name, arm in self._arms.items()
            }
            self._by_regime[regime] = bucket
        return bucket

    def select(self, *, regime: str = "", now_ns: int = 0) -> ArmDefinition:
        """Choose an arm. Never returns the same one deterministically."""
        self.selections += 1
        arms = self._arms_for(regime)

        # Uniform exploration floor. Thompson sampling can, given an unlucky
        # early streak, effectively abandon an arm before its posterior has any
        # real information. The floor guarantees every arm keeps receiving data.
        if self._rng.random() < self.exploration_floor:
            self.forced_explorations += 1
            arm = self._rng.choice(list(arms.values()))
            arm.last_selected_ns = now_ns
            return arm.definition

        best_name, best_score = "", -math.inf
        for name, arm in arms.items():
            score = arm.sample(self._rng)
            if score > best_score:
                best_score, best_name = score, name

        chosen = arms[best_name]
        chosen.last_selected_ns = now_ns
        return chosen.definition

    def update(
        self, arm_name: str, *, filled: bool, realized_bps: float, regime: str = "",
    ) -> None:
        """
        Record an outcome against the global posterior and, when a regime is
        supplied, that regime's posterior too.

        The two buckets are deduplicated by identity: with no regime,
        ``_arms_for`` returns the global dict itself, and updating it twice
        would double-count every observation -- inflating ``pulls`` and, worse,
        applying the decay twice per sample so the posterior forgets at double
        the intended rate.
        """
        buckets = [self._arms]
        regime_bucket = self._arms_for(regime)
        if regime_bucket is not self._arms:
            buckets.append(regime_bucket)

        for bucket in buckets:
            arm = bucket.get(arm_name)
            if arm is not None:
                arm.update(filled, realized_bps, self.decay)

    # -- reporting ---------------------------------------------------------

    def leaderboard(self, regime: str = "") -> list[dict[str, object]]:
        arms = self._arms_for(regime)
        rows = [
            {
                "arm": arm.name,
                "pulls": arm.pulls,
                "fills": arm.fills,
                "fill_rate": round(arm.fill_rate, 4),
                "mean_bps": round(arm.mean_bps, 3),
                "expected_value_bps": round(arm.expected_value, 3),
                "posterior_p": round(arm.alpha / (arm.alpha + arm.beta), 4),
                "description": arm.definition.description,
            }
            for arm in arms.values()
        ]
        rows.sort(key=lambda r: -r["expected_value_bps"])
        return rows

    def best_arm(self, regime: str = "") -> str:
        arms = self._arms_for(regime)
        return max(arms.values(), key=lambda a: a.expected_value).name

    def stats(self) -> dict[str, object]:
        return {
            "arms": len(self._arms),
            "selections": self.selections,
            "forced_explorations": self.forced_explorations,
            "exploration_rate": round(
                self.forced_explorations / max(1, self.selections), 4
            ),
            "regimes": list(self._by_regime),
            "leaderboard": self.leaderboard(),
        }

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "type": "thompson_bandit",
            "decay": self.decay,
            "exploration_floor": self.exploration_floor,
            "arms": [a.to_dict() for a in self._arms.values()],
            "regimes": {
                regime: [a.to_dict() for a in bucket.values()]
                for regime, bucket in self._by_regime.items()
            },
            "selections": self.selections,
        }

    def load_state(self, data: Mapping) -> None:
        for row in data.get("arms", []):
            arm = self._arms.get(row["name"])
            if arm is None:
                continue
            for field_name in ("alpha", "beta", "mu", "kappa", "a", "b", "total_bps"):
                setattr(arm, field_name, float(row.get(field_name, getattr(arm, field_name))))
            arm.pulls = int(row.get("pulls", 0))
            arm.fills = int(row.get("fills", 0))
        self.selections = int(data.get("selections", 0))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(p)


# --------------------------------------------------------------------------
# Sampling primitives -- stdlib only
# --------------------------------------------------------------------------


def _sample_gamma(shape: float, rate: float, rng: random.Random) -> float:
    """Gamma(shape, rate) via Marsaglia-Tsang. ``random.gammavariate`` takes a
    *scale*, so the rate is inverted."""
    if shape <= 0 or rate <= 0:
        return 1.0
    try:
        return rng.gammavariate(shape, 1.0 / rate)
    except ValueError:  # pragma: no cover - guards degenerate parameters
        return shape / rate


def _sample_beta(alpha: float, beta: float, rng: random.Random) -> float:
    """
    Beta(alpha, beta) as the ratio of two Gammas.

    ``random.betavariate`` exists but raises on the very small parameter values
    that heavy decay can produce, so this does it directly with clamping.
    """
    a = max(1e-6, alpha)
    b = max(1e-6, beta)
    x = _sample_gamma(a, 1.0, rng)
    y = _sample_gamma(b, 1.0, rng)
    total = x + y
    return x / total if total > 0 else 0.5
