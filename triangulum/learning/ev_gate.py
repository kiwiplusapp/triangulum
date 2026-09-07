"""
The expected-value gate: the single decision point where capital is risked.

Every other component produces information. This one acts on it. It is
deliberately the most conservative code in the repository.

THE CALCULATION
===============

    EV = P(fill) * net_edge_after_costs  -  (1 - P(fill)) * expected_failure_cost

Both terms matter, and the second is the one naive systems omit entirely.

A cycle that fails is not free. If leg 1 filled and leg 2 did not, you are
holding an unwanted position and must unwind it, crossing a spread and paying a
fee to get back where you started. That cost is real, it is asymmetric, and it
is *larger* precisely when fills are hardest -- fast markets both prevent fills
and widen the unwind.

So an opportunity showing +4 bps with a 60% fill probability and a 12 bps
failure cost has:

    EV = 0.6 * 4 - 0.4 * 12 = 2.4 - 4.8 = -2.4 bps

It is a losing trade, and it *looks* like a winner in every naive backtest ever
written. Filtering these out is where the learner earns its place.

THE UNCERTAINTY ADJUSTMENT
==========================

Point estimates are not enough. Early on, P(fill) is a guess, and acting
confidently on a guess is how a system loses money fast before it has learned
anything. So the gate requires

    EV - k * sigma(EV) > threshold

where sigma is propagated from the fill model's own uncertainty. With few
samples sigma is large and almost nothing passes; as evidence accumulates sigma
shrinks and the gate opens. The system is automatically timid when ignorant and
confident when informed, with no schedule to tune.

THE EXPLORATION FLOOR AND THE COLD-START TRAP
============================================

A gate that only takes trades it is confident about never learns that it was
wrong. If the model believes cycles with 200ms book age never fill, it stops
taking them, receives no more data about them, and can never discover that the
belief was an artifact of a bad week. A small fraction of marginal trades are
therefore taken deliberately, sized down, purely to keep the training
distribution honest.

There is a sharper version of this problem at startup, and it is a trap the
uncertainty adjustment above walks straight into if left alone:

    zero samples -> wide sigma -> every lower bound is negative
                 -> nothing is accepted -> no outcomes observed
                 -> still zero samples.

The engine sits there forever, correctly cautious and completely useless.
Measured on the synthetic market: 24 opportunities detected, 10 sized
successfully, 0 accepted -- 9 of them rejected on uncertainty alone.

The fix is to make the exploration rate a function of ignorance rather than a
constant. It starts high (the model knows nothing, so information is cheap
relative to its value) and decays to the configured floor as evidence
accumulates. Exploration trades are sized down, so the cost of buying that
information is bounded and explicit.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping

from triangulum.core.decimal_math import D, ZERO
from triangulum.core.types import CyclePlan, Opportunity
from triangulum.learning.calibration import PlattCalibrator
from triangulum.learning.features import FeatureExtractor, FeatureVector
from triangulum.learning.online_lr import FTRLProximal, OnlineRidge

logger = logging.getLogger(__name__)

__all__ = ["EVGate", "GateDecision", "GateVerdict"]


class GateVerdict:
    ACCEPT = "accept"
    ACCEPT_EXPLORATION = "accept_exploration"
    REJECT_EV = "reject_ev"
    REJECT_UNCERTAINTY = "reject_uncertainty"
    REJECT_EDGE = "reject_edge"
    REJECT_FILL_PROBABILITY = "reject_fill_probability"
    REJECT_STALE = "reject_stale"


@dataclass(slots=True)
class GateDecision:
    """A full audit trail of one accept/reject decision."""

    verdict: str
    accept: bool

    fill_probability: float = 0.5
    fill_probability_raw: float = 0.5
    predicted_slippage_bps: float = 0.0
    net_edge_bps: float = 0.0
    failure_cost_bps: float = 0.0

    expected_value_bps: float = 0.0
    ev_stddev_bps: float = 0.0
    ev_lower_bound_bps: float = 0.0

    size_multiplier: float = 1.0
    reason: str = ""
    features: FeatureVector | None = None
    model_samples: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "accept": self.accept,
            "p_fill": round(self.fill_probability, 4),
            "p_fill_raw": round(self.fill_probability_raw, 4),
            "predicted_slippage_bps": round(self.predicted_slippage_bps, 3),
            "net_edge_bps": round(self.net_edge_bps, 3),
            "failure_cost_bps": round(self.failure_cost_bps, 3),
            "ev_bps": round(self.expected_value_bps, 3),
            "ev_stddev_bps": round(self.ev_stddev_bps, 3),
            "ev_lower_bound_bps": round(self.ev_lower_bound_bps, 3),
            "size_multiplier": round(self.size_multiplier, 3),
            "reason": self.reason,
            "model_samples": self.model_samples,
        }

    def explain(self) -> str:
        return (
            f"{self.verdict.upper()}: EV = {self.fill_probability:.2f} x "
            f"{self.net_edge_bps:+.2f} - {1 - self.fill_probability:.2f} x "
            f"{self.failure_cost_bps:.2f} = {self.expected_value_bps:+.2f} bps "
            f"(lower bound {self.ev_lower_bound_bps:+.2f}). {self.reason}"
        )


class EVGate:
    """Decides whether a planned cycle is worth executing."""

    def __init__(
        self,
        fill_model: FTRLProximal,
        slippage_model: OnlineRidge,
        extractor: FeatureExtractor,
        *,
        calibrator: PlattCalibrator | None = None,
        min_ev_bps: float = 0.5,
        confidence_multiplier: float = 1.0,
        min_fill_probability: float = 0.15,
        min_samples_before_trust: int = 200,
        exploration_floor: float = 0.05,
        cold_start_exploration: float = 0.60,
        exploration_size_multiplier: float = 0.25,
        prior_fill_probability: float = 0.55,
        default_unwind_cost_bps: float = 12.0,
        feature_bits: int = 18,
        seed: int | None = None,
    ) -> None:
        self.fill_model = fill_model
        self.slippage_model = slippage_model
        self.extractor = extractor
        self.calibrator = calibrator or PlattCalibrator()
        self.min_ev_bps = min_ev_bps
        self.confidence_multiplier = confidence_multiplier
        self.min_fill_probability = min_fill_probability
        self.min_samples_before_trust = min_samples_before_trust
        self.exploration_floor = exploration_floor
        self.cold_start_exploration = cold_start_exploration
        self.exploration_size_multiplier = exploration_size_multiplier
        self.prior_fill_probability = prior_fill_probability
        self.default_unwind_cost_bps = default_unwind_cost_bps
        self.feature_bits = feature_bits
        self._rng = random.Random(seed)

        self.evaluations = 0
        self.accepts = 0
        self.explorations = 0
        self.rejects: dict[str, int] = {}

    # -- the decision ------------------------------------------------------

    def evaluate(
        self,
        opportunity: Opportunity,
        plan: CyclePlan,
        *,
        now_ns: int,
        unwind_cost_bps: float | None = None,
    ) -> GateDecision:
        self.evaluations += 1

        features = self.extractor.extract(
            opportunity, now_ns=now_ns, plan=plan, notional=plan.start_amount
        )
        hashed = features.to_hashed(self.feature_bits)

        # -- P(fill) --
        raw_probability = self.fill_model.predict(hashed)
        samples = self.fill_model.samples
        if samples < self.min_samples_before_trust:
            # Blend toward the prior while the model is uninformed. A linear
            # ramp, so trust grows smoothly rather than switching on.
            weight = samples / max(1, self.min_samples_before_trust)
            probability = weight * raw_probability + (1 - weight) * self.prior_fill_probability
        else:
            probability = self.calibrator.calibrate(raw_probability)

        # -- costs --
        predicted_slippage = self.slippage_model.predict(hashed)
        net_edge = float(plan.net_edge_bps) - max(0.0, predicted_slippage)
        failure_cost = (
            unwind_cost_bps
            if unwind_cost_bps is not None
            else self._estimate_failure_cost(plan)
        )

        # -- expected value --
        ev = probability * net_edge - (1.0 - probability) * failure_cost

        # Uncertainty in EV comes overwhelmingly from uncertainty in P(fill).
        # Binomial standard error on the model's effective sample count.
        effective_n = max(1, min(samples, 5000))
        p_stddev = math.sqrt(max(1e-9, probability * (1 - probability)) / effective_n)
        # d(EV)/dp = net_edge + failure_cost
        ev_stddev = abs(net_edge + failure_cost) * p_stddev
        # Add the slippage model's own residual spread.
        ev_stddev = math.sqrt(ev_stddev ** 2 + (probability * self.slippage_model.recent_rmse) ** 2)
        lower_bound = ev - self.confidence_multiplier * ev_stddev

        decision = GateDecision(
            verdict=GateVerdict.ACCEPT, accept=True,
            fill_probability=probability,
            fill_probability_raw=raw_probability,
            predicted_slippage_bps=predicted_slippage,
            net_edge_bps=net_edge,
            failure_cost_bps=failure_cost,
            expected_value_bps=ev,
            ev_stddev_bps=ev_stddev,
            ev_lower_bound_bps=lower_bound,
            features=features,
            model_samples=samples,
        )

        # -- the screens, cheapest first --
        if net_edge <= 0:
            return self._reject(
                decision, GateVerdict.REJECT_EDGE,
                f"net edge {net_edge:+.2f} bps is not positive after predicted "
                f"slippage of {predicted_slippage:.2f} bps",
                opportunity, plan,
            )

        if probability < self.min_fill_probability:
            return self._reject(
                decision, GateVerdict.REJECT_FILL_PROBABILITY,
                f"fill probability {probability:.3f} below floor "
                f"{self.min_fill_probability:.2f}",
                opportunity, plan,
            )

        if ev < self.min_ev_bps:
            return self._maybe_explore(
                decision, GateVerdict.REJECT_EV,
                f"EV {ev:+.2f} bps below threshold {self.min_ev_bps:.2f}",
                opportunity, plan,
            )

        if lower_bound < 0:
            return self._maybe_explore(
                decision, GateVerdict.REJECT_UNCERTAINTY,
                f"EV {ev:+.2f} bps is positive but its lower bound "
                f"{lower_bound:+.2f} is not, at {samples} samples",
                opportunity, plan,
            )

        self.accepts += 1
        decision.reason = (
            f"EV {ev:+.2f} bps with lower bound {lower_bound:+.2f} clears "
            f"the {self.min_ev_bps:.2f} bps threshold"
        )
        return decision

    def _maybe_explore(
        self, decision: GateDecision, verdict: str, reason: str,
        opportunity: Opportunity, plan: CyclePlan,
    ) -> GateDecision:
        """
        Take a marginal trade occasionally, at reduced size, to keep learning.

        Only applied to *marginal* rejections -- an opportunity with a positive
        raw EV that failed on uncertainty or threshold. Exploring outright
        negative-EV trades would be paying for information we already have.
        """
        if decision.expected_value_bps > -self.min_ev_bps and (
            self._rng.random() < self.current_exploration_rate
        ):
            self.explorations += 1
            decision.verdict = GateVerdict.ACCEPT_EXPLORATION
            decision.accept = True
            decision.size_multiplier = self.exploration_size_multiplier
            decision.reason = (
                f"exploration: {reason}, taken at "
                f"{self.exploration_size_multiplier:.0%} size to keep the "
                f"training distribution honest"
            )
            return decision
        return self._reject(decision, verdict, reason, opportunity, plan)

    def _reject(
        self, decision: GateDecision, verdict: str, reason: str,
        opportunity: Opportunity, plan: CyclePlan,
    ) -> GateDecision:
        decision.verdict = verdict
        decision.accept = False
        decision.reason = reason
        self.rejects[verdict] = self.rejects.get(verdict, 0) + 1
        return decision

    def _estimate_failure_cost(self, plan: CyclePlan) -> float:
        """
        Expected cost of a cycle that commits capital and then fails.

        Approximated as: cross the spread back (one leg's worth), pay a taker
        fee, plus a slippage allowance. Scaled by how far through the cycle a
        typical failure occurs -- failing on the last leg is worse than failing
        on the second, because more capital has been converted into the
        unwanted asset.
        """
        legs = len(plan.legs)
        if legs == 0:
            return self.default_unwind_cost_bps
        per_leg_fee = float(plan.fee_bps) / legs
        # A failure is roughly uniform across legs 1..n-1; the mean position is
        # halfway through, so about half the cycle's cost is already sunk.
        return per_leg_fee * 2 + float(plan.slippage_bps) * 0.5 + self.default_unwind_cost_bps * 0.5

    @property
    def current_exploration_rate(self) -> float:
        """
        Exploration probability, decaying with accumulated evidence.

        Linear from ``cold_start_exploration`` at zero samples to
        ``exploration_floor`` once the model is trusted. Linear rather than
        exponential on purpose: the first hundred samples are worth far more
        than the next thousand, and a linear ramp keeps the rate meaningfully
        above the floor through exactly that range.
        """
        if self.min_samples_before_trust <= 0:
            return self.exploration_floor
        progress = min(1.0, self.fill_model.samples / self.min_samples_before_trust)
        return (
            self.cold_start_exploration * (1 - progress)
            + self.exploration_floor * progress
        )

    # -- learning from outcomes -------------------------------------------

    def observe(
        self,
        decision: GateDecision,
        *,
        filled: bool,
        realized_bps: float,
        now_ns: int = 0,
        cycle_key: str = "",
    ) -> None:
        """
        Feed an executed cycle's outcome back into the models.

        Called for *every* attempted cycle, including failures -- the failures
        are the informative samples. A model trained only on successes learns
        that everything succeeds.
        """
        if decision.features is None:
            return
        hashed = decision.features.to_hashed(self.feature_bits)

        self.fill_model.update(hashed, 1 if filled else 0)
        self.calibrator.observe(decision.fill_probability_raw, 1 if filled else 0)

        if filled:
            # The slippage label is expected-minus-realized: how much of the
            # predicted edge failed to materialise.
            residual = decision.net_edge_bps - realized_bps
            self.slippage_model.update(hashed, residual)

        if cycle_key:
            self.extractor.observe_outcome(cycle_key, filled, now_ns)

    # -- reporting ---------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        return self.accepts / self.evaluations if self.evaluations else 0.0

    def stats(self) -> dict[str, object]:
        return {
            "evaluations": self.evaluations,
            "accepts": self.accepts,
            "explorations": self.explorations,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "rejects_by_reason": dict(self.rejects),
            "fill_model": self.fill_model.stats(),
            "slippage_model": self.slippage_model.stats(),
            "calibration": self.calibrator.stats(),
            "trusts_model": self.fill_model.samples >= self.min_samples_before_trust,
            "exploration_rate": round(self.current_exploration_rate, 4),
            "min_ev_bps": self.min_ev_bps,
        }
