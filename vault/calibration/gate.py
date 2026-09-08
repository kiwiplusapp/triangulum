"""
The capital gate.

This is the answer to "make it generate returns safely", and it is a mechanism
rather than a warning.

    Position size is a function of DEMONSTRATED calibration.
    Before there is a track record, that function returns zero.

Not a small number. Zero. The agent cannot risk capital on its own say-so,
because its own say-so is exactly the thing under test. It earns size by being
right, measured, over a sample large enough to distinguish skill from luck.

## The five gates, in order

1. **Sample.** Fewer than ``min_samples`` resolved calls -> size zero. At
   n=30 a 60% hit rate has a 95% confidence interval of roughly [42%, 75%];
   it cannot be told apart from a coin flip, so it is not paid.

   The default is 100, which is what ``scoring._adequacy`` has always said is
   needed before a genuine 60% edge separates from 50%. It was 50, which
   contradicted that docstring and mattered: measured across 40 seeds, the
   no-edge "lucky" profile got capital in 15% of runs at n=50 and 2.5% at
   n=100, while the honest 65% forecaster unlocked in 100% of runs either
   way. Raising it cost the real edge nothing and closed most of the leak.

2. **Skill.** The Brier score must beat the 0.25 baseline. A system scoring
   worse than "always say 0.5" is not a forecasting system, however articulate
   its reasoning.

3. **Significance.** The *always-valid* lower bound on the hit rate must
   exceed 0.5. This is the gate that stops a lucky streak from unlocking
   capital: 7 wins out of 10 looks impressive and has a lower bound of 0.40.

   It must be the always-valid bound and not the ordinary Wilson bound,
   because this gate is consulted after every single resolved call. A 95%
   bound checked once excludes a coin flip 95% of the time; checked after
   each of 300 calls it is crossed by a genuinely edgeless forecaster in
   **14.1% of runs** (measured, 2,000 simulations). The time-uniform bound
   holds that to 0.5%. The cost is real and worth stating: a true 65%
   forecaster now waits a median of 94 resolved calls for capital instead of
   50. See ``vault/resolve/scoring.py :: sequential_wilson_interval``.

4. **Calibration.** Overconfidence beyond ``max_overconfidence`` scales size
   down proportionally. A model whose 0.80 calls come in at 0.60 is not
   useless -- its ordering may be fine -- but its stated probabilities cannot
   be used for sizing until they are corrected.

## Sizing

Fractional Kelly on the *lower bound* of the measured edge, not the point
estimate. Full Kelly is famously optimal for terminal wealth and famously
unusable in practice: it assumes the edge is known exactly, and it prescribes
position sizes whose drawdowns no human tolerates. Quarter-Kelly on a
conservative edge estimate is the standard professional compromise, and using
the lower confidence bound rather than the mean means the size shrinks
automatically when the sample is small.

    f* = (p*b - q) / b        Kelly fraction
    f  = f* * kelly_fraction  scaled down
    where p is the ALWAYS-VALID LOWER BOUND of the measured hit rate

The result is a system whose risk grows only as its evidence does.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from vault.resolve.scoring import CalibrationScore

logger = logging.getLogger(__name__)

__all__ = ["CapitalGate", "GateVerdict", "SizingDecision"]


class GateVerdict:
    APPROVED = "approved"
    BLOCKED_NO_TRACK_RECORD = "blocked_no_track_record"
    BLOCKED_INSUFFICIENT_SAMPLE = "blocked_insufficient_sample"
    BLOCKED_NO_SKILL = "blocked_no_skill"
    BLOCKED_NOT_SIGNIFICANT = "blocked_not_significant"
    BLOCKED_MISCALIBRATED = "blocked_miscalibrated"
    BLOCKED_MANUALLY = "blocked_manually"


@dataclass(slots=True)
class SizingDecision:
    """What the gate decided, and exactly why."""

    verdict: str
    approved: bool
    fraction_of_capital: float = 0.0
    notional: float = 0.0

    kelly_raw: float = 0.0
    kelly_scaled: float = 0.0
    edge_used: float = 0.0
    payoff_ratio: float = 0.0
    confidence_haircut: float = 1.0

    reason: str = ""
    requirements: list[str] = field(default_factory=list)
    samples_needed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "approved": self.approved,
            "fraction_of_capital": round(self.fraction_of_capital, 5),
            "notional": round(self.notional, 2),
            "kelly_raw": round(self.kelly_raw, 5),
            "kelly_scaled": round(self.kelly_scaled, 5),
            "edge_used": round(self.edge_used, 4),
            "payoff_ratio": round(self.payoff_ratio, 3),
            "confidence_haircut": round(self.confidence_haircut, 3),
            "reason": self.reason,
            "requirements": self.requirements,
            "samples_needed": self.samples_needed,
        }

    def explain(self) -> str:
        if self.approved:
            return (
                f"APPROVED: {self.fraction_of_capital:.2%} of capital "
                f"({self.notional:,.2f}). Kelly {self.kelly_raw:.3f} scaled to "
                f"{self.kelly_scaled:.3f}, edge {self.edge_used:+.3f} from the "
                f"lower bound. {self.reason}"
            )
        lines = [f"BLOCKED ({self.verdict}): {self.reason}"]
        if self.requirements:
            lines.append("To unlock capital:")
            lines.extend(f"  - {r}" for r in self.requirements)
        return "\n".join(lines)


class CapitalGate:
    """Derives position size from demonstrated calibration."""

    def __init__(
        self,
        *,
        min_samples: int = 100,
        kelly_fraction: float = 0.25,
        max_position_fraction: float = 0.05,
        max_overconfidence: float = 0.10,
        require_significance: bool = True,
        manual_block: bool = False,
    ) -> None:
        self.min_samples = min_samples
        self.kelly_fraction = kelly_fraction
        self.max_position_fraction = max_position_fraction
        self.max_overconfidence = max_overconfidence
        self.require_significance = require_significance
        self.manual_block = manual_block

        self.evaluations = 0
        self.approvals = 0
        self.blocks: dict[str, int] = {}

    def evaluate(
        self,
        score: CalibrationScore,
        *,
        capital: float,
        stated_probability: float,
        magnitude_pct: float,
        invalidation_pct: float,
    ) -> SizingDecision:
        """
        Decide how much capital, if any, this call may risk.

        ``stated_probability`` is what the model claims for THIS call.
        ``score`` is what it has actually achieved historically. The gate uses
        the historical record to decide whether the claim can be trusted at
        all, and only then lets the claim influence size.
        """
        self.evaluations += 1

        if self.manual_block:
            return self._block(
                GateVerdict.BLOCKED_MANUALLY,
                "the gate is manually blocked; no capital may be risked",
                [],
            )

        # ---- gate 1: is there a track record at all? ----
        if score.n == 0:
            return self._block(
                GateVerdict.BLOCKED_NO_TRACK_RECORD,
                "no resolved calls. The agent has not demonstrated anything yet, "
                "so it may not risk capital.",
                [
                    f"accumulate {self.min_samples} resolved calls",
                    "the system runs in paper mode meanwhile; the calls are real "
                    "and scored, only the capital is not at risk",
                ],
                samples_needed=self.min_samples,
            )

        # ---- gate 2: is the sample large enough to mean anything? ----
        if score.n < self.min_samples:
            return self._block(
                GateVerdict.BLOCKED_INSUFFICIENT_SAMPLE,
                f"only {score.n} resolved calls against a {self.min_samples} "
                f"minimum. The 95% confidence interval on the hit rate is "
                f"[{score.hit_rate_low:.1%}, {score.hit_rate_high:.1%}] and the "
                f"always-valid bound the gate tests is "
                f"{score.hit_rate_low_sequential:.1%}, which cannot be "
                f"distinguished from a coin flip.",
                [f"{self.min_samples - score.n} more resolved calls"],
                samples_needed=self.min_samples - score.n,
            )

        # ---- gate 3: does it beat the baseline? ----
        if not score.beats_baseline:
            return self._block(
                GateVerdict.BLOCKED_NO_SKILL,
                f"Brier score {score.brier:.4f} does not beat the {score.brier_baseline:.4f} "
                f"baseline of always saying 0.50. The agent's stated confidences "
                f"are worse than useless.",
                [
                    "the model, prompt, or feature set needs work",
                    "capital stays at zero until the Brier score drops below 0.25",
                ],
            )

        # ---- gate 4: is the edge statistically real, across every look? ----
        #
        # Tests the ALWAYS-VALID bound, not the fixed-sample one. The gate is
        # re-evaluated after every resolved call, and a fixed 95% bound checked
        # repeatedly is not a 95% bound: measured over 2,000 simulated runs, a
        # forecaster with no edge at all crossed the fixed bound in 14.1% of
        # 300-call runs, against 0.5% for this one. See
        # vault/resolve/scoring.py :: sequential_wilson_interval.
        if self.require_significance and score.hit_rate_low_sequential <= 0.5:
            return self._block(
                GateVerdict.BLOCKED_NOT_SIGNIFICANT,
                f"hit rate {score.hit_rate:.1%} looks positive but its "
                f"always-valid lower bound is "
                f"{score.hit_rate_low_sequential:.1%}, which does not exclude a "
                f"coin flip once you account for having checked after every "
                f"one of {score.n} calls. This is the gate that stops a lucky "
                f"streak from unlocking capital.",
                [
                    "more resolved calls, to tighten the sequence",
                    "or a genuinely higher hit rate",
                ],
            )

        # ---- gate 5: are the stated probabilities usable for sizing? ----
        haircut = 1.0
        if score.overconfidence > self.max_overconfidence:
            haircut = max(0.0, 1 - (score.overconfidence - self.max_overconfidence) / 0.20)
            if haircut <= 0.05:
                return self._block(
                    GateVerdict.BLOCKED_MISCALIBRATED,
                    f"overconfident by {score.overconfidence:.1%} (states "
                    f"{score.mean_probability:.2f} on average, achieves "
                    f"{score.hit_rate:.2f}). The stated probabilities cannot be "
                    f"used for sizing until they are corrected.",
                    ["calibration feedback is already in the prompt; give it "
                     "more resolved calls to correct against"],
                )

        # ---- sizing ----
        # Use the ALWAYS-VALID LOWER BOUND of the measured hit rate, not this
        # call's stated probability and not the historical mean. The bound
        # shrinks automatically with sample size, so a small sample sizes small
        # without needing a separate rule -- and because it is time-uniform, it
        # does not reward the system for having been asked many times.
        measured_p = score.hit_rate_low_sequential

        # Blend the call's own confidence in, but only to the extent the model
        # has earned it: a call stated at 0.80 by a model whose 0.80s come in at
        # 0.60 gets sized on something much closer to 0.60.
        if score.mean_probability > 0.5:
            confidence_ratio = min(
                1.5, max(0.5, stated_probability / score.mean_probability)
            )
        else:
            confidence_ratio = 1.0
        effective_p = min(0.90, max(0.50, measured_p * confidence_ratio))

        # Payoff ratio: expected favourable move against the stop. This is what
        # makes b in the Kelly formula, and it is why invalidation_pct is a
        # required field on the schema.
        payoff = magnitude_pct / max(0.01, invalidation_pct)

        kelly_raw = _kelly(effective_p, payoff)
        if kelly_raw <= 0:
            return self._block(
                GateVerdict.BLOCKED_NO_SKILL,
                f"Kelly fraction is {kelly_raw:.4f} at an effective probability "
                f"of {effective_p:.3f} and a payoff ratio of {payoff:.2f}. The "
                f"expected value is not positive at this stop distance.",
                ["a wider stop, a larger expected move, or a higher hit rate"],
            )

        scaled = kelly_raw * self.kelly_fraction * haircut
        fraction = min(self.max_position_fraction, scaled)

        self.approvals += 1
        return SizingDecision(
            verdict=GateVerdict.APPROVED,
            approved=True,
            fraction_of_capital=fraction,
            notional=capital * fraction,
            kelly_raw=kelly_raw,
            kelly_scaled=scaled,
            edge_used=effective_p - 0.5,
            payoff_ratio=payoff,
            confidence_haircut=haircut,
            reason=(
                f"{score.n} resolved calls, hit rate {score.hit_rate:.1%} "
                f"(lower bound {score.hit_rate_low:.1%}), Brier {score.brier:.4f} "
                f"vs {score.brier_baseline:.2f} baseline"
                + (f", size cut {1 - haircut:.0%} for overconfidence" if haircut < 1 else "")
            ),
        )

    def _block(
        self, verdict: str, reason: str, requirements: list[str],
        *, samples_needed: int = 0,
    ) -> SizingDecision:
        self.blocks[verdict] = self.blocks.get(verdict, 0) + 1
        return SizingDecision(
            verdict=verdict, approved=False, reason=reason,
            requirements=requirements, samples_needed=samples_needed,
        )

    def progress(self, score: CalibrationScore) -> dict[str, Any]:
        """
        How far the agent is from unlocking capital, gate by gate.

        The most useful panel on the dashboard while the system is young: it
        turns "not trading yet" into a concrete checklist with a number next to
        each item.
        """
        gates = [
            {
                "gate": "sample size",
                "requirement": f">= {self.min_samples} resolved calls",
                "current": f"{score.n}",
                "passed": score.n >= self.min_samples,
                "progress": min(1.0, score.n / self.min_samples) if self.min_samples else 1.0,
            },
            {
                "gate": "beats baseline",
                "requirement": "Brier < 0.2500",
                "current": f"{score.brier:.4f}" if score.n else "n/a",
                "passed": bool(score.n) and score.beats_baseline,
                "progress": (
                    min(1.0, max(0.0, (0.35 - score.brier) / 0.10)) if score.n else 0.0
                ),
            },
            {
                "gate": "statistically real",
                "requirement": "always-valid lower bound on hit rate > 50%",
                "current": (
                    f"{score.hit_rate_low_sequential:.1%}" if score.n else "n/a"
                ),
                "passed": bool(score.n) and score.hit_rate_low_sequential > 0.5,
                "progress": (
                    min(1.0, max(0.0, score.hit_rate_low_sequential / 0.5))
                    if score.n else 0.0
                ),
            },
            {
                "gate": "calibrated",
                "requirement": f"overconfidence <= {self.max_overconfidence:.0%}",
                "current": f"{score.overconfidence:+.1%}" if score.n else "n/a",
                "passed": bool(score.n) and score.overconfidence <= self.max_overconfidence,
                "progress": (
                    min(1.0, max(0.0, 1 - max(0.0, score.overconfidence) / 0.25))
                    if score.n else 0.0
                ),
            },
        ]
        return {
            "gates": gates,
            "passed": sum(1 for g in gates if g["passed"]),
            "total": len(gates),
            "unlocked": all(g["passed"] for g in gates),
            "overall_progress": round(sum(g["progress"] for g in gates) / len(gates), 4),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "kelly_fraction": self.kelly_fraction,
            "max_position_fraction": self.max_position_fraction,
            "max_overconfidence": self.max_overconfidence,
            "manual_block": self.manual_block,
            "evaluations": self.evaluations,
            "approvals": self.approvals,
            "blocks_by_reason": dict(self.blocks),
        }


def _kelly(p: float, payoff: float) -> float:
    """
    Kelly fraction: (p*b - q) / b, with b the payoff ratio and q = 1 - p.

    Returns 0 when the edge is not positive. Note that Kelly is extremely
    sensitive to p -- a 5-percentage-point error in the win probability can
    double or halve the prescribed size, which is the whole reason this system
    sizes on a measured lower bound rather than on a stated confidence.
    """
    if payoff <= 0:
        return 0.0
    q = 1 - p
    fraction = (p * payoff - q) / payoff
    return max(0.0, fraction)
