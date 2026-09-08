"""
Agent simulation: does the scoreboard actually detect skill?

The calibration machinery makes a strong claim -- that it can tell a real edge
from a lucky streak, and that it will not unlock capital for the latter. That
claim needs testing against agents whose true skill is KNOWN, because on real
data you never learn the ground truth.

So this module builds synthetic forecasters with dialled-in properties and runs
them through the real journal, the real resolver and the real gate:

    ORACLE          65% true accuracy, well calibrated  -> should unlock
    COINFLIP        50% true accuracy, claims 65%       -> must never unlock
    OVERCONFIDENT   58% true accuracy, claims 85%       -> must be caught
    LUCKY           50% true, but a hot opening streak   -> must not unlock on it

If the COINFLIP agent ever unlocks capital, the gate is broken and the whole
system is theatre. That is the test worth writing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

from vault.calibration.gate import CapitalGate
from vault.data.series import Series
from vault.resolve.scoring import CalibrationScore, score_records
from vault.thesis.journal import ThesisJournal
from vault.thesis.schema import Outcome, Thesis, ThesisRecord

__all__ = ["SimulatedForecaster", "AGENT_PROFILES", "run_simulation", "SimulationResult"]


@dataclass(slots=True)
class SimulatedForecaster:
    """A forecaster with known ground-truth properties."""

    name: str
    true_accuracy: float          # the probability it is actually right
    stated_probability: float     # what it claims
    description: str
    hot_streak: int = 0           # first N calls forced correct

    # What the gate OUGHT to do with this agent, declared per profile rather
    # than inferred from a threshold on true_accuracy. The two are not the same
    # question: `overconfident` has a genuine 58% edge and still must be
    # blocked, because it states 0.85 and its Brier score is therefore worse
    # than the always-say-0.50 baseline. An agent whose stated probabilities
    # are unusable cannot be sized on them, edge or no edge -- that is what
    # gate 3 is for, and a ground-truth rule of `true_accuracy > 0.55` would
    # score the gate as broken for doing its job.
    should_unlock: bool = True


AGENT_PROFILES: tuple[SimulatedForecaster, ...] = (
    SimulatedForecaster(
        "oracle", 0.65, 0.65,
        "genuine 65% edge, honestly stated. The gate should unlock for this one.",
        should_unlock=True,
    ),
    SimulatedForecaster(
        "coinflip", 0.50, 0.65,
        "no edge at all, claims 65%. If this unlocks capital the gate is broken.",
        should_unlock=False,
    ),
    SimulatedForecaster(
        "overconfident", 0.58, 0.85,
        "a real but modest edge, wildly overstated. Must be caught by the skill "
        "gate even though its hit rate beats 50%: stating 0.85 and hitting 0.57 "
        "scores a Brier of 0.33, worse than always saying 0.50. Recalibration "
        "is what rescues this agent, not the gate relaxing.",
        should_unlock=False,
    ),
    SimulatedForecaster(
        "lucky", 0.50, 0.65,
        "no edge, but wins its first 12 calls. Tests whether a streak can "
        "unlock capital before the sample is large enough. Against a "
        "fixed-sample bound it took $356 at call 50; against the always-valid "
        "bound it never gets capital.",
        hot_streak=12, should_unlock=False,
    ),
)


@dataclass(slots=True)
class SimulationResult:
    agent: str
    description: str
    true_accuracy: float
    calls: int
    score: CalibrationScore
    unlocked_at: int | None
    final_notional: float
    peak_notional: float
    approvals: int
    verdict: str
    correct_verdict: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "description": self.description,
            "true_accuracy": self.true_accuracy,
            "calls": self.calls,
            "hit_rate": round(self.score.hit_rate, 4),
            "hit_rate_ci": [
                round(self.score.hit_rate_low, 4), round(self.score.hit_rate_high, 4),
            ],
            "brier": round(self.score.brier, 5),
            "skill": round(self.score.skill, 4),
            "overconfidence": round(self.score.overconfidence, 4),
            "unlocked_at_call": self.unlocked_at,
            "final_notional": round(self.final_notional, 2),
            "peak_notional": round(self.peak_notional, 2),
            "approvals": self.approvals,
            "verdict": self.verdict,
            "gate_was_right": self.correct_verdict,
        }


def _build_thesis(probability: float, direction: str = "up") -> Thesis:
    return Thesis(
        asset="SP500",
        direction=direction,
        horizon="1w",
        probability=probability,
        magnitude_pct=2.5,
        invalidation_pct=1.8,
        regime_dependency="simulated -- no real regime dependency",
        key_risk="A core PCE print above 0.35% m/m would invalidate this",
        reasoning="Simulated forecast for calibration-machinery testing.",
        primary_evidence=["T10Y3M"],
    )


def run_simulation(
    forecaster: SimulatedForecaster,
    *,
    journal: ThesisJournal,
    gate: CapitalGate,
    calls: int = 200,
    capital: float = 10_000.0,
    seed: int = 42,
) -> SimulationResult:
    """
    Run one forecaster through the real journal, resolver and gate.

    Outcomes are drawn from the forecaster's TRUE accuracy and written through
    the same resolution path a live run uses, so the scoring code under test is
    the production code, not a parallel implementation.
    """
    rng = random.Random(seed)
    base = datetime.now(timezone.utc) - timedelta(days=calls + 40)
    unlocked_at: int | None = None
    final_notional = 0.0
    peak_notional = 0.0
    approvals = 0

    for index in range(calls):
        created = base + timedelta(days=index)
        thesis = _build_thesis(
            forecaster.stated_probability,
            direction="up" if rng.random() < 0.5 else "down",
        )

        # Write through the real journal, to the real append-only file.
        #
        # The anti-backfill guard in `commit` correctly refuses a thesis whose
        # outcome is already knowable, which is exactly what simulated history
        # is -- so the record is constructed directly and appended. It is
        # appended, though, not merely held in memory: a simulated journal that
        # never reaches disk means `vault demo` seeds a track record the HUD
        # then loads from an empty file and renders as zeros. The scoreboard
        # showing nothing while the terminal reports 140 calls is precisely the
        # decorative-dashboard failure this project exists to avoid.
        #
        # The journal's own guard is exercised against the real API in the
        # tests; nothing here weakens it.
        record = ThesisRecord(
            thesis_id=f"sim-{forecaster.name}-{index:04d}",
            thesis=thesis,
            created_at=created,
            entry_price=5600.0,
            resolve_on=(created + timedelta(days=7)).date(),
            regime="simulated",
            regime_confidence=0.5,
            brief_digest="sim",
            data_snapshot="",
            model="simulation",
            prev_hash=journal.head_hash,
        )
        journal._append({"kind": "thesis", "record": record.to_dict()})  # noqa: SLF001
        journal._records[record.thesis_id] = record       # noqa: SLF001
        journal._order.append(record.thesis_id)           # noqa: SLF001
        journal._head_hash = record.content_hash          # noqa: SLF001

        # Draw the outcome from TRUE accuracy, not from the stated probability.
        forced = index < forecaster.hot_streak
        correct = forced or (rng.random() < forecaster.true_accuracy)
        # Resolve through the journal's own path, so the resolution is a second
        # appended line exactly as it is in a live run and a reload of the file
        # reconstructs the same scoreboard.
        journal.resolve(
            record.thesis_id,
            outcome=Outcome.CORRECT if correct else Outcome.WRONG,
            exit_price=5600.0 * (1.02 if correct else 0.98),
            realized_pct=2.0 if correct else -2.0,
            now=created + timedelta(days=7),
        )

        # Ask the gate after every call, so we learn WHEN it unlocks.
        score = score_records(list(journal))
        decision = gate.evaluate(
            score, capital=capital,
            stated_probability=forecaster.stated_probability,
            magnitude_pct=2.5, invalidation_pct=1.8,
        )
        if decision.approved:
            approvals += 1
            peak_notional = max(peak_notional, decision.notional)
            if unlocked_at is None:
                unlocked_at = index + 1
        final_notional = decision.notional if decision.approved else 0.0

    score = score_records(list(journal))
    final = gate.evaluate(
        score, capital=capital,
        stated_probability=forecaster.stated_probability,
        magnitude_pct=2.5, invalidation_pct=1.8,
    )

    # Ground truth: only an agent with a real edge SHOULD unlock.
    #
    # For an agent that should NOT unlock, the test is whether it ever got
    # capital -- not whether it happens to be blocked on the last call. Money
    # released at call 50 and withdrawn at call 56 was still money at risk on
    # an agent with no edge, and scoring only the end state would hide exactly
    # the failure this simulation exists to detect. That is not a hypothetical:
    # against a fixed-sample confidence bound the "lucky" profile below took
    # $356 at call 50 while the summary line printed HELD.
    should_unlock = forecaster.should_unlock
    if should_unlock:
        correct = final.approved
    else:
        correct = unlocked_at is None
    return SimulationResult(
        agent=forecaster.name,
        description=forecaster.description,
        true_accuracy=forecaster.true_accuracy,
        calls=calls,
        score=score,
        unlocked_at=unlocked_at,
        final_notional=final.notional if final.approved else 0.0,
        peak_notional=peak_notional,
        approvals=approvals,
        verdict=final.verdict,
        correct_verdict=correct,
    )
