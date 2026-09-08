"""
Capital gate tests.

The gate is the answer to "make it generate returns safely". Safety here is a
mechanism, not a disclaimer: position size is a function of demonstrated
calibration, and before there is a track record that function returns zero.

The load-bearing test is ``test_a_lucky_streak_does_not_unlock_capital``. It
encodes a bug that was live: with a fixed-sample confidence bound, a forecaster
with no edge whatsoever but twelve rigged opening wins was approved for $356 at
call 50 -- while the simulation's summary line printed HELD, because it scored
only the final call rather than the whole run. Both halves of that failure are
tested here.
"""

from __future__ import annotations

import pytest

from vault.calibration.gate import CapitalGate, GateVerdict
from vault.resolve.scoring import CalibrationScore, score_records

from tests.unit.vault_factories import make_record, resolve_record


CAPITAL = 10_000.0


def _score(*, n: int, hit_rate: float, probability: float = 0.65) -> CalibrationScore:
    """Build a real score from real records, not a hand-set dataclass."""
    wins = round(n * hit_rate)
    records = [
        resolve_record(
            make_record(index=i, probability=probability), correct=(i < wins)
        )
        for i in range(n)
    ]
    return score_records(records)


def _evaluate(gate: CapitalGate, score: CalibrationScore, *, probability: float = 0.65):
    return gate.evaluate(
        score, capital=CAPITAL, stated_probability=probability,
        magnitude_pct=2.5, invalidation_pct=1.8,
    )


# ---------------------------------------------------------------------------
# Gate 1: nothing on the board
# ---------------------------------------------------------------------------


def test_an_agent_with_no_track_record_gets_exactly_zero():
    """Not a small number. Zero."""
    decision = _evaluate(CapitalGate(), CalibrationScore())
    assert not decision.approved
    assert decision.verdict == GateVerdict.BLOCKED_NO_TRACK_RECORD
    assert decision.notional == 0.0
    assert decision.fraction_of_capital == 0.0


def test_the_block_explains_how_to_unlock():
    """A gate that says only "no" is a gate nobody can work with."""
    decision = _evaluate(CapitalGate(min_samples=100), CalibrationScore())
    assert decision.requirements
    assert decision.samples_needed == 100
    assert "100 resolved calls" in decision.explain()


# ---------------------------------------------------------------------------
# Gate 2: sample size
# ---------------------------------------------------------------------------


def test_a_short_but_flawless_record_is_still_not_enough():
    """Ten from ten is not evidence; it is ten."""
    decision = _evaluate(CapitalGate(min_samples=100), _score(n=10, hit_rate=1.0))
    assert not decision.approved
    assert decision.verdict == GateVerdict.BLOCKED_INSUFFICIENT_SAMPLE
    assert decision.samples_needed == 90


# ---------------------------------------------------------------------------
# Gate 3: skill against the honest baseline
# ---------------------------------------------------------------------------


def test_worse_than_a_coin_flip_is_blocked_however_large_the_sample():
    score = _score(n=200, hit_rate=0.52, probability=0.9)
    assert not score.beats_baseline
    decision = _evaluate(CapitalGate(), score, probability=0.9)
    assert not decision.approved
    assert decision.verdict == GateVerdict.BLOCKED_NO_SKILL


# ---------------------------------------------------------------------------
# Gate 4: the one that stops luck -- the load-bearing test
# ---------------------------------------------------------------------------


def test_a_lucky_streak_does_not_unlock_capital():
    """
    A no-edge forecaster with twelve rigged opening wins, evaluated after
    every call exactly as the live system evaluates it.

    Against a fixed-sample bound this took $356 at call 50. The gate must now
    release nothing at ANY point in the run -- the peak matters, not the end
    state, because money released at call 50 and withdrawn at call 56 was
    still money at risk on an agent with no edge.
    """
    import random

    rng = random.Random(42)
    gate = CapitalGate()
    records = []
    peak = 0.0

    for i in range(300):
        correct = True if i < 12 else rng.random() < 0.50
        records.append(resolve_record(make_record(index=i), correct=correct))
        decision = _evaluate(gate, score_records(records))
        if decision.approved:
            peak = max(peak, decision.notional)

    assert peak == 0.0, (
        f"a forecaster with no edge was funded up to ${peak:,.2f}; the "
        f"sequential bound is not doing its job"
    )


def test_the_gate_tests_the_sequential_bound_not_the_fixed_one():
    """
    Pin the actual mechanism, so a future edit back to `hit_rate_low` fails
    loudly rather than silently reopening the leak.
    """
    score = _score(n=100, hit_rate=0.60)
    # A sample shaped so the two bounds disagree about the coin flip.
    assert score.hit_rate_low > 0.5 >= score.hit_rate_low_sequential, (
        "this test needs a sample where the fixed bound clears 0.5 and the "
        "sequential bound does not; adjust n/hit_rate if the constants moved"
    )
    decision = _evaluate(CapitalGate(min_samples=50), score)
    assert not decision.approved
    assert decision.verdict == GateVerdict.BLOCKED_NOT_SIGNIFICANT


def test_significance_can_be_switched_off_only_deliberately():
    score = _score(n=100, hit_rate=0.60)
    assert not _evaluate(CapitalGate(min_samples=50), score).approved
    relaxed = CapitalGate(min_samples=50, require_significance=False)
    assert _evaluate(relaxed, score).approved


# ---------------------------------------------------------------------------
# Gate 5: calibration
# ---------------------------------------------------------------------------


def test_overconfidence_cuts_size_before_it_blocks():
    """
    A model whose ordering is fine but whose numbers are inflated is not
    useless -- it is unusable at face value. Size down, then block.
    """
    honest = _score(n=200, hit_rate=0.66, probability=0.65)
    inflated = _score(n=200, hit_rate=0.66, probability=0.78)

    a = _evaluate(CapitalGate(), honest, probability=0.65)
    b = _evaluate(CapitalGate(), inflated, probability=0.78)
    assert a.approved and a.confidence_haircut == 1.0
    if b.approved:
        assert b.confidence_haircut < 1.0
    else:
        assert b.verdict == GateVerdict.BLOCKED_MISCALIBRATED


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_a_real_demonstrated_edge_does_unlock():
    """The gate must be passable, or it is theatre rather than a mechanism."""
    decision = _evaluate(CapitalGate(), _score(n=250, hit_rate=0.66))
    assert decision.approved
    assert decision.notional > 0
    assert decision.verdict == GateVerdict.APPROVED


def test_size_never_exceeds_the_hard_cap():
    """Kelly on a strong edge prescribes sizes no one should actually take."""
    gate = CapitalGate(max_position_fraction=0.05)
    decision = _evaluate(gate, _score(n=500, hit_rate=0.85, probability=0.85),
                         probability=0.85)
    assert decision.approved
    assert decision.fraction_of_capital <= 0.05
    assert decision.notional <= CAPITAL * 0.05


def test_size_grows_with_evidence_at_a_fixed_hit_rate():
    """
    The core property: risk grows only as evidence does. Same measured skill,
    more of it, larger position -- because the lower bound tightens.
    """
    # The cap is lifted for this test only: at the production 5% cap the last
    # two sizes both saturate and the growth being measured is invisible.
    gate = CapitalGate(max_position_fraction=0.25)
    sizes = [
        _evaluate(gate, _score(n=n, hit_rate=0.66)).fraction_of_capital
        for n in (150, 300, 600)
    ]
    assert all(s > 0 for s in sizes), sizes
    assert sizes[0] < sizes[1] < sizes[2], sizes


def test_a_stop_too_tight_for_the_expected_move_is_refused():
    """Positive hit rate, negative expected value. Kelly returns zero."""
    # A record strong enough to clear gates 1-4, so the refusal is unambiguously
    # the expected-value check and not an earlier gate firing first.
    gate = CapitalGate()
    decision = gate.evaluate(
        _score(n=300, hit_rate=0.66), capital=CAPITAL, stated_probability=0.65,
        magnitude_pct=0.5, invalidation_pct=5.0,
    )
    assert not decision.approved
    assert decision.verdict == GateVerdict.BLOCKED_NO_SKILL


def test_manual_block_overrides_everything():
    gate = CapitalGate(manual_block=True)
    decision = _evaluate(gate, _score(n=1000, hit_rate=0.80, probability=0.80),
                         probability=0.80)
    assert not decision.approved
    assert decision.verdict == GateVerdict.BLOCKED_MANUALLY


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def test_progress_turns_not_trading_yet_into_a_checklist():
    progress = CapitalGate(min_samples=100).progress(CalibrationScore())
    assert progress["total"] == 4
    assert progress["passed"] == 0
    assert not progress["unlocked"]
    assert all(g["requirement"] for g in progress["gates"])


def test_progress_agrees_with_the_decision_it_describes():
    """A dashboard that says "unlocked" while the gate says no is worse than
    no dashboard."""
    gate = CapitalGate()
    for n, hit_rate in ((0, 0.0), (40, 0.7), (150, 0.52), (250, 0.66)):
        score = _score(n=n, hit_rate=hit_rate) if n else CalibrationScore()
        assert gate.progress(score)["unlocked"] == _evaluate(gate, score).approved, (
            f"progress and evaluate disagree at n={n}, hit_rate={hit_rate}"
        )


def test_stats_counts_blocks_by_reason():
    gate = CapitalGate()
    _evaluate(gate, CalibrationScore())
    _evaluate(gate, _score(n=10, hit_rate=1.0))
    stats = gate.stats()
    assert stats["evaluations"] == 2
    assert stats["approvals"] == 0
    assert stats["blocks_by_reason"][GateVerdict.BLOCKED_NO_TRACK_RECORD] == 1
