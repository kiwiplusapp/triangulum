"""
Vault integration tests: the whole loop, on fixtures, without an API key.

The claim this file exists to defend is the one the system makes about itself:

    Position size is a function of demonstrated calibration, and before
    there is a track record that function returns zero.

``test_no_edge_agents_are_never_funded`` measures that across seeds rather
than asserting it from one run. That distinction is the point: the first
version of the simulation checked only the final call, printed HELD, and was
wrong -- on that very seed the no-edge "lucky" profile had taken $356 of
capital at call 50 and given it back six calls later. A summary line that can
disagree with the table above it is exactly the failure mode this system is
built to avoid, so the assertion now looks at the peak across the whole run,
over many seeds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from vault.calibration.gate import CapitalGate, GateVerdict
from vault.pipeline import Vault
from vault.simulate import AGENT_PROFILES, run_simulation
from vault.thesis.journal import ThesisJournal


SEEDS = 20
CALLS = 300
ERROR_BUDGET = 0.05        # the rate the gate advertises, and is held to


@pytest.fixture()
def vault(tmp_path: Path) -> Vault:
    return Vault(
        journal_path=tmp_path / "journal.ndjson",
        cache_dir=tmp_path / "cache",
        capital=10_000.0,
        use_fixtures=True,
        dry_run=True,
    )


# ---------------------------------------------------------------------------
# The pipeline, end to end
# ---------------------------------------------------------------------------


def test_a_full_cycle_runs_without_credentials(vault: Vault):
    """
    Everything except the model call must work with no API key at all, or the
    system cannot be inspected, tested, or trusted by anyone without one.
    """
    result = vault.run(commit=False)
    assert result is not None
    assert vault.series, "the scan produced no series"
    assert result.summary()


def test_the_scan_produces_a_usable_universe(vault: Vault):
    series = vault.scan()
    assert len(series) > 20
    assert all(s.points for s in series.values())
    assert vault.data_mode == "fixture"


def test_the_regime_read_is_a_probability_distribution(vault: Vault):
    vault.scan()
    regime = vault.classify()
    total = sum(regime.probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= regime.confidence <= 1.0
    assert regime.quadrant in regime.probabilities


def test_the_brief_names_the_series_it_is_built_from(vault: Vault):
    vault.scan()
    brief = vault.brief(vault.classify())
    assert brief["signals"]
    assert brief["regime"]


def test_an_empty_journal_yields_a_blocked_gate(vault: Vault):
    """The state every honest system starts in."""
    score, _ = vault.score()
    assert score.n == 0
    decision = vault.gate.evaluate(
        score, capital=vault.capital, stated_probability=0.65,
        magnitude_pct=2.5, invalidation_pct=1.8,
    )
    assert not decision.approved
    assert decision.notional == 0.0
    assert decision.verdict == GateVerdict.BLOCKED_NO_TRACK_RECORD


def test_the_snapshot_is_serialisable_and_agrees_with_the_gate(vault: Vault):
    """The HUD reads this; it must not be able to show a state the gate denies."""
    import json

    result = vault.run(commit=False)
    snapshot = vault.snapshot()
    json.dumps(snapshot)          # must not raise

    assert snapshot["score"]["n"] == 0
    assert snapshot["gate_progress"]["unlocked"] is False
    assert snapshot["data_mode"] == "fixture"
    assert snapshot["open_positions"] == []

    # Whatever the run decided about size, the HUD must show the same thing.
    if result.sizing is not None:
        assert result.sizing.approved is False
        assert result.sizing.notional == 0.0
        assert result.to_dict()["sizing"]["approved"] is False


def test_running_twice_does_not_corrupt_the_journal(vault: Vault):
    vault.run(commit=False)
    vault.run(commit=False)
    assert vault.journal.verify_chain().valid


# ---------------------------------------------------------------------------
# The load-bearing claim
# ---------------------------------------------------------------------------


def _sweep(profile, *, seeds: int = SEEDS, min_samples: int = 100):
    """Run one profile over many seeds; return (funded_rate, peak_notional)."""
    funded = 0
    peak = 0.0
    for seed in range(seeds):
        journal = ThesisJournal(Path(tempfile.mkdtemp()) / "j.ndjson")
        result = run_simulation(
            profile, journal=journal, gate=CapitalGate(min_samples=min_samples),
            calls=CALLS, capital=10_000.0, seed=seed,
        )
        if result.unlocked_at is not None:
            funded += 1
        peak = max(peak, result.peak_notional)
    return funded / seeds, peak


@pytest.mark.parametrize(
    "profile",
    [p for p in AGENT_PROFILES if not p.should_unlock],
    ids=lambda p: p.name,
)
def test_no_edge_agents_are_never_funded(profile):
    """
    Held to the error rate the gate advertises, measured over the whole run
    rather than at its end.

    ``lucky`` is the adversarial case: no edge at all, but its first twelve
    calls are rigged wins. It is the profile that walked through the old
    fixed-sample bound.
    """
    rate, peak = _sweep(profile)
    assert rate <= ERROR_BUDGET, (
        f"{profile.name} has no usable edge but was funded in {rate:.0%} of "
        f"runs (budget {ERROR_BUDGET:.0%}), peaking at ${peak:,.0f}"
    )


def test_a_genuine_edge_is_funded():
    """
    The other half. A gate that never opens is not safe, it is broken -- and
    it would be trivially easy to pass every test above by returning zero.
    """
    oracle = next(p for p in AGENT_PROFILES if p.name == "oracle")
    rate, peak = _sweep(oracle)
    assert rate >= 0.90, f"a true 65% forecaster was funded in only {rate:.0%} of runs"
    assert peak > 0


def test_the_old_fixed_bound_would_fail_this_suite():
    """
    Pin the regression itself. If someone points the gate back at the
    fixed-sample bound, this test says so in as many words rather than
    letting the leak return silently.
    """
    import random

    from vault.resolve.scoring import sequential_wilson_interval, wilson_interval

    leaked = 0
    trials = 300
    rng = random.Random(19)
    for _ in range(trials):
        successes = 0
        for n in range(1, CALLS + 1):
            if rng.random() < 0.5:
                successes += 1
            if n >= 100 and wilson_interval(successes, n)[0] > 0.5:
                leaked += 1
                break
    assert leaked / trials > ERROR_BUDGET, (
        "the fixed-sample bound no longer leaks under repeated looks, which "
        "means this test has stopped measuring the regression it was written for"
    )

    # And the bound actually in use does not.
    held = 0
    rng = random.Random(19)
    for _ in range(trials):
        successes = 0
        for n in range(1, CALLS + 1):
            if rng.random() < 0.5:
                successes += 1
            if n >= 100 and sequential_wilson_interval(successes, n)[0] > 0.5:
                held += 1
                break
    assert held / trials <= ERROR_BUDGET


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------


def test_recalibration_rescues_an_overconfident_forecaster():
    """
    ``overconfident`` has a real 58% edge stated as 85%, which scores a Brier
    worse than always saying 0.50 -- so the skill gate blocks it, correctly.
    The fix is to correct its numbers, not to relax the gate.
    """
    from vault.calibration.recalibrate import fit_recalibrator
    from vault.resolve.scoring import score_records

    profile = next(p for p in AGENT_PROFILES if p.name == "overconfident")
    journal = ThesisJournal(Path(tempfile.mkdtemp()) / "j.ndjson")
    run_simulation(
        profile, journal=journal, gate=CapitalGate(),
        calls=CALLS, capital=10_000.0, seed=3,
    )
    records = list(journal)
    before = score_records(records)
    assert not before.beats_baseline, "this profile is supposed to fail the skill gate"

    recalibrator, report = fit_recalibrator(records)
    assert recalibrator is not None, report.reason
    assert report.fitted
    assert report.brier_after < report.brier_before, report.summary()
    assert report.helped

    # And the correction is what the gate needed: the corrected probabilities
    # beat the 0.25 baseline the raw ones failed.
    assert report.brier_after < 0.25
