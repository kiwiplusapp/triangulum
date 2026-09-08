"""
Journal tests: the hash chain, and the two guards that make it worth having.

A track record you can edit after the fact is not a track record. Two specific
failures are encoded here because both were live:

1. **Backfill.** The anti-backfill check originally compared the resolution
   date against the record's OWN created_at. Both move together when a commit
   is backdated, so a "prediction" made thirty days ago with a seven-day
   horizon resolved twenty-three days ago and sailed straight through. The
   check must compare against the wall clock: the only question that matters
   is whether the outcome is knowable NOW.

2. **False tampering.** ``json.dumps(5600)`` is ``"5600"`` and
   ``json.dumps(5600.0)`` is ``"5600.0"``. An integer entry price hashed at
   commit and a float one after a round-trip through ``from_dict`` produced
   different hashes, so the chain reported tampering on an untouched record.
   An integrity check that cries wolf is worse than none.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from vault.thesis.journal import ThesisJournal
from vault.thesis.schema import Outcome

from tests.unit.vault_factories import make_thesis


@pytest.fixture()
def journal(tmp_path: Path) -> ThesisJournal:
    return ThesisJournal(tmp_path / "journal.ndjson")


# ---------------------------------------------------------------------------
# Committing
# ---------------------------------------------------------------------------


def test_a_committed_thesis_is_readable_back(journal: ThesisJournal):
    record = journal.commit(
        make_thesis(), entry_price=5600.0, regime="goldilocks",
        regime_confidence=0.6, brief_digest="d",
    )
    assert len(journal) == 1
    assert journal.get(record.thesis_id) is record
    assert record.outcome == Outcome.PENDING
    assert record.resolve_on > datetime.now(timezone.utc).date()


def test_a_backdated_thesis_is_refused(journal: ThesisJournal):
    """
    The guard that stops a backfilled prediction entering the track record.

    Backdating moves created_at AND resolve_on together, so this only fails
    if the check is against the wall clock rather than the record's own
    timestamp.
    """
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    with pytest.raises(ValueError, match="not in the future"):
        journal.commit(
            make_thesis(horizon="1w"), entry_price=5600.0, regime="goldilocks",
            regime_confidence=0.6, brief_digest="d", now=long_ago,
        )
    assert len(journal) == 0


def test_a_thesis_resolving_today_is_refused(journal: ThesisJournal):
    """Knowable today is not a forecast, even by one day."""
    with pytest.raises(ValueError, match="not in the future"):
        journal.commit(
            make_thesis(horizon="1w"), entry_price=5600.0, regime="g",
            regime_confidence=0.6, brief_digest="d",
            now=datetime.now(timezone.utc) - timedelta(days=7),
        )


def test_a_nonpositive_entry_price_is_refused(journal: ThesisJournal):
    with pytest.raises(ValueError, match="entry price"):
        journal.commit(
            make_thesis(), entry_price=0.0, regime="g",
            regime_confidence=0.6, brief_digest="d",
        )


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def _commit_several(journal: ThesisJournal, n: int = 5):
    return [
        journal.commit(
            make_thesis(probability=0.55 + i * 0.05), entry_price=5600.0 + i,
            regime="goldilocks", regime_confidence=0.6, brief_digest=f"d{i}",
        )
        for i in range(n)
    ]


def test_the_chain_verifies_on_an_untouched_journal(journal: ThesisJournal):
    _commit_several(journal)
    verification = journal.verify_chain()
    assert verification.valid, verification.reason
    assert verification.records_checked == 5


def test_each_record_links_to_its_predecessor(journal: ThesisJournal):
    records = _commit_several(journal, 4)
    for earlier, later in zip(records, records[1:]):
        assert later.prev_hash == earlier.content_hash


def test_editing_a_committed_thesis_breaks_the_chain(journal: ThesisJournal):
    """The point of the whole construction."""
    records = _commit_several(journal, 5)
    records[1].entry_price = 4200.0        # rewrite history

    verification = journal.verify_chain()
    assert not verification.valid
    assert verification.broken_at == 1
    assert "edited after it was committed" in verification.reason


def test_a_reloaded_journal_still_verifies(journal: ThesisJournal):
    """
    The regression for the int/float hash bug. A round-trip through JSON must
    not, by itself, look like tampering.
    """
    _commit_several(journal, 4)
    reloaded = ThesisJournal(journal.path)
    assert reloaded.load() == 4
    verification = reloaded.verify_chain()
    assert verification.valid, verification.reason


def test_an_integer_entry_price_survives_the_round_trip(tmp_path: Path):
    """The exact shape that broke it: 5600 rather than 5600.0."""
    journal = ThesisJournal(tmp_path / "j.ndjson")
    journal.commit(
        make_thesis(), entry_price=5600, regime="g",     # int, deliberately
        regime_confidence=0.6, brief_digest="d",
    )
    reloaded = ThesisJournal(tmp_path / "j.ndjson")
    reloaded.load()
    assert reloaded.verify_chain().valid


def test_head_hash_advances_with_every_commit(journal: ThesisJournal):
    seen = {journal.head_hash}
    for _ in range(3):
        journal.commit(
            make_thesis(), entry_price=5600.0, regime="g",
            regime_confidence=0.6, brief_digest="d",
        )
        assert journal.head_hash not in seen
        seen.add(journal.head_hash)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolving_does_not_break_the_chain(journal: ThesisJournal):
    """
    Outcomes are appended, not written into the hashed content. Grading a
    call must never look like editing it.
    """
    records = _commit_several(journal, 3)
    journal.resolve(
        records[0].thesis_id, outcome=Outcome.CORRECT,
        exit_price=5740.0, realized_pct=2.5,
    )
    assert journal.verify_chain().valid
    assert journal.get(records[0].thesis_id).outcome == Outcome.CORRECT


def test_a_resolution_survives_a_reload(journal: ThesisJournal):
    records = _commit_several(journal, 2)
    journal.resolve(
        records[0].thesis_id, outcome=Outcome.WRONG,
        exit_price=5500.0, realized_pct=-1.8,
    )
    reloaded = ThesisJournal(journal.path)
    reloaded.load()
    assert reloaded.get(records[0].thesis_id).outcome == Outcome.WRONG
    assert reloaded.verify_chain().valid


def test_the_journal_file_is_append_only_ndjson(journal: ThesisJournal):
    _commit_several(journal, 3)
    lines = Path(journal.path).read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert json.loads(line)["kind"] == "thesis"


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


def test_pending_and_scoreable_partition_the_journal(journal: ThesisJournal):
    records = _commit_several(journal, 4)
    journal.resolve(records[0].thesis_id, outcome=Outcome.CORRECT,
                    exit_price=5740.0, realized_pct=2.5)
    journal.resolve(records[1].thesis_id, outcome=Outcome.UNRESOLVABLE,
                    exit_price=None, realized_pct=None)

    assert len(journal.scoreable()) == 1          # UNRESOLVABLE is not scoreable
    assert len(journal.resolved()) == 2
    assert len(journal.open_positions()) == 2


def test_pending_can_be_filtered_by_due_date(journal: ThesisJournal):
    _commit_several(journal, 3)
    assert journal.pending(due_on_or_before=date.today()) == []
    far_future = date.today() + timedelta(days=400)
    assert len(journal.pending(due_on_or_before=far_future)) == 3
