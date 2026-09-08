"""
Test factories for vault records.

These build ``ThesisRecord`` objects directly rather than going through
``ThesisJournal.commit``, because commit deliberately refuses any thesis whose
resolution date is already knowable -- which is exactly the shape a test
fixture needs. The journal's own guard is exercised against the real API in
``test_vault_journal.py``; everything else gets records built here.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from vault.thesis.schema import Outcome, Thesis, ThesisRecord

__all__ = ["make_thesis", "make_record", "resolve_record", "GENESIS"]

GENESIS = "0" * 64


def make_thesis(
    *,
    probability: float = 0.65,
    direction: str = "up",
    asset: str = "SP500",
    horizon: str = "1w",
    magnitude_pct: float = 2.5,
    invalidation_pct: float = 1.8,
) -> Thesis:
    return Thesis(
        asset=asset,
        direction=direction,
        horizon=horizon,
        probability=probability,
        magnitude_pct=magnitude_pct,
        invalidation_pct=invalidation_pct,
        regime_dependency="Holds while the curve keeps re-steepening from inversion.",
        key_risk="A core PCE print above 0.35% m/m would invalidate this",
        reasoning="Test fixture; the reasoning field is not under test here.",
        primary_evidence=["T10Y3M"],
    )


def make_record(
    *,
    index: int = 0,
    probability: float = 0.65,
    direction: str = "up",
    entry_price: float = 5600.0,
    created_at: datetime | None = None,
    resolve_on: date | None = None,
    prev_hash: str = GENESIS,
    regime: str = "goldilocks",
    horizon: str = "1w",
    magnitude_pct: float = 2.5,
    invalidation_pct: float = 1.8,
) -> ThesisRecord:
    created = created_at or (datetime.now(timezone.utc) - timedelta(days=30 + index))
    thesis = make_thesis(
        probability=probability, direction=direction, horizon=horizon,
        magnitude_pct=magnitude_pct, invalidation_pct=invalidation_pct,
    )
    return ThesisRecord(
        thesis_id=f"th-test-{index:04d}",
        thesis=thesis,
        created_at=created,
        entry_price=entry_price,
        resolve_on=resolve_on or thesis.resolves_on(created.date()),
        regime=regime,
        regime_confidence=0.6,
        brief_digest="test-brief",
        data_snapshot="",
        model="test",
        prev_hash=prev_hash,
    )


def resolve_record(
    record: ThesisRecord, *, correct: bool = True, invalidated: bool = False,
) -> ThesisRecord:
    """Mark a record resolved in place and return it."""
    if invalidated:
        record.outcome = Outcome.INVALIDATED
        record.realized_pct = -record.thesis.invalidation_pct
        record.exit_price = record.thesis.invalidation_price(record.entry_price)
        record.invalidated_on = record.resolve_on - timedelta(days=2)
    else:
        record.outcome = Outcome.CORRECT if correct else Outcome.WRONG
        move = record.thesis.magnitude_pct * (1 if correct else -1) * record.thesis.sign
        record.realized_pct = move
        record.exit_price = record.entry_price * (1 + move / 100)
    record.resolved_at = datetime.now(timezone.utc)
    return record
