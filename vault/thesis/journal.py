"""
The thesis journal: append-only, hash-chained, tamper-evident.

The problem this solves is the one that quietly ruins every informal
track record:

    A forecast that can be edited after the outcome is known is not a forecast.

Not because anyone intends to cheat -- because memory is reconstructive. Six
weeks later, a call that said "SPX down 3% in two weeks" gets remembered as
"I was cautious", and a system with an editable log will happily agree.

So the journal is:

**Append-only.** Records are written with ``a`` mode to an NDJSON file. There
is no update path and no delete path. Resolution writes a SEPARATE record that
references the original by id, so the original line is never touched.

**Hash-chained.** Each record's hash covers its content plus the previous
record's hash. Editing record 40 of 200 changes its hash, which invalidates
41 through 200. ``verify_chain`` finds the exact index where a chain was
broken.

**Committed before the outcome is knowable.** ``commit`` records the entry
price and the resolution date at write time. A thesis whose resolution date is
in the past when it is committed is rejected outright -- that is the signature
of backfilling.

This does not make the journal cryptographically unforgeable by its own owner;
anyone with the file can rewrite the whole chain. It makes tampering *visible*
to anyone who kept an earlier copy of the head hash, and -- more usefully --
it makes accidental self-deception structurally impossible.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from vault.thesis.schema import Outcome, Thesis, ThesisRecord

logger = logging.getLogger(__name__)

__all__ = ["ThesisJournal", "ChainVerification", "GENESIS_HASH"]

GENESIS_HASH = "0" * 64


@dataclass(slots=True)
class ChainVerification:
    valid: bool
    records_checked: int
    broken_at: int | None = None
    reason: str = ""
    head_hash: str = ""

    def summary(self) -> str:
        if self.valid:
            return (
                f"chain intact across {self.records_checked} records "
                f"(head {self.head_hash[:12]}...)"
            )
        return (
            f"CHAIN BROKEN at record {self.broken_at} of {self.records_checked}: "
            f"{self.reason}"
        )


class ThesisJournal:
    """Append-only NDJSON journal of committed theses and their resolutions."""

    def __init__(self, path: str | Path = "data/vault/journal.ndjson") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ThesisRecord] = {}
        self._order: list[str] = []
        self._head_hash = GENESIS_HASH
        self.load()

    # -- loading -----------------------------------------------------------

    def load(self) -> int:
        """Replay the journal from disk. Resolutions are applied in order."""
        self._records.clear()
        self._order.clear()
        self._head_hash = GENESIS_HASH
        if not self.path.exists():
            return 0

        loaded = 0
        with open(self.path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(
                        "journal line %d is corrupt and was skipped; the chain "
                        "will not verify past this point", line_number,
                    )
                    continue

                kind = raw.get("kind", "thesis")
                if kind == "thesis":
                    record = ThesisRecord.from_dict(raw["record"])
                    self._records[record.thesis_id] = record
                    self._order.append(record.thesis_id)
                    self._head_hash = record.content_hash
                    loaded += 1
                elif kind == "resolution":
                    self._apply_resolution(raw)
        return loaded

    def _apply_resolution(self, raw: dict[str, Any]) -> None:
        record = self._records.get(raw.get("thesis_id", ""))
        if record is None:
            logger.warning("resolution for unknown thesis %s", raw.get("thesis_id"))
            return
        record.outcome = raw.get("outcome", Outcome.PENDING)
        resolved = raw.get("resolved_at")
        record.resolved_at = datetime.fromisoformat(resolved) if resolved else None
        record.exit_price = raw.get("exit_price")
        record.realized_pct = raw.get("realized_pct")
        invalidated = raw.get("invalidated_on")
        record.invalidated_on = date.fromisoformat(invalidated) if invalidated else None
        record.notes = raw.get("notes", "")

    # -- writing -----------------------------------------------------------

    def commit(
        self,
        thesis: Thesis,
        *,
        entry_price: float,
        regime: str,
        regime_confidence: float,
        brief_digest: str,
        data_snapshot: str = "",
        model: str = "",
        now: datetime | None = None,
    ) -> ThesisRecord:
        """
        Commit a thesis. This is the point of no return.

        Rejects a thesis whose resolution date is already in the past. That is
        the one shape a legitimate forward-looking call can never have, and
        catching it here is what stops a backfilled "prediction" from entering
        the track record.
        """
        created = now or datetime.now(timezone.utc)
        resolve_on = thesis.resolves_on(created.date())

        # Compare against the WALL CLOCK, not against the supplied timestamp.
        #
        # Checking `resolve_on <= created.date()` is no check at all: a
        # backdated commit moves both sides together, so a thesis "made" 30
        # days ago with a 7-day horizon resolves 23 days ago and sails through.
        # The only question that matters is whether the outcome is knowable
        # NOW -- if it is, this is not a forecast.
        today = datetime.now(timezone.utc).date()
        if resolve_on <= today:
            raise ValueError(
                f"resolution date {resolve_on} is not in the future (today is "
                f"{today}) -- the outcome of this thesis is already knowable, "
                f"so it cannot enter the track record"
            )
        if entry_price <= 0:
            raise ValueError(f"entry price must be positive, got {entry_price}")

        thesis_id = f"th-{created.strftime('%Y%m%d-%H%M%S')}-{len(self._order):04d}"
        record = ThesisRecord(
            thesis_id=thesis_id,
            thesis=thesis,
            created_at=created,
            entry_price=entry_price,
            resolve_on=resolve_on,
            regime=regime,
            regime_confidence=regime_confidence,
            brief_digest=brief_digest,
            data_snapshot=data_snapshot,
            model=model,
            prev_hash=self._head_hash,
        )

        self._append({"kind": "thesis", "record": record.to_dict()})
        self._records[thesis_id] = record
        self._order.append(thesis_id)
        self._head_hash = record.content_hash

        logger.info(
            "committed %s: %s %s over %s at p=%.2f, entry %.4f, resolves %s",
            thesis_id, thesis.direction.value if hasattr(thesis.direction, "value") else thesis.direction,
            thesis.asset.value if hasattr(thesis.asset, "value") else thesis.asset,
            thesis.horizon.value if hasattr(thesis.horizon, "value") else thesis.horizon,
            thesis.probability, entry_price, resolve_on,
        )
        return record

    def resolve(
        self,
        thesis_id: str,
        *,
        outcome: str,
        exit_price: float | None,
        realized_pct: float | None,
        invalidated_on: date | None = None,
        notes: str = "",
        now: datetime | None = None,
    ) -> ThesisRecord:
        """
        Record a resolution as a NEW journal line.

        The original thesis line is never modified. That is what makes the
        chain verifiable after resolution: the hashes cover only the committed
        content, and outcomes accumulate alongside rather than inside them.
        """
        record = self._records.get(thesis_id)
        if record is None:
            raise KeyError(f"unknown thesis {thesis_id!r}")
        if record.is_resolved:
            raise ValueError(
                f"{thesis_id} is already resolved as {record.outcome}; a "
                f"resolution is final"
            )

        resolved_at = now or datetime.now(timezone.utc)
        payload = {
            "kind": "resolution",
            "thesis_id": thesis_id,
            "outcome": outcome,
            "resolved_at": resolved_at.isoformat(),
            "exit_price": exit_price,
            "realized_pct": realized_pct,
            "invalidated_on": invalidated_on.isoformat() if invalidated_on else None,
            "notes": notes,
        }
        self._append(payload)
        self._apply_resolution(payload)
        logger.info(
            "resolved %s: %s (entry %.4f -> exit %s, %.2f%%)",
            thesis_id, outcome, record.entry_price,
            f"{exit_price:.4f}" if exit_price else "n/a",
            realized_pct if realized_pct is not None else 0.0,
        )
        return record

    def _append(self, payload: dict[str, Any]) -> None:
        """
        Append one line, flushed and fsynced.

        fsync on every write is slow and correct. A thesis that exists in the
        page cache and not on disk when the process dies is a thesis that can be
        silently re-made with hindsight.
        """
        line = json.dumps(payload, separators=(",", ":"), default=str)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # -- verification ------------------------------------------------------

    def verify_chain(self) -> ChainVerification:
        """Recompute every hash and confirm each links to its predecessor."""
        previous = GENESIS_HASH
        for index, thesis_id in enumerate(self._order):
            record = self._records[thesis_id]
            if record.prev_hash != previous:
                return ChainVerification(
                    valid=False, records_checked=len(self._order), broken_at=index,
                    reason=(
                        f"{thesis_id} claims prev_hash {record.prev_hash[:12]}... "
                        f"but the previous record hashes to {previous[:12]}..."
                    ),
                )
            if not record.verify():
                return ChainVerification(
                    valid=False, records_checked=len(self._order), broken_at=index,
                    reason=(
                        f"{thesis_id} content does not match its stored hash -- "
                        f"the record was edited after it was committed"
                    ),
                )
            previous = record.content_hash

        return ChainVerification(
            valid=True, records_checked=len(self._order), head_hash=previous,
        )

    # -- querying ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self) -> Iterator[ThesisRecord]:
        for thesis_id in self._order:
            yield self._records[thesis_id]

    def get(self, thesis_id: str) -> ThesisRecord | None:
        return self._records.get(thesis_id)

    @property
    def head_hash(self) -> str:
        return self._head_hash

    def pending(self, *, due_on_or_before: date | None = None) -> list[ThesisRecord]:
        """Unresolved theses, optionally only those already due."""
        cutoff = due_on_or_before or date.today()
        return [
            r for r in self
            if not r.is_resolved and r.resolve_on <= cutoff
        ]

    def open_positions(self) -> list[ThesisRecord]:
        """Unresolved theses whose horizon has not yet elapsed."""
        today = date.today()
        return [r for r in self if not r.is_resolved and r.resolve_on > today]

    def resolved(self) -> list[ThesisRecord]:
        return [r for r in self if r.is_resolved]

    def scoreable(self) -> list[ThesisRecord]:
        return [r for r in self if r.is_scoreable]

    def by_asset(self, asset: str) -> list[ThesisRecord]:
        return [
            r for r in self
            if (r.thesis.asset.value if hasattr(r.thesis.asset, "value")
                else r.thesis.asset) == asset
        ]

    def by_regime(self, regime: str) -> list[ThesisRecord]:
        return [r for r in self if r.regime == regime]

    def stats(self) -> dict[str, Any]:
        verification = self.verify_chain()
        scoreable = self.scoreable()
        return {
            "path": str(self.path),
            "total": len(self._order),
            "pending": len([r for r in self if not r.is_resolved]),
            "due_now": len(self.pending()),
            "resolved": len(self.resolved()),
            "scoreable": len(scoreable),
            "correct": sum(1 for r in scoreable if r.outcome == Outcome.CORRECT),
            "wrong": sum(1 for r in scoreable if r.outcome == Outcome.WRONG),
            "invalidated": sum(1 for r in scoreable if r.outcome == Outcome.INVALIDATED),
            "chain_valid": verification.valid,
            "chain_summary": verification.summary(),
            "head_hash": self._head_hash,
        }
