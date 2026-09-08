"""
Automatic resolution.

The screenshot that prompted this project showed a beautiful macro terminal
whose hit-rate panel read `0 runs / 0 resolved`. Everything else was built; the
scoreboard was decoration.

This module is that scoreboard, working. It runs unattended, it cannot be
argued with, and it applies the same rules to every call.

## The rules

A thesis resolves when its horizon elapses. Before then it is checked daily
against its invalidation level, and hitting that level ends it early.

    CORRECT       price moved in the stated direction over the horizon
    WRONG         it did not
    INVALIDATED   the adverse move hit the stop before the horizon
    UNRESOLVABLE  no price data at the horizon -- excluded from scoring

## Two decisions that matter

**INVALIDATED scores as wrong.** The thesis said up, price went far enough down
to hit the stop, and the fact that it may have recovered afterwards is not
relevant -- a real position would have been closed. Counting invalidations as
"neither" would let a system quietly discard its worst calls.

**A flat close is WRONG, not a tie.** If the asset ends exactly where it
started, the directional call did not happen. Ties are vanishingly rare in
practice, and a "tie" category is a place for losses to hide.

## The revision problem

Macro series get revised; market prices do not. Every resolvable asset in the
schema is a market price for exactly this reason. A thesis about "next month's
CPI print" would be graded against a number that changes twice after
publication, and there is no honest way to score that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from vault.data.series import Series
from vault.thesis.journal import ThesisJournal
from vault.thesis.schema import Outcome, ThesisRecord

logger = logging.getLogger(__name__)

__all__ = ["Resolver", "ResolutionReport", "resolve_one"]


@dataclass(slots=True)
class ResolutionReport:
    checked: int = 0
    resolved: int = 0
    correct: int = 0
    wrong: int = 0
    invalidated: int = 0
    unresolvable: int = 0
    still_open: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        if not self.resolved:
            return f"checked {self.checked}, nothing due for resolution"
        return (
            f"resolved {self.resolved} of {self.checked}: "
            f"{self.correct} correct, {self.wrong} wrong, "
            f"{self.invalidated} invalidated, {self.unresolvable} unresolvable "
            f"({self.still_open} still open)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "resolved": self.resolved,
            "correct": self.correct,
            "wrong": self.wrong,
            "invalidated": self.invalidated,
            "unresolvable": self.unresolvable,
            "still_open": self.still_open,
            "details": self.details,
        }


def resolve_one(
    record: ThesisRecord,
    prices: Series,
    *,
    today: date | None = None,
) -> tuple[str, float | None, float | None, date | None, str]:
    """
    Determine one thesis's outcome.

    Returns ``(outcome, exit_price, realized_pct, invalidated_on, note)``.

    The invalidation scan walks every observation between commit and horizon,
    not just the endpoint. A thesis that was stopped out on day three and
    recovered by day seven is INVALIDATED, because that is what would have
    happened to the position. Grading only the endpoint would systematically
    flatter any strategy with wide stops.
    """
    today = today or date.today()
    thesis = record.thesis
    entry = record.entry_price
    sign = thesis.sign

    if entry <= 0:
        return Outcome.UNRESOLVABLE, None, None, None, "entry price was not positive"

    stop = thesis.invalidation_price(entry)

    # ---- 1. did the stop trigger before the horizon? ----
    window = [
        point for point in prices.points
        if record.created_at.date() < point.on <= min(record.resolve_on, today)
    ]
    for point in window:
        breached = (
            point.value <= stop if sign > 0 else point.value >= stop
        )
        if breached:
            realized = (point.value / entry - 1) * 100
            return (
                Outcome.INVALIDATED, point.value, realized, point.on,
                f"stop at {stop:.4f} breached on {point.on} "
                f"({point.value:.4f} vs entry {entry:.4f})",
            )

    # ---- 2. has the horizon elapsed? ----
    if record.resolve_on > today:
        return Outcome.PENDING, None, None, None, "horizon has not elapsed"

    # ---- 3. grade at the horizon ----
    # Tolerance covers weekends and holidays: a Saturday horizon resolves
    # against Friday's close rather than being marked unresolvable.
    exit_price = prices.value_on(record.resolve_on, tolerance_days=6)
    if exit_price is None or exit_price <= 0:
        return (
            Outcome.UNRESOLVABLE, None, None, None,
            f"no price for {prices.key} within 6 days of {record.resolve_on}",
        )

    realized = (exit_price / entry - 1) * 100
    moved_with_thesis = (realized > 0) if sign > 0 else (realized < 0)

    if realized == 0:
        return (
            Outcome.WRONG, exit_price, 0.0, None,
            "price closed exactly flat; a directional call that did not happen "
            "is wrong, not a tie",
        )

    outcome = Outcome.CORRECT if moved_with_thesis else Outcome.WRONG
    return (
        outcome, exit_price, realized, None,
        f"entry {entry:.4f} -> exit {exit_price:.4f} ({realized:+.2f}%)",
    )


class Resolver:
    """Resolves every due thesis against price data."""

    def __init__(self, journal: ThesisJournal) -> None:
        self.journal = journal
        self.runs = 0

    def run(
        self,
        prices_by_asset: Mapping[str, Series],
        *,
        today: date | None = None,
        include_open: bool = True,
    ) -> ResolutionReport:
        """
        Resolve everything that can be resolved.

        ``include_open`` also scans un-expired theses for stop breaches, which
        is what makes invalidation an early exit rather than a post-hoc
        reinterpretation at the horizon.
        """
        self.runs += 1
        today = today or date.today()
        report = ResolutionReport()

        candidates = [r for r in self.journal if not r.is_resolved]
        if not include_open:
            candidates = [r for r in candidates if r.resolve_on <= today]

        for record in candidates:
            report.checked += 1
            asset = (
                record.thesis.asset.value
                if hasattr(record.thesis.asset, "value") else record.thesis.asset
            )
            prices = prices_by_asset.get(asset)

            if prices is None or not prices.points:
                if record.resolve_on <= today:
                    self._commit(record, Outcome.UNRESOLVABLE, None, None, None,
                                 f"no price series for {asset}", report, today)
                else:
                    report.still_open += 1
                continue

            outcome, exit_price, realized, invalidated_on, note = resolve_one(
                record, prices, today=today,
            )

            if outcome == Outcome.PENDING:
                report.still_open += 1
                continue

            self._commit(record, outcome, exit_price, realized, invalidated_on,
                         note, report, today)

        logger.info("resolver: %s", report.summary())
        return report

    def _commit(
        self, record: ThesisRecord, outcome: str, exit_price: float | None,
        realized: float | None, invalidated_on: date | None, note: str,
        report: ResolutionReport, today: date,
    ) -> None:
        self.journal.resolve(
            record.thesis_id,
            outcome=outcome,
            exit_price=exit_price,
            realized_pct=realized,
            invalidated_on=invalidated_on,
            notes=note,
        )
        report.resolved += 1
        if outcome == Outcome.CORRECT:
            report.correct += 1
        elif outcome == Outcome.WRONG:
            report.wrong += 1
        elif outcome == Outcome.INVALIDATED:
            report.invalidated += 1
        else:
            report.unresolvable += 1

        asset = (
            record.thesis.asset.value
            if hasattr(record.thesis.asset, "value") else record.thesis.asset
        )
        direction = (
            record.thesis.direction.value
            if hasattr(record.thesis.direction, "value") else record.thesis.direction
        )
        report.details.append({
            "thesis_id": record.thesis_id,
            "asset": asset,
            "direction": direction,
            "probability": record.thesis.probability,
            "outcome": outcome,
            "entry": record.entry_price,
            "exit": exit_price,
            "realized_pct": round(realized, 3) if realized is not None else None,
            "brier": record.brier,
            "note": note,
        })
