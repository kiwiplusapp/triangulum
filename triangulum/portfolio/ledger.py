"""
Double-entry ledger.

Every fill produces two entries: one debit, one credit. The invariant is that
they always balance, and the ledger checks it on every write rather than
trusting itself.

Why double-entry rather than simply tracking balances: in a triangular cycle
the same capital passes through three assets in under a second, and fees are
charged in three different currencies. A single-entry "balance" model cannot
answer "where did the 0.4 bps go?" -- it only knows the endpoint. Double-entry
can decompose the outcome into spread cost, fee cost, slippage and quantization
drag, which is the difference between knowing you lost money and knowing why.

The reconciliation check against venue balances is not optional. Any divergence
beyond tolerance means one of: an unrecorded fill, a fee model error, a rounding
bug, or a fill we never saw. All four are reasons to stop trading, because all
four mean the engine's model of its own position is wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterator, Mapping, Sequence

from triangulum.core.clock import Clock, SystemClock
from triangulum.core.decimal_math import D, ZERO, bps, safe_div
from triangulum.core.errors import ReconciliationMismatch
from triangulum.core.types import (
    Asset, ExecutionResult, Fill, Liquidity, Side, new_id,
)

logger = logging.getLogger(__name__)

__all__ = ["Ledger", "Entry", "EntryType", "Lot", "PnLBreakdown"]


class EntryType:
    TRADE = "trade"
    FEE = "fee"
    REBATE = "rebate"
    TRANSFER = "transfer"
    ADJUSTMENT = "adjustment"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"


@dataclass(frozen=True, slots=True)
class Entry:
    """One side of a double-entry pair."""

    entry_id: str
    ts_ns: int
    asset: str
    amount: Decimal          # signed: positive = credit, negative = debit
    entry_type: str
    reference: str = ""      # cycle id, order id, etc.
    venue: str = ""
    note: str = ""

    @property
    def is_credit(self) -> bool:
        return self.amount > 0


@dataclass(slots=True)
class Lot:
    """A FIFO cost-basis lot."""

    asset: str
    quantity: Decimal
    cost_per_unit: Decimal   # in base currency
    acquired_ns: int

    @property
    def cost_basis(self) -> Decimal:
        return self.quantity * self.cost_per_unit


@dataclass(slots=True)
class PnLBreakdown:
    """
    Decomposition of realized P&L.

    The sum of the components equals the total. When it does not, the
    attribution is wrong somewhere and the ledger says so rather than hiding a
    residual in one of the buckets.
    """

    gross_spread: Decimal = ZERO      # what the price moves gave us
    fees_paid: Decimal = ZERO
    rebates_earned: Decimal = ZERO
    slippage: Decimal = ZERO          # realized worse than planned
    quantization: Decimal = ZERO      # lot-rounding loss
    unwind_cost: Decimal = ZERO
    total: Decimal = ZERO

    @property
    def residual(self) -> Decimal:
        """Unattributed remainder. Should be ~0; a large value is a bug."""
        attributed = (
            self.gross_spread - self.fees_paid + self.rebates_earned
            - self.slippage - self.quantization - self.unwind_cost
        )
        return self.total - attributed

    def to_dict(self) -> dict[str, str]:
        return {
            "gross_spread": str(self.gross_spread),
            "fees_paid": str(self.fees_paid),
            "rebates_earned": str(self.rebates_earned),
            "slippage": str(self.slippage),
            "quantization": str(self.quantization),
            "unwind_cost": str(self.unwind_cost),
            "total": str(self.total),
            "residual": str(self.residual),
        }


class Ledger:
    """Double-entry ledger with FIFO lot accounting."""

    def __init__(
        self,
        base_currency: str = "USDT",
        *,
        clock: Clock | None = None,
        max_entries: int = 200_000,
    ) -> None:
        self.base_currency = base_currency
        self.clock = clock or SystemClock()
        self.max_entries = max_entries

        self._entries: list[Entry] = []
        self._balances: dict[str, Decimal] = {}
        self._lots: dict[str, list[Lot]] = {}
        self._by_venue: dict[str, dict[str, Decimal]] = {}

        self.realized_pnl = ZERO
        self.total_fees = ZERO
        self.total_rebates = ZERO
        self.cycles_recorded = 0
        self.breakdown = PnLBreakdown()
        self._imbalances = 0

    # -- writing -----------------------------------------------------------

    def post(
        self,
        *,
        asset: str,
        amount: Decimal,
        entry_type: str,
        reference: str = "",
        venue: str = "",
        note: str = "",
        ts_ns: int = 0,
    ) -> Entry:
        entry = Entry(
            entry_id=new_id("e-"),
            ts_ns=ts_ns or self.clock.wall_ns(),
            asset=asset,
            amount=amount,
            entry_type=entry_type,
            reference=reference,
            venue=venue,
            note=note,
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            # Keep the tail; the full history lives in the SQLite store.
            self._entries = self._entries[-(self.max_entries // 2):]

        self._balances[asset] = self._balances.get(asset, ZERO) + amount
        if venue:
            venue_book = self._by_venue.setdefault(venue, {})
            venue_book[asset] = venue_book.get(asset, ZERO) + amount
        return entry

    def record_fill(self, fill: Fill, *, reference: str = "") -> tuple[Entry, Entry]:
        """
        Record a fill as a balanced pair of entries, plus the fee.

        BUY:  debit quote (price*qty), credit base (qty), fee in base
        SELL: debit base (qty), credit quote (price*qty), fee in quote
        """
        venue = fill.symbol.venue
        base = fill.symbol.base.code
        quote = fill.symbol.quote.code
        notional = fill.price * fill.quantity
        ref = reference or fill.order_id

        if fill.side is Side.BUY:
            debit = self.post(asset=quote, amount=-notional, entry_type=EntryType.TRADE,
                              reference=ref, venue=venue,
                              note=f"buy {fill.quantity} {base} @ {fill.price}")
            credit = self.post(asset=base, amount=fill.quantity, entry_type=EntryType.TRADE,
                               reference=ref, venue=venue)
        else:
            debit = self.post(asset=base, amount=-fill.quantity, entry_type=EntryType.TRADE,
                              reference=ref, venue=venue,
                              note=f"sell {fill.quantity} {base} @ {fill.price}")
            credit = self.post(asset=quote, amount=notional, entry_type=EntryType.TRADE,
                               reference=ref, venue=venue)

        if fill.fee != 0:
            is_rebate = fill.fee < 0
            self.post(
                asset=fill.fee_asset.code,
                amount=-fill.fee,     # a positive fee debits, a rebate credits
                entry_type=EntryType.REBATE if is_rebate else EntryType.FEE,
                reference=ref, venue=venue,
                note=f"{fill.liquidity.value} fee",
            )
            if is_rebate:
                self.total_rebates += -fill.fee
                self.breakdown.rebates_earned += -fill.fee
            else:
                self.total_fees += fill.fee
                self.breakdown.fees_paid += fill.fee

        self._update_lots(fill)
        return debit, credit

    def _update_lots(self, fill: Fill) -> None:
        """FIFO lot tracking, for cost basis and tax-style reporting."""
        base = fill.symbol.base.code
        lots = self._lots.setdefault(base, [])
        if fill.side is Side.BUY:
            lots.append(Lot(
                asset=base, quantity=fill.quantity,
                cost_per_unit=fill.price, acquired_ns=fill.ts_ns,
            ))
        else:
            remaining = fill.quantity
            while remaining > 0 and lots:
                lot = lots[0]
                consumed = min(remaining, lot.quantity)
                lot.quantity -= consumed
                remaining -= consumed
                if lot.quantity <= 0:
                    lots.pop(0)

    def record_cycle(self, result: ExecutionResult) -> None:
        """Record every fill in a cycle and attribute its P&L."""
        self.cycles_recorded += 1
        for fill in result.fills:
            self.record_fill(fill, reference=result.cycle_id)

        if not result.outcome.committed_capital:
            return

        pnl = result.realized_pnl
        self.realized_pnl += pnl
        self.breakdown.total += pnl

        # Attribute: the plan said what it expected; the difference between
        # expected and realized is slippage, and the difference between gross
        # and net is fees plus quantization.
        plan = result.plan
        expected = plan.expected_end_amount - plan.start_amount
        self.breakdown.gross_spread += expected + result.total_fees
        self.breakdown.slippage += max(ZERO, expected - pnl)
        self.breakdown.quantization += (
            plan.start_amount * plan.slippage_bps / D(10_000)
        )
        if result.outcome.value.startswith("partial"):
            self.breakdown.unwind_cost += abs(min(ZERO, pnl))

    def deposit(self, asset: str, amount: Decimal, venue: str = "") -> Entry:
        return self.post(asset=asset, amount=amount, entry_type=EntryType.DEPOSIT,
                         venue=venue, note="initial funding")

    # -- reading -----------------------------------------------------------

    def balance(self, asset: str) -> Decimal:
        return self._balances.get(asset, ZERO)

    def balances(self) -> Mapping[str, Decimal]:
        return {k: v for k, v in self._balances.items() if v != 0}

    def venue_balances(self, venue: str) -> Mapping[str, Decimal]:
        return dict(self._by_venue.get(venue, {}))

    def cost_basis(self, asset: str) -> Decimal:
        lots = self._lots.get(asset, [])
        return sum((l.cost_basis for l in lots), ZERO)

    def average_cost(self, asset: str) -> Decimal:
        lots = self._lots.get(asset, [])
        quantity = sum((l.quantity for l in lots), ZERO)
        return safe_div(self.cost_basis(asset), quantity)

    def equity(self, prices: Mapping[str, Decimal]) -> Decimal:
        """
        Mark all balances into the base currency.

        ``prices`` maps ``"ASSET/BASE"`` to a rate. Assets with no price are
        counted at zero and logged -- silently ignoring them would make the
        equity curve drift upward as untradeable dust accumulated.
        """
        total = ZERO
        unpriced: list[str] = []
        for asset, amount in self._balances.items():
            if amount == 0:
                continue
            if asset == self.base_currency:
                total += amount
                continue
            rate = prices.get(f"{asset}/{self.base_currency}")
            if rate:
                total += amount * rate
                continue
            inverse = prices.get(f"{self.base_currency}/{asset}")
            if inverse and inverse > 0:
                total += amount / inverse
                continue
            unpriced.append(asset)
        if unpriced:
            logger.debug("ledger: no price for %s; excluded from equity", unpriced)
        return total

    def entries(
        self, *, since_ns: int = 0, asset: str = "", reference: str = "",
    ) -> list[Entry]:
        return [
            e for e in self._entries
            if (not since_ns or e.ts_ns >= since_ns)
            and (not asset or e.asset == asset)
            and (not reference or e.reference == reference)
        ]

    # -- integrity ---------------------------------------------------------

    def verify_balanced(self, prices: Mapping[str, Decimal]) -> bool:
        """
        Check that trade entries net to zero in base-currency terms.

        Trades only -- fees legitimately reduce the total, and deposits
        legitimately increase it. What must balance is the *exchange* of value.
        """
        total = ZERO
        for entry in self._entries:
            if entry.entry_type != EntryType.TRADE:
                continue
            if entry.asset == self.base_currency:
                total += entry.amount
                continue
            rate = prices.get(f"{entry.asset}/{self.base_currency}")
            if rate:
                total += entry.amount * rate
        balanced = abs(total) < D("0.01")
        if not balanced:
            self._imbalances += 1
            logger.error("ledger imbalance of %s in base currency terms", total)
        return balanced

    def reconcile(
        self,
        venue_balances: Mapping[str, Decimal],
        *,
        tolerance_bps: Decimal = D("5"),
        prices: Mapping[str, Decimal] | None = None,
    ) -> dict[str, Decimal]:
        """
        Compare internal balances against the venue's own.

        Returns the per-asset divergence. Raises when any divergence exceeds
        tolerance, because a ledger that disagrees with reality is a ledger
        that cannot size the next trade correctly.
        """
        divergences: dict[str, Decimal] = {}
        breaches: list[str] = []

        for asset in set(list(self._balances) + list(venue_balances)):
            internal = self._balances.get(asset, ZERO)
            external = venue_balances.get(asset, ZERO)
            difference = external - internal
            if difference == 0:
                continue
            divergences[asset] = difference
            reference = max(abs(internal), abs(external))
            if reference <= 0:
                continue
            deviation = bps(abs(difference) / reference)
            if deviation > tolerance_bps:
                breaches.append(
                    f"{asset}: internal {internal}, venue {external} "
                    f"({deviation:.1f} bps apart)"
                )

        if breaches:
            raise ReconciliationMismatch(
                "ledger diverged from venue balances: " + "; ".join(breaches),
                divergences={k: str(v) for k, v in divergences.items()},
            )
        return divergences

    def stats(self) -> dict[str, object]:
        return {
            "entries": len(self._entries),
            "cycles_recorded": self.cycles_recorded,
            "realized_pnl": str(self.realized_pnl),
            "total_fees": str(self.total_fees),
            "total_rebates": str(self.total_rebates),
            "balances": {k: str(v) for k, v in self.balances().items()},
            "breakdown": self.breakdown.to_dict(),
            "imbalances_detected": self._imbalances,
            "base_currency": self.base_currency,
        }
