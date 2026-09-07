"""
Exception hierarchy for Triangulum.

Design notes
------------
Every error carries a ``retryable`` flag and a ``severity``. The supervisor and
the circuit breakers key off these instead of doing ``isinstance`` gymnastics,
so adding a new venue-specific error never requires touching the risk layer.

Severity semantics:
    DEBUG    - expected, high-frequency, not worth logging at INFO
    WARNING  - degraded but self-healing (a reconnect, a rate-limit backoff)
    ERROR    - the current operation failed; the engine keeps running
    CRITICAL - the engine's assumptions are violated; trip the kill switch
"""

from __future__ import annotations

import enum
from typing import Any


class Severity(enum.IntEnum):
    DEBUG = 10
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class TriangulumError(Exception):
    """Root of the exception tree. Never raised directly."""

    retryable: bool = False
    severity: Severity = Severity.ERROR

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def __str__(self) -> str:  # pragma: no cover - trivial
        if not self.context:
            return self.message
        ctx = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} [{ctx}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": self.message,
            "retryable": self.retryable,
            "severity": int(self.severity),
            "context": self.context,
        }


# --------------------------------------------------------------------------
# Configuration / startup
# --------------------------------------------------------------------------


class ConfigError(TriangulumError):
    """Malformed or contradictory configuration. Always fatal at startup."""

    severity = Severity.CRITICAL


class UnsupportedVenueError(ConfigError):
    pass


class CapitalInadequateError(ConfigError):
    """
    Raised when the configured capital cannot satisfy the venue's minimum
    notional across every leg of the shortest cycle. This is the error that
    saves a small account from bleeding on quantization drag.
    """


# --------------------------------------------------------------------------
# Transport / venue
# --------------------------------------------------------------------------


class VenueError(TriangulumError):
    """Base for anything an exchange did to us."""

    def __init__(self, message: str = "", *, venue: str = "", **context: Any) -> None:
        super().__init__(message, venue=venue, **context)
        self.venue = venue


class ConnectionLost(VenueError):
    retryable = True
    severity = Severity.WARNING


class RateLimited(VenueError):
    retryable = True
    severity = Severity.WARNING

    def __init__(self, message: str = "", *, retry_after: float = 1.0, **ctx: Any) -> None:
        super().__init__(message, retry_after=retry_after, **ctx)
        self.retry_after = retry_after


class AuthenticationError(VenueError):
    severity = Severity.CRITICAL


class VenueMaintenance(VenueError):
    retryable = True
    severity = Severity.WARNING


class BadResponse(VenueError):
    """The venue returned something we could not parse."""

    retryable = True


class OrderRejected(VenueError):
    """
    The venue refused the order. Not retryable as-is: the caller must re-plan
    (resize, re-price) because the book has moved or a constraint was violated.
    """

    def __init__(self, message: str = "", *, code: str = "", **ctx: Any) -> None:
        super().__init__(message, code=code, **ctx)
        self.code = code


class InsufficientBalance(OrderRejected):
    severity = Severity.ERROR


class MinNotionalError(OrderRejected):
    severity = Severity.DEBUG


# --------------------------------------------------------------------------
# Market data integrity
# --------------------------------------------------------------------------


class MarketDataError(TriangulumError):
    pass


class BookChecksumMismatch(MarketDataError):
    """
    An incremental order-book update produced a book whose checksum disagrees
    with the venue's. The only safe response is to drop the book and resnapshot;
    trading on a corrupted book is how accounts die.
    """

    retryable = True
    severity = Severity.WARNING


class SequenceGap(MarketDataError):
    retryable = True
    severity = Severity.WARNING


class StaleBook(MarketDataError):
    """Book age exceeded the configured freshness budget."""

    severity = Severity.WARNING


class CrossedBook(MarketDataError):
    """best_bid >= best_ask on a single venue: almost always a data bug."""

    severity = Severity.WARNING


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


class ExecutionError(TriangulumError):
    pass


class LegFailed(ExecutionError):
    def __init__(self, message: str = "", *, leg_index: int = -1, **ctx: Any) -> None:
        super().__init__(message, leg_index=leg_index, **ctx)
        self.leg_index = leg_index


class CycleAborted(ExecutionError):
    """Cycle abandoned before any capital was committed. Cheap, expected."""

    severity = Severity.DEBUG


class UnwindFailed(ExecutionError):
    """
    We hold an unwanted inventory position and could not flatten it. This is the
    single worst state the engine can be in: escalate immediately.
    """

    severity = Severity.CRITICAL


class ReconciliationMismatch(ExecutionError):
    """Venue balances diverged from the internal ledger beyond tolerance."""

    severity = Severity.CRITICAL


# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------


class RiskError(TriangulumError):
    severity = Severity.WARNING


class LimitBreached(RiskError):
    def __init__(self, message: str = "", *, limit: str = "", value: Any = None,
                 threshold: Any = None, **ctx: Any) -> None:
        super().__init__(message, limit=limit, value=value, threshold=threshold, **ctx)
        self.limit = limit


class CircuitOpen(RiskError):
    """A breaker is open; the strategy must not emit orders."""


class KillSwitchEngaged(RiskError):
    severity = Severity.CRITICAL


class LiveTradingNotArmed(RiskError):
    """
    Live mode requires three independent confirmations. This error means at
    least one is missing, and it is deliberately not bypassable in code.
    """

    severity = Severity.CRITICAL


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------


class LearningError(TriangulumError):
    pass


class ModelNotFitted(LearningError):
    severity = Severity.WARNING


class FeatureMismatch(LearningError):
    severity = Severity.ERROR


class ConceptDriftDetected(LearningError):
    """Not an error so much as a signal; carried as an exception for routing."""

    severity = Severity.WARNING


__all__ = [name for name in dir() if not name.startswith("_")]
