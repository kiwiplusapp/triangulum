"""
What a trading signal is, in this system.

A signal here is not "a hint that something might go up". It is a named,
reproducible function from observable series to a number, carrying with it
everything needed to grade it later:

    - which series it read, so a stale or missing input is visible rather
      than silently defaulting to zero
    - how strong the reading is, on a scale comparable across signals
    - how confident the signal is in its own reading, separately from how
      strong that reading is

The last distinction is the one most signal libraries collapse, and it costs
them. "Momentum is strongly positive" and "momentum is strongly positive, but
the price series has not updated in nine days" are different statements, and
a system that cannot tell them apart will size the second like the first.

Every signal is scored independently against forward returns (see
``scoring.py``). A signal that cannot demonstrate an edge gets weight zero --
the same rule the capital gate applies to the agent as a whole, applied one
level down. Signals earn their weight; they are not granted it because they
are famous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SignalReading", "SignalFamily", "Stance",
    "squash", "STALENESS_FULL_CONFIDENCE_DAYS", "STALENESS_ZERO_CONFIDENCE_DAYS",
]


class SignalFamily:
    """
    Coarse grouping, used to stop the composite double-counting.

    Six momentum signals that all read the S&P are not six independent votes,
    and averaging them as if they were is how a composite ends up with a
    confidence interval far narrower than its actual information content.
    The composite weights within a family before it weights across families.
    """

    CURVE = "curve"
    INFLATION = "inflation"
    CREDIT = "credit"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    LABOUR = "labour"
    LIQUIDITY = "liquidity"
    HOUSING = "housing"

    ALL = (
        CURVE, INFLATION, CREDIT, MOMENTUM, VOLATILITY,
        CURRENCY, COMMODITY, LABOUR, LIQUIDITY, HOUSING,
    )


class Stance:
    """What a signal is saying, in risk terms."""

    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"


# A signal reading from a series that updated today is worth its full weight.
# One from a series that last updated six weeks ago is worth nothing, and the
# decay between them is linear rather than a cliff, so a monthly series
# published on an irregular calendar degrades smoothly instead of flickering
# in and out of the composite.
STALENESS_FULL_CONFIDENCE_DAYS = 7
STALENESS_ZERO_CONFIDENCE_DAYS = 45


@dataclass(slots=True)
class SignalReading:
    """One signal's view of the world right now."""

    key: str
    label: str
    family: str

    strength: float = 0.0            # [-1, +1]; sign is risk-on positive
    raw: float | None = None         # the underlying number, in its own units
    zscore: float | None = None
    confidence: float = 0.0          # [0, 1]; data quality, NOT conviction

    inputs: tuple[str, ...] = ()
    staleness_days: int = 0
    usable: bool = False
    note: str = ""
    unavailable_reason: str = ""

    @property
    def stance(self) -> str:
        if not self.usable or abs(self.strength) < 0.15:
            return Stance.NEUTRAL
        return Stance.RISK_ON if self.strength > 0 else Stance.RISK_OFF

    @property
    def weighted_strength(self) -> float:
        """Strength discounted by how much the inputs can be trusted."""
        return self.strength * self.confidence if self.usable else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "strength": round(self.strength, 4),
            "weighted_strength": round(self.weighted_strength, 4),
            "raw": round(self.raw, 5) if self.raw is not None else None,
            "zscore": round(self.zscore, 3) if self.zscore is not None else None,
            "confidence": round(self.confidence, 3),
            "stance": self.stance,
            "inputs": list(self.inputs),
            "staleness_days": self.staleness_days,
            "usable": self.usable,
            "note": self.note,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def unavailable(cls, key: str, label: str, family: str, reason: str,
                    inputs: tuple[str, ...] = ()) -> "SignalReading":
        """
        A signal that could not be computed.

        Returned rather than omitted, and never silently coerced to 0.0. A
        missing signal and a neutral signal are different facts: the first
        says the system is partly blind, the second says it looked and saw
        nothing. A feature vector that encodes both as zero teaches the model
        that blindness is a market condition.
        """
        return cls(key=key, label=label, family=family, inputs=inputs,
                   usable=False, unavailable_reason=reason)


def squash(value: float, scale: float) -> float:
    """
    Map an unbounded reading onto [-1, +1] via tanh.

    ``scale`` is the value that maps to roughly 0.76 -- i.e. "a clearly
    significant move", chosen per signal from the historical distribution of
    that particular series rather than from a single global constant. Using
    tanh rather than clipping keeps the derivative non-zero at the extremes,
    which matters because these readings feed a gradient-trained model
    downstream: a clipped feature is a feature the network cannot learn from
    once it saturates.
    """
    if scale <= 0 or not math.isfinite(value):
        return 0.0
    return math.tanh(value / scale)


def confidence_from_staleness(days: int) -> float:
    """Linear decay from full confidence to zero. See the constants above."""
    if days <= STALENESS_FULL_CONFIDENCE_DAYS:
        return 1.0
    if days >= STALENESS_ZERO_CONFIDENCE_DAYS:
        return 0.0
    span = STALENESS_ZERO_CONFIDENCE_DAYS - STALENESS_FULL_CONFIDENCE_DAYS
    return 1.0 - (days - STALENESS_FULL_CONFIDENCE_DAYS) / span
