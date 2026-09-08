"""
Fee models for the prediction venues.

## Why these are structured as declared assumptions rather than constants

Fees decide whether a 2-cent edge is a trade or a loss, and neither venue
publishes a machine-readable schedule on the endpoints used here. Kalshi's
market payloads carry no fee fields at all. Polymarket's carry
``takerBaseFee: 1000`` and ``makerBaseFee: 1000`` with a ``feeType`` of
``politics_fees`` or ``sports_fees_v3`` -- integers whose *scale* is not stated
anywhere in the response.

Guessing that scale wrong is not a rounding error. If 1000 means 0.10% the
scanner will find trades; if it means 10% every one of them is a loss. So the
scale is not guessed. Each model carries:

    - the formula, written out
    - ``verified``: whether it has been checked against a real fill
    - ``source``: where the numbers came from
    - ``confidence``: what to do with the output

and every opportunity reports edge **gross and net side by side** with the fee
assumption named, so a wrong assumption is visible rather than silently
baked into a number that looks authoritative.

**Nothing here has been verified against a real fill.** The defaults are
deliberately pessimistic. Before trading, place one small order on each venue,
read the actual fee off the confirmation, and set the model from that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vault.markets.types import Venue

__all__ = ["FeeModel", "KALSHI_FEES", "POLYMARKET_FEES", "FEE_MODELS", "fee_model_for"]


@dataclass(slots=True, frozen=True)
class FeeModel:
    """What a venue charges, and how much we trust that we know."""

    venue: str
    formula: str
    source: str
    verified: bool = False
    confidence: str = "unverified"

    # Kalshi-style: fee = rate * contracts * price * (1 - price), rounded up to
    # the cent. The p(1-p) shape means fees peak at 50c and vanish at the
    # extremes, which matters a lot here -- most arbitrage legs are cheap
    # contracts near 0 or 1, where this model charges very little.
    curve_rate: float = 0.0

    # Flat proportional taker fee on notional, as a fraction.
    proportional_rate: float = 0.0

    # Per-order fixed cost in dollars (gas, withdrawal amortisation).
    fixed_per_order: float = 0.0

    # Rounding granularity in dollars. Kalshi rounds fees UP to the cent, which
    # on a 1-contract order can exceed the edge entirely.
    round_up_to: float = 0.0

    def cost(self, contracts: float, price: float, *, taker: bool = True) -> float:
        """Fee in dollars for filling ``contracts`` at ``price``."""
        if contracts <= 0:
            return 0.0
        price = min(1.0, max(0.0, price))
        fee = 0.0
        if self.curve_rate:
            fee += self.curve_rate * contracts * price * (1.0 - price)
        if self.proportional_rate:
            fee += self.proportional_rate * contracts * price
        if self.round_up_to > 0:
            fee = math.ceil(fee / self.round_up_to) * self.round_up_to
        return fee + self.fixed_per_order

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "formula": self.formula,
            "source": self.source,
            "verified": self.verified,
            "confidence": self.confidence,
            "curve_rate": self.curve_rate,
            "proportional_rate": self.proportional_rate,
            "fixed_per_order": self.fixed_per_order,
            "round_up_to": self.round_up_to,
        }


KALSHI_FEES = FeeModel(
    venue=Venue.KALSHI,
    formula="ceil_to_cent(0.07 * contracts * price * (1 - price))",
    source=(
        "Kalshi's published trading-fee schedule. NOT confirmed against a fill "
        "from this account, and not exposed on any API endpoint used here."
    ),
    verified=False,
    confidence=(
        "The 0.07 coefficient and the p(1-p) shape are the widely documented "
        "form. The rounding-up-to-the-cent is what bites on small orders: on "
        "a single contract at 50c the fee is 2c, which is 4% of notional and "
        "larger than most edges this scanner will find. Size accordingly."
    ),
    curve_rate=0.07,
    round_up_to=0.01,
)


POLYMARKET_FEES = FeeModel(
    venue=Venue.POLYMARKET,
    formula="proportional_rate * contracts * price, plus a fixed per-order cost",
    source=(
        "Market payloads report takerBaseFee=1000 and makerBaseFee=1000 with "
        "feeType 'politics_fees' or 'sports_fees_v3'. The SCALE of 1000 is not "
        "stated in the response and is not assumed here."
    ),
    verified=False,
    confidence=(
        "Assumes 1000 means 10 basis points (1000 / 1e5), the least "
        "unfavourable of the plausible readings that is still non-zero. If the "
        "true scale is 1000/1e4 = 1%, every net edge below 2% reported here is "
        "wrong by a factor that flips its sign. VERIFY BEFORE TRADING: place "
        "one $5 order and read the fee off the fill. The fixed cost covers "
        "on-chain settlement, which is small on Polygon but not zero and does "
        "not scale down with order size."
    ),
    proportional_rate=0.0010,
    fixed_per_order=0.02,
)


FEE_MODELS: dict[str, FeeModel] = {
    Venue.KALSHI: KALSHI_FEES,
    Venue.POLYMARKET: POLYMARKET_FEES,
}


def fee_model_for(venue: str) -> FeeModel:
    """
    The model for a venue, or a deliberately punitive fallback.

    An unknown venue gets a 2% proportional charge rather than zero. Defaulting
    an unknown fee to zero is how a scanner reports free money on a venue
    nobody has modelled.
    """
    model = FEE_MODELS.get(venue)
    if model is not None:
        return model
    return FeeModel(
        venue=venue,
        formula="2% of notional (placeholder)",
        source="no model for this venue",
        verified=False,
        confidence=(
            "This venue has no fee model. The 2% placeholder is punitive on "
            "purpose so that unmodelled venues cannot manufacture edge."
        ),
        proportional_rate=0.02,
    )
