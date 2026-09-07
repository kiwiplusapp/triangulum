"""
Exact decimal arithmetic for price/quantity math.

Why this module exists
----------------------
Arbitrage edges live in the 1-30 bps range. A float64 has ~15-17 significant
decimal digits, which sounds like plenty until you chain three multiplications
of prices that span 8 orders of magnitude (BTC at 60_000 vs SHIB at 0.000008)
and then compare the product against 1.0 to decide whether to commit capital.
The accumulated representation error is small in absolute terms but it is *not*
small relative to a 3 bps edge, and it is systematically biased by the venue's
own rounding rules.

So: every quantity of money and every price in Triangulum is a ``Decimal``, and
every rounding is explicit about its direction. Rounding in the wrong direction
is not a cosmetic bug -- it is the difference between an order that fills and an
order the venue rejects for min-notional, or between a cycle that nets +2 bps
and one that nets -1 bps.

Rounding policy
---------------
The house rule is *always round against yourself*:

    - quantity you are selling / spending  -> ROUND_DOWN  (never overcommit)
    - quantity you expect to receive       -> ROUND_DOWN  (never overestimate)
    - price you pay when buying            -> ROUND_UP    (assume the worse fill)
    - price you receive when selling       -> ROUND_DOWN  (assume the worse fill)
    - fees you will pay                    -> ROUND_UP    (never underestimate)

Applied consistently, the engine's projected P&L becomes a *lower bound* on the
realized P&L modulo market movement. That property is worth more than a fraction
of a basis point of theoretical precision, because it means a backtest that
shows a profit is not showing you a rounding artifact.
"""

from __future__ import annotations

import math
from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    ROUND_UP,
    Context,
    Decimal,
    InvalidOperation,
    localcontext,
)
from typing import Iterable, Sequence

__all__ = [
    "D",
    "ZERO",
    "ONE",
    "DECIMAL_CONTEXT",
    "PRICE_PRECISION",
    "QTY_PRECISION",
    "to_decimal",
    "quantize_down",
    "quantize_up",
    "quantize_nearest",
    "round_to_step",
    "round_price_for_side",
    "floor_to_step",
    "ceil_to_step",
    "step_of",
    "decimals_of_step",
    "safe_div",
    "bps",
    "from_bps",
    "apply_bps",
    "pct",
    "geometric_product",
    "log_sum",
    "ln",
    "exp",
    "clamp",
    "is_close",
    "sign",
    "quantization_drag_bps",
]


# High working precision. 34 digits is IEEE-754 decimal128; it is far beyond
# anything a venue will accept but keeps intermediate products clean.
DECIMAL_CONTEXT = Context(prec=34, rounding=ROUND_HALF_EVEN)

ZERO = Decimal(0)
ONE = Decimal(1)

# Display/storage precision. Venues never exceed these in practice.
PRICE_PRECISION = Decimal("0.00000000000001")  # 1e-14
QTY_PRECISION = Decimal("0.00000000000001")


def D(value: object) -> Decimal:
    """
    Coerce to Decimal without ever going through binary float when avoidable.

    ``D(0.1)`` is a bug waiting to happen (0.1 is not representable in binary),
    so floats are routed through ``repr`` which gives the shortest string that
    round-trips -- i.e. ``D(0.1) == Decimal("0.1")`` rather than
    ``Decimal("0.1000000000000000055511151231257827021181583404541015625")``.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, str):
        return Decimal(value)
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"cannot convert non-finite float {value!r} to Decimal")
        return Decimal(repr(value))
    if value is None:
        raise TypeError("cannot convert None to Decimal")
    return Decimal(str(value))


# Alias with a more searchable name.
to_decimal = D


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------


def quantize_down(value: Decimal, exp: Decimal) -> Decimal:
    """Truncate ``value`` toward zero to the precision of ``exp``."""
    with localcontext(DECIMAL_CONTEXT):
        return value.quantize(exp, rounding=ROUND_DOWN)


def quantize_up(value: Decimal, exp: Decimal) -> Decimal:
    """Round ``value`` away from zero to the precision of ``exp``."""
    with localcontext(DECIMAL_CONTEXT):
        return value.quantize(exp, rounding=ROUND_UP)


def quantize_nearest(value: Decimal, exp: Decimal) -> Decimal:
    with localcontext(DECIMAL_CONTEXT):
        return value.quantize(exp, rounding=ROUND_HALF_EVEN)


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    """
    Largest multiple of ``step`` that is <= ``value``.

    This is the operation venues apply to your order quantity. Getting it wrong
    by one ulp is the single most common cause of "LOT_SIZE" rejections.
    """
    if step <= 0:
        return value
    with localcontext(DECIMAL_CONTEXT):
        n = (value / step).to_integral_value(rounding=ROUND_FLOOR)
        return (n * step).normalize() + ZERO


def ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Smallest multiple of ``step`` that is >= ``value``."""
    if step <= 0:
        return value
    with localcontext(DECIMAL_CONTEXT):
        n = (value / step).to_integral_value(rounding=ROUND_CEILING)
        return (n * step).normalize() + ZERO


def round_to_step(value: Decimal, step: Decimal, *, mode: str = "down") -> Decimal:
    if mode == "down":
        return floor_to_step(value, step)
    if mode == "up":
        return ceil_to_step(value, step)
    if mode == "nearest":
        if step <= 0:
            return value
        with localcontext(DECIMAL_CONTEXT):
            n = (value / step).to_integral_value(rounding=ROUND_HALF_EVEN)
            return (n * step).normalize() + ZERO
    raise ValueError(f"unknown rounding mode {mode!r}")


def round_price_for_side(price: Decimal, tick: Decimal, side: str) -> Decimal:
    """
    Round a price to the venue tick, biased against us.

    Buying: round the limit price *up* so the order is at least as aggressive as
    intended. Selling: round *down*. This guarantees a marketable order stays
    marketable after quantization, which matters for IOC legs where a one-tick
    slip means a no-fill and an aborted cycle.
    """
    s = side.lower()
    if s in ("buy", "bid", "b"):
        return ceil_to_step(price, tick)
    if s in ("sell", "ask", "a", "s"):
        return floor_to_step(price, tick)
    raise ValueError(f"unknown side {side!r}")


def step_of(decimals: int) -> Decimal:
    """``step_of(3) -> Decimal('0.001')``."""
    if decimals < 0:
        raise ValueError("decimals must be >= 0")
    return Decimal(1).scaleb(-decimals)


def decimals_of_step(step: Decimal) -> int:
    """
    Inverse of :func:`step_of` for the common case of power-of-ten steps.

    For non-power-of-ten steps (Kraken uses e.g. 0.00005 on some pairs) this
    returns the number of decimals needed to *express* the step, which is what
    formatting code actually wants.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    t = step.normalize()
    exponent = t.as_tuple().exponent
    return max(0, -int(exponent))


# --------------------------------------------------------------------------
# Arithmetic helpers
# --------------------------------------------------------------------------


def safe_div(numerator: Decimal, denominator: Decimal, default: Decimal = ZERO) -> Decimal:
    """Division that yields ``default`` instead of raising on a zero divisor."""
    if denominator == 0:
        return default
    with localcontext(DECIMAL_CONTEXT):
        try:
            return numerator / denominator
        except (InvalidOperation, ZeroDivisionError):
            return default


def bps(fraction: Decimal | float) -> Decimal:
    """Fraction -> basis points. ``bps(0.0001) == 1``."""
    return D(fraction) * Decimal(10_000)


def from_bps(basis_points: Decimal | float) -> Decimal:
    """Basis points -> fraction. ``from_bps(1) == Decimal('0.0001')``."""
    return D(basis_points) / Decimal(10_000)


def apply_bps(value: Decimal, basis_points: Decimal | float) -> Decimal:
    """Scale ``value`` by ``(1 + bps/10000)``."""
    with localcontext(DECIMAL_CONTEXT):
        return value * (ONE + from_bps(basis_points))


def pct(fraction: Decimal | float) -> Decimal:
    return D(fraction) * Decimal(100)


def geometric_product(values: Iterable[Decimal]) -> Decimal:
    """
    Product of a sequence, in full decimal precision.

    Used for the cycle gross return: multiply the per-leg conversion rates and
    compare to 1. Keeping this in Decimal rather than summing logs avoids the
    log/exp round trip for the final go/no-go decision -- we use logs to *find*
    cycles and exact decimals to *validate* them.
    """
    with localcontext(DECIMAL_CONTEXT):
        acc = ONE
        for v in values:
            acc *= v
        return acc


def ln(value: Decimal) -> float:
    """
    Natural log as a float.

    Deliberately returns float: the graph layer wants speed and the shortest
    path algorithms are numerically tolerant. Any decision that touches capital
    re-derives the exact value with :func:`geometric_product`.
    """
    if value <= 0:
        return float("-inf")
    return math.log(float(value))


def exp(value: float) -> Decimal:
    return D(math.exp(value))


def log_sum(values: Sequence[float]) -> float:
    """Kahan-compensated sum, because cycle weights are small and numerous."""
    total = 0.0
    compensation = 0.0
    for v in values:
        y = v - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total


def clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    if low > high:
        raise ValueError(f"clamp bounds inverted: {low} > {high}")
    return max(low, min(high, value))


def is_close(a: Decimal, b: Decimal, *, tol: Decimal = Decimal("1e-12")) -> bool:
    return abs(a - b) <= tol


def sign(value: Decimal) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


# --------------------------------------------------------------------------
# Quantization drag -- the small-account killer
# --------------------------------------------------------------------------


def quantization_drag_bps(
    notional: Decimal,
    lot_step: Decimal,
    price: Decimal,
) -> Decimal:
    """
    Expected loss, in bps of notional, from truncating an order to the lot step.

    On average a uniform truncation loses half a lot step, so the expected drag
    is ``0.5 * lot_step * price / notional``. At $100 of capital this is the
    term nobody models and everybody pays.

    Worked example -- Binance BTCUSDT, lot step 0.00001 BTC at $60,000/BTC,
    a $100 account running a 3-leg cycle so roughly $33 crosses each leg:
        half-lot value = 0.000005 * 60000 = $0.30
        on a $33 leg   = 91 bps of drag.

    Ninety-one basis points. Per leg. Against an edge measured in single digits.
    That one number is why this engine refuses to route a cycle whose legs
    cannot clear the lot grid with room to spare, and why
    :class:`CapitalInadequateError` exists. The same cycle at $10,000 of capital
    pays 0.9 bps of drag -- the drag is inversely proportional to size, which is
    the precise sense in which arbitrage has a minimum viable account.
    """
    if notional <= 0:
        return ZERO
    with localcontext(DECIMAL_CONTEXT):
        half_lot_value = (lot_step * price) / Decimal(2)
        return bps(half_lot_value / notional)
