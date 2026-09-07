"""
Fee engine.

Fees are the largest single cost in this strategy and the easiest thing to model
wrongly. Four mistakes, each of which silently biases every P&L number:

**1. Charging the fee in the wrong asset.** Spot venues charge a BUY fee in the
asset you *receive* (base) and a SELL fee in the asset you receive (quote). So a
BUY of 1 BTC at 10 bps delivers 0.999 BTC -- it does not deduct USDT. Model it
as a quote-side deduction and your BTC balance is systematically 10 bps too high
at every leg, which compounds around the cycle.

**2. Subtracting fees at the end instead of inside each leg.** Fees compound
multiplicatively: three 10 bps legs cost ``1 - 0.999^3 = 29.97`` bps, not 30. A
0.03 bps error sounds negligible until you remember the median opportunity is
2 bps.

**3. Ignoring the token discount.** Paying fees in BNB/KCS/OKB cuts them 20-25%.
On a 3-leg cycle that is 7.5 bps -- larger than most opportunities. It also
requires holding the token, whose price risk is a real, unhedged cost that
nobody accounts for.

**4. Assuming the entry tier.** Fee tiers depend on 30-day volume. A bot doing
$3,000/day reaches VIP 1 on some venues within a month. Modelling the entry tier
forever means under-trading opportunities that are genuinely profitable at your
real tier -- so the engine tracks rolling volume and re-tiers itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping, Sequence

from triangulum.core.decimal_math import D, ONE, ZERO, bps, from_bps, safe_div
from triangulum.core.ringbuffer import TimedRingBuffer
from triangulum.core.types import Asset, Fill, Liquidity, Side, Symbol

logger = logging.getLogger(__name__)

__all__ = ["FeeTier", "FeeSchedule", "FeeEngine", "FeeQuote"]


@dataclass(frozen=True, slots=True)
class FeeTier:
    """One rung of a venue's volume ladder."""

    name: str
    min_volume_30d: Decimal
    maker_bps: Decimal
    taker_bps: Decimal

    def with_discount(self, pct: Decimal) -> "FeeTier":
        factor = ONE - pct / D(100)
        return FeeTier(
            name=f"{self.name}+discount",
            min_volume_30d=self.min_volume_30d,
            maker_bps=self.maker_bps * factor,
            taker_bps=self.taker_bps * factor,
        )


@dataclass(slots=True)
class FeeSchedule:
    """A venue's full ladder plus its token-discount terms."""

    venue: str
    tiers: tuple[FeeTier, ...]
    discount_asset: str = ""
    discount_pct: Decimal = ZERO

    def tier_for(self, volume_30d: Decimal) -> FeeTier:
        chosen = self.tiers[0]
        for tier in self.tiers:
            if volume_30d >= tier.min_volume_30d:
                chosen = tier
            else:
                break
        return chosen


# Published entry ladders. Deliberately conservative: where a venue's schedule
# is ambiguous the worse number is used, so the engine never discovers that its
# real costs are higher than modelled.
DEFAULT_SCHEDULES: Mapping[str, FeeSchedule] = {
    "binance": FeeSchedule(
        venue="binance",
        tiers=(
            FeeTier("VIP 0", D("0"), D("10"), D("10")),
            FeeTier("VIP 1", D("1000000"), D("9"), D("10")),
            FeeTier("VIP 2", D("5000000"), D("8"), D("10")),
            FeeTier("VIP 3", D("20000000"), D("7"), D("9")),
        ),
        discount_asset="BNB", discount_pct=D("25"),
    ),
    "okx": FeeSchedule(
        venue="okx",
        tiers=(
            FeeTier("Lv1", D("0"), D("8"), D("10")),
            FeeTier("Lv2", D("5000000"), D("7"), D("9")),
            FeeTier("Lv3", D("10000000"), D("6"), D("8")),
        ),
        discount_asset="OKB", discount_pct=D("20"),
    ),
    "kucoin": FeeSchedule(
        venue="kucoin",
        tiers=(
            FeeTier("Lv0", D("0"), D("10"), D("10")),
            FeeTier("Lv1", D("50"), D("9"), D("10")),
        ),
        discount_asset="KCS", discount_pct=D("20"),
    ),
    "mexc": FeeSchedule(
        venue="mexc",
        tiers=(FeeTier("Standard", D("0"), D("0"), D("5")),),
    ),
    "bybit": FeeSchedule(
        venue="bybit",
        tiers=(FeeTier("Standard", D("0"), D("10"), D("10")),),
    ),
    "kraken": FeeSchedule(
        venue="kraken",
        tiers=(
            FeeTier("Starter", D("0"), D("25"), D("40")),
            FeeTier("Intermediate", D("50000"), D("20"), D("35")),
        ),
    ),
    "coinbase": FeeSchedule(
        venue="coinbase",
        tiers=(
            FeeTier("Entry", D("0"), D("40"), D("60")),
            FeeTier("Tier 2", D("1000"), D("25"), D("40")),
        ),
    ),
    "gateio": FeeSchedule(
        venue="gateio",
        tiers=(FeeTier("VIP 0", D("0"), D("9"), D("9")),),
        discount_asset="GT", discount_pct=D("25"),
    ),
    "paper": FeeSchedule(
        venue="paper",
        tiers=(FeeTier("Sim", D("0"), D("10"), D("10")),),
    ),
    "oanda": FeeSchedule(venue="oanda", tiers=(FeeTier("FX", D("0"), ZERO, ZERO),)),
    "alpaca": FeeSchedule(venue="alpaca", tiers=(FeeTier("Free", D("0"), ZERO, ZERO),)),
}


@dataclass(frozen=True, slots=True)
class FeeQuote:
    """The fee for one prospective fill."""

    amount: Decimal
    asset: Asset
    rate_bps: Decimal
    liquidity: Liquidity

    @property
    def is_rebate(self) -> bool:
        return self.rate_bps < 0


class FeeEngine:
    """
    Computes fees, tracks rolling volume, and re-tiers automatically.

    ``net_conversion_rate`` is the method the graph and planner actually depend
    on: it answers "how much of the target asset do I end up holding per unit
    spent", with the fee applied on the correct side. Everything else is
    bookkeeping around that one function.
    """

    def __init__(
        self,
        *,
        schedules: Mapping[str, FeeSchedule] | None = None,
        use_discount_asset: bool = False,
        overrides: Mapping[str, tuple[Decimal, Decimal]] | None = None,
    ) -> None:
        self._schedules = dict(schedules or DEFAULT_SCHEDULES)
        self.use_discount_asset = use_discount_asset
        self._overrides = dict(overrides or {})
        # 30 days of notional per venue. 200k entries is ample at any realistic
        # trade rate and bounds the memory.
        self._volume: dict[str, TimedRingBuffer] = {}
        self._paid: dict[str, Decimal] = {}
        self._rebates: dict[str, Decimal] = {}

    # -- configuration -----------------------------------------------------

    def override(self, venue: str, maker_bps: Decimal, taker_bps: Decimal) -> None:
        """
        Pin a venue's fees to your account's actual rates.

        Do this. The published entry tier is a guess about your account; the fee
        page in your account settings is the truth, and the gap between them is
        larger than every edge this engine looks for.
        """
        self._overrides[venue] = (maker_bps, taker_bps)

    def schedule(self, venue: str) -> FeeSchedule:
        return self._schedules.get(
            venue, FeeSchedule(venue=venue, tiers=(FeeTier("unknown", ZERO, D("10"), D("10")),))
        )

    # -- volume tracking ---------------------------------------------------

    def record_volume(self, venue: str, notional: Decimal, ts_ns: int) -> None:
        buf = self._volume.setdefault(venue, TimedRingBuffer(200_000))
        buf.push(ts_ns, float(notional))

    def volume_30d(self, venue: str, now_ns: int) -> Decimal:
        buf = self._volume.get(venue)
        if buf is None:
            return ZERO
        cutoff = now_ns - 30 * 24 * 3600 * 1_000_000_000
        return D(str(buf.sum_since(cutoff)))

    def current_tier(self, venue: str, now_ns: int = 0) -> FeeTier:
        if venue in self._overrides:
            maker, taker = self._overrides[venue]
            return FeeTier("override", ZERO, maker, taker)
        schedule = self.schedule(venue)
        tier = schedule.tier_for(self.volume_30d(venue, now_ns))
        if self.use_discount_asset and schedule.discount_pct > 0:
            tier = tier.with_discount(schedule.discount_pct)
        return tier

    def rate_bps(self, venue: str, liquidity: Liquidity, now_ns: int = 0) -> Decimal:
        tier = self.current_tier(venue, now_ns)
        return tier.maker_bps if liquidity is Liquidity.MAKER else tier.taker_bps

    # -- the core calculation ----------------------------------------------

    def quote_fee(
        self,
        symbol: Symbol,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        liquidity: Liquidity = Liquidity.TAKER,
        *,
        now_ns: int = 0,
    ) -> FeeQuote:
        """
        Fee for a prospective fill, in the asset the venue will actually charge.

        BUY  -> charged in ``symbol.base``  (you receive less base)
        SELL -> charged in ``symbol.quote`` (you receive less quote)
        """
        rate = self.rate_bps(symbol.venue, liquidity, now_ns)
        fraction = from_bps(rate)
        if side is Side.BUY:
            return FeeQuote(quantity * fraction, symbol.base, rate, liquidity)
        return FeeQuote(quantity * price * fraction, symbol.quote, rate, liquidity)

    def net_conversion_rate(
        self,
        symbol: Symbol,
        side: Side,
        execution_price: Decimal,
        liquidity: Liquidity = Liquidity.TAKER,
        *,
        now_ns: int = 0,
    ) -> Decimal:
        """
        Units of the received asset per unit of the spent asset, net of fees.

        This is the quantity the graph's edge rate is built from, and the
        multiplicative form is what makes cycle returns compose correctly:

            BUY  (spend quote, receive base):  (1/price) * (1 - fee)
            SELL (spend base, receive quote):  price     * (1 - fee)
        """
        if execution_price <= 0:
            return ZERO
        fraction = from_bps(self.rate_bps(symbol.venue, liquidity, now_ns))
        if side is Side.BUY:
            return (ONE / execution_price) * (ONE - fraction)
        return execution_price * (ONE - fraction)

    def cycle_fee_bps(
        self,
        venues: Sequence[str],
        liquidities: Sequence[Liquidity],
        *,
        now_ns: int = 0,
    ) -> Decimal:
        """
        Exact multiplicative fee cost of a whole cycle.

        ``1 - prod(1 - f_i)``, not ``sum(f_i)``. For three 10 bps legs that is
        29.97 bps rather than 30.00 -- the difference is small in absolute terms
        and not small relative to the opportunities being evaluated.
        """
        survival = ONE
        for venue, liquidity in zip(venues, liquidities):
            survival *= ONE - from_bps(self.rate_bps(venue, liquidity, now_ns))
        return bps(ONE - survival)

    # -- reconciliation ----------------------------------------------------

    def record_fill(self, fill: Fill, now_ns: int = 0) -> None:
        venue = fill.symbol.venue
        self.record_volume(venue, fill.notional, now_ns or fill.ts_ns)
        if fill.fee < 0:
            self._rebates[venue] = self._rebates.get(venue, ZERO) - fill.fee
        else:
            self._paid[venue] = self._paid.get(venue, ZERO) + fill.fee

    def check_estimate(
        self, fill: Fill, estimated: Decimal, *, tolerance: Decimal = D("0.2")
    ) -> bool:
        """
        Compare a realized fee against what we predicted.

        Persistent disagreement means the tier or discount configuration is
        wrong, and every edge calculation in the system is biased by the same
        amount. Worth an alert, not a silent accrual.
        """
        if estimated <= 0:
            return fill.fee <= 0
        deviation = abs(safe_div(fill.fee - estimated, estimated))
        if deviation > tolerance:
            logger.warning(
                "fee estimate off by %.1f%% on %s: predicted %s, charged %s -- "
                "check the configured fee tier for %s",
                float(deviation * 100), fill.symbol.canonical,
                estimated, fill.fee, fill.symbol.venue,
            )
            return False
        return True

    def stats(self, now_ns: int = 0) -> dict[str, object]:
        return {
            "tiers": {
                venue: {
                    "tier": self.current_tier(venue, now_ns).name,
                    "maker_bps": str(self.current_tier(venue, now_ns).maker_bps),
                    "taker_bps": str(self.current_tier(venue, now_ns).taker_bps),
                    "volume_30d": str(self.volume_30d(venue, now_ns)),
                }
                for venue in set(list(self._volume) + list(self._overrides))
            },
            "fees_paid": {k: str(v) for k, v in self._paid.items()},
            "rebates_earned": {k: str(v) for k, v in self._rebates.items()},
            "use_discount_asset": self.use_discount_asset,
        }
