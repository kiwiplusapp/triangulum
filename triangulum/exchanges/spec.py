"""
Venue specifications: fees, trading rules, and capital adequacy.

This is the module that decides whether your account can trade at all.

Every venue imposes three constraints that interact badly with small capital:

    ``min_notional``  the smallest order value the venue accepts, in quote
                      currency. $1 on Binance, $10 on Coinbase, ~$0.10 on
                      KuCoin. A three-leg cycle needs *every* leg to clear it.

    ``lot_step``      the quantity granularity. Truncating to it costs, on
                      average, half a step -- see
                      ``decimal_math.quantization_drag_bps``. This is the
                      dominant cost at small size and it is invisible in every
                      naive backtest.

    ``tick_size``     the price granularity. Matters for maker legs, where
                      being one tick too passive means no fill and one tick too
                      aggressive means crossing and paying taker.

The numbers below are the published defaults as of the last review and are
deliberately *pessimistic* where a venue's schedule is tiered: the engine
should never discover that its real costs are higher than modelled. They are
refreshed at startup from each venue's live instrument endpoint; these values
are the offline fallback and the basis for ``triangulum doctor``, which tells
you what your capital can actually do before you fund anything.

VERIFY THESE AGAINST YOUR OWN ACCOUNT'S FEE PAGE. Fee tiers depend on 30-day
volume and on holding the venue token, and the difference between the top and
bottom tier is larger than every edge this engine will ever find.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Mapping

from triangulum.core.decimal_math import D, ZERO, bps, quantization_drag_bps
from triangulum.core.errors import CapitalInadequateError

__all__ = [
    "SymbolSpec",
    "VenueSpec",
    "VENUE_SPECS",
    "get_venue_spec",
    "CapitalAdequacyReport",
    "assess_capital_adequacy",
]


@dataclass(slots=True)
class SymbolSpec:
    """Trading rules for one instrument."""

    venue_symbol: str
    base: str
    quote: str

    tick_size: Decimal = D("0.01")
    lot_step: Decimal = D("0.00000001")
    min_quantity: Decimal = ZERO
    max_quantity: Decimal = D("1e12")
    min_notional: Decimal = D("1")
    max_notional: Decimal = D("1e12")

    # Per-symbol fee override; falls back to the venue default when None.
    maker_fee_bps: Decimal | None = None
    taker_fee_bps: Decimal | None = None

    base_precision: int = 8
    quote_precision: int = 8
    active: bool = True
    # Some venues bill spot fees in the base asset on a BUY (you receive
    # slightly less base) and in the quote on a SELL. Getting this backwards
    # produces a systematic, direction-dependent error in the cycle rate.
    fee_in_base_on_buy: bool = True

    def drag_bps_at(self, notional: Decimal, price: Decimal) -> Decimal:
        return quantization_drag_bps(notional, self.lot_step, price)

    def clears_min_notional(self, notional: Decimal, *, buffer: Decimal = D("1.15")) -> bool:
        return notional >= self.min_notional * buffer


@dataclass(slots=True)
class VenueSpec:
    """Venue-level defaults and capabilities."""

    name: str
    display_name: str

    default_maker_bps: Decimal
    default_taker_bps: Decimal

    # Token discount: holding/paying with the venue token cuts fees.
    discount_asset: str = ""
    discount_pct: Decimal = ZERO

    # Typical smallest min-notional across the venue's major pairs, in USD.
    typical_min_notional_usd: Decimal = D("10")

    supports_post_only: bool = True
    supports_fok: bool = True
    supports_ioc: bool = True
    # Some venues expose an atomic multi-leg or OCO primitive; none currently
    # offer true atomic triangular execution, which is the whole reason this
    # engine needs an unwinder.
    supports_atomic_multi_leg: bool = False

    maker_rebate: bool = False          # negative maker fee at some tier
    asset_class: str = "crypto"
    fiat_onramp: bool = False

    rate_limit_rps: float = 10.0
    order_rate_limit_rps: float = 5.0
    ws_book_channels: int = 200

    notes: str = ""
    fee_schedule_url: str = ""

    def effective_maker_bps(self, *, use_discount: bool = False) -> Decimal:
        fee = self.default_maker_bps
        if use_discount and self.discount_pct > 0:
            fee = fee * (D(1) - self.discount_pct / D(100))
        return fee

    def effective_taker_bps(self, *, use_discount: bool = False) -> Decimal:
        fee = self.default_taker_bps
        if use_discount and self.discount_pct > 0:
            fee = fee * (D(1) - self.discount_pct / D(100))
        return fee

    def round_trip_bps(self, legs: int = 3, *, maker_legs: int = 0,
                       use_discount: bool = False) -> Decimal:
        """
        Total fee cost of an N-leg cycle.

        This single number is the hurdle every opportunity must clear before it
        is even worth evaluating. On a 3-leg all-taker cycle at 10 bps/leg it is
        30 bps -- and the median triangular dislocation on a liquid venue is
        roughly one tenth of that.
        """
        maker_legs = max(0, min(maker_legs, legs))
        taker_legs = legs - maker_legs
        return (
            self.effective_maker_bps(use_discount=use_discount) * maker_legs
            + self.effective_taker_bps(use_discount=use_discount) * taker_legs
        )


# --------------------------------------------------------------------------
# The venues
# --------------------------------------------------------------------------

VENUE_SPECS: Mapping[str, VenueSpec] = {
    "binance": VenueSpec(
        name="binance",
        display_name="Binance Spot",
        default_maker_bps=D("10"),
        default_taker_bps=D("10"),
        discount_asset="BNB",
        discount_pct=D("25"),
        typical_min_notional_usd=D("5"),
        maker_rebate=False,
        rate_limit_rps=20.0,
        order_rate_limit_rps=10.0,
        ws_book_channels=1024,
        fee_schedule_url="https://www.binance.com/en/fee/schedule",
        notes=(
            "Deepest triangular graph in crypto: ~1400 spot pairs across USDT, "
            "USDC, FDUSD, BTC, ETH and BNB quotes, which is where the cycle "
            "count comes from. 10/10 bps base, 7.5 bps taker paying in BNB. "
            "$5 min notional is the lowest of the majors. The realistic default "
            "choice for a small account -- not because the edges are large, but "
            "because it is the only venue where a $100 account clears min "
            "notional on every leg of a 3-cycle with room to spare."
        ),
    ),
    "kucoin": VenueSpec(
        name="kucoin",
        display_name="KuCoin Spot",
        default_maker_bps=D("10"),
        default_taker_bps=D("10"),
        discount_asset="KCS",
        discount_pct=D("20"),
        typical_min_notional_usd=D("0.1"),
        rate_limit_rps=10.0,
        order_rate_limit_rps=5.0,
        fee_schedule_url="https://www.kucoin.com/vip/level",
        notes=(
            "Min notional near $0.10 -- by far the friendliest to tiny capital, "
            "and the reason it is the recommended second venue. Long tail of "
            "illiquid altcoin pairs means more apparent cycles, most of which "
            "are traps: the edge is real but the depth behind it is not."
        ),
    ),
    "okx": VenueSpec(
        name="okx",
        display_name="OKX Spot",
        default_maker_bps=D("8"),
        default_taker_bps=D("10"),
        discount_asset="OKB",
        discount_pct=D("20"),
        typical_min_notional_usd=D("1"),
        rate_limit_rps=20.0,
        order_rate_limit_rps=10.0,
        fee_schedule_url="https://www.okx.com/fees",
        notes=(
            "8 bps maker is materially better than Binance's 10, which matters "
            "a great deal for maker-leg execution: an MTT cycle here costs "
            "28 bps against Binance's 30. Publishes an order-book checksum, so "
            "book integrity is verifiable rather than assumed."
        ),
    ),
    "bybit": VenueSpec(
        name="bybit",
        display_name="Bybit Spot",
        default_maker_bps=D("10"),
        default_taker_bps=D("10"),
        typical_min_notional_usd=D("1"),
        rate_limit_rps=20.0,
        order_rate_limit_rps=10.0,
        fee_schedule_url="https://www.bybit.com/en/help-center/article/Trading-Fee-Structure",
        notes="Fast matching, thinner spot books than Binance. Good cross-venue leg.",
    ),
    "kraken": VenueSpec(
        name="kraken",
        display_name="Kraken Spot",
        default_maker_bps=D("25"),
        default_taker_bps=D("40"),
        typical_min_notional_usd=D("10"),
        fiat_onramp=True,
        rate_limit_rps=1.0,
        order_rate_limit_rps=1.0,
        fee_schedule_url="https://www.kraken.com/features/fee-schedule",
        notes=(
            "40 bps taker means a 3-leg all-taker cycle costs 120 bps. No "
            "triangular opportunity of that size exists on a venue this liquid. "
            "Kraken is here for cross-venue arbitrage against its deep fiat "
            "books and as a fiat on-ramp -- NOT for single-venue triangles. "
            "Its aggressive rate limits also make it a poor primary."
        ),
    ),
    "coinbase": VenueSpec(
        name="coinbase",
        display_name="Coinbase Advanced Trade",
        default_maker_bps=D("40"),
        default_taker_bps=D("60"),
        typical_min_notional_usd=D("1"),
        fiat_onramp=True,
        rate_limit_rps=10.0,
        order_rate_limit_rps=5.0,
        fee_schedule_url="https://www.coinbase.com/advanced-fees",
        notes=(
            "60 bps taker at the entry tier: 180 bps for a 3-leg cycle. "
            "Structurally unable to support triangular arbitrage at retail "
            "volume. Included as a price reference and cross-venue quote leg."
        ),
    ),
    "mexc": VenueSpec(
        name="mexc",
        display_name="MEXC Spot",
        default_maker_bps=D("0"),
        default_taker_bps=D("5"),
        typical_min_notional_usd=D("1"),
        maker_rebate=False,
        rate_limit_rps=20.0,
        order_rate_limit_rps=10.0,
        fee_schedule_url="https://www.mexc.com/fee",
        notes=(
            "0 bps maker / 5 bps taker on many spot pairs is the single most "
            "favourable retail fee structure in crypto, and it changes the "
            "arithmetic completely: an MTT cycle costs 10 bps here versus 30 on "
            "Binance. That is the difference between an empty opportunity set "
            "and a populated one. Weigh against materially thinner books, "
            "occasional withdrawal friction, and a shorter operating history -- "
            "venue risk on a small account is concentrated risk."
        ),
    ),
    "gateio": VenueSpec(
        name="gateio",
        display_name="Gate.io Spot",
        default_maker_bps=D("9"),
        default_taker_bps=D("9"),
        discount_asset="GT",
        discount_pct=D("25"),
        typical_min_notional_usd=D("1"),
        rate_limit_rps=15.0,
        notes="Wide altcoin coverage, decent fees with GT. Similar profile to KuCoin.",
    ),
    # ---- non-crypto ----
    "oanda": VenueSpec(
        name="oanda",
        display_name="OANDA (FX)",
        default_maker_bps=D("0"),
        default_taker_bps=D("0"),
        typical_min_notional_usd=D("1"),
        asset_class="fx",
        supports_post_only=False,
        maker_rebate=False,
        rate_limit_rps=5.0,
        notes=(
            "Zero commission, but the cost is entirely in the spread (~0.8-1.5 "
            "pips on EUR/USD = 8-15 bps round trip) and OANDA quotes crosses "
            "DERIVED from the majors. EUR/USD x USD/JPY x JPY/EUR is therefore "
            "closed BY CONSTRUCTION -- the triangle cannot pay because the third "
            "price is computed from the first two. Retail FX triangular "
            "arbitrage has been dead since roughly 2010. This adapter exists for "
            "statistical/mean-reversion strategies and as an honest "
            "demonstration of why the non-crypto request cannot be satisfied the "
            "way it was framed."
        ),
    ),
    "alpaca": VenueSpec(
        name="alpaca",
        display_name="Alpaca (US Equities)",
        default_maker_bps=D("0"),
        default_taker_bps=D("0"),
        typical_min_notional_usd=D("1"),
        asset_class="equity",
        supports_post_only=True,
        rate_limit_rps=3.0,
        notes=(
            "Commission-free US equities. There is no triangular structure in "
            "single-listed equities -- there is no currency cycle to close. "
            "Genuine equity arbitrage (ADR/ordinary, index/constituent, "
            "merger spreads) needs market-data and borrow infrastructure that "
            "costs more per month than this account holds. Wired up for the "
            "statistical strategy and for paper experimentation only."
        ),
    ),
    "paper": VenueSpec(
        name="paper",
        display_name="Simulated Venue",
        default_maker_bps=D("10"),
        default_taker_bps=D("10"),
        typical_min_notional_usd=D("5"),
        notes="Local matching engine used for paper trading and backtests.",
    ),
}


def get_venue_spec(name: str) -> VenueSpec:
    spec = VENUE_SPECS.get(name.lower())
    if spec is None:
        raise KeyError(
            f"unknown venue {name!r}; known: {sorted(VENUE_SPECS)}"
        )
    return spec


# --------------------------------------------------------------------------
# Capital adequacy -- run this before funding anything
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CapitalAdequacyReport:
    """
    Whether a given amount of capital can profitably trade a given venue.

    The verdict combines three costs that all scale badly with small size:
    fees (constant in bps), quantization drag (inversely proportional to
    notional), and min-notional feasibility (a hard cliff).
    """

    venue: str
    capital: Decimal
    cycle_legs: int
    notional_per_leg: Decimal
    fee_bps: Decimal
    drag_bps: Decimal
    total_cost_bps: Decimal
    min_notional_ok: bool
    required_edge_bps: Decimal
    feasible: bool
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "FEASIBLE" if self.feasible else "NOT FEASIBLE"
        return (
            f"{self.venue}: {verdict} at {self.capital} over {self.cycle_legs} legs "
            f"({self.notional_per_leg:.2f}/leg) -- fees {self.fee_bps:.1f}bps + "
            f"drag {self.drag_bps:.1f}bps = {self.total_cost_bps:.1f}bps required edge"
        )


def assess_capital_adequacy(
    venue: str,
    capital: Decimal,
    *,
    cycle_legs: int = 3,
    maker_legs: int = 0,
    use_discount: bool = False,
    representative_price: Decimal = D("60000"),
    representative_lot_step: Decimal = D("0.00001"),
    capital_fraction: Decimal = D("0.95"),
    raise_on_infeasible: bool = False,
) -> CapitalAdequacyReport:
    """
    Compute the edge a cycle must show before it is worth trading, at this size.

    The output is the honest answer to "can $100 do this?" -- expressed not as
    an opinion but as a number of basis points that the market must hand you
    before you break even.
    """
    spec = get_venue_spec(venue)
    warnings: list[str] = []

    # In a cycle, roughly the full working capital crosses each leg in turn.
    notional_per_leg = capital * capital_fraction

    fee_bps = spec.round_trip_bps(
        cycle_legs, maker_legs=maker_legs, use_discount=use_discount
    )

    # Drag is per leg and additive across the cycle.
    per_leg_drag = quantization_drag_bps(
        notional_per_leg, representative_lot_step, representative_price
    )
    drag_bps = per_leg_drag * cycle_legs

    min_notional_ok = notional_per_leg >= spec.typical_min_notional_usd * D("1.15")
    if not min_notional_ok:
        warnings.append(
            f"notional per leg ({notional_per_leg:.2f}) is below "
            f"{spec.typical_min_notional_usd} x 1.15 buffer: most orders will be "
            f"rejected outright"
        )

    total_cost_bps = fee_bps + drag_bps
    # Require the edge to exceed cost with a margin -- an edge exactly equal to
    # cost is a coin flip that pays nothing when it lands.
    required_edge_bps = total_cost_bps * D("1.3")

    if drag_bps > fee_bps:
        warnings.append(
            f"quantization drag ({drag_bps:.1f} bps) exceeds total fees "
            f"({fee_bps:.1f} bps): this account is too small for this "
            f"instrument's lot grid, and the dominant cost is rounding, not "
            f"trading"
        )
    if required_edge_bps > D("50"):
        warnings.append(
            f"required edge of {required_edge_bps:.0f} bps is far outside the "
            f"observed distribution of triangular dislocations (typically "
            f"1-8 bps); expect approximately zero qualifying opportunities"
        )
    if spec.default_taker_bps >= D("25") and maker_legs == 0:
        warnings.append(
            f"{spec.display_name} charges {spec.default_taker_bps} bps taker; "
            f"all-taker cycles here are structurally unprofitable"
        )

    feasible = min_notional_ok and required_edge_bps <= D("35")

    report = CapitalAdequacyReport(
        venue=venue,
        capital=capital,
        cycle_legs=cycle_legs,
        notional_per_leg=notional_per_leg,
        fee_bps=fee_bps,
        drag_bps=drag_bps,
        total_cost_bps=total_cost_bps,
        min_notional_ok=min_notional_ok,
        required_edge_bps=required_edge_bps,
        feasible=feasible,
        warnings=warnings,
    )

    if raise_on_infeasible and not feasible:
        raise CapitalInadequateError(
            report.summary(), venue=venue, capital=str(capital), warnings=warnings
        )
    return report


# --------------------------------------------------------------------------
# Lot-value screening -- the decisive constraint for a small account
# --------------------------------------------------------------------------
#
# The finding that shapes this entire engine's instrument selection:
#
#   drag_bps_per_leg = (lot_step * price / 2) / notional * 10000
#
# The numerator is the *value of one lot* -- and it varies by two orders of
# magnitude across instruments on the SAME venue with the SAME fee schedule:
#
#   Binance spot, $95 per leg, 3-leg cycle
#   ------------------------------------------------------------------
#   BTCUSDT    step 1e-5   @ $60000   lot = $0.600   ->  94.7 bps/cycle
#   ETHUSDT    step 1e-4   @  $3000   lot = $0.300   ->  47.4 bps/cycle
#   SOLUSDT    step 1e-3   @   $150   lot = $0.150   ->  23.7 bps/cycle
#   DOGEUSDT   step 1      @  $0.10   lot = $0.100   ->  15.8 bps/cycle
#   XRPUSDT    step 0.1    @  $0.55   lot = $0.055   ->   8.7 bps/cycle
#   ADAUSDT    step 0.1    @  $0.45   lot = $0.045   ->   7.1 bps/cycle
#   TRXUSDT    step 0.1    @  $0.12   lot = $0.012   ->   1.9 bps/cycle
#
# A fifty-fold difference in cost, driven entirely by which instruments the
# cycle passes through, on a dimension that no published treatment of
# triangular arbitrage discusses because everyone writing them assumes
# institutional size where the term vanishes.
#
# Consequence for a $100 account: BTC- and ETH-anchored triangles are not
# merely unprofitable, they are unprofitable by an order of magnitude, and no
# fee tier or execution cleverness recovers it. Cycles must be routed through
# low-lot-value instruments. ``max_lot_value_for_budget`` turns a drag budget
# into the hard screening threshold the strategy layer applies to every edge.


def lot_value(lot_step: Decimal, price: Decimal) -> Decimal:
    """Quote-currency value of one lot. The numerator of quantization drag."""
    return lot_step * price


def max_lot_value_for_budget(
    notional_per_leg: Decimal,
    drag_budget_bps: Decimal,
    *,
    legs: int = 3,
) -> Decimal:
    """
    Largest tolerable lot value given a per-cycle drag budget.

    Inverting the drag formula: with ``legs`` legs each contributing
    ``(lot_value / 2) / notional`` of drag, the budget caps lot value at

        lot_value <= 2 * notional * (budget_bps / 10000) / legs

    Worked: $95/leg, a 5 bps total drag budget, 3 legs
        -> 2 * 95 * 0.0005 / 3 = $0.0317 maximum lot value.
    Which admits XRP, ADA, TRX, MATIC and excludes BTC, ETH, SOL, DOGE.
    """
    if notional_per_leg <= 0 or legs <= 0:
        return ZERO
    return (D(2) * notional_per_leg * (drag_budget_bps / D(10_000))) / D(legs)


def screen_symbol_for_capital(
    spec: SymbolSpec,
    reference_price: Decimal,
    notional_per_leg: Decimal,
    *,
    drag_budget_bps: Decimal = D("5"),
    legs: int = 3,
) -> tuple[bool, Decimal, str]:
    """
    Decide whether an instrument is usable at this size.

    Returns ``(usable, drag_bps_per_leg, reason)``. Applied by the graph builder
    to every candidate edge, so an unusable instrument never even becomes a
    node in the cycle search -- the alternative is discovering the problem after
    the planner has spent CPU on a cycle it can never route.
    """
    lv = lot_value(spec.lot_step, reference_price)
    cap = max_lot_value_for_budget(notional_per_leg, drag_budget_bps, legs=legs)
    drag = quantization_drag_bps(notional_per_leg, spec.lot_step, reference_price)

    if not spec.active:
        return False, drag, "instrument inactive"
    if not spec.clears_min_notional(notional_per_leg):
        return (
            False, drag,
            f"notional {notional_per_leg:.2f} below min {spec.min_notional} +buffer",
        )
    if lv > cap:
        return (
            False, drag,
            f"lot value {lv:.4f} exceeds {cap:.4f} budget "
            f"({drag:.1f} bps/leg drag at this size)",
        )
    return True, drag, "ok"
