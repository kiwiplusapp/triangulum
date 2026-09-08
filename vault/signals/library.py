"""
The signal library.

Twenty named signals across ten families. Each is a pure function from the
series universe to a :class:`SignalReading`, registered by decorator so the
catalogue is the code rather than a list that drifts out of sync with it.

## Sign convention

**Positive strength is always risk-on.** Every signal is written so that +1
means "conditions favour risk assets" and -1 means the opposite, whatever the
underlying series does. Widening credit spreads produce a NEGATIVE reading
even though the spread itself went up. Getting this wrong in one signal out of
twenty is close to undetectable by eye and quietly poisons every composite and
every model trained on it, so each function states its inversion explicitly
and ``test_vault_signals.py`` asserts the direction of every one.

## On scale constants

The ``scale`` passed to :func:`squash` is the move size that should read as
clearly significant for that particular series. These are not tuned to make
backtests look good -- they are set from the rough historical dispersion of
each series (a 50bp move in the 10y is large; a 50bp move in HY OAS is
noise) and then left alone. Tuning them against outcomes would be fitting the
signal definitions to the sample, which is the thing the whole calibration
apparatus exists to detect.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Mapping

from vault.data.series import Series
from vault.signals.types import (
    SignalFamily,
    SignalReading,
    confidence_from_staleness,
    squash,
)

logger = logging.getLogger(__name__)

__all__ = ["SIGNALS", "signal", "evaluate_all", "SignalSpec"]


class SignalSpec:
    """A registered signal: metadata plus the function that computes it."""

    __slots__ = ("key", "label", "family", "requires", "fn", "rationale")

    def __init__(self, key: str, label: str, family: str,
                 requires: tuple[str, ...], fn: Callable, rationale: str) -> None:
        self.key = key
        self.label = label
        self.family = family
        self.requires = requires
        self.fn = fn
        self.rationale = rationale

    def evaluate(self, series: Mapping[str, Series]) -> SignalReading:
        missing = [key for key in self.requires if not series.get(key)]
        if missing:
            return SignalReading.unavailable(
                self.key, self.label, self.family,
                f"missing input series: {', '.join(missing)}", self.requires,
            )
        try:
            reading = self.fn(series)
        except Exception as exc:                       # pragma: no cover
            logger.exception("signal %s raised", self.key)
            return SignalReading.unavailable(
                self.key, self.label, self.family,
                f"{type(exc).__name__}: {exc}", self.requires,
            )
        if reading is None:
            return SignalReading.unavailable(
                self.key, self.label, self.family,
                "insufficient history to compute", self.requires,
            )
        return reading


SIGNALS: dict[str, SignalSpec] = {}


def signal(key: str, label: str, family: str, requires: tuple[str, ...],
           rationale: str):
    """Register a signal. The decorated function returns a SignalReading."""

    def decorate(fn: Callable) -> Callable:
        if key in SIGNALS:
            raise ValueError(f"duplicate signal key {key!r}")
        SIGNALS[key] = SignalSpec(key, label, family, requires, fn, rationale)
        return fn

    return decorate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reading(spec_key: str, *, strength: float, raw: float | None,
             inputs: tuple[str, ...], staleness: int, note: str,
             zscore: float | None = None) -> SignalReading:
    spec = SIGNALS[spec_key]
    confidence = confidence_from_staleness(staleness)
    # Three states, kept distinct: computed and usable; computed but too stale
    # to weight; and not computable at all (SignalReading.unavailable). The
    # middle one is the interesting one -- the signal HAS a reading, it is
    # simply describing a world 69 days old -- and collapsing it into "missing"
    # would hide the difference between an input the system lacks and an input
    # the system has but should not act on.
    return SignalReading(
        key=spec.key, label=spec.label, family=spec.family,
        strength=max(-1.0, min(1.0, strength)), raw=raw, zscore=zscore,
        confidence=confidence, inputs=inputs, staleness_days=staleness,
        usable=confidence > 0.0, note=note,
        unavailable_reason=(
            "" if confidence > 0.0 else
            f"computed, but its newest input is {staleness} days old"
        ),
    )


def _annualised_3m(index: Series) -> float | None:
    """3-month change on a monthly index, annualised. The inflation-run rate."""
    change = index.pct_change(3)
    if change is None:
        return None
    return ((1 + change / 100) ** 4 - 1) * 100


# ---------------------------------------------------------------------------
# curve
# ---------------------------------------------------------------------------


@signal("curve_slope", "Yield curve slope (10y-3m)", SignalFamily.CURVE,
        ("T10Y3M",),
        "Inversion has led every US recession since 1955, at 6-18 months. It "
        "is a slow signal and is weighted as one: it says what the next year "
        "looks like, not the next week.")
def _curve_slope(series: Mapping[str, Series]) -> SignalReading | None:
    curve = series["T10Y3M"]
    level = curve.last
    if level is None:
        return None
    # Inverted curve is risk-off, so the sign passes through unchanged: a
    # negative spread produces a negative reading.
    return _reading(
        "curve_slope", strength=squash(level, 1.2), raw=level,
        inputs=("T10Y3M",), staleness=curve.staleness_days,
        zscore=curve.zscore(504),
        note=(f"10y-3m at {level:+.2f}%. "
              + ("Inverted." if level < 0 else "Positively sloped.")),
    )


@signal("curve_impulse", "Curve steepening impulse", SignalFamily.CURVE,
        ("T10Y2Y",),
        "The DIRECTION of the curve matters separately from its level. "
        "Re-steepening from deep inversion (bull or bear) has historically "
        "marked the transition from 'recession expected' to 'recession here'.")
def _curve_impulse(series: Mapping[str, Series]) -> SignalReading | None:
    curve = series["T10Y2Y"]
    change = curve.change(63)          # ~3 months of business days
    level = curve.last
    if change is None or level is None:
        return None
    # Steepening from inversion is a late-cycle warning, not a green light:
    # the sign is INVERTED when the curve is still inverted, and passes
    # through when it is not. This is the one signal whose sign is
    # state-dependent, and it is deliberate.
    inverted = level < 0
    strength = squash(-change if inverted else change, 0.35)
    return _reading(
        "curve_impulse", strength=strength, raw=change,
        inputs=("T10Y2Y",), staleness=curve.staleness_days,
        note=(f"10y-2y moved {change:+.2f}% over 3 months from a "
              f"{'inverted' if inverted else 'positive'} base."),
    )


@signal("carry", "Cash versus duration", SignalFamily.CURVE,
        ("DFF", "DGS10"),
        "When cash out-yields the 10y, holding duration is a bet on cuts "
        "rather than a carry trade. Deeply negative carry is a risk-off "
        "condition for anything long-duration, equities included.")
def _carry(series: Mapping[str, Series]) -> SignalReading | None:
    cash, ten = series["DFF"].last, series["DGS10"].last
    if cash is None or ten is None:
        return None
    spread = ten - cash
    staleness = max(series["DFF"].staleness_days, series["DGS10"].staleness_days)
    return _reading(
        "carry", strength=squash(spread, 1.0), raw=spread,
        inputs=("DFF", "DGS10"), staleness=staleness,
        note=f"10y yields {spread:+.2f}% over effective fed funds.",
    )


# ---------------------------------------------------------------------------
# inflation
# ---------------------------------------------------------------------------


@signal("inflation_impulse", "Core inflation run-rate vs trend",
        SignalFamily.INFLATION, ("PCEPILFE",),
        "The 3-month annualised rate against the 12-month rate is the "
        "standard way to see a turn before the year-over-year number shows "
        "it, because YoY carries eleven months of history it cannot shed.")
def _inflation_impulse(series: Mapping[str, Series]) -> SignalReading | None:
    pce = series["PCEPILFE"]
    fast, slow = _annualised_3m(pce), pce.year_over_year()
    if fast is None or slow is None:
        return None
    acceleration = fast - slow
    # Accelerating inflation is risk-off: it removes the cuts the market is
    # priced for. Hence the inversion.
    return _reading(
        "inflation_impulse", strength=squash(-acceleration, 0.8),
        raw=acceleration, inputs=("PCEPILFE",), staleness=pce.staleness_days,
        note=(f"Core PCE running {fast:.2f}% annualised over 3m against "
              f"{slow:.2f}% YoY ({acceleration:+.2f}pp)."),
    )


@signal("breakeven_impulse", "Inflation expectations impulse",
        SignalFamily.INFLATION, ("T10YIE",),
        "Breakevens are the market's own inflation forecast, updated "
        "continuously. A sharp move is a repricing of the policy path.")
def _breakeven_impulse(series: Mapping[str, Series]) -> SignalReading | None:
    breakeven = series["T10YIE"]
    change = breakeven.change(21)
    if change is None:
        return None
    return _reading(
        "breakeven_impulse", strength=squash(-change, 0.20), raw=change,
        inputs=("T10YIE",), staleness=breakeven.staleness_days,
        zscore=breakeven.zscore(252),
        note=f"10y breakevens moved {change:+.2f}% over a month.",
    )


@signal("real_yield_impulse", "Real yield impulse", SignalFamily.INFLATION,
        ("DFII10",),
        "The real yield is the discount rate for every long-duration cash "
        "flow. Rising real yields compress multiples mechanically, "
        "independently of growth.")
def _real_yield_impulse(series: Mapping[str, Series]) -> SignalReading | None:
    real = series["DFII10"]
    change = real.change(21)
    if change is None:
        return None
    return _reading(
        "real_yield_impulse", strength=squash(-change, 0.30), raw=change,
        inputs=("DFII10",), staleness=real.staleness_days,
        zscore=real.zscore(504),
        note=f"10y real yield moved {change:+.2f}% over a month.",
    )


# ---------------------------------------------------------------------------
# credit
# ---------------------------------------------------------------------------


@signal("credit_stress", "High-yield spread level", SignalFamily.CREDIT,
        ("BAMLH0A0HYM2",),
        "Credit leads equity at turns. HY spreads widen before earnings "
        "estimates fall, because lenders reprice default risk before "
        "analysts reprice growth.")
def _credit_stress(series: Mapping[str, Series]) -> SignalReading | None:
    spread = series["BAMLH0A0HYM2"]
    zscore = spread.zscore(504)
    level = spread.last
    if zscore is None or level is None:
        return None
    # Wide spreads are risk-off: invert.
    return _reading(
        "credit_stress", strength=squash(-zscore, 1.5), raw=level,
        inputs=("BAMLH0A0HYM2",), staleness=spread.staleness_days,
        zscore=zscore,
        note=f"HY OAS at {level:.2f}%, {zscore:+.1f} sigma against 2 years.",
    )


@signal("credit_impulse", "Credit spread impulse", SignalFamily.CREDIT,
        ("BAMLH0A0HYM2",),
        "The rate of change matters more than the level for timing. Spreads "
        "can sit wide for months; they widen FAST only around events.")
def _credit_impulse(series: Mapping[str, Series]) -> SignalReading | None:
    spread = series["BAMLH0A0HYM2"]
    change = spread.change(21)
    if change is None:
        return None
    return _reading(
        "credit_impulse", strength=squash(-change, 0.60), raw=change,
        inputs=("BAMLH0A0HYM2",), staleness=spread.staleness_days,
        note=f"HY OAS moved {change:+.2f}% over a month.",
    )


@signal("quality_spread", "High-yield versus investment-grade",
        SignalFamily.CREDIT, ("BAMLH0A0HYM2", "BAMLC0A0CM"),
        "The HY/IG ratio strips out the common duration and rates component "
        "and leaves the part that is specifically about default risk, which "
        "is the part that matters for equity.")
def _quality_spread(series: Mapping[str, Series]) -> SignalReading | None:
    high_yield, investment = series["BAMLH0A0HYM2"], series["BAMLC0A0CM"]
    hy, ig = high_yield.last, investment.last
    if hy is None or ig is None or ig <= 0:
        return None
    ratio = hy / ig
    staleness = max(high_yield.staleness_days, investment.staleness_days)
    # Historical HY/IG sits around 3.5-4.5x; above that is stress.
    return _reading(
        "quality_spread", strength=squash(-(ratio - 4.0), 1.2), raw=ratio,
        inputs=("BAMLH0A0HYM2", "BAMLC0A0CM"), staleness=staleness,
        note=f"HY trades at {ratio:.2f}x IG.",
    )


# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------


@signal("equity_momentum_fast", "Equity momentum (20d vs 100d)",
        SignalFamily.MOMENTUM, ("SP500",),
        "Time-series momentum is among the most replicated anomalies in the "
        "literature and among the easiest to over-trade. It is included at "
        "two horizons so the model can learn which one, if either, is "
        "informative in this regime.")
def _equity_momentum_fast(series: Mapping[str, Series]) -> SignalReading | None:
    index = series["SP500"]
    momentum = index.momentum(20, 100)
    if momentum is None:
        return None
    return _reading(
        "equity_momentum_fast", strength=squash(momentum, 3.0), raw=momentum,
        inputs=("SP500",), staleness=index.staleness_days,
        note=f"20d mean sits {momentum:+.2f}% against the 100d mean.",
    )


@signal("equity_momentum_slow", "Equity momentum (100d vs 200d)",
        SignalFamily.MOMENTUM, ("SP500",),
        "The slow leg. Trend followers live here; it whipsaws far less and "
        "turns far later than the fast leg.")
def _equity_momentum_slow(series: Mapping[str, Series]) -> SignalReading | None:
    index = series["SP500"]
    momentum = index.momentum(100, 200)
    if momentum is None:
        return None
    return _reading(
        "equity_momentum_slow", strength=squash(momentum, 4.0), raw=momentum,
        inputs=("SP500",), staleness=index.staleness_days,
        note=f"100d mean sits {momentum:+.2f}% against the 200d mean.",
    )


@signal("equity_mean_reversion", "Short-horizon stretch", SignalFamily.MOMENTUM,
        ("SP500",),
        "Deliberately OPPOSED to the momentum signals at short horizons. "
        "Both cannot be right at once, and forcing the model to resolve the "
        "conflict from data is more honest than picking a side in advance.")
def _equity_mean_reversion(series: Mapping[str, Series]) -> SignalReading | None:
    index = series["SP500"]
    zscore = index.zscore(20)
    if zscore is None:
        return None
    return _reading(
        "equity_mean_reversion", strength=squash(-zscore, 2.0), raw=zscore,
        inputs=("SP500",), staleness=index.staleness_days, zscore=zscore,
        note=f"S&P sits {zscore:+.1f} sigma against its 20-day mean.",
    )


@signal("breadth_proxy", "Nasdaq versus S&P", SignalFamily.MOMENTUM,
        ("NASDAQ100", "SP500"),
        "A crude breadth proxy: when the concentrated index leads the broad "
        "one hard, the rally is narrow. Narrow rallies are fragile.")
def _breadth_proxy(series: Mapping[str, Series]) -> SignalReading | None:
    nasdaq, sp500 = series["NASDAQ100"], series["SP500"]
    ndx, spx = nasdaq.pct_change(63), sp500.pct_change(63)
    if ndx is None or spx is None:
        return None
    divergence = ndx - spx
    staleness = max(nasdaq.staleness_days, sp500.staleness_days)
    # Extreme leadership in EITHER direction is the warning; the sign here is
    # the magnitude of the gap, penalised.
    return _reading(
        "breadth_proxy", strength=squash(-abs(divergence) + 3.0, 4.0),
        raw=divergence, inputs=("NASDAQ100", "SP500"), staleness=staleness,
        note=f"Nasdaq has out/under-performed by {divergence:+.2f}% over 3 months.",
    )


@signal("risk_appetite", "Speculative risk appetite", SignalFamily.MOMENTUM,
        ("BTCUSD",),
        "Crypto is the cleanest available read on marginal speculative "
        "appetite: no earnings, no carry, pure risk preference. It is used "
        "as a SENTIMENT input, not as a tradeable asset.")
def _risk_appetite(series: Mapping[str, Series]) -> SignalReading | None:
    btc = series["BTCUSD"]
    momentum = btc.momentum(20, 90)
    if momentum is None:
        return None
    return _reading(
        "risk_appetite", strength=squash(momentum, 12.0), raw=momentum,
        inputs=("BTCUSD",), staleness=btc.staleness_days,
        note=f"Bitcoin 20d/90d momentum {momentum:+.2f}%.",
    )


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------


@signal("vol_regime", "Volatility regime", SignalFamily.VOLATILITY,
        ("VIXCLS",),
        "The level of implied vol, as a percentile rather than an absolute. "
        "A VIX of 20 means something different after a year at 12 than after "
        "a year at 30.")
def _vol_regime(series: Mapping[str, Series]) -> SignalReading | None:
    vix = series["VIXCLS"]
    percentile = vix.percentile_rank(504)
    level = vix.last
    if percentile is None or level is None:
        return None
    # High vol percentile is risk-off. Centre on the median and scale so the
    # extremes reach roughly +/-0.8.
    return _reading(
        "vol_regime", strength=squash(-(percentile - 0.5), 0.35), raw=level,
        inputs=("VIXCLS",), staleness=vix.staleness_days,
        note=f"VIX at {level:.1f}, {percentile:.0%} of its 2-year range.",
    )


@signal("vol_of_vol", "Realised volatility impulse", SignalFamily.VOLATILITY,
        ("SP500",),
        "Realised vol rising fast is a regime change in progress. It leads "
        "the drawdown; the drawdown does not lead it.")
def _vol_of_vol(series: Mapping[str, Series]) -> SignalReading | None:
    index = series["SP500"]
    fast = index.realized_volatility(10)
    slow = index.realized_volatility(60)
    if fast is None or slow is None or slow <= 0:
        return None
    ratio = fast / slow
    return _reading(
        "vol_of_vol", strength=squash(-(ratio - 1.0), 0.45), raw=ratio,
        inputs=("SP500",), staleness=index.staleness_days,
        note=f"10-day realised vol at {ratio:.2f}x the 60-day.",
    )


# ---------------------------------------------------------------------------
# currency, commodity
# ---------------------------------------------------------------------------


@signal("dollar_impulse", "Broad dollar impulse", SignalFamily.CURRENCY,
        ("DTWEXBGS",),
        "A rising dollar tightens global financial conditions mechanically, "
        "through dollar-denominated debt outside the US. It is a risk-off "
        "signal for almost everything except US cash.")
def _dollar_impulse(series: Mapping[str, Series]) -> SignalReading | None:
    dollar = series["DTWEXBGS"]
    change = dollar.pct_change(63)
    if change is None:
        return None
    return _reading(
        "dollar_impulse", strength=squash(-change, 3.0), raw=change,
        inputs=("DTWEXBGS",), staleness=dollar.staleness_days,
        zscore=dollar.zscore(504),
        note=f"Broad dollar {change:+.2f}% over 3 months.",
    )


@signal("oil_impulse", "Crude oil impulse", SignalFamily.COMMODITY,
        ("DCOILWTICO",),
        "Oil is a growth signal and an inflation signal at once, and which "
        "one dominates depends on whether the move is demand-led or "
        "supply-led. The model is given the reading and left to work out "
        "which regime it is in from the other inputs.")
def _oil_impulse(series: Mapping[str, Series]) -> SignalReading | None:
    oil = series["DCOILWTICO"]
    change = oil.pct_change(63)
    if change is None:
        return None
    # A large move in EITHER direction is destabilising; a moderate rise is
    # growth-positive. Peak reading around +10%.
    return _reading(
        "oil_impulse", strength=squash(-(abs(change) - 10.0), 20.0), raw=change,
        inputs=("DCOILWTICO",), staleness=oil.staleness_days,
        note=f"WTI {change:+.2f}% over 3 months.",
    )


# ---------------------------------------------------------------------------
# labour, liquidity, housing
# ---------------------------------------------------------------------------


@signal("labour_momentum", "Jobless claims momentum", SignalFamily.LABOUR,
        ("ICSA",),
        "Weekly, barely revised, and among the earliest reads on the labour "
        "market. Claims turn up before payrolls turn down.")
def _labour_momentum(series: Mapping[str, Series]) -> SignalReading | None:
    claims = series["ICSA"]
    fast = claims.rolling_mean(4)
    slow = claims.rolling_mean(26)
    if fast is None or slow is None or slow <= 0:
        return None
    ratio = (fast / slow - 1) * 100
    # Rising claims are risk-off: invert.
    return _reading(
        "labour_momentum", strength=squash(-ratio, 6.0), raw=ratio,
        inputs=("ICSA",), staleness=claims.staleness_days,
        note=f"4-week claims {ratio:+.1f}% against the 26-week average.",
    )


@signal("financial_conditions", "Financial conditions", SignalFamily.LIQUIDITY,
        ("NFCI",),
        "The Chicago Fed index, which aggregates over 100 measures of risk, "
        "credit and leverage. Positive readings are tighter than average.")
def _financial_conditions(series: Mapping[str, Series]) -> SignalReading | None:
    nfci = series["NFCI"]
    level = nfci.last
    change = nfci.change(13)
    if level is None:
        return None
    # Positive NFCI is TIGHT, which is risk-off: invert the level, and weight
    # the impulse alongside it.
    impulse = change if change is not None else 0.0
    strength = squash(-level, 0.5) * 0.6 + squash(-impulse, 0.25) * 0.4
    return _reading(
        "financial_conditions", strength=strength, raw=level,
        inputs=("NFCI",), staleness=nfci.staleness_days, zscore=nfci.zscore(252),
        note=(f"NFCI at {level:+.2f} "
              f"({'tighter' if level > 0 else 'looser'} than average), "
              f"{impulse:+.2f} over a quarter."),
    )


@signal("net_liquidity", "Net central-bank liquidity", SignalFamily.LIQUIDITY,
        ("WALCL", "RRPONTSYD", "WTREGEN"),
        "Fed balance sheet less the reverse repo facility less the Treasury "
        "General Account. This is the quantity of reserves actually "
        "available to the private system, and it moves for mechanical "
        "reasons -- debt issuance, tax dates -- that have nothing to do with "
        "the economy, which is what makes it informative.")
def _net_liquidity(series: Mapping[str, Series]) -> SignalReading | None:
    balance_sheet, repo, tga = series["WALCL"], series["RRPONTSYD"], series["WTREGEN"]
    # WALCL is in $M; RRPONTSYD and WTREGEN are in $B. Convert to $B.
    now = balance_sheet.last
    if now is None or repo.last is None or tga.last is None:
        return None

    def net_on(offset: int) -> float | None:
        points = balance_sheet.points
        if len(points) <= offset:
            return None
        when = points[-1 - offset].on
        sheet = balance_sheet.value_on(when, tolerance_days=10)
        rrp = repo.value_on(when, tolerance_days=10)
        gen = tga.value_on(when, tolerance_days=10)
        if sheet is None or rrp is None or gen is None:
            return None
        return sheet / 1000.0 - rrp - gen

    current, prior = net_on(0), net_on(13)
    if current is None or prior is None or prior == 0:
        return None
    change = (current / prior - 1) * 100
    staleness = max(balance_sheet.staleness_days, repo.staleness_days,
                    tga.staleness_days)
    return _reading(
        "net_liquidity", strength=squash(change, 2.5), raw=current,
        inputs=("WALCL", "RRPONTSYD", "WTREGEN"), staleness=staleness,
        note=(f"Net liquidity ${current:,.0f}B, {change:+.2f}% over a quarter."),
    )


@signal("housing_impulse", "Housing permits momentum", SignalFamily.HOUSING,
        ("PERMIT",),
        "Building permits are the longest-leading component of the Conference "
        "Board's leading index. Housing turns first on the way down and "
        "first on the way up, because it is the most rate-sensitive sector.")
def _housing_impulse(series: Mapping[str, Series]) -> SignalReading | None:
    permits = series["PERMIT"]
    change = permits.year_over_year()
    if change is None:
        return None
    return _reading(
        "housing_impulse", strength=squash(change, 12.0), raw=change,
        inputs=("PERMIT",), staleness=permits.staleness_days,
        note=f"Building permits {change:+.1f}% year over year.",
    )


@signal("money_growth", "Money supply growth", SignalFamily.LIQUIDITY,
        ("M2SL",),
        "M2 year-over-year. Contracting broad money is historically rare and "
        "historically followed by trouble; it is included for the tails "
        "rather than for day-to-day signal.")
def _money_growth(series: Mapping[str, Series]) -> SignalReading | None:
    m2 = series["M2SL"]
    change = m2.year_over_year()
    if change is None:
        return None
    return _reading(
        "money_growth", strength=squash(change, 5.0), raw=change,
        inputs=("M2SL",), staleness=m2.staleness_days,
        note=f"M2 {change:+.1f}% year over year.",
    )


# ---------------------------------------------------------------------------


def evaluate_all(series: Mapping[str, Series]) -> list[SignalReading]:
    """
    Evaluate every registered signal, in a stable order.

    Order is the registration order and must stay stable: it is the column
    order of the feature vector the network trains on, and a reordering would
    silently permute a trained model's inputs.
    """
    return [spec.evaluate(series) for spec in SIGNALS.values()]
