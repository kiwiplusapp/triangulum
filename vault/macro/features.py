"""
The FLOW stage: curve structure, credit, cross-asset, and named signals.

Turns raw series into the readings a macro analyst would actually cite, so the
brief handed to the model contains conclusions with their evidence rather than
a wall of numbers. A model given thirty raw series will invent a narrative; a
model given "the 10Y-3M is at -0.42 and has been inverted for 14 months"
reasons about a fact.

Every named signal carries its own staleness, so a signal built on a monthly
series that is seven weeks old cannot masquerade as current.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

from vault.data.series import Series, correlation
from vault.macro.regime import RegimeRead, sahm_rule

__all__ = ["build_curve_read", "build_cross_asset_read", "named_signals", "build_brief"]


def _fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "unavailable"
    return f"{value:,.{digits}f}{suffix}"


def build_curve_read(series: Mapping[str, Series]) -> dict[str, str]:
    """Yield curve and real-rate structure, as sentences."""
    out: dict[str, str] = {}

    ten_three = series.get("T10Y3M")
    ten_two = series.get("T10Y2Y")
    if ten_three and ten_three.points:
        level = ten_three.last
        state = "inverted" if level < 0 else "positive"
        months = _months_inverted(ten_three)
        out["10Y-3M"] = (
            f"{_fmt(level, 2, '%')} -- {state}"
            + (f", and has been for ~{months} months" if months else "")
            + f". 3m change {_fmt(ten_three.change(63), 2, 'pp')}"
        )
    if ten_two and ten_two.points:
        out["10Y-2Y"] = (
            f"{_fmt(ten_two.last, 2, '%')}, "
            f"3m change {_fmt(ten_two.change(63), 2, 'pp')}"
        )

    real = series.get("DFII10")
    if real and real.points:
        out["10Y real yield"] = (
            f"{_fmt(real.last, 2, '%')} "
            f"({_fmt(real.percentile_rank(504), 2)} percentile over 2 years), "
            f"3m change {_fmt(real.change(63), 2, 'pp')}. "
            f"Rising real yields compress valuations."
        )

    breakeven = series.get("T10YIE")
    forward = series.get("T5YIFR")
    if breakeven and breakeven.points:
        out["inflation expectations"] = (
            f"10Y breakeven {_fmt(breakeven.last, 2, '%')} "
            f"(3m {_fmt(breakeven.change(63), 2, 'pp')})"
            + (f", 5y5y forward {_fmt(forward.last, 2, '%')}" if forward and forward.points else "")
        )

    policy = series.get("DFF")
    if policy and policy.points:
        out["policy rate"] = (
            f"effective fed funds {_fmt(policy.last, 2, '%')}, "
            f"6m change {_fmt(policy.change(126), 2, 'pp')}"
        )
    return out


def build_cross_asset_read(series: Mapping[str, Series]) -> dict[str, str]:
    """Risk appetite, the dollar, commodities, and correlation state."""
    out: dict[str, str] = {}

    high_yield = series.get("BAMLH0A0HYM2")
    if high_yield and high_yield.points:
        percentile = high_yield.percentile_rank(504)
        out["credit"] = (
            f"HY OAS {_fmt(high_yield.last, 2, '%')} "
            f"({_fmt(percentile, 2)} percentile over 2 years), "
            f"1m change {_fmt(high_yield.change(21), 2, 'pp')}. "
            f"Credit leads equity drawdowns more reliably than equity momentum does."
        )

    vix = series.get("VIXCLS")
    if vix and vix.points:
        out["volatility"] = (
            f"VIX {_fmt(vix.last, 1)}, "
            f"z-score {_fmt(vix.zscore(252), 2)} over a year"
        )

    equity = series.get("SP500")
    if equity and equity.points:
        out["equity"] = (
            f"S&P 500 {_fmt(equity.last, 0)}, "
            f"1m {_fmt(equity.pct_change(21), 2, '%')}, "
            f"3m {_fmt(equity.pct_change(63), 2, '%')}, "
            f"drawdown from peak {_fmt(equity.drawdown(), 2, '%')}, "
            f"20d realized vol {_fmt(equity.realized_volatility(20), 1, '%')}"
        )

    dollar = series.get("DTWEXBGS")
    if dollar and dollar.points:
        out["dollar"] = (
            f"broad dollar index {_fmt(dollar.last, 2)}, "
            f"3m {_fmt(dollar.pct_change(63), 2, '%')}. "
            f"A stronger dollar is a headwind for commodities and EM."
        )

    oil = series.get("DCOILWTICO")
    if oil and oil.points:
        out["oil"] = (
            f"WTI {_fmt(oil.last, 2, ' USD')}, "
            f"3m {_fmt(oil.pct_change(63), 2, '%')}"
        )

    btc = series.get("BTCUSD")
    if btc and btc.points:
        out["bitcoin"] = (
            f"{_fmt(btc.last, 0, ' USD')}, "
            f"1m {_fmt(btc.pct_change(21), 2, '%')}, "
            f"20d realized vol {_fmt(btc.realized_volatility(20), 0, '%')}"
        )

    # Correlation regime. Computed on RETURNS -- correlating the levels of two
    # trending series produces a number near 1 that means only "both went up".
    if equity and series.get("DGS10"):
        rho = correlation(equity, series["DGS10"], window=60, on_returns=True)
        if rho is not None:
            reading = (
                "positive -- the market is trading growth, so higher yields are "
                "read as better growth"
                if rho > 0.15 else
                "negative -- the market is trading rates, so higher yields hurt "
                "equities" if rho < -0.15 else
                "near zero -- no clear rates/equity regime"
            )
            out["equity-rates correlation"] = f"{rho:+.2f} over 60 days, {reading}"

    return out


def named_signals(series: Mapping[str, Series], regime: RegimeRead) -> list[str]:
    """
    Discrete, checkable signals. Each one is a fact with a threshold, not an
    interpretation -- the model does the interpreting.
    """
    signals: list[str] = []

    sahm = sahm_rule(series.get("UNRATE"))
    if sahm.get("available"):
        state = "TRIGGERED" if sahm["triggered"] else "not triggered"
        signals.append(
            f"Sahm rule {state}: {sahm['value']:+.2f}pp against a 0.50 threshold "
            f"(3m avg {sahm['current_3m_avg']:.2f}% vs 12m low {sahm['min_12m']:.2f}%, "
            f"as of {sahm['as_of']})"
        )

    curve = series.get("T10Y3M")
    if curve and curve.points:
        if curve.last < 0:
            signals.append(
                f"Yield curve INVERTED at {curve.last:.2f}%. Inversion leads "
                f"recession by 6-18 months -- a slow signal that does not "
                f"justify a fast call."
            )
        elif curve.change(63) is not None and curve.change(63) > 0.3 and curve.last < 0.5:
            signals.append(
                f"Curve re-steepening from inversion ({curve.change(63):+.2f}pp over "
                f"3 months). Bear steepening after inversion has historically "
                f"coincided with the onset of recession rather than its avoidance."
            )

    high_yield = series.get("BAMLH0A0HYM2")
    if high_yield and high_yield.points:
        change = high_yield.change(21)
        if change is not None and change > 0.4:
            signals.append(
                f"High-yield spreads widened {change:+.2f}pp in a month to "
                f"{high_yield.last:.2f}% -- credit is repricing risk."
            )
        elif high_yield.percentile_rank(504) is not None and high_yield.percentile_rank(504) < 0.1:
            signals.append(
                f"HY spreads at {high_yield.last:.2f}%, in the bottom decile of "
                f"2 years. Credit is priced for perfection, which is an "
                f"asymmetry, not a forecast."
            )

    claims = series.get("ICSA")
    if claims and claims.points:
        change = claims.pct_change(13)
        if change is not None and change > 12:
            signals.append(
                f"Initial claims up {change:.1f}% over 13 weeks to "
                f"{claims.last:,.0f} -- the highest-frequency labour signal is "
                f"deteriorating."
            )

    m2 = series.get("M2SL")
    if m2 and m2.points:
        yoy = m2.year_over_year()
        if yoy is not None and yoy < 0:
            signals.append(
                f"M2 contracting {yoy:.1f}% year over year. Outside 2023, this "
                f"has essentially not happened since the 1930s."
            )

    if regime.transitioning:
        second, probability = regime.runner_up
        signals.append(
            f"Regime is in transition: {regime.quadrant} at "
            f"{regime.probabilities.get(regime.quadrant, 0):.0%} against {second} "
            f"at {probability:.0%}. Transitions are when a single label is most "
            f"confidently wrong."
        )

    if regime.max_staleness_days > 45:
        signals.append(
            f"The regime classification's oldest input is "
            f"{regime.max_staleness_days} days old. Weight it accordingly."
        )
    return signals


def build_brief(
    series: Mapping[str, Series],
    regime: RegimeRead,
    coverage: dict[str, Any],
    *,
    generated_at: str,
    data_mode: str = "live",
    specs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble everything the model sees. This is the SCAN -> FLOW output."""
    thresholds = {"daily": 5, "weekly": 14, "monthly": 50, "quarterly": 130, "irregular": 30}

    rows: list[dict[str, Any]] = []
    for key in sorted(series):
        row = series[key].describe()
        spec = (specs or {}).get(key)
        row["note"] = getattr(spec, "direction_note", "") if spec else ""
        row["category"] = getattr(spec, "category", "") if spec else ""
        row["stale"] = row["staleness_days"] > thresholds.get(series[key].frequency, 30)
        rows.append(row)

    return {
        "generated_at": generated_at,
        "data_mode": data_mode,
        "regime": regime.to_dict(),
        "curve": build_curve_read(series),
        "cross_asset": build_cross_asset_read(series),
        "signals": named_signals(series, regime),
        "series": rows,
        "coverage": coverage,
    }


def _months_inverted(curve: Series) -> int | None:
    """How many months the curve has been continuously below zero."""
    if not curve.points or curve.last >= 0:
        return None
    count = 0
    for point in reversed(curve.points):
        if point.value >= 0:
            break
        count += 1
    return max(1, round(count / 21)) if count else None
