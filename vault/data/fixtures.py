"""
Offline fixture data.

Three jobs:

1. **Tests run without a network.** A test suite that needs FRED to be up is a
   test suite that fails for reasons unrelated to the code.
2. **The demo works anywhere.** `vault demo` shows the whole pipeline -- scan,
   regime, thesis, resolution, calibration -- on a plane, behind a corporate
   proxy, or before the user has thought about data sources at all.
3. **Deterministic scoring tests.** Calibration maths needs known inputs to be
   verified against a hand-computed answer.

THESE ARE NOT REAL MARKET DATA. Every series is synthesised from a plausible
starting level and a seeded random walk, with the macro relationships that
matter for regime classification deliberately built in (a curve that inverts,
inflation that decelerates then reaccelerates, claims that drift up). The
generator is labelled in every series' `source` field as ``fixture`` so nothing
downstream can mistake it for the real thing, and the HUD shows a banner when
any fixture series is loaded.

Using these to judge whether the strategy works would be circular. They exist
to prove the *machinery* works.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

from vault.data.series import Frequency, Series

__all__ = ["build_fixture_universe", "FIXTURE_SOURCE"]

FIXTURE_SOURCE = "fixture"


def _walk(
    start: float, days: int, *, drift: float = 0.0, vol: float = 0.01,
    rng: random.Random, floor: float | None = None, ceiling: float | None = None,
    seasonal: float = 0.0,
) -> list[float]:
    """Geometric-ish random walk with optional drift, bounds, and a seasonal term."""
    values = [start]
    for i in range(1, days):
        shock = rng.gauss(drift, vol)
        season = seasonal * math.sin(2 * math.pi * i / 252) if seasonal else 0.0
        nxt = values[-1] * (1 + shock) + season
        if floor is not None:
            nxt = max(floor, nxt)
        if ceiling is not None:
            nxt = min(ceiling, nxt)
        values.append(nxt)
    return values


def _daily(key: str, label: str, units: str, values: list[float], *,
           end: date, category_freq: str = Frequency.DAILY) -> Series:
    """Business-day spine ending today."""
    points: list[tuple[date, float]] = []
    cursor = end
    for value in reversed(values):
        while cursor.weekday() >= 5:
            cursor -= timedelta(days=1)
        points.append((cursor, value))
        cursor -= timedelta(days=1)
    return Series.from_pairs(
        key, reversed(points), source=FIXTURE_SOURCE, units=units,
        label=f"{label} [FIXTURE]", frequency=category_freq,
    )


def _monthly(key: str, label: str, units: str, values: list[float], *,
             end: date, lag_days: int = 30) -> Series:
    """
    Monthly spine with a publication lag.

    The lag is the point. Core PCE for month M is published around M+45 days,
    so the freshest observation a system can possibly have is already six weeks
    old. Fixtures that pretend monthly data is same-day would let the regime
    classifier look better than it can ever be in production.
    """
    points: list[tuple[date, float]] = []
    cursor = end - timedelta(days=lag_days)
    for value in reversed(values):
        points.append((cursor.replace(day=1), value))
        cursor = (cursor.replace(day=1) - timedelta(days=1))
    return Series.from_pairs(
        key, reversed(points), source=FIXTURE_SOURCE, units=units,
        label=f"{label} [FIXTURE]", frequency=Frequency.MONTHLY,
    )


def _weekly(key: str, label: str, units: str, values: list[float], *,
            end: date, lag_days: int = 5) -> Series:
    points: list[tuple[date, float]] = []
    cursor = end - timedelta(days=lag_days)
    for value in reversed(values):
        points.append((cursor, value))
        cursor -= timedelta(days=7)
    return Series.from_pairs(
        key, reversed(points), source=FIXTURE_SOURCE, units=units,
        label=f"{label} [FIXTURE]", frequency=Frequency.WEEKLY,
    )


def build_fixture_universe(
    *, seed: int = 20260101, days: int = 900, end: date | None = None,
    scenario: str = "late_cycle",
) -> dict[str, Series]:
    """
    Build a complete, internally consistent fixture universe.

    ``scenario`` shapes the macro story so the regime classifier has something
    real to find:

        late_cycle   curve inverted then re-steepening, claims drifting up,
                     inflation decelerating -> should read DEFLATION/GOLDILOCKS
        reflation    growth and inflation both accelerating
        stagflation  growth rolling over while inflation reaccelerates
    """
    rng = random.Random(seed)
    end = end or date.today()
    months = max(30, days // 21)
    weeks = max(60, days // 5)

    # Scenario knobs: (growth drift, inflation drift, curve level, oil drift)
    knobs = {
        "late_cycle": (-0.00006, -0.00010, -0.35, -0.0004),
        "reflation": (+0.00010, +0.00012, +0.90, +0.0011),
        "stagflation": (-0.00010, +0.00018, +0.15, +0.0016),
    }.get(scenario, (-0.00006, -0.00010, -0.35, -0.0004))
    growth_drift, inflation_drift, curve_level, oil_drift = knobs

    out: dict[str, Series] = {}

    # ---- rates -----------------------------------------------------------
    ten_year = _walk(4.28, days, drift=0.00004, vol=0.011, rng=rng, floor=0.3, ceiling=8.0)
    out["DGS10"] = _daily("DGS10", "US 10Y Treasury yield", "%", ten_year, end=end)

    two_year = [max(0.2, y - curve_level + rng.gauss(0, 0.04)) for y in ten_year]
    out["DGS2"] = _daily("DGS2", "US 2Y Treasury yield", "%", two_year, end=end)

    three_month = [max(0.1, y + 0.18 + rng.gauss(0, 0.03)) for y in two_year]
    out["DGS3MO"] = _daily("DGS3MO", "US 3M Treasury yield", "%", three_month, end=end)

    out["T10Y2Y"] = _daily("T10Y2Y", "10Y-2Y spread", "%",
                           [a - b for a, b in zip(ten_year, two_year)], end=end)
    out["T10Y3M"] = _daily("T10Y3M", "10Y-3M spread", "%",
                           [a - b for a, b in zip(ten_year, three_month)], end=end)

    breakeven = _walk(2.31, days, drift=inflation_drift, vol=0.007, rng=rng, floor=0.4, ceiling=4.5)
    out["T10YIE"] = _daily("T10YIE", "10Y breakeven inflation", "%", breakeven, end=end)
    out["T5YIFR"] = _daily("T5YIFR", "5y5y forward inflation", "%",
                           [b + 0.09 + rng.gauss(0, 0.02) for b in breakeven], end=end)
    out["DFII10"] = _daily("DFII10", "US 10Y real yield (TIPS)", "%",
                           [n - b for n, b in zip(ten_year, breakeven)], end=end)
    out["DFF"] = _daily("DFF", "Effective fed funds rate", "%",
                        [max(0.05, t - 0.05) for t in three_month], end=end)

    # ---- inflation indices ----------------------------------------------
    # Level indices. The classifier reads 3-month annualized against YoY, so the
    # month-to-month path is what matters, not the absolute level.
    core_cpi = [325.0]
    for _ in range(months - 1):
        core_cpi.append(core_cpi[-1] * (1 + 0.0022 + inflation_drift * 18 + rng.gauss(0, 0.0009)))
    out["CPILFESL"] = _monthly("CPILFESL", "Core CPI", "index", core_cpi, end=end, lag_days=16)
    out["CPIAUCSL"] = _monthly("CPIAUCSL", "CPI, all items", "index",
                               [v * 0.985 for v in core_cpi], end=end, lag_days=16)

    core_pce = [126.0]
    for _ in range(months - 1):
        core_pce.append(core_pce[-1] * (1 + 0.0019 + inflation_drift * 16 + rng.gauss(0, 0.0007)))
    out["PCEPILFE"] = _monthly("PCEPILFE", "Core PCE", "index", core_pce, end=end, lag_days=45)

    # ---- labour ----------------------------------------------------------
    claims = _walk(221_000, weeks, drift=-growth_drift * 9, vol=0.028, rng=rng, floor=180_000)
    out["ICSA"] = _weekly("ICSA", "Initial jobless claims", "count", claims, end=end)

    payrolls = [158_000.0]
    for _ in range(months - 1):
        payrolls.append(payrolls[-1] * (1 + 0.0009 + growth_drift * 7 + rng.gauss(0, 0.0004)))
    out["PAYEMS"] = _monthly("PAYEMS", "Nonfarm payrolls", "thousands", payrolls, end=end, lag_days=8)

    unemployment = [4.05]
    for _ in range(months - 1):
        unemployment.append(max(3.2, unemployment[-1] - growth_drift * 260 + rng.gauss(0, 0.045)))
    out["UNRATE"] = _monthly("UNRATE", "Unemployment rate", "%", unemployment, end=end, lag_days=8)

    # ---- activity & housing ----------------------------------------------
    industrial = _walk(103.0, months, drift=growth_drift * 12, vol=0.004, rng=rng)
    out["INDPRO"] = _monthly("INDPRO", "Industrial production", "index", industrial, end=end, lag_days=17)

    permits = _walk(1_430.0, months, drift=growth_drift * 22, vol=0.021, rng=rng, floor=700)
    out["PERMIT"] = _monthly("PERMIT", "Building permits", "thousands", permits, end=end, lag_days=19)
    out["HOUST"] = _monthly("HOUST", "Housing starts", "thousands",
                            [p * 0.96 for p in permits], end=end, lag_days=19)
    out["RSAFS"] = _monthly("RSAFS", "Retail sales", "$M",
                            _walk(710_000, months, drift=0.0025 + growth_drift * 8,
                                  vol=0.006, rng=rng), end=end, lag_days=16)

    # ---- liquidity -------------------------------------------------------
    out["M2SL"] = _monthly("M2SL", "M2 money stock", "$B",
                           _walk(21_600, months, drift=0.0016, vol=0.0022, rng=rng),
                           end=end, lag_days=30)
    out["WALCL"] = _weekly("WALCL", "Fed balance sheet", "$M",
                           _walk(6_620_000, weeks, drift=-0.0011, vol=0.0016, rng=rng),
                           end=end, lag_days=2)
    out["RRPONTSYD"] = _daily("RRPONTSYD", "Overnight reverse repo", "$B",
                              _walk(148.0, days, drift=-0.0022, vol=0.10, rng=rng, floor=0.0),
                              end=end)
    out["WTREGEN"] = _weekly("WTREGEN", "Treasury General Account", "$B",
                             _walk(742.0, weeks, drift=0.0009, vol=0.055, rng=rng, floor=50),
                             end=end, lag_days=2)
    out["NFCI"] = _weekly("NFCI", "Chicago Fed financial conditions", "index",
                          _walk(-0.42, weeks, drift=0.0, vol=0.05, rng=rng), end=end, lag_days=6)

    # ---- credit & risk ---------------------------------------------------
    high_yield = _walk(3.05, days, drift=-growth_drift * 12, vol=0.019, rng=rng, floor=2.4, ceiling=11.0)
    out["BAMLH0A0HYM2"] = _daily("BAMLH0A0HYM2", "US high-yield OAS", "%", high_yield, end=end)
    out["BAMLC0A0CM"] = _daily("BAMLC0A0CM", "US investment-grade OAS", "%",
                               [h * 0.29 for h in high_yield], end=end)
    out["VIXCLS"] = _daily("VIXCLS", "VIX", "index",
                           _walk(15.4, days, drift=0.0, vol=0.055, rng=rng, floor=9.0, ceiling=70.0),
                           end=end)

    # ---- markets ---------------------------------------------------------
    out["SP500"] = _daily("SP500", "S&P 500", "index",
                          _walk(5_600, days, drift=0.00032 + growth_drift * 3,
                                vol=0.0091, rng=rng), end=end)
    out["NASDAQ100"] = _daily("NASDAQ100", "Nasdaq 100", "index",
                              _walk(19_800, days, drift=0.00038 + growth_drift * 4,
                                    vol=0.0117, rng=rng), end=end)
    out["DTWEXBGS"] = _daily("DTWEXBGS", "Broad dollar index", "index",
                             _walk(121.4, days, drift=0.00004, vol=0.0031, rng=rng), end=end)
    out["DEXUSEU"] = _daily("DEXUSEU", "USD per EUR", "rate",
                            _walk(1.084, days, drift=-0.00003, vol=0.0038, rng=rng), end=end)
    out["DEXJPUS"] = _daily("DEXJPUS", "JPY per USD", "rate",
                            _walk(152.4, days, drift=0.00006, vol=0.0042, rng=rng), end=end)
    out["DCOILWTICO"] = _daily("DCOILWTICO", "WTI crude oil", "$/bbl",
                               _walk(71.8, days, drift=oil_drift, vol=0.019, rng=rng, floor=20),
                               end=end)

    # ---- crypto ----------------------------------------------------------
    out["BTCUSD"] = _daily("BTCUSD", "Bitcoin", "USD",
                           _walk(79_000, min(days, 300), drift=0.0006, vol=0.028, rng=rng,
                                 floor=1000), end=end)
    out["ETHUSD"] = _daily("ETHUSD", "Ethereum", "USD",
                           _walk(2_450, min(days, 300), drift=0.0004, vol=0.034, rng=rng,
                                 floor=50), end=end)

    return out
