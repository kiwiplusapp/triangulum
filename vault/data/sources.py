"""
Data sources. All free, all keyless, all public.

    FRED (St. Louis Fed)  every macro series, plus SP500, VIX, the broad dollar
                          index, WTI, credit spreads and the Fed balance sheet.
                          The keyless CSV endpoint (`fredgraph.csv?id=`) is used
                          rather than the API, so the system works with no
                          signup at all. An optional FRED_API_KEY upgrades to
                          the JSON API for vintage (as-first-published) data.
    Coinbase Exchange     crypto daily candles, keyless.
    CoinGecko             crypto spot, keyless, used as a fallback.

Why this matters more than it sounds: a "quant terminal" whose data layer needs
four paid subscriptions is a terminal nobody actually runs. Everything below
works on a laptop with no accounts, which is the difference between a system
that gets used daily and one that gets screenshotted once.

REVISIONS. Macro series are revised after publication. An initial payrolls
print of +150k can become +90k two months later. The keyless CSV endpoint
serves the *current* (revised) vintage, so a thesis scored months later would
be graded against numbers the model never saw. The cache's snapshot mechanism
is the mitigation: what the model read is frozen at commit time. With a
FRED_API_KEY the true as-first-published vintage is available and
:func:`fetch_fred_vintage` uses it.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from vault.data.cache import SeriesCache
from vault.data.http import TransportError, http_get
from vault.data.series import Frequency, Point, Series

logger = logging.getLogger(__name__)

__all__ = [
    "FRED_SERIES", "CRYPTO_SERIES", "SeriesSpec",
    "DataHub", "fetch_fred", "fetch_coinbase", "fetch_coingecko_spot",
]

_UA = "Vault/1.0 (macro research; contact via repository)"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
COINBASE = "https://api.exchange.coinbase.com"
COINGECKO = "https://api.coingecko.com/api/v3"


@dataclass(frozen=True, slots=True)
class SeriesSpec:
    """One series and what it means, so the synthesis layer can label it."""

    key: str
    source_id: str
    label: str
    units: str
    frequency: str
    category: str
    # How the model should read a rise in this series. Encoded here rather than
    # left to the LLM to infer, because "a higher DXY is dollar strength" is a
    # fact, not a judgement, and hard-coding it removes a whole class of
    # confident-sounding error.
    direction_note: str = ""


# --------------------------------------------------------------------------
# The macro universe
# --------------------------------------------------------------------------

FRED_SERIES: tuple[SeriesSpec, ...] = (
    # ---- rates & curve ----
    SeriesSpec("DGS10", "DGS10", "US 10Y Treasury yield", "%", Frequency.DAILY, "rates",
               "Higher = tighter financial conditions, headwind for long-duration assets."),
    SeriesSpec("DGS2", "DGS2", "US 2Y Treasury yield", "%", Frequency.DAILY, "rates",
               "Tracks expected policy path over ~2 years."),
    SeriesSpec("DGS3MO", "DGS3MO", "US 3M Treasury yield", "%", Frequency.DAILY, "rates",
               "Proxy for the current policy rate."),
    SeriesSpec("T10Y2Y", "T10Y2Y", "10Y-2Y spread", "%", Frequency.DAILY, "rates",
               "Negative = inverted curve. Historically precedes recession by 6-18 months."),
    SeriesSpec("T10Y3M", "T10Y3M", "10Y-3M spread", "%", Frequency.DAILY, "rates",
               "The NY Fed's preferred recession signal. Negative = inverted."),
    SeriesSpec("DFII10", "DFII10", "US 10Y real yield (TIPS)", "%", Frequency.DAILY, "rates",
               "The real cost of capital. Rising real yields compress valuations."),
    SeriesSpec("T10YIE", "T10YIE", "10Y breakeven inflation", "%", Frequency.DAILY, "inflation",
               "Market-implied average inflation over 10 years."),
    SeriesSpec("T5YIFR", "T5YIFR", "5y5y forward inflation", "%", Frequency.DAILY, "inflation",
               "Long-run inflation expectations. The Fed watches this closely."),
    SeriesSpec("DFF", "DFF", "Effective fed funds rate", "%", Frequency.DAILY, "rates",
               "The actual policy rate."),

    # ---- growth ----
    SeriesSpec("UNRATE", "UNRATE", "Unemployment rate", "%", Frequency.MONTHLY, "labor",
               "Rising = labor market cooling. The Sahm rule triggers on a 0.5pp rise."),
    SeriesSpec("PAYEMS", "PAYEMS", "Nonfarm payrolls", "thousands", Frequency.MONTHLY, "labor",
               "Level series -- read the monthly change, not the level."),
    SeriesSpec("ICSA", "ICSA", "Initial jobless claims", "count", Frequency.WEEKLY, "labor",
               "The highest-frequency real labor signal. Rising = deterioration."),
    SeriesSpec("INDPRO", "INDPRO", "Industrial production", "index", Frequency.MONTHLY, "growth",
               "Goods-side activity. Leads the manufacturing cycle."),
    SeriesSpec("RSAFS", "RSAFS", "Retail sales", "$M", Frequency.MONTHLY, "growth",
               "Consumer demand, nominal."),
    SeriesSpec("HOUST", "HOUST", "Housing starts", "thousands", Frequency.MONTHLY, "housing",
               "The most rate-sensitive part of the real economy. Leads by ~2 quarters."),
    SeriesSpec("PERMIT", "PERMIT", "Building permits", "thousands", Frequency.MONTHLY, "housing",
               "Leads housing starts."),

    # ---- inflation ----
    SeriesSpec("CPIAUCSL", "CPIAUCSL", "CPI, all items", "index", Frequency.MONTHLY, "inflation",
               "Level series -- read year-over-year."),
    SeriesSpec("CPILFESL", "CPILFESL", "Core CPI", "index", Frequency.MONTHLY, "inflation",
               "Ex food and energy. The Fed's focus. Read year-over-year."),
    SeriesSpec("PCEPILFE", "PCEPILFE", "Core PCE", "index", Frequency.MONTHLY, "inflation",
               "The Fed's actual target measure. Read year-over-year."),

    # ---- liquidity ----
    SeriesSpec("M2SL", "M2SL", "M2 money stock", "$B", Frequency.MONTHLY, "liquidity",
               "Read year-over-year. Contraction is historically rare and disinflationary."),
    SeriesSpec("WALCL", "WALCL", "Fed balance sheet", "$M", Frequency.WEEKLY, "liquidity",
               "Falling = quantitative tightening, a liquidity drain."),
    SeriesSpec("RRPONTSYD", "RRPONTSYD", "Overnight reverse repo", "$B", Frequency.DAILY, "liquidity",
               "Cash parked at the Fed. Draining RRP has offset QT."),
    SeriesSpec("WTREGEN", "WTREGEN", "Treasury General Account", "$B", Frequency.WEEKLY, "liquidity",
               "A rising TGA drains liquidity from the system."),
    SeriesSpec("NFCI", "NFCI", "Chicago Fed financial conditions", "index", Frequency.WEEKLY, "conditions",
               "Positive = tighter than average. A broad conditions summary."),

    # ---- credit & risk ----
    SeriesSpec("BAMLH0A0HYM2", "BAMLH0A0HYM2", "US high-yield OAS", "%", Frequency.DAILY, "credit",
               "Widening = credit stress. The most reliable risk-off tell."),
    SeriesSpec("BAMLC0A0CM", "BAMLC0A0CM", "US investment-grade OAS", "%", Frequency.DAILY, "credit",
               "Widening = broad credit repricing."),
    SeriesSpec("VIXCLS", "VIXCLS", "VIX", "index", Frequency.DAILY, "risk",
               "Implied equity volatility. Spikes on risk-off."),

    # ---- markets ----
    SeriesSpec("SP500", "SP500", "S&P 500", "index", Frequency.DAILY, "equity",
               "Only ~10 years of history on FRED."),
    SeriesSpec("NASDAQ100", "NASDAQ100", "Nasdaq 100", "index", Frequency.DAILY, "equity", ""),
    SeriesSpec("DTWEXBGS", "DTWEXBGS", "Broad dollar index", "index", Frequency.DAILY, "fx",
               "Higher = dollar strength. A headwind for commodities and EM."),
    SeriesSpec("DEXUSEU", "DEXUSEU", "USD per EUR", "rate", Frequency.DAILY, "fx",
               "Note the quoting: HIGHER means a WEAKER dollar."),
    SeriesSpec("DEXJPUS", "DEXJPUS", "JPY per USD", "rate", Frequency.DAILY, "fx",
               "Higher = stronger dollar against the yen."),
    SeriesSpec("DCOILWTICO", "DCOILWTICO", "WTI crude oil", "$/bbl", Frequency.DAILY, "commodity",
               "Both a growth signal and an inflation input."),
)

CRYPTO_SERIES: tuple[tuple[str, str, str], ...] = (
    ("BTCUSD", "BTC-USD", "Bitcoin"),
    ("ETHUSD", "ETH-USD", "Ethereum"),
)

_BY_KEY = {spec.key: spec for spec in FRED_SERIES}


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------


def _http_get(url: str, *, timeout: float = 25.0) -> bytes:
    """Fetch through whichever transport can actually reach the network here."""
    return http_get(url, timeout=timeout, user_agent=_UA)


def fetch_fred(spec: SeriesSpec, *, start: date | None = None) -> Series | None:
    """
    Fetch one FRED series through the keyless CSV endpoint.

    FRED writes missing observations as ``.`` -- holidays in a daily yield
    series, for instance. Those rows are dropped rather than forward-filled: a
    forward-filled holiday makes a series look fresher than it is, and
    staleness is a signal this system actively uses.
    """
    params = {"id": spec.source_id}
    if start:
        params["cosd"] = start.isoformat()
    url = f"{FRED_CSV}?{urllib.parse.urlencode(params)}"

    try:
        raw = _http_get(url).decode("utf-8", errors="replace")
    except (TransportError, urllib.error.URLError, OSError, TimeoutError) as exc:
        logger.warning("FRED fetch failed for %s: %s", spec.source_id, exc)
        return None

    reader = csv.reader(io.StringIO(raw))
    try:
        header = next(reader)
    except StopIteration:
        return None
    if len(header) < 2:
        logger.warning("FRED returned an unexpected header for %s: %r", spec.source_id, header)
        return None

    pairs: list[tuple[date, float]] = []
    for row in reader:
        if len(row) < 2:
            continue
        raw_date, raw_value = row[0].strip(), row[1].strip()
        if not raw_date or raw_value in ("", "."):
            continue          # genuine gap; do NOT forward-fill
        try:
            pairs.append((date.fromisoformat(raw_date), float(raw_value)))
        except ValueError:
            continue

    if not pairs:
        return None
    return Series.from_pairs(
        spec.key, pairs,
        source="FRED", units=spec.units, label=spec.label, frequency=spec.frequency,
        fetched_at=datetime.now(timezone.utc),
    )


def fetch_fred_vintage(spec: SeriesSpec, as_of: date) -> Series | None:
    """
    As-first-published values, using the FRED JSON API. Needs FRED_API_KEY.

    This is the correct input for honest backtesting: it returns what was
    actually knowable on ``as_of``, before revisions. Without a key the system
    falls back to current vintages and the journal records that it did, so a
    later reader knows which grading standard applied.
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return None
    params = {
        "series_id": spec.source_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": as_of.isoformat(),
        "realtime_end": as_of.isoformat(),
    }
    try:
        raw = _http_get(f"{FRED_API}?{urllib.parse.urlencode(params)}")
        payload = json.loads(raw)
    except (TransportError, urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("FRED vintage fetch failed for %s: %s", spec.source_id, exc)
        return None

    pairs = []
    for obs in payload.get("observations", []):
        value = obs.get("value", ".")
        if value in (".", "", None):
            continue
        try:
            pairs.append((date.fromisoformat(obs["date"]), float(value)))
        except (ValueError, KeyError):
            continue
    if not pairs:
        return None
    return Series.from_pairs(
        f"{spec.key}@{as_of.isoformat()}", pairs,
        source="FRED-vintage", units=spec.units, label=f"{spec.label} (as of {as_of})",
        frequency=spec.frequency,
    )


def fetch_coinbase(product: str, *, days: int = 300) -> Series | None:
    """
    Daily closes from Coinbase Exchange. Keyless.

    Coinbase caps a candles request at 300 buckets, so ``days`` above that is
    silently truncated by the venue rather than erroring -- requesting 1000 and
    assuming you got 1000 is a way to compute a "3-year" momentum from ten
    months of data.
    """
    days = min(days, 300)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    params = {
        "granularity": 86400,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    url = f"{COINBASE}/products/{product}/candles?{urllib.parse.urlencode(params)}"
    try:
        payload = json.loads(_http_get(url))
    except (TransportError, urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("Coinbase fetch failed for %s: %s", product, exc)
        return None

    # [ time, low, high, open, close, volume ], newest first.
    pairs: list[tuple[date, float]] = []
    for row in payload if isinstance(payload, list) else []:
        if len(row) < 5:
            continue
        try:
            pairs.append((
                datetime.fromtimestamp(int(row[0]), tz=timezone.utc).date(),
                float(row[4]),
            ))
        except (ValueError, TypeError, OSError):
            continue
    if not pairs:
        return None
    return Series.from_pairs(
        product.replace("-", ""), pairs,
        source="Coinbase", units="USD", label=product, frequency=Frequency.DAILY,
        fetched_at=datetime.now(timezone.utc),
    )


def fetch_coingecko_spot(ids: Sequence[str] = ("bitcoin", "ethereum")) -> dict[str, float]:
    """Current spot. Used only as a freshness cross-check on Coinbase closes."""
    params = {"ids": ",".join(ids), "vs_currencies": "usd"}
    try:
        payload = json.loads(_http_get(f"{COINGECKO}/simple/price?{urllib.parse.urlencode(params)}"))
    except (TransportError, urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("CoinGecko spot fetch failed: %s", exc)
        return {}
    return {k: float(v.get("usd", 0)) for k, v in payload.items() if isinstance(v, dict)}


# --------------------------------------------------------------------------
# The hub
# --------------------------------------------------------------------------


class DataHub:
    """Fetches, caches and reports on the full series universe."""

    def __init__(
        self,
        cache: SeriesCache | None = None,
        *,
        history_years: int = 12,
        polite_delay_sec: float = 0.25,
    ) -> None:
        self.cache = cache or SeriesCache()
        self.history_start = date.today() - timedelta(days=365 * history_years)
        self.polite_delay = polite_delay_sec
        self._series: dict[str, Series] = {}
        self.failures: list[str] = []

    def scan(self, *, include_crypto: bool = True,
             only: Sequence[str] = ()) -> dict[str, Series]:
        """
        Fetch (or read from cache) the whole universe.

        Failures are collected rather than raised. A macro synthesis missing
        one series is degraded; a synthesis that refuses to run because a
        single endpoint had a bad minute is useless. The failure list travels
        with the brief so the model is told what it is missing.
        """
        wanted = set(only) if only else None
        self.failures = []

        for spec in FRED_SERIES:
            if wanted and spec.key not in wanted:
                continue
            series = self.cache.get_or_fetch(
                spec.key,
                lambda s=spec: fetch_fred(s, start=self.history_start),
                frequency=spec.frequency,
            )
            if series is None:
                self.failures.append(spec.key)
                continue
            self._series[spec.key] = series
            if self.cache.misses and self.polite_delay:
                time.sleep(self.polite_delay)

        if include_crypto:
            for key, product, label in CRYPTO_SERIES:
                if wanted and key not in wanted:
                    continue
                series = self.cache.get_or_fetch(
                    key, lambda p=product: fetch_coinbase(p), frequency=Frequency.DAILY,
                )
                if series is None:
                    self.failures.append(key)
                    continue
                self._series[key] = series

        return dict(self._series)

    # -- access ------------------------------------------------------------

    def get(self, key: str) -> Series | None:
        return self._series.get(key)

    def require(self, key: str) -> Series:
        series = self._series.get(key)
        if series is None:
            raise KeyError(f"series {key!r} is not loaded (failures: {self.failures})")
        return series

    def spec(self, key: str) -> SeriesSpec | None:
        return _BY_KEY.get(key)

    def by_category(self, category: str) -> dict[str, Series]:
        return {
            k: s for k, s in self._series.items()
            if (_BY_KEY.get(k) and _BY_KEY[k].category == category)
        }

    @property
    def keys(self) -> list[str]:
        return sorted(self._series)

    # -- reporting ---------------------------------------------------------

    def coverage(self) -> dict[str, Any]:
        """
        What we have, how fresh it is, and what is missing.

        ``stale`` uses a per-frequency threshold: a daily series untouched for
        five days is stale; a monthly one is not stale until about fifty. A
        single global threshold would either flag every monthly series forever
        or never flag a broken daily feed.
        """
        thresholds = {
            Frequency.DAILY: 5, Frequency.WEEKLY: 14,
            Frequency.MONTHLY: 50, Frequency.QUARTERLY: 130,
            Frequency.IRREGULAR: 30,
        }
        stale: list[dict[str, Any]] = []
        for key, series in self._series.items():
            limit = thresholds.get(series.frequency, 30)
            if series.staleness_days > limit:
                stale.append({
                    "key": key, "as_of": series.as_of.isoformat() if series.as_of else None,
                    "days": series.staleness_days, "limit": limit,
                })
        return {
            "loaded": len(self._series),
            "expected": len(FRED_SERIES) + len(CRYPTO_SERIES),
            "failed": self.failures,
            "stale": sorted(stale, key=lambda r: -r["days"]),
            "cache": self.cache.stats(),
        }

    def describe_all(self) -> list[dict[str, Any]]:
        rows = []
        for key in sorted(self._series):
            row = self._series[key].describe()
            spec = _BY_KEY.get(key)
            if spec:
                row["category"] = spec.category
                row["note"] = spec.direction_note
            rows.append(row)
        return rows
