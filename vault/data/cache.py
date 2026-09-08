"""
Disk cache for fetched series.

Two jobs, and the second one matters more than it looks:

1. **Do not hammer free endpoints.** FRED and CoinGecko are public goods. A
   scan every five minutes that re-downloads sixty years of daily yields is
   abusive and will get the IP blocked.

2. **Make a run reproducible after the fact.** When a thesis is later resolved
   and scored, you need to know what the model actually saw -- not what the
   series looks like now, after revisions. Macro data is revised: an initial
   payrolls print of +150k becomes +90k two months later. Scoring a forecast
   against revised data credits or blames the model for information it did not
   have. So every fetch is snapshotted, and the thesis journal records the
   cache keys it read.

TTLs are per-frequency, because a daily yield and a monthly CPI print have
completely different useful lifetimes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from vault.data.series import Frequency, Point, Series

logger = logging.getLogger(__name__)

__all__ = ["SeriesCache", "DEFAULT_TTL"]

# Seconds. A daily series is worth re-fetching hourly; a monthly release does
# not change intraday and re-fetching it is pure noise.
DEFAULT_TTL: dict[str, float] = {
    Frequency.DAILY: 3600.0,
    Frequency.WEEKLY: 6 * 3600.0,
    Frequency.MONTHLY: 12 * 3600.0,
    Frequency.QUARTERLY: 24 * 3600.0,
    Frequency.IRREGULAR: 3600.0,
}


@dataclass(slots=True)
class CacheEntry:
    key: str
    fetched_at: float
    frequency: str
    payload: dict

    @property
    def age_sec(self) -> float:
        return time.time() - self.fetched_at

    def expired(self, ttl: dict[str, float] | None = None) -> bool:
        table = ttl or DEFAULT_TTL
        return self.age_sec > table.get(self.frequency, 3600.0)


class SeriesCache:
    """Gzipped-JSON cache of fetched series, one file per key."""

    def __init__(self, directory: str | Path = "data/vault/cache",
                 *, ttl: dict[str, float] | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl or dict(DEFAULT_TTL)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _path(self, key: str) -> Path:
        # Hash the key so arbitrary series ids cannot escape the directory.
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:48]
        return self.directory / f"{safe}.{digest}.json.gz"

    def get(self, key: str, *, ignore_ttl: bool = False) -> Series | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError, EOFError):
            logger.warning("cache entry %s is unreadable; refetching", path.name)
            self.misses += 1
            return None

        entry = CacheEntry(
            key=raw.get("key", key),
            fetched_at=float(raw.get("fetched_at", 0)),
            frequency=raw.get("frequency", Frequency.IRREGULAR),
            payload=raw,
        )
        if not ignore_ttl and entry.expired(self.ttl):
            self.misses += 1
            return None

        self.hits += 1
        return _deserialize(raw)

    def put(self, series: Series) -> None:
        path = self._path(series.key)
        payload = _serialize(series)
        tmp = path.with_suffix(".tmp")
        try:
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=5) as handle:
                json.dump(payload, handle)
            tmp.replace(path)
            self.writes += 1
        except OSError:
            logger.exception("failed to write cache entry for %s", series.key)

    def get_or_fetch(
        self, key: str, fetcher: Callable[[], Series | None],
        *, frequency: str = Frequency.DAILY,
    ) -> Series | None:
        """
        Cache-aside with a stale fallback.

        If the fetch fails and a stale entry exists, the stale entry is returned
        WITH its true staleness intact. A scan on stale data is a legitimate
        degraded mode -- a scan that silently invents fresh data is not, and the
        staleness field is what lets the synthesis layer see the difference.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        try:
            fetched = fetcher()
        except Exception as exc:
            logger.warning("fetch failed for %s: %s", key, exc)
            fetched = None
        if fetched is not None and fetched.points:
            self.put(fetched)
            return fetched
        stale = self.get(key, ignore_ttl=True)
        if stale is not None:
            logger.warning(
                "using STALE cache for %s (last observation %s, %d days old)",
                key, stale.as_of, stale.staleness_days,
            )
        return stale

    def snapshot(self, label: str, keys: list[str]) -> Path:
        """
        Freeze the current state of ``keys`` under an immutable label.

        Called when a thesis is committed. Scoring later reads the snapshot, not
        the live series, so a forecast is graded against the data the model
        actually saw -- macro releases get revised, and grading against revised
        data credits or blames the model for information it never had.
        """
        target = self.directory.parent / "snapshots" / label
        target.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for key in keys:
            series = self.get(key, ignore_ttl=True)
            if series is None:
                continue
            path = target / f"{_safe(key)}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(_serialize(series), handle)
            written.append(key)
        (target / "MANIFEST.json").write_text(
            json.dumps({
                "label": label,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "keys": written,
            }, indent=2),
            encoding="utf-8",
        )
        return target

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "directory": str(self.directory),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "entries": len(list(self.directory.glob("*.json.gz"))),
        }


def _serialize(series: Series) -> dict:
    return {
        "key": series.key,
        "label": series.label,
        "source": series.source,
        "units": series.units,
        "frequency": series.frequency,
        "fetched_at": time.time(),
        "points": [[p.on.isoformat(), p.value] for p in series.points],
    }


def _deserialize(raw: dict) -> Series:
    return Series(
        key=raw["key"],
        points=[Point(date.fromisoformat(d), float(v)) for d, v in raw.get("points", [])],
        source=raw.get("source", ""),
        units=raw.get("units", ""),
        label=raw.get("label", ""),
        frequency=raw.get("frequency", Frequency.IRREGULAR),
        fetched_at=datetime.fromtimestamp(raw.get("fetched_at", 0), tz=timezone.utc),
    )


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:64]
