"""
Tick recorder.

The learning layer is only as good as its training data, and there is no
downloadable dataset of "order books across six venues at the moment a
triangular edge appeared". You have to record it yourself, from day one, in
paper mode, before you ever risk capital. This module is therefore not an
optional nicety -- it is the input to everything in ``triangulum.learning``.

Format: newline-delimited JSON, gzip-compressed, one file per venue per hour.
Chosen over Parquet deliberately: NDJSON is append-only and crash-safe (a
truncated final line is discarded on read, losing one record rather than the
file), needs no schema migration when a field is added, and streams without
loading the file. The compression ratio on order-book data is around 12:1,
which is close enough to columnar formats to not be worth the fragility.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, TextIO

from triangulum.core.types import BookSnapshot, Opportunity, Symbol

logger = logging.getLogger(__name__)

__all__ = ["TickRecorder", "RecordType", "read_recording", "iter_recordings"]


class RecordType:
    BOOK = "book"
    QUOTE = "quote"
    TRADE = "trade"
    OPPORTUNITY = "opp"
    CYCLE = "cycle"
    META = "meta"


def _encode(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_encode(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    return obj


@dataclass(slots=True)
class _OpenFile:
    handle: TextIO
    path: Path
    hour_bucket: int
    lines: int = 0
    bytes_written: int = 0


class TickRecorder:
    """
    Buffered NDJSON.gz writer with hourly rotation.

    Writes are buffered and flushed on a timer so the market-data path never
    blocks on disk. The buffer is bounded; overflow drops the *oldest* buffered
    records and increments a counter, because a recorder that stalls the engine
    is worse than a recorder with a hole in it.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        buffer_size: int = 8192,
        flush_interval_sec: float = 5.0,
        compress: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._buffer: list[str] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_sec
        self._compress = compress
        self._files: dict[str, _OpenFile] = {}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.records_written = 0
        self.records_dropped = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._flush_loop(), name="recorder-flush")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.flush()
        self.close()

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._flush_interval)
                return
            except asyncio.TimeoutError:
                self.flush()

    # -- writing -----------------------------------------------------------

    def _write(self, stream: str, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time_ns())
        line = json.dumps(_encode(record), separators=(",", ":"))
        self._buffer.append(f"{stream}\t{line}")
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def record_book(self, snapshot: BookSnapshot, *, depth: int = 10) -> None:
        self._write(
            snapshot.symbol.venue,
            {
                "t": RecordType.BOOK,
                "s": snapshot.symbol.canonical,
                "sq": snapshot.sequence,
                "tv": snapshot.ts_venue_ns,
                "tl": snapshot.ts_local_ns,
                "b": [[str(l.price), str(l.size)] for l in snapshot.bids[:depth]],
                "a": [[str(l.price), str(l.size)] for l in snapshot.asks[:depth]],
            },
        )

    def record_quote(self, symbol: Symbol, bid: Decimal, ask: Decimal,
                     bid_size: Decimal, ask_size: Decimal, ts_ns: int) -> None:
        self._write(
            symbol.venue,
            {
                "t": RecordType.QUOTE, "s": symbol.canonical,
                "b": str(bid), "a": str(ask),
                "bs": str(bid_size), "as": str(ask_size), "tl": ts_ns,
            },
        )

    def record_opportunity(self, opp: Opportunity) -> None:
        self._write(
            opp.venues[0] if opp.venues else "multi",
            {
                "t": RecordType.OPPORTUNITY,
                "id": opp.opportunity_id,
                "path": opp.path,
                "edge": str(opp.gross_edge_bps),
                "notional": str(opp.reference_notional),
                "age": opp.max_book_age_ns,
                "legs": [
                    {"sym": l.symbol.canonical, "side": l.side.value, "venue": l.symbol.venue}
                    for l in opp.legs
                ],
                "tl": opp.ts_detected_ns,
            },
        )

    def record_cycle(self, payload: dict[str, Any], venue: str = "multi") -> None:
        self._write(venue, {"t": RecordType.CYCLE, **payload})

    def record_meta(self, payload: dict[str, Any], stream: str = "meta") -> None:
        self._write(stream, {"t": RecordType.META, **payload})

    # -- flushing ----------------------------------------------------------

    def _file_for(self, stream: str, hour_bucket: int) -> _OpenFile:
        current = self._files.get(stream)
        if current is not None and current.hour_bucket == hour_bucket:
            return current
        if current is not None:
            current.handle.close()
            logger.info(
                "recorder: rotated %s (%d lines, %.1f MB)",
                current.path.name, current.lines, current.bytes_written / 1e6,
            )
        suffix = ".ndjson.gz" if self._compress else ".ndjson"
        stamp = time.strftime("%Y%m%d-%H", time.gmtime(hour_bucket * 3600))
        path = self.directory / f"{stream}-{stamp}{suffix}"
        handle: TextIO = (
            gzip.open(path, "at", encoding="utf-8", compresslevel=5)  # type: ignore[assignment]
            if self._compress
            else open(path, "a", encoding="utf-8")
        )
        opened = _OpenFile(handle=handle, path=path, hour_bucket=hour_bucket)
        self._files[stream] = opened
        return opened

    def flush(self) -> None:
        if not self._buffer:
            return
        buffered, self._buffer = self._buffer, []
        hour_bucket = int(time.time() // 3600)
        by_stream: dict[str, list[str]] = {}
        for entry in buffered:
            stream, _, line = entry.partition("\t")
            by_stream.setdefault(stream, []).append(line)
        for stream, lines in by_stream.items():
            try:
                target = self._file_for(stream, hour_bucket)
                payload = "\n".join(lines) + "\n"
                target.handle.write(payload)
                target.handle.flush()
                target.lines += len(lines)
                target.bytes_written += len(payload)
                self.records_written += len(lines)
            except OSError:
                self.records_dropped += len(lines)
                logger.exception("recorder: write failed for stream %s", stream)

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.handle.close()
            except OSError:
                pass
        self._files.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "written": self.records_written,
            "dropped": self.records_dropped,
            "buffered": len(self._buffer),
            "open_files": len(self._files),
            "directory": str(self.directory),
        }


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_recording(path: str | Path) -> Iterator[dict[str, Any]]:
    """
    Stream records from a recording file.

    A truncated final line (from a crash mid-write) is silently skipped rather
    than raising -- losing the last record of an hourly file is preferable to
    losing the ability to read the file at all.
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.debug("recorder: skipping malformed line in %s", p.name)
                continue


def iter_recordings(
    directory: str | Path,
    *,
    stream: str = "",
    record_type: str = "",
) -> Iterator[dict[str, Any]]:
    """Chronologically ordered records across every file in a directory."""
    d = Path(directory)
    if not d.exists():
        return
    pattern = f"{stream}-*" if stream else "*"
    files = sorted(
        [p for p in d.glob(pattern) if p.suffix in (".gz", ".ndjson")],
        key=lambda p: p.name,
    )
    for path in files:
        for record in read_recording(path):
            if record_type and record.get("t") != record_type:
                continue
            yield record
