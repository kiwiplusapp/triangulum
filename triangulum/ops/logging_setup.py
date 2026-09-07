"""
Structured logging.

Two formats. Human format for a terminal, JSON for anything that ships logs
somewhere. The JSON path exists because when this engine does something
expensive at 04:12 UTC you want to grep for the cycle id across every component
that touched it, and `grep '"cycle_id":"cyc-abc"'` beats reading prose.

A ring-buffer handler feeds the dashboard's event stream without a second
logging path, so what you see in the HUD is exactly what is in the log file.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque

__all__ = ["setup_logging", "RingBufferHandler", "JsonFormatter", "HumanFormatter"]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("tri_"):
                payload[key[4:]] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Terminal format with ANSI colour when the stream is a TTY."""

    COLORS = {
        "DEBUG": "\033[38;5;244m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;203m",
        "CRITICAL": "\033[1;38;5;196m",
    }
    RESET = "\033[0m"

    def __init__(self, *, color: bool = True) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        millis = int(record.msecs)
        name = record.name.replace("triangulum.", "")
        level = record.levelname[:4]
        if self.color:
            tint = self.COLORS.get(record.levelname, "")
            level = f"{tint}{level}{self.RESET}"
            name = f"\033[38;5;245m{name}{self.RESET}"
        line = f"{stamp}.{millis:03d} {level} {name:<22} {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class RingBufferHandler(logging.Handler):
    """Keeps the last N records for the dashboard, and optionally forwards them."""

    def __init__(self, capacity: int = 1000,
                 sink: Callable[[str, str, int], None] | None = None) -> None:
        super().__init__()
        self.buffer: Deque[dict] = deque(maxlen=capacity)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "ts": int(record.created * 1e9),
            }
            self.buffer.append(entry)
            if self.sink is not None:
                self.sink(entry["level"], entry["message"], entry["ts"])
        except Exception:      # pragma: no cover - a logging handler must never raise
            pass

    def recent(self, count: int = 200) -> list[dict]:
        return list(self.buffer)[-count:]


_ring: RingBufferHandler | None = None


def setup_logging(
    *,
    level: str = "INFO",
    json_output: bool = False,
    log_file: str = "",
    ring_capacity: int = 1000,
) -> RingBufferHandler:
    global _ring

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(
        JsonFormatter() if json_output else HumanFormatter(color=sys.stderr.isatty())
    )
    root.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    _ring = RingBufferHandler(capacity=ring_capacity)
    root.addHandler(_ring)

    # These are noisy and never actionable.
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    return _ring


def get_ring_handler() -> RingBufferHandler | None:
    return _ring
