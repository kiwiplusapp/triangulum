"""
Alerting.

Deliberately narrow: only conditions a human must act on within minutes.

An alerting system that fires on everything is an alerting system nobody reads,
and the failure mode it produces -- a real CRITICAL buried in noise -- is worse
than no alerting at all. So the default threshold is CRITICAL, and there are
exactly three of those: a stranded position, a tripped kill switch, and a
reconciliation mismatch.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["Alerter", "Alert"]


@dataclass(slots=True)
class Alert:
    severity: str
    title: str
    message: str
    ts_ns: int = 0
    context: dict | None = None

    def format(self) -> str:
        lines = [f"[{self.severity.upper()}] {self.title}", self.message]
        if self.context:
            lines.append(
                "\n".join(f"  {k}: {v}" for k, v in self.context.items())
            )
        return "\n".join(lines)


class Alerter:
    """Sends alerts to a webhook and/or Telegram, with deduplication."""

    def __init__(
        self,
        *,
        webhook_url: str = "",
        telegram_token: str = "",
        telegram_chat_id: str = "",
        min_severity: str = "CRITICAL",
        dedupe_window_sec: float = 300.0,
    ) -> None:
        self.webhook_url = webhook_url
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.min_severity = min_severity.upper()
        self.dedupe_window_sec = dedupe_window_sec
        self._last_sent: dict[str, float] = {}
        self.sent = 0
        self.suppressed = 0
        self.failures = 0

    _ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

    def _should_send(self, alert: Alert) -> bool:
        if self._ORDER.get(alert.severity.upper(), 0) < self._ORDER.get(self.min_severity, 50):
            return False
        key = f"{alert.severity}:{alert.title}"
        now = time.time()
        last = self._last_sent.get(key, 0.0)
        if now - last < self.dedupe_window_sec:
            self.suppressed += 1
            return False
        self._last_sent[key] = now
        return True

    def send(self, alert: Alert) -> bool:
        if not self._should_send(alert):
            return False
        delivered = False
        if self.webhook_url:
            delivered |= self._post(self.webhook_url, {
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "context": alert.context or {},
                "ts": alert.ts_ns or time.time_ns(),
            })
        if self.telegram_token and self.telegram_chat_id:
            delivered |= self._post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                {"chat_id": self.telegram_chat_id, "text": alert.format()},
            )
        if delivered:
            self.sent += 1
        return delivered

    def _post(self, url: str, payload: dict) -> bool:
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5):
                return True
        except (urllib.error.URLError, OSError) as exc:
            self.failures += 1
            # An alerting failure must never propagate: the alert exists because
            # something is already wrong, and raising here would compound it.
            logger.error("alert delivery failed: %s", exc)
            return False

    def critical(self, title: str, message: str, **context: Any) -> bool:
        return self.send(Alert("CRITICAL", title, message, time.time_ns(), context))

    def warning(self, title: str, message: str, **context: Any) -> bool:
        return self.send(Alert("WARNING", title, message, time.time_ns(), context))

    def stats(self) -> dict[str, Any]:
        return {
            "sent": self.sent,
            "suppressed": self.suppressed,
            "failures": self.failures,
            "min_severity": self.min_severity,
            "channels": [
                c for c, on in (
                    ("webhook", bool(self.webhook_url)),
                    ("telegram", bool(self.telegram_token and self.telegram_chat_id)),
                ) if on
            ],
        }
