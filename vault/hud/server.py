"""
HUD server for Vault.

Reuses Triangulum's stdlib-only HTTP server -- same reasoning, same code, no
duplication. Only the routes and the static root differ.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

from triangulum.api.server import DashboardServer

logger = logging.getLogger(__name__)

__all__ = ["VaultServer"]

_STATIC_ROOT = Path(__file__).parent


class VaultServer(DashboardServer):
    """Serves the Vault HUD and exposes run/state endpoints."""

    def __init__(self, vault, *, host: str = "127.0.0.1", port: int = 8899,
                 auth_token: str = "") -> None:
        super().__init__(
            host=host, port=port, auth_token=auth_token,
            snapshot_provider=vault.snapshot,
            broadcast_interval_ms=2000.0,
        )
        self.vault = vault
        self.static_root = _STATIC_ROOT
        self._run_lock = threading.Lock()

        self.get_routes["/api/state"] = vault.snapshot
        self.get_routes["/api/journal"] = self._journal
        self.get_routes["/api/brief"] = lambda: (
            vault.last_run.brief if vault.last_run else {}
        )
        self.get_routes["/api/signals"] = self._signals
        self.get_routes["/api/learning"] = self._learning
        self.post_routes["/api/run"] = self._run
        self.post_routes["/api/learn"] = self._learn

    def _run(self, _body: dict) -> dict[str, Any]:
        """
        Trigger one pipeline cycle.

        Serialised behind a lock: two concurrent runs would interleave journal
        writes and produce a chain whose prev_hash links do not match, which
        would look exactly like tampering.
        """
        if not self._run_lock.acquire(blocking=False):
            return {"error": "a run is already in progress"}
        try:
            result = self.vault.run()
            return result.to_dict()
        except Exception as exc:
            logger.exception("run failed")
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._run_lock.release()

    def _signals(self) -> dict[str, Any]:
        from vault.signals.library import evaluate_all

        if not self.vault.series:
            self.vault.scan()
        readings = evaluate_all(self.vault.series)
        return {
            "usable": sum(1 for r in readings if r.usable),
            "total": len(readings),
            "signals": [r.to_dict() for r in readings],
        }

    def _learning(self) -> dict[str, Any]:
        if self.vault.learning is None:
            return {
                "trained": False,
                "reason": (
                    "nothing learned yet -- POST /api/learn, or run "
                    "`vault learn`, to score the signals and fit the models"
                ),
            }
        return {"trained": True, **self.vault.learning.to_dict()}

    def _learn(self, body: dict) -> dict[str, Any]:
        """
        Fit the signal scorecard and the models.

        Behind the same lock as a run: learning replaces the predictor the
        run path reads, and interleaving the two would let a cycle size a
        call against a half-swapped model.
        """
        if not self._run_lock.acquire(blocking=False):
            return {"error": "a run or a learning pass is already in progress"}
        try:
            result = self.vault.learn(
                target=str(body.get("target", "SP500")),
                horizon_days=int(body.get("horizon_days", 21)),
                step_days=int(body.get("step_days", 5)),
                lookback_days=int(body.get("lookback_days", 2000)),
                folds=int(body.get("folds", 5)),
                epochs=int(body.get("epochs", 150)),
            )
            return result.to_dict()
        except Exception as exc:
            logger.exception("learning failed")
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._run_lock.release()

    def _journal(self) -> dict[str, Any]:
        return {
            "stats": self.vault.journal.stats(),
            "records": [r.to_dict() for r in list(self.vault.journal)[-100:]],
        }
