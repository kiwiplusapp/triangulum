"""
Model persistence.

A model that vanishes on restart is not a model that learns. This engine is
meant to run for months, accumulating evidence; losing it to a container restart
would reset the system to ignorance every time.

Requirements this satisfies:

**Atomic writes.** Write to a temp file, fsync, rename. A crash mid-checkpoint
must never leave a half-written model that the engine loads and trades on.

**Versioned, never overwritten in place.** Each checkpoint is a new numbered
file, and ``latest`` is a pointer. When a model degrades you can roll back to a
known-good one instead of discovering it is unrecoverable.

**Metadata alongside weights.** Sample count, loss, calibration error and the
config hash. A weights file without its provenance is unusable six weeks later.

**Config-hash guarding.** If the feature set changed, the stored weights refer
to different features and loading them is worse than starting fresh. The hash
catches this and the loader refuses rather than silently producing nonsense.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

__all__ = ["ModelStore", "Checkpoint"]


@dataclass(slots=True)
class Checkpoint:
    version: int
    path: Path
    created_at: float
    samples: int
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_hash: str = ""

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "path": str(self.path),
            "created_at": self.created_at,
            "samples": self.samples,
            "metrics": self.metrics,
            "feature_hash": self.feature_hash,
        }


class ModelStore:
    """Versioned, atomic model checkpointing."""

    def __init__(self, directory: str | Path, *, keep_versions: int = 10) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_versions = keep_versions
        self._index_path = self.directory / "index.json"
        self._index: dict[str, list[dict]] = self._load_index()

    def _load_index(self) -> dict[str, list[dict]]:
        if not self._index_path.exists():
            return {}
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("model index unreadable; starting a fresh one")
            return {}

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
        tmp.replace(self._index_path)

    @staticmethod
    def feature_hash(feature_names: tuple[str, ...]) -> str:
        """
        Fingerprint of the feature set.

        Loading weights trained on a different feature set produces confident
        nonsense -- the weights are indexed by hashed feature name, so a renamed
        or removed feature silently maps to a stale coordinate. Better to
        refuse.
        """
        joined = "|".join(sorted(feature_names))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    # -- writing -----------------------------------------------------------

    def save(
        self,
        name: str,
        payload: Mapping[str, Any],
        *,
        samples: int = 0,
        metrics: Mapping[str, Any] | None = None,
        feature_hash: str = "",
    ) -> Checkpoint:
        versions = self._index.setdefault(name, [])
        version = (versions[-1]["version"] + 1) if versions else 1
        path = self.directory / f"{name}.v{version:05d}.json"

        document = {
            "name": name,
            "version": version,
            "created_at": time.time(),
            "samples": samples,
            "metrics": dict(metrics or {}),
            "feature_hash": feature_hash,
            "payload": payload,
        }

        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        tmp.replace(path)

        # ``latest`` is a copy rather than a symlink: symlinks do not survive
        # every container filesystem, and a broken pointer is worse than a
        # duplicated file.
        latest = self.directory / f"{name}.latest.json"
        shutil.copy2(path, latest)

        checkpoint = Checkpoint(
            version=version, path=path, created_at=document["created_at"],
            samples=samples, metrics=dict(metrics or {}), feature_hash=feature_hash,
        )
        versions.append(checkpoint.to_dict())
        self._prune(name)
        self._save_index()
        logger.info(
            "checkpointed %s v%d (%d samples) -> %s", name, version, samples, path.name
        )
        return checkpoint

    def _prune(self, name: str) -> None:
        versions = self._index.get(name, [])
        while len(versions) > self.keep_versions:
            oldest = versions.pop(0)
            try:
                Path(oldest["path"]).unlink(missing_ok=True)
            except OSError:
                pass

    # -- reading -----------------------------------------------------------

    def load(
        self, name: str, *, version: int | None = None, expect_feature_hash: str = "",
    ) -> dict | None:
        """
        Load a checkpoint's payload, or None when unavailable or incompatible.

        Returning None rather than raising is deliberate: a missing or stale
        model is a normal startup condition (first run, changed features), and
        the correct response is to begin learning fresh, not to fail to boot.
        """
        if version is not None:
            path = self.directory / f"{name}.v{version:05d}.json"
        else:
            path = self.directory / f"{name}.latest.json"
        if not path.exists():
            logger.info("no checkpoint for %s; starting fresh", name)
            return None

        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("checkpoint %s is unreadable (%s); starting fresh", path.name, exc)
            return None

        stored_hash = document.get("feature_hash", "")
        if expect_feature_hash and stored_hash and stored_hash != expect_feature_hash:
            logger.warning(
                "checkpoint %s was trained on a different feature set "
                "(%s != %s); refusing to load it and starting fresh",
                path.name, stored_hash, expect_feature_hash,
            )
            return None

        logger.info(
            "loaded %s v%s (%s samples)",
            name, document.get("version"), document.get("samples"),
        )
        return document.get("payload")

    def history(self, name: str) -> list[dict]:
        return list(self._index.get(name, []))

    def rollback(self, name: str, version: int) -> bool:
        """Promote an older checkpoint to ``latest``."""
        path = self.directory / f"{name}.v{version:05d}.json"
        if not path.exists():
            return False
        shutil.copy2(path, self.directory / f"{name}.latest.json")
        logger.warning("rolled %s back to v%d", name, version)
        return True

    def stats(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "models": {
                name: {
                    "versions": len(versions),
                    "latest_version": versions[-1]["version"] if versions else 0,
                    "latest_samples": versions[-1]["samples"] if versions else 0,
                }
                for name, versions in self._index.items()
            },
        }
