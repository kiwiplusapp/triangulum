"""
The thesis engine: the SCAN -> MACRO -> FLOW -> BIAS stage.

Calls Claude with the market brief and a strict output schema, then commits the
result to the journal. The interesting engineering here is not the API call --
it is everything around it that stops a language model's fluency from being
mistaken for accuracy.

**Structured output, not prose parsing.** The response is validated against the
Pydantic schema by the SDK. There is no regex over free text, so a thesis that
does not fit the falsifiable shape cannot enter the journal at all.

**Abstention is respected.** The model can decline. An abstention is recorded
in the run log and never enters the scored track record.

**One thesis per scan, committed immediately.** Not a list of five to choose
from -- picking the best of five after seeing them is a subtle way to cheat the
track record, because the selection uses information the score will not credit.

**Adaptive thinking, high effort.** This is a hard reasoning task where an
error costs money, which is exactly the shape the effort parameter exists for.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from vault.thesis.journal import ThesisJournal
from vault.thesis.prompts import ABSTAIN_SENTINEL, SYSTEM_PROMPT, build_user_prompt
from vault.thesis.schema import PYDANTIC, Thesis, ThesisRecord, thesis_json_schema

logger = logging.getLogger(__name__)

__all__ = ["ThesisEngine", "ThesisResult", "DEFAULT_MODEL"]

DEFAULT_MODEL = "claude-opus-5"


@dataclass(slots=True)
class ThesisResult:
    """The outcome of one BIAS stage."""

    abstained: bool
    thesis: Thesis | None = None
    record: ThesisRecord | None = None
    reason: str = ""
    raw_response: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.thesis is not None or self.abstained

    def summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        if self.abstained:
            return f"ABSTAINED: {self.reason}"
        t = self.thesis
        asset = t.asset.value if hasattr(t.asset, "value") else t.asset
        direction = t.direction.value if hasattr(t.direction, "value") else t.direction
        horizon = t.horizon.value if hasattr(t.horizon, "value") else t.horizon
        return (
            f"{asset} {direction.upper()} over {horizon} at p={t.probability:.2f} "
            f"(expect {t.magnitude_pct:.1f}%, invalidate at {t.invalidation_pct:.1f}%)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstained": self.abstained,
            "thesis": self.thesis.to_dict() if self.thesis else None,
            "thesis_id": self.record.thesis_id if self.record else None,
            "reason": self.reason,
            "model": self.model,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "cache_read": self.cache_read_tokens,
            },
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


class ThesisEngine:
    """Turns a market brief into a committed, falsifiable call."""

    def __init__(
        self,
        journal: ThesisJournal,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_tokens: int = 8000,
        api_key: str = "",
        dry_run: bool = False,
    ) -> None:
        self.journal = journal
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: Any = None

        self.calls = 0
        self.abstentions = 0
        self.failures = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0

    # -- client ------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the anthropic SDK is required for the thesis engine. "
                "Install it with: pip install anthropic"
            ) from exc
        # A bare constructor also picks up an `ant auth login` profile, so an
        # unset ANTHROPIC_API_KEY is not necessarily missing credentials.
        self._client = (
            anthropic.Anthropic(api_key=self._api_key)
            if self._api_key else anthropic.Anthropic()
        )
        return self._client

    @property
    def has_credentials(self) -> bool:
        if self._api_key:
            return True
        return bool(os.environ.get("ANTHROPIC_AUTH_TOKEN")) or _has_cli_profile()

    # -- the call ----------------------------------------------------------

    def generate(
        self,
        brief: dict[str, Any],
        *,
        entry_prices: dict[str, float],
        calibration: dict[str, Any] | None = None,
        data_snapshot: str = "",
        commit: bool = True,
    ) -> ThesisResult:
        """
        Produce one thesis from the brief, and commit it.

        ``entry_prices`` must contain a current price for every asset the model
        may pick. A thesis about an asset with no price cannot be resolved, so
        it is rejected rather than committed -- an unresolvable call is exactly
        the kind of entry that inflates a track record without ever being
        graded.
        """
        self.calls += 1
        started = datetime.now(timezone.utc)
        brief_digest = _digest(brief)
        user_prompt = build_user_prompt(brief, calibration=calibration)

        if self.dry_run:
            return ThesisResult(
                abstained=True, reason="dry run -- no API call made",
                model=self.model,
            )

        try:
            parsed, raw_text, usage = self._call_model(user_prompt)
        except Exception as exc:
            self.failures += 1
            logger.exception("thesis generation failed")
            return ThesisResult(
                abstained=False, error=f"{type(exc).__name__}: {exc}", model=self.model,
            )

        latency = (datetime.now(timezone.utc) - started).total_seconds() * 1000
        self.total_input_tokens += usage.get("input", 0)
        self.total_output_tokens += usage.get("output", 0)
        self.total_cache_read_tokens += usage.get("cache_read", 0)

        result = ThesisResult(
            abstained=False, raw_response=raw_text, model=self.model,
            input_tokens=usage.get("input", 0), output_tokens=usage.get("output", 0),
            cache_read_tokens=usage.get("cache_read", 0), latency_ms=latency,
        )

        if parsed is None:
            self.failures += 1
            result.error = "model returned no parseable thesis"
            return result

        # Abstention: the model declined, using the sentinel.
        reasoning = getattr(parsed, "reasoning", "") or ""
        if reasoning.strip().upper().startswith(ABSTAIN_SENTINEL):
            self.abstentions += 1
            result.abstained = True
            result.reason = reasoning.strip()[len(ABSTAIN_SENTINEL):].strip(" .:-") or \
                "no thesis supported by the evidence"
            logger.info("abstained: %s", result.reason)
            return result

        result.thesis = parsed

        asset_key = parsed.asset.value if hasattr(parsed.asset, "value") else parsed.asset
        entry = entry_prices.get(asset_key)
        if entry is None or entry <= 0:
            self.failures += 1
            result.error = (
                f"no current price for {asset_key}; the thesis cannot be resolved "
                f"and will not be committed"
            )
            logger.error("%s", result.error)
            return result

        if commit:
            regime = brief.get("regime", {})
            result.record = self.journal.commit(
                parsed,
                entry_price=entry,
                regime=regime.get("quadrant", "unknown"),
                regime_confidence=float(regime.get("confidence", 0.0)),
                brief_digest=brief_digest,
                data_snapshot=data_snapshot,
                model=self.model,
            )
        return result

    def _call_model(self, user_prompt: str) -> tuple[Any, str, dict[str, int]]:
        client = self._get_client()

        # The system prompt is frozen and cached; the volatile brief follows the
        # breakpoint, so every scan re-reads ~2k cached tokens at ~10% cost.
        system = [{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }]

        if PYDANTIC:
            response = client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user_prompt}],
                output_format=Thesis,
            )
            parsed = response.parsed_output
        else:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {
                        "type": "json_schema",
                        "schema": thesis_json_schema(),
                    },
                },
                messages=[{"role": "user", "content": user_prompt}],
            )
            parsed = _parse_without_pydantic(response)

        raw_text = "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        )
        usage = {
            "input": getattr(response.usage, "input_tokens", 0),
            "output": getattr(response.usage, "output_tokens", 0),
            "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }
        # A refusal is an HTTP 200 with stop_reason set; check before reading.
        if getattr(response, "stop_reason", "") == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(
                f"model declined: {getattr(details, 'category', 'unknown')}"
            )
        return parsed, raw_text, usage

    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "calls": self.calls,
            "abstentions": self.abstentions,
            "failures": self.failures,
            "abstention_rate": round(self.abstentions / self.calls, 3) if self.calls else 0.0,
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "cache_read": self.total_cache_read_tokens,
                "cache_hit_rate": round(
                    self.total_cache_read_tokens
                    / max(1, self.total_input_tokens + self.total_cache_read_tokens),
                    3,
                ),
            },
            "has_credentials": self.has_credentials,
            "dry_run": self.dry_run,
        }


def _digest(brief: dict[str, Any]) -> str:
    """
    Stable hash of the brief the model saw.

    Committed with the thesis so a later reader can confirm which inputs
    produced which call -- and detect a brief that was regenerated between
    scan and commit.
    """
    payload = json.dumps(brief, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _parse_without_pydantic(response: Any) -> Any:
    """Fallback JSON extraction when pydantic is unavailable."""
    from vault.thesis.schema import _PlainThesis

    for block in response.content:
        if getattr(block, "type", "") != "text":
            continue
        text = block.text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            return _PlainThesis(json.loads(text[start:end + 1]))
        except json.JSONDecodeError:
            continue
    return None


def _has_cli_profile() -> bool:
    """Whether `ant auth login` has stored a profile the SDK can pick up."""
    from pathlib import Path
    return (Path.home() / ".config" / "anthropic").exists()
