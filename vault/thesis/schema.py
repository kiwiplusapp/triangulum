"""
The falsifiable thesis.

This schema is the most important file in the project, because it is what makes
the difference between a system and a horoscope.

A macro "call" is only worth recording if it can be shown to be WRONG. That
requires four things, and every one of them is a required field below:

    asset          what, exactly, is being predicted
    direction      up or down -- not "constructive", not "cautiously bullish"
    horizon        by when
    probability    with what stated confidence, as a number

Plus one field that most forecasting systems omit and that changes behaviour
more than any other:

    invalidation   the price level at which this thesis is dead

Without invalidation, a losing call gets quietly reinterpreted as "early". With
it, the thesis has a tripwire that fires before the horizon, and the resolver
marks it INVALIDATED rather than waiting to see if it comes back.

WHAT IS DELIBERATELY FORBIDDEN

The schema rejects hedged language by construction. There is no "neutral"
direction, no probability outside [0.05, 0.95] (a 0.99 call is either a lie or
a trivially true one), and `reasoning` is capped so it cannot become a wall of
qualifications that would make any outcome look anticipated.

The `key_risk` field is required and must name a specific, observable event --
because a forecaster who cannot say what would change their mind has not made a
forecast.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC = True
except ImportError:                                    # pragma: no cover
    PYDANTIC = False
    BaseModel = object                                 # type: ignore[misc,assignment]

    def Field(*args, **kwargs):                        # type: ignore[misc]
        return kwargs.get("default")

    def field_validator(*args, **kwargs):              # type: ignore[misc]
        return lambda fn: fn

    def model_validator(*args, **kwargs):              # type: ignore[misc]
        return lambda fn: fn

__all__ = [
    "Direction", "Horizon", "TradeableAsset", "Thesis", "ThesisRecord",
    "Outcome", "HORIZON_DAYS", "thesis_json_schema",
]


class Direction(str, Enum):
    """
    Up or down. There is no third option, on purpose.

    "Neutral" is not a forecast -- it is the absence of one, and allowing it
    lets a system accumulate a track record of calls that can never be scored.
    If the model has no view, it should decline to produce a thesis at all,
    which is a separate and honest outcome (``abstained``).
    """

    UP = "up"
    DOWN = "down"


class Horizon(str, Enum):
    ONE_DAY = "1d"
    THREE_DAYS = "3d"
    ONE_WEEK = "1w"
    TWO_WEEKS = "2w"
    ONE_MONTH = "1m"
    THREE_MONTHS = "3m"


HORIZON_DAYS: dict[str, int] = {
    Horizon.ONE_DAY: 1,
    Horizon.THREE_DAYS: 3,
    Horizon.ONE_WEEK: 7,
    Horizon.TWO_WEEKS: 14,
    Horizon.ONE_MONTH: 30,
    Horizon.THREE_MONTHS: 90,
}


class TradeableAsset(str, Enum):
    """
    The universe a thesis may be about.

    Restricted to instruments with a free, unambiguous, daily settlement price,
    because a forecast that cannot be resolved automatically will not be
    resolved at all. That constraint is the reason this list is short.
    """

    SP500 = "SP500"
    NASDAQ100 = "NASDAQ100"
    US10Y_YIELD = "DGS10"
    US2Y_YIELD = "DGS2"
    DOLLAR_INDEX = "DTWEXBGS"
    WTI_OIL = "DCOILWTICO"
    HY_SPREAD = "BAMLH0A0HYM2"
    VIX = "VIXCLS"
    BITCOIN = "BTCUSD"
    ETHEREUM = "ETHUSD"


class Outcome(str, Enum):
    PENDING = "pending"
    CORRECT = "correct"
    WRONG = "wrong"
    INVALIDATED = "invalidated"     # stop level hit before the horizon
    UNRESOLVABLE = "unresolvable"   # no price data at the horizon


class Thesis(BaseModel):
    """
    One falsifiable directional call. This is what the model must produce.

    Every field is required. A model that wants to hedge has to do it by
    lowering ``probability``, which is exactly where hedging belongs -- it makes
    the hedge measurable instead of rhetorical.
    """

    asset: TradeableAsset = Field(
        description="The instrument this thesis is about. Must be one of the enum values."
    )
    direction: Direction = Field(
        description="up or down over the horizon. There is no neutral option."
    )
    horizon: Horizon = Field(
        description="The window over which this resolves."
    )
    probability: float = Field(
        ge=0.05, le=0.95,
        description=(
            "Your genuine probability that the direction is correct, in [0.05, 0.95]. "
            "This is scored with a Brier score against outcomes, so systematic "
            "overconfidence is measured and penalised. 0.5 means no view."
        ),
    )
    magnitude_pct: float = Field(
        ge=0.0, le=60.0,
        description=(
            "Expected absolute move over the horizon, in percent. Used for sizing, "
            "not for scoring direction."
        ),
    )
    invalidation_pct: float = Field(
        gt=0.0, le=30.0,
        description=(
            "Adverse move from the entry price, in percent, at which this thesis "
            "is dead. If price moves this far against the direction before the "
            "horizon, the call is marked INVALIDATED rather than left to run."
        ),
    )
    regime_dependency: str = Field(
        max_length=200,
        description=(
            "The macro regime this thesis depends on, and what a change in it "
            "would mean for the call."
        ),
    )
    key_risk: str = Field(
        max_length=250,
        description=(
            "The single most likely specific, OBSERVABLE event that would make "
            "this wrong. Must name something checkable -- a data release, a "
            "policy decision, a level being breached. Not 'markets could move "
            "against us'."
        ),
    )
    reasoning: str = Field(
        max_length=1200,
        description=(
            "The causal chain, in at most 1200 characters. Cite the specific "
            "series and readings that drive it. Brevity is enforced so the "
            "reasoning cannot become a hedge that makes every outcome look "
            "anticipated."
        ),
    )
    primary_evidence: list[str] = Field(
        min_length=1, max_length=5,
        description=(
            "1-5 series keys from the brief that most drive this call, e.g. "
            "['T10Y3M', 'ICSA']. Used to attribute performance to inputs later."
        ),
    )

    if PYDANTIC:
        @field_validator("key_risk")
        @classmethod
        def _risk_must_be_specific(cls, v: str) -> str:
            vague = (
                "market conditions", "unexpected events", "volatility",
                "uncertainty", "black swan", "things could change",
                "macro factors", "various factors",
            )
            lowered = v.lower()
            if any(phrase in lowered for phrase in vague) and len(v) < 80:
                raise ValueError(
                    "key_risk must name a specific observable event (a release, a "
                    "decision, a level), not a generic gesture at uncertainty"
                )
            return v

        @model_validator(mode="after")
        def _coherent(self):
            # A high-confidence call with a tight stop is internally inconsistent:
            # it says "very likely right" and "kill it on a small adverse move".
            if self.probability >= 0.75 and self.invalidation_pct < self.magnitude_pct * 0.4:
                raise ValueError(
                    f"probability {self.probability} is high but invalidation "
                    f"{self.invalidation_pct}% is tight relative to the expected "
                    f"move of {self.magnitude_pct}% -- these disagree"
                )
            return self

    # -- derived -----------------------------------------------------------

    @property
    def horizon_days(self) -> int:
        return HORIZON_DAYS[self.horizon]

    @property
    def sign(self) -> int:
        return 1 if self.direction == Direction.UP else -1

    @property
    def edge_over_coinflip(self) -> float:
        return self.probability - 0.5

    def resolves_on(self, from_date: date) -> date:
        return from_date + timedelta(days=self.horizon_days)

    def invalidation_price(self, entry: float) -> float:
        """The price at which this thesis is dead."""
        move = entry * self.invalidation_pct / 100
        return entry - move if self.direction == Direction.UP else entry + move

    def target_price(self, entry: float) -> float:
        move = entry * self.magnitude_pct / 100
        return entry + move if self.direction == Direction.UP else entry - move

    def to_dict(self) -> dict[str, Any]:
        if PYDANTIC:
            return json.loads(self.model_dump_json())
        return dict(self.__dict__)


def thesis_json_schema() -> dict[str, Any]:
    """
    JSON schema for the raw-schema path, when pydantic is unavailable.

    Kept in sync with the model by construction: it is generated from it when
    pydantic is present, and hand-written only as the fallback.
    """
    if PYDANTIC:
        return Thesis.model_json_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "asset", "direction", "horizon", "probability", "magnitude_pct",
            "invalidation_pct", "regime_dependency", "key_risk", "reasoning",
            "primary_evidence",
        ],
        "properties": {
            "asset": {"type": "string", "enum": [a.value for a in TradeableAsset]},
            "direction": {"type": "string", "enum": ["up", "down"]},
            "horizon": {"type": "string", "enum": [h.value for h in Horizon]},
            "probability": {"type": "number", "minimum": 0.05, "maximum": 0.95},
            "magnitude_pct": {"type": "number", "minimum": 0.0, "maximum": 60.0},
            "invalidation_pct": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 30.0},
            "regime_dependency": {"type": "string", "maxLength": 200},
            "key_risk": {"type": "string", "maxLength": 250},
            "reasoning": {"type": "string", "maxLength": 1200},
            "primary_evidence": {
                "type": "array", "items": {"type": "string"},
                "minItems": 1, "maxItems": 5,
            },
        },
    }


class ThesisRecord:
    """
    A committed thesis: the call plus everything needed to grade it later.

    Immutable once created. The `content_hash` covers the thesis and the market
    state it was made against, so a later reader can verify that neither was
    edited after the outcome became known.
    """

    __slots__ = (
        "thesis_id", "thesis", "created_at", "entry_price", "resolve_on",
        "regime", "regime_confidence", "brief_digest", "data_snapshot",
        "model", "prev_hash", "content_hash", "outcome", "resolved_at",
        "exit_price", "realized_pct", "invalidated_on", "notes",
    )

    def __init__(
        self,
        thesis_id: str,
        thesis: Thesis,
        created_at: datetime,
        entry_price: float,
        resolve_on: date,
        regime: str,
        regime_confidence: float,
        brief_digest: str,
        data_snapshot: str,
        model: str,
        prev_hash: str,
    ) -> None:
        self.thesis_id = thesis_id
        self.thesis = thesis
        self.created_at = created_at
        # Coerce to float at construction. The hash is computed over a JSON
        # rendering, and json.dumps(5600) is "5600" while json.dumps(5600.0) is
        # "5600.0" -- so an int entry price hashed at commit time and a float
        # one after a round-trip through from_dict produce DIFFERENT hashes.
        # The chain then reports tampering on an untouched record, which is
        # worse than no integrity check at all: it cries wolf.
        self.entry_price = float(entry_price)
        self.resolve_on = resolve_on
        self.regime = regime
        self.regime_confidence = regime_confidence
        self.brief_digest = brief_digest
        self.data_snapshot = data_snapshot
        self.model = model
        self.prev_hash = prev_hash
        self.content_hash = self._compute_hash()

        self.outcome: str = Outcome.PENDING
        self.resolved_at: datetime | None = None
        self.exit_price: float | None = None
        self.realized_pct: float | None = None
        self.invalidated_on: date | None = None
        self.notes: str = ""

    def _compute_hash(self) -> str:
        """
        SHA-256 over the committed content, chained to the previous record.

        Chaining is what makes the journal tamper-EVIDENT rather than merely
        append-only: altering any earlier record changes its hash, which breaks
        every subsequent link. The outcome fields are deliberately excluded --
        they are written after resolution, and including them would make the
        chain unverifiable by design.
        """
        payload = json.dumps({
            "thesis_id": self.thesis_id,
            "thesis": self.thesis.to_dict(),
            "created_at": self.created_at.isoformat(),
            # Fixed precision everywhere a float enters the hash, so a
            # round-trip through JSON cannot change the rendering.
            "entry_price": round(float(self.entry_price), 10),
            "resolve_on": self.resolve_on.isoformat(),
            "regime": self.regime,
            "regime_confidence": round(float(self.regime_confidence), 6),
            "brief_digest": self.brief_digest,
            "data_snapshot": self.data_snapshot,
            "model": self.model,
            "prev_hash": self.prev_hash,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        return self._compute_hash() == self.content_hash

    @property
    def is_resolved(self) -> bool:
        return self.outcome != Outcome.PENDING

    @property
    def is_scoreable(self) -> bool:
        """
        Whether this record contributes to the calibration score.

        INVALIDATED counts as WRONG for scoring: the thesis said the move would
        go one way and it went far enough the other way to hit the stop. Only
        UNRESOLVABLE is excluded, and that is a data failure, not a forecast.
        """
        return self.outcome in (Outcome.CORRECT, Outcome.WRONG, Outcome.INVALIDATED)

    @property
    def was_correct(self) -> bool | None:
        if not self.is_scoreable:
            return None
        return self.outcome == Outcome.CORRECT

    @property
    def brier(self) -> float | None:
        """
        Brier score for this single call: (probability - actual)^2.

        Lower is better. 0.25 is what you get by always saying 0.5. A score
        above 0.25 means the stated confidence is actively misleading -- worse
        than admitting no view.
        """
        if not self.is_scoreable:
            return None
        actual = 1.0 if self.outcome == Outcome.CORRECT else 0.0
        return (self.thesis.probability - actual) ** 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "thesis_id": self.thesis_id,
            "thesis": self.thesis.to_dict(),
            "created_at": self.created_at.isoformat(),
            # Fixed precision everywhere a float enters the hash, so a
            # round-trip through JSON cannot change the rendering.
            "entry_price": round(float(self.entry_price), 10),
            "resolve_on": self.resolve_on.isoformat(),
            "regime": self.regime,
            "regime_confidence": round(float(self.regime_confidence), 6),
            "brief_digest": self.brief_digest,
            "data_snapshot": self.data_snapshot,
            "model": self.model,
            "prev_hash": self.prev_hash,
            "content_hash": self.content_hash,
            "outcome": self.outcome,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "exit_price": self.exit_price,
            "realized_pct": self.realized_pct,
            "invalidated_on": self.invalidated_on.isoformat() if self.invalidated_on else None,
            "brier": self.brier,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ThesisRecord":
        thesis = (
            Thesis(**raw["thesis"]) if PYDANTIC
            else _PlainThesis(raw["thesis"])            # type: ignore[assignment]
        )
        record = cls.__new__(cls)
        record.thesis_id = raw["thesis_id"]
        record.thesis = thesis
        record.created_at = datetime.fromisoformat(raw["created_at"])
        record.entry_price = float(raw["entry_price"])
        record.resolve_on = date.fromisoformat(raw["resolve_on"])
        record.regime = raw.get("regime", "")
        record.regime_confidence = float(raw.get("regime_confidence", 0.0))
        record.brief_digest = raw.get("brief_digest", "")
        record.data_snapshot = raw.get("data_snapshot", "")
        record.model = raw.get("model", "")
        record.prev_hash = raw.get("prev_hash", "")
        record.content_hash = raw.get("content_hash", "")
        record.outcome = raw.get("outcome", Outcome.PENDING)
        resolved = raw.get("resolved_at")
        record.resolved_at = datetime.fromisoformat(resolved) if resolved else None
        record.exit_price = raw.get("exit_price")
        record.realized_pct = raw.get("realized_pct")
        invalidated = raw.get("invalidated_on")
        record.invalidated_on = date.fromisoformat(invalidated) if invalidated else None
        record.notes = raw.get("notes", "")
        return record

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ThesisRecord({self.thesis_id} {self.thesis.asset} "
            f"{self.thesis.direction} p={self.thesis.probability:.2f} "
            f"{self.outcome})"
        )


class _PlainThesis:
    """Duck-typed stand-in used only when pydantic is unavailable."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.__dict__.update(data)

    @property
    def horizon_days(self) -> int:
        return HORIZON_DAYS.get(self.horizon, 7)

    @property
    def sign(self) -> int:
        return 1 if self.direction == "up" else -1

    def invalidation_price(self, entry: float) -> float:
        move = entry * self.invalidation_pct / 100
        return entry - move if self.direction == "up" else entry + move

    def target_price(self, entry: float) -> float:
        move = entry * self.magnitude_pct / 100
        return entry + move if self.direction == "up" else entry - move

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)
