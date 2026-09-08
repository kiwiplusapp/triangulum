"""
The pipeline: SCAN -> MACRO -> FLOW -> BIAS -> COMMIT -> RESOLVE -> SCORE -> GATE.

One run of this is one complete cycle of the system. The ordering is not
arbitrary and cannot be rearranged:

    RESOLVE runs BEFORE BIAS.

Old calls are graded before a new one is made, so the calibration feedback the
model receives always reflects everything currently knowable. Generating a new
thesis first and resolving afterwards would feed the model a track record one
cycle out of date -- a small thing that compounds into the model correcting
against a stale picture of its own accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from vault.calibration.gate import CapitalGate, SizingDecision
from vault.calibration.recalibrate import RecalibrationReport, fit_recalibrator
from vault.data.cache import SeriesCache
from vault.data.fixtures import FIXTURE_SOURCE, build_fixture_universe
from vault.data.series import Series
from vault.data.sources import FRED_SERIES, DataHub
from vault.macro.features import build_brief
from vault.macro.regime import RegimeRead, classify_regime
from vault.resolve.resolver import ResolutionReport, Resolver
from vault.nn.ensemble import Prediction, Predictor
from vault.nn.pipeline import LearningResult, learn
from vault.resolve.scoring import CalibrationScore, score_records
from vault.signals.library import evaluate_all
from vault.signals.types import SignalReading
from vault.thesis.engine import ThesisEngine, ThesisResult
from vault.thesis.journal import ThesisJournal
from vault.thesis.schema import TradeableAsset

logger = logging.getLogger(__name__)

__all__ = ["Vault", "RunResult"]

_SPECS = {spec.key: spec for spec in FRED_SERIES}


@dataclass(slots=True)
class RunResult:
    """Everything one cycle produced."""

    started_at: datetime
    data_mode: str
    series_loaded: int
    regime: RegimeRead | None = None
    brief: dict[str, Any] = field(default_factory=dict)
    resolution: ResolutionReport | None = None
    score: CalibrationScore | None = None
    recalibration: RecalibrationReport | None = None
    thesis: ThesisResult | None = None
    sizing: SizingDecision | None = None
    signals: list[SignalReading] = field(default_factory=list)
    prior: Prediction | None = None
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"run at {self.started_at.isoformat(timespec='seconds')} "
                 f"[{self.data_mode}] -- {self.series_loaded} series"]
        if self.regime:
            lines.append(f"  regime:   {self.regime.summary()}")
        if self.resolution:
            lines.append(f"  resolve:  {self.resolution.summary()}")
        if self.score:
            lines.append(
                f"  score:    {self.score.n} resolved, hit rate "
                f"{self.score.hit_rate:.1%}, Brier {self.score.brier:.4f}, "
                f"{self.score.adequacy}"
            )
        if self.prior:
            lines.append(
                f"  prior:    P(up) {self.prior.probability:.3f} vs base rate "
                f"{self.prior.base_rate:.3f} ({self.prior.confidence})"
            )
        if self.signals:
            usable = sum(1 for s in self.signals if s.usable)
            lines.append(f"  signals:  {usable}/{len(self.signals)} usable")
        if self.thesis:
            lines.append(f"  thesis:   {self.thesis.summary()}")
        if self.sizing:
            lines.append(f"  capital:  {self.sizing.explain().splitlines()[0]}")
        for error in self.errors:
            lines.append(f"  ERROR:    {error}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "data_mode": self.data_mode,
            "series_loaded": self.series_loaded,
            "regime": self.regime.to_dict() if self.regime else None,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "score": self.score.to_dict() if self.score else None,
            "recalibration": self.recalibration.to_dict() if self.recalibration else None,
            "thesis": self.thesis.to_dict() if self.thesis else None,
            "sizing": self.sizing.to_dict() if self.sizing else None,
            "signals": [s.to_dict() for s in self.signals],
            "prior": self.prior.to_dict() if self.prior else None,
            "errors": self.errors,
        }


class Vault:
    """The agent. Owns the data, the journal, the models and the gate."""

    def __init__(
        self,
        *,
        journal_path: str | Path = "data/vault/journal.ndjson",
        cache_dir: str | Path = "data/vault/cache",
        capital: float = 10_000.0,
        model: str = "claude-opus-5",
        effort: str = "high",
        min_samples: int = 100,
        kelly_fraction: float = 0.25,
        max_position_fraction: float = 0.05,
        use_fixtures: bool = False,
        fixture_scenario: str = "late_cycle",
        dry_run: bool = False,
    ) -> None:
        self.capital = capital
        self.use_fixtures = use_fixtures
        self.fixture_scenario = fixture_scenario

        self.journal = ThesisJournal(journal_path)
        self.hub = DataHub(SeriesCache(cache_dir))
        self.resolver = Resolver(self.journal)
        self.engine = ThesisEngine(self.journal, model=model, effort=effort, dry_run=dry_run)
        self.gate = CapitalGate(
            min_samples=min_samples,
            kelly_fraction=kelly_fraction,
            max_position_fraction=max_position_fraction,
        )

        self.series: dict[str, Series] = {}
        self.last_run: RunResult | None = None
        self.runs = 0
        self.stage = "idle"

        # Starts untrained, which means every prior it produces is the base
        # rate. `vault learn` fits it; nothing here assumes it has been.
        self.predictor = Predictor()
        self.learning: LearningResult | None = None

    # -- stages ------------------------------------------------------------

    def scan(self) -> dict[str, Series]:
        """SCAN: load every series, live or from fixtures."""
        self.stage = "scan"
        if self.use_fixtures:
            self.series = build_fixture_universe(scenario=self.fixture_scenario)
            logger.info(
                "loaded %d FIXTURE series (scenario=%s) -- these are synthetic",
                len(self.series), self.fixture_scenario,
            )
        else:
            self.series = dict(self.hub.scan())
            if not self.series:
                logger.error(
                    "no live series loaded (failures: %s). Falling back to "
                    "fixtures so the pipeline can still run, CLEARLY LABELLED.",
                    self.hub.failures,
                )
                self.series = build_fixture_universe(scenario=self.fixture_scenario)
                self.use_fixtures = True
        return self.series

    @property
    def data_mode(self) -> str:
        if self.use_fixtures:
            return "fixture"
        if any(s.source == FIXTURE_SOURCE for s in self.series.values()):
            return "mixed"
        return "live"

    def classify(self) -> RegimeRead:
        """MACRO: the growth/inflation quadrant."""
        self.stage = "macro"
        return classify_regime(self.series)

    def brief(self, regime: RegimeRead) -> dict[str, Any]:
        """FLOW: curve, credit, cross-asset and named signals."""
        self.stage = "flow"
        coverage = (
            {"loaded": len(self.series), "failed": [], "stale": []}
            if self.use_fixtures else self.hub.coverage()
        )
        return build_brief(
            self.series, regime, coverage,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            data_mode=self.data_mode,
            specs=_SPECS,
        )

    def resolve(self) -> ResolutionReport:
        """RESOLVE: grade every due call against price data."""
        self.stage = "resolve"
        prices = {
            asset.value: self.series[asset.value]
            for asset in TradeableAsset
            if asset.value in self.series
        }
        return self.resolver.run(prices)

    def score(self) -> tuple[CalibrationScore, RecalibrationReport | None]:
        """
        SCORE: the scoreboard, and a recalibration fit when there is enough data.

        The recalibrated score is what the gate sees. An agent whose ordering is
        informative but whose numbers are inflated gets its probabilities
        corrected rather than being blocked forever -- which is what the Murphy
        decomposition says to do when reliability is poor and resolution is not.
        """
        self.stage = "score"
        records = list(self.journal)
        raw = score_records(records)

        recalibrator, report = fit_recalibrator(records)
        if recalibrator is None or not report.helped:
            return raw, report

        adjusted = _rescore_with(records, recalibrator)
        logger.info(
            "recalibration improved Brier %.4f -> %.4f; sizing on the corrected "
            "probabilities", raw.brier, adjusted.brier,
        )
        return adjusted, report

    def entry_prices(self) -> dict[str, float]:
        """Current price for every asset a thesis may name."""
        out: dict[str, float] = {}
        for asset in TradeableAsset:
            series = self.series.get(asset.value)
            if series and series.points and series.last > 0:
                out[asset.value] = float(series.last)
        return out

    # -- the run -----------------------------------------------------------

    def run(self, *, commit: bool = True) -> RunResult:
        """One full cycle."""
        self.runs += 1
        started = datetime.now(timezone.utc)
        result = RunResult(started_at=started, data_mode="unknown", series_loaded=0)

        try:
            self.scan()
            result.data_mode = self.data_mode
            result.series_loaded = len(self.series)
        except Exception as exc:
            result.errors.append(f"scan failed: {exc}")
            logger.exception("scan failed")
            self.stage = "idle"
            return result

        # RESOLVE BEFORE BIAS. The model must be told about every outcome that
        # is knowable now, before it makes the next call.
        try:
            result.resolution = self.resolve()
        except Exception as exc:
            result.errors.append(f"resolution failed: {exc}")
            logger.exception("resolution failed")

        try:
            score, recalibration = self.score()
            result.score = score
            result.recalibration = recalibration
        except Exception as exc:
            result.errors.append(f"scoring failed: {exc}")
            logger.exception("scoring failed")
            score = None

        try:
            result.regime = self.classify()
            result.brief = self.brief(result.regime)
        except Exception as exc:
            result.errors.append(f"macro synthesis failed: {exc}")
            logger.exception("macro synthesis failed")
            self.stage = "idle"
            return result

        # FLOW: the signal readings, and the quantitative prior built from
        # whichever of them have demonstrated an edge. On an untrained
        # predictor the prior IS the base rate, which is the honest default
        # and costs nothing.
        self.stage = "flow"
        try:
            result.signals = evaluate_all(self.series)
            result.prior = self.predictor.predict(self.series)
            result.brief["prior"] = result.prior.to_dict()
            result.brief["signals"] = [
                s.to_dict() for s in result.signals if s.usable
            ]
        except Exception as exc:
            result.errors.append(f"signal evaluation failed: {exc}")
            logger.exception("signal evaluation failed")

        # BIAS
        self.stage = "bias"
        if not self.engine.has_credentials and not self.engine.dry_run:
            result.errors.append(
                "no Anthropic credentials; set ANTHROPIC_API_KEY or run "
                "`ant auth login`. The scan, regime and scoreboard above are "
                "complete -- only the thesis step needs the API."
            )
            self.stage = "idle"
            return result

        try:
            result.thesis = self.engine.generate(
                result.brief,
                entry_prices=self.entry_prices(),
                calibration=score.to_dict() if score else None,
                commit=commit,
            )
        except Exception as exc:
            result.errors.append(f"thesis generation failed: {exc}")
            logger.exception("thesis generation failed")
            self.stage = "idle"
            return result

        # GATE
        self.stage = "gate"
        if result.thesis and result.thesis.thesis and score is not None:
            thesis = result.thesis.thesis
            result.sizing = self.gate.evaluate(
                score,
                capital=self.capital,
                stated_probability=float(thesis.probability),
                magnitude_pct=float(thesis.magnitude_pct),
                invalidation_pct=float(thesis.invalidation_pct),
            )

        self.stage = "idle"
        self.last_run = result
        return result

    def learn(
        self, *, target: str = "SP500", horizon_days: int = 21,
        step_days: int = 5, lookback_days: int = 2000, folds: int = 5,
        epochs: int = 150,
    ) -> LearningResult:
        """
        Score every signal, fit the models, and adopt whatever earned its way in.

        Safe to call on an untrained system and safe to call repeatedly: if
        nothing clears the bars, the predictor is left producing the base rate
        and the report says which hurdle each candidate failed.
        """
        if not self.series:
            self.scan()
        self.stage = "learn"
        try:
            result = learn(
                self.series, target=target, horizon_days=horizon_days,
                step_days=step_days, lookback_days=lookback_days,
                folds=folds, epochs=epochs,
            )
            self.predictor = result.predictor
            self.learning = result
            return result
        finally:
            self.stage = "idle"

    # -- state -------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Full state for the HUD and the CLI."""
        records = list(self.journal)
        score, recalibration = (
            self.score() if records else (score_records([]), None)
        )
        open_positions = self.journal.open_positions()

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage": self.stage,
            "runs": self.runs,
            "capital": self.capital,
            "data_mode": self.data_mode,
            "series_loaded": len(self.series),
            "journal": self.journal.stats(),
            "score": score.to_dict(),
            "recalibration": recalibration.to_dict() if recalibration else None,
            "gate": self.gate.stats(),
            "gate_progress": self.gate.progress(score),
            "engine": self.engine.stats(),
            "regime": (
                self.last_run.regime.to_dict()
                if self.last_run and self.last_run.regime else None
            ),
            "brief": self.last_run.brief if self.last_run else {},
            "signals": [
                s.to_dict() for s in (
                    self.last_run.signals if self.last_run
                    else evaluate_all(self.series) if self.series else []
                )
            ],
            "prior": (
                self.last_run.prior.to_dict()
                if self.last_run and self.last_run.prior else None
            ),
            "predictor": {
                "summary": self.predictor.summary(),
                "model_kind": self.predictor.model_kind,
                "trained": self.learning is not None,
            },
            "learning": self.learning.to_dict() if self.learning else None,
            "last_run": self.last_run.to_dict() if self.last_run else None,
            "open_positions": [
                {
                    "thesis_id": r.thesis_id,
                    "asset": _enum(r.thesis.asset),
                    "direction": _enum(r.thesis.direction),
                    "probability": r.thesis.probability,
                    "entry": r.entry_price,
                    "resolve_on": r.resolve_on.isoformat(),
                    "days_left": (r.resolve_on - date.today()).days,
                    "invalidation": r.thesis.invalidation_price(r.entry_price),
                    "regime": r.regime,
                }
                for r in open_positions
            ],
            "recent_calls": [
                {
                    "thesis_id": r.thesis_id,
                    "asset": _enum(r.thesis.asset),
                    "direction": _enum(r.thesis.direction),
                    "horizon": _enum(r.thesis.horizon),
                    "probability": r.thesis.probability,
                    "outcome": r.outcome,
                    "realized_pct": r.realized_pct,
                    "brier": r.brier,
                    "regime": r.regime,
                    "created_at": r.created_at.isoformat(timespec="seconds"),
                    "key_risk": getattr(r.thesis, "key_risk", ""),
                }
                for r in records[-40:][::-1]
            ],
        }


def _rescore_with(records, recalibrator) -> CalibrationScore:
    """Re-score with recalibrated probabilities, without mutating the journal."""
    import copy

    adjusted = []
    for record in records:
        if not record.is_scoreable:
            continue
        clone = copy.copy(record)
        clone.thesis = copy.copy(record.thesis)
        try:
            clone.thesis.probability = recalibrator.apply(float(record.thesis.probability))
        except Exception:
            continue
        adjusted.append(clone)
    return score_records(adjusted)


def _enum(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)
