"""
Health checks.

One question: *is this engine safe to leave running?* Not "is it profitable" --
that is the metrics layer's job -- but "is anything about its state such that
continuing to trade is unwise".

Each check returns a severity. CRITICAL means stop now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["HealthCheck", "HealthReport", "check_engine_health"]


@dataclass(slots=True)
class HealthCheck:
    name: str
    ok: bool
    severity: str          # info | warning | critical
    message: str
    detail: Any = None


@dataclass(slots=True)
class HealthReport:
    checks: list[HealthCheck] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not any(c.severity == "critical" and not c.ok for c in self.checks)

    @property
    def degraded(self) -> bool:
        return any(not c.ok for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "degraded": self.degraded,
            "checks": [
                {"name": c.name, "ok": c.ok, "severity": c.severity,
                 "message": c.message, "detail": c.detail}
                for c in self.checks
            ],
        }

    def summary(self) -> str:
        failing = [c for c in self.checks if not c.ok]
        if not failing:
            return f"healthy ({len(self.checks)} checks passed)"
        worst = "critical" if any(c.severity == "critical" for c in failing) else "warning"
        return f"{worst}: " + "; ".join(c.message for c in failing[:3])


def check_engine_health(snapshot: dict) -> HealthReport:
    report = HealthReport()
    risk = snapshot.get("risk", {})
    books = snapshot.get("books", {})
    graph = snapshot.get("graph", {})
    executor = snapshot.get("executor", {})
    gate = snapshot.get("gate", {})

    def add(name: str, ok: bool, severity: str, message: str, detail: Any = None) -> None:
        report.checks.append(HealthCheck(name, ok, severity, message, detail))

    add("kill_switch", not risk.get("kill_switch"), "critical",
        f"kill switch engaged: {risk.get('kill_switch_reason', '')}"
        if risk.get("kill_switch") else "kill switch clear")

    stuck = executor.get("stuck", 0)
    add("stranded_inventory", stuck == 0, "critical",
        f"{stuck} cycles left inventory stranded -- a human must flatten it"
        if stuck else "no stranded inventory")

    connected = books.get("connected_venues") or []
    add("venue_connectivity", bool(connected), "critical",
        "no venue connected" if not connected else f"connected: {', '.join(connected)}")

    tradeable = books.get("tradeable", 0)
    total = books.get("books", 0) or 1
    ratio = tradeable / total
    add("book_freshness", ratio > 0.5, "warning",
        f"only {tradeable}/{total} books are tradeable" if ratio <= 0.5
        else f"{tradeable}/{total} books fresh", round(ratio, 3))

    gaps = books.get("sequence_gaps", 0)
    checksums = books.get("checksum_failures", 0)
    add("data_integrity", gaps + checksums < 20, "warning",
        f"{gaps} sequence gaps and {checksums} checksum failures -- the feed is "
        f"degraded" if gaps + checksums >= 20 else "market data intact")

    add("graph", graph.get("edges_usable", 0) > 0, "warning",
        "graph has no usable edges; the engine cannot find cycles"
        if not graph.get("edges_usable") else
        f"{graph.get('edges_usable')} usable edges")

    drawdown = float(risk.get("drawdown_pct", 0) or 0)
    limit = float((risk.get("limits") or {}).get("max_drawdown_pct", 10) or 10)
    add("drawdown", drawdown < limit * 0.8, "warning",
        f"drawdown {drawdown:.2f}% is within 20% of the {limit}% limit"
        if drawdown >= limit * 0.8 else f"drawdown {drawdown:.2f}%")

    model = gate.get("fill_model") or {}
    skill = float(model.get("skill", 0) or 0)
    samples = int(model.get("samples", 0) or 0)
    add("model_skill", samples < 200 or skill > 0, "warning",
        f"fill model skill is {skill:.3f} after {samples} samples -- it is "
        f"performing worse than the base rate and the gate should not be trusted"
        if samples >= 200 and skill <= 0 else
        f"skill {skill:.3f} over {samples} samples")

    calibration = gate.get("calibration") or {}
    ece = float(calibration.get("expected_calibration_error", 0) or 0)
    add("calibration", samples < 200 or ece < 0.15, "warning",
        f"calibration error {ece:.3f} is high; the EV maths is multiplying by a "
        f"probability that is not real" if samples >= 200 and ece >= 0.15
        else f"calibration error {ece:.3f}")

    return report
