"""
MCP server: lets Claude Code inspect and steer a running engine.

Speaks the Model Context Protocol over stdio, so ``claude`` can attach to a
live engine and answer questions like "why did nothing trade in the last hour?"
by reading the actual rejection histogram rather than guessing from logs.

Deliberate boundary on what is exposed:

    READ    engine state, metrics, the decision log, model diagnostics,
            capital adequacy, backtests
    WRITE   halt, release a halt, adjust risk limits DOWNWARD, tune the EV
            threshold UPWARD

Nothing here can loosen a risk limit, arm live trading, place an order, or
raise position size. An assistant reasoning over a trading engine should be
able to make it *safer* without a human, and should require a human to make it
riskier. The asymmetry is the whole point, and it is enforced here rather than
being left to the model's judgement.

The engine is reached over its own HTTP API rather than by importing it, so
this server can attach to an engine already running in another process --
which is the normal case.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

__all__ = ["MCPServer", "ToolDefinition", "main"]

PROTOCOL_VERSION = "2024-11-05"


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Any]
    mutates: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class EngineClient:
    """Thin HTTP client for a running engine's control plane."""

    def __init__(self, base_url: str = "http://127.0.0.1:8787", token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _request(self, path: str, payload: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            return {
                "error": f"cannot reach the engine at {self.base_url}: {exc}",
                "hint": "start it with `triangulum paper` or `triangulum demo`",
            }

    def state(self) -> dict:
        return self._request("/api/state")

    def metrics(self) -> dict:
        return self._request("/api/metrics")

    def config(self) -> dict:
        return self._request("/api/config")

    def command(self, name: str, **kwargs: Any) -> dict:
        return self._request("/api/command", {"command": name, **kwargs})


class MCPServer:
    """Minimal MCP server over stdio."""

    def __init__(self, client: EngineClient) -> None:
        self.client = client
        self.tools: dict[str, ToolDefinition] = {}
        self._register_all()

    # -- tools -------------------------------------------------------------

    def register(self, tool: ToolDefinition) -> None:
        self.tools[tool.name] = tool

    def _register_all(self) -> None:
        self.register(ToolDefinition(
            name="engine_status",
            description=(
                "Current engine state: mode, equity, regime, cycle counts, graph "
                "size, risk limits, model diagnostics and venue health. Start here "
                "for any question about what the engine is doing."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=lambda _: _summarize_state(self.client.state()),
        ))

        self.register(ToolDefinition(
            name="why_no_trades",
            description=(
                "Diagnose why the engine is not trading. Walks the decision funnel "
                "from detection to execution and reports where opportunities are "
                "dying, with the specific remedy for the dominant cause. This is "
                "the tool to reach for when the engine looks idle."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=lambda _: _diagnose(self.client.state()),
        ))

        self.register(ToolDefinition(
            name="recent_decisions",
            description=(
                "The last N accept/reject decisions with full expected-value "
                "reasoning: fill probability, predicted slippage, EV, and the "
                "verdict with its stated reason."
            ),
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 20}},
            },
            handler=lambda args: (self.client.state().get("recent_decisions") or [])[
                -int(args.get("limit", 20)):
            ],
        ))

        self.register(ToolDefinition(
            name="performance",
            description=(
                "Performance metrics and Monte Carlo distribution: return, Sharpe, "
                "Sortino, drawdown, hit rate, expectancy, tail risk, and progress "
                "against the configured target. Includes an explicit sample-"
                "adequacy verdict -- read it before believing any ratio."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=lambda _: self.client.metrics(),
        ))

        self.register(ToolDefinition(
            name="capital_adequacy",
            description=(
                "Given a capital amount, compute the edge in basis points that the "
                "market must provide before the configuration breaks even, per "
                "venue, decomposed into fees and quantization drag. Answers 'can "
                "$X actually do this?' with a number instead of an opinion."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "capital": {"type": "number"},
                    "legs": {"type": "integer", "default": 3},
                    "maker_legs": {"type": "integer", "default": 0},
                },
                "required": ["capital"],
            },
            handler=_capital_adequacy,
        ))

        self.register(ToolDefinition(
            name="model_diagnostics",
            description=(
                "Fill-model skill and calibration, slippage-model fit, bandit arm "
                "leaderboard, drift status. Use this to judge whether the learned "
                "gate is trustworthy yet."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=lambda _: _model_report(self.client.state()),
        ))

        self.register(ToolDefinition(
            name="halt",
            description=(
                "Engage the kill switch. The engine stops opening new cycles "
                "immediately; in-flight cycles still complete or unwind. Always "
                "safe to call."
            ),
            input_schema={
                "type": "object",
                "properties": {"reason": {"type": "string"}},
            },
            handler=lambda args: self.client.command(
                "engage_kill_switch", reason=args.get("reason", "requested via MCP")
            ),
            mutates=True,
        ))

        self.register(ToolDefinition(
            name="release_halt",
            description=(
                "Release the kill switch and resume trading. Refuses while "
                "stranded inventory is outstanding -- flatten the position first. "
                "This is the one mutation that increases risk, and it is "
                "deliberately gated on the engine's own state rather than on "
                "anyone's judgement."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=self._release_halt,
            mutates=True,
        ))

        self.register(ToolDefinition(
            name="tighten_risk",
            description=(
                "Lower a risk limit. Only accepts values MORE conservative than "
                "the current setting; a request to loosen one is refused with an "
                "explanation. Use to reduce exposure without a human in the loop."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "string",
                        "enum": [
                            "max_daily_loss_pct", "max_drawdown_pct",
                            "max_cycle_notional_pct", "max_consecutive_losses",
                            "min_expected_value_bps",
                        ],
                    },
                    "value": {"type": "number"},
                },
                "required": ["limit", "value"],
            },
            handler=self._tighten,
            mutates=True,
        ))

    def _release_halt(self, _args: dict) -> dict:
        state = self.client.state()
        risk = state.get("risk", {})
        reason = str(risk.get("kill_switch_reason", ""))
        if "unhedged" in reason or "stranded" in reason:
            return {
                "refused": True,
                "reason": (
                    "the halt was caused by inventory the engine could not "
                    "unwind. Releasing it while that position is open means "
                    "trading on top of an unhedged exposure. Flatten it first, "
                    "then release."
                ),
                "kill_switch_reason": reason,
            }
        return self.client.command("release_kill_switch")

    def _tighten(self, args: dict) -> dict:
        name = str(args.get("limit", ""))
        value = float(args.get("value", 0))
        config = self.client.config()
        risk = config.get("risk", {})
        learning = config.get("learning", {})
        current = risk.get(name, learning.get(name))

        if current is None:
            return {"error": f"unknown limit {name!r}"}

        # For every limit here, a SMALLER number is safer -- except the EV
        # threshold, where a LARGER number means "demand more edge before
        # risking capital", which is the conservative direction.
        tighter = value > float(current) if name == "min_expected_value_bps" \
            else value < float(current)
        if not tighter:
            return {
                "refused": True,
                "current": current,
                "requested": value,
                "reason": (
                    f"this would loosen {name} from {current} to {value}. This "
                    f"interface only tightens limits. Widening risk requires a "
                    f"human editing the config."
                ),
            }
        return self.client.command("set_risk_limit", limit=name, value=value)

    # -- protocol ----------------------------------------------------------

    def handle(self, message: dict) -> dict | None:
        method = message.get("method", "")
        request_id = message.get("id")

        if method == "initialize":
            return _ok(request_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "triangulum", "version": "1.0.0"},
            })
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return _ok(request_id, {
                "tools": [t.to_dict() for t in self.tools.values()]
            })
        if method == "tools/call":
            params = message.get("params", {})
            name = params.get("name", "")
            tool = self.tools.get(name)
            if tool is None:
                return _error(request_id, -32601, f"unknown tool {name!r}")
            try:
                result = tool.handler(params.get("arguments", {}) or {})
            except Exception as exc:
                logger.exception("tool %s failed", name)
                return _ok(request_id, {
                    "content": [{"type": "text", "text": f"error: {exc}"}],
                    "isError": True,
                })
            return _ok(request_id, {
                "content": [
                    {"type": "text", "text": json.dumps(result, indent=2, default=str)}
                ]
            })
        return _error(request_id, -32601, f"unknown method {method!r}")

    def serve(self) -> None:
        """Read JSON-RPC from stdin, write responses to stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = self.handle(message)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()


def _ok(request_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


# --------------------------------------------------------------------------
# Report builders
# --------------------------------------------------------------------------


def _summarize_state(state: dict) -> dict:
    if "error" in state:
        return state
    risk = state.get("risk", {})
    executor = state.get("executor", {})
    gate = state.get("gate", {})
    return {
        "mode": state.get("mode"),
        "running": state.get("running"),
        "halted": risk.get("kill_switch"),
        "halt_reason": risk.get("kill_switch_reason"),
        "equity": state.get("equity"),
        "starting_equity": state.get("starting_equity"),
        "total_return_pct": risk.get("total_return_pct"),
        "drawdown_pct": risk.get("drawdown_pct"),
        "regime": state.get("regime"),
        "opportunities_seen": state.get("opportunities_seen"),
        "cycles_planned": state.get("cycles_planned"),
        "cycles": {
            "attempted": executor.get("attempted"),
            "completed": executor.get("completed"),
            "aborted": executor.get("aborted_pre_trade"),
            "unwound": executor.get("unwound"),
            "stuck": executor.get("stuck"),
        },
        "gate_acceptance_rate": gate.get("acceptance_rate"),
        "model_trusted": gate.get("trusts_model"),
        "graph": state.get("graph", {}),
        "venues": list((state.get("venues") or {}).keys()),
    }


def _diagnose(state: dict) -> dict:
    """Walk the funnel and name the dominant constraint, with its remedy."""
    if "error" in state:
        return state

    graph = state.get("graph", {})
    gate = state.get("gate", {})
    executor = state.get("executor", {})
    risk = state.get("risk", {})

    detected = state.get("opportunities_seen", 0)
    planned = state.get("cycles_planned", 0)
    evaluated = gate.get("evaluations", 0)
    accepted = gate.get("accepts", 0)
    completed = executor.get("completed", 0)

    findings: list[dict] = []

    if risk.get("kill_switch"):
        findings.append({
            "stage": "halted",
            "severity": "critical",
            "finding": f"The kill switch is engaged: {risk.get('kill_switch_reason')}",
            "remedy": (
                "Resolve the underlying cause, then release the halt. If it was "
                "stranded inventory, flatten the position manually first."
            ),
        })

    if graph.get("edges_usable", 0) == 0:
        excluded = graph.get("excluded", {})
        top = max(excluded.items(), key=lambda kv: kv[1])[0] if excluded else "unknown"
        findings.append({
            "stage": "graph",
            "severity": "critical",
            "finding": f"No usable graph edges. Dominant exclusion: {top}.",
            "detail": excluded,
            "remedy": {
                "lot_value": (
                    "Quantization drag exceeds the budget at this capital. Raise "
                    "capital, raise strategy.drag_budget_bps, or restrict the "
                    "universe to low-lot-value instruments. Run capital_adequacy."
                ),
                "stale": "Books are older than max_book_age_ms. Check feed health.",
                "min_notional": "Per-leg size is below the venue minimum. Raise capital.",
                "no_book": "No market data. Check venue connectivity.",
            }.get(top, "Inspect graph.excluded for the breakdown."),
        })
    elif detected == 0:
        findings.append({
            "stage": "detection",
            "severity": "warning",
            "finding": (
                f"The graph has {graph.get('edges_usable')} usable edges but no "
                f"cycle has cleared the screening threshold."
            ),
            "remedy": (
                "This is the normal state. Real triangular dislocations are rare "
                "and small. Verify strategy.min_gross_edge_bps is not set above "
                "what the market produces, then wait."
            ),
        })
    elif planned < detected * 0.2:
        findings.append({
            "stage": "sizing",
            "severity": "warning",
            "finding": (
                f"{detected} opportunities detected but only {planned} could be "
                f"sized ({planned / max(1, detected):.0%})."
            ),
            "remedy": (
                "Most opportunities fail min-notional or lot quantization. This "
                "is a capital-size constraint, not a tuning problem."
            ),
        })
    elif accepted == 0 and evaluated > 0:
        rejects = gate.get("rejects_by_reason", {})
        top = max(rejects.items(), key=lambda kv: kv[1])[0] if rejects else "unknown"
        findings.append({
            "stage": "ev_gate",
            "severity": "info",
            "finding": (
                f"{evaluated} cycles reached the EV gate; none were accepted. "
                f"Dominant rejection: {top}."
            ),
            "detail": rejects,
            "remedy": {
                "reject_uncertainty": (
                    "The model has too few samples for its confidence bound to "
                    "clear zero. Exploration decays this automatically as samples "
                    "accumulate; check gate.exploration_rate."
                ),
                "reject_ev": (
                    "Expected value is genuinely below threshold. The opportunities "
                    "are real but too small to pay for their own fill risk."
                ),
                "reject_edge": (
                    "Net edge is negative after predicted slippage -- the "
                    "opportunities are illusory."
                ),
                "reject_fill_probability": (
                    "The model predicts these will not fill, most often because "
                    "the books are stale."
                ),
            }.get(top, "Inspect gate.rejects_by_reason."),
        })
    elif completed < accepted * 0.5 and accepted > 3:
        findings.append({
            "stage": "execution",
            "severity": "warning",
            "finding": (
                f"{accepted} cycles accepted but only {completed} completed. Legs "
                f"are not filling."
            ),
            "remedy": (
                "Raise execution.taker_price_offset_ticks, or check venue latency. "
                "The fill model should learn this on its own within a few hundred "
                "samples."
            ),
        })

    if not findings:
        findings.append({
            "stage": "healthy",
            "severity": "info",
            "finding": "The pipeline is converting opportunities into cycles.",
            "remedy": "No action needed.",
        })

    return {
        "funnel": {
            "detected": detected, "sized": planned, "reached_gate": evaluated,
            "accepted": accepted, "completed": completed,
        },
        "findings": findings,
    }


def _model_report(state: dict) -> dict:
    if "error" in state:
        return state
    gate = state.get("gate", {})
    return {
        "fill_model": gate.get("fill_model"),
        "slippage_model": gate.get("slippage_model"),
        "calibration": gate.get("calibration"),
        "exploration_rate": gate.get("exploration_rate"),
        "trusted": gate.get("trusts_model"),
        "bandit": state.get("bandit"),
        "drift": state.get("drift"),
        "interpretation": {
            "skill": (
                "Above 0 means the fill model beats always-predicting the base "
                "rate. Below 0 means it is actively harmful and the gate should "
                "not be trusted."
            ),
            "expected_calibration_error": (
                "Below 0.05 is good. The EV calculation multiplies by P(fill), so "
                "a miscalibrated probability produces a wrong decision even when "
                "the ranking is right."
            ),
        },
    }


def _capital_adequacy(args: dict) -> dict:
    from decimal import Decimal

    from triangulum.exchanges.spec import (
        assess_capital_adequacy, max_lot_value_for_budget,
    )

    capital = Decimal(str(args.get("capital", 100)))
    legs = int(args.get("legs", 3))
    maker_legs = int(args.get("maker_legs", 0))

    scenarios = {
        "btc_anchored": (Decimal("0.00001"), Decimal("62000")),
        "low_lot": (Decimal("0.1"), Decimal("0.12")),
    }
    out: dict = {"capital": str(capital), "legs": legs, "scenarios": {}}

    for label, (lot, price) in scenarios.items():
        rows = {}
        for venue in ("binance", "mexc", "okx", "kucoin", "kraken", "coinbase"):
            report = assess_capital_adequacy(
                venue, capital, cycle_legs=legs, maker_legs=maker_legs,
                use_discount=True,
                representative_lot_step=lot, representative_price=price,
            )
            rows[venue] = {
                "fee_bps": float(report.fee_bps),
                "drag_bps": float(report.drag_bps),
                "required_edge_bps": float(report.required_edge_bps),
                "feasible": report.feasible,
            }
        out["scenarios"][label] = rows

    out["max_lot_value_usd"] = {
        "5bps_budget": str(max_lot_value_for_budget(
            capital * Decimal("0.95"), Decimal("5"), legs=legs)),
        "10bps_budget": str(max_lot_value_for_budget(
            capital * Decimal("0.95"), Decimal("10"), legs=legs)),
    }
    out["note"] = (
        "Observed triangular dislocations on a liquid venue run 1-8 bps and last "
        "50-300ms. Compare required_edge_bps against that range."
    )
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Triangulum MCP server")
    parser.add_argument("--url", default=os.environ.get(
        "TRIANGULUM_URL", "http://127.0.0.1:8787"))
    parser.add_argument("--token", default=os.environ.get(
        "TRIANGULUM_DASHBOARD_TOKEN", ""))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    MCPServer(EngineClient(args.url, args.token)).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
