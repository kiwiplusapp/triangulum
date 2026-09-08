"""
Prompts for the thesis engine.

Two design rules govern everything here.

**The prompt must not be able to hedge.** The schema already forbids a neutral
direction and caps the reasoning length; the prompt reinforces it by explicitly
naming the failure mode. A model told only "analyse the macro data" produces
balanced, defensible, unfalsifiable prose -- which is the correct output for an
essay and useless as a forecast.

**Abstaining is a first-class answer.** A forecaster who always has a view is
not a forecaster. The prompt makes clear that returning no thesis is a valid,
respected outcome when the evidence does not support one, and the calibration
layer never penalises an abstention. Without that permission the model will
manufacture a 0.55 call on noise, which is how a track record fills with
coin-flips that dilute any real edge.

CACHING. The system prompt is frozen and cached; the volatile brief goes in the
user turn after the cache breakpoint. Every scan re-reads the same ~2k-token
system block, so this is roughly a 90% saving on that portion.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "ABSTAIN_SENTINEL"]

ABSTAIN_SENTINEL = "NO_THESIS"


SYSTEM_PROMPT = """\
You are the analytical core of Vault, a macro research system that keeps score \
on itself.

Every call you make is committed to a hash-chained, append-only journal before \
its outcome is knowable, resolved automatically against market data at its \
horizon, and scored with a Brier score against the 0.5 baseline. Your \
calibration is measured and displayed. Systematic overconfidence is not a \
stylistic flaw here -- it is a number on a dashboard.

## What you are producing

A single falsifiable directional call, or nothing.

A call must state: the asset, up or down, by when, with what probability, how \
far you expect it to move, and the adverse move at which the thesis is dead. \
It must name one specific observable event that would prove it wrong.

## What disqualifies a call

- Hedged direction. There is no "neutral". If you have no view, abstain.
- A probability that does not reflect genuine belief. You are scored on \
calibration, not on boldness. Calls at 0.55 that are right 55% of the time \
score better than calls at 0.85 that are right 60% of the time.
- A key risk that gestures at uncertainty ("markets may move", "volatility"). \
Name a release, a decision, or a level.
- Reasoning that anticipates every outcome. If the reasoning would look \
correct whichever way the asset moves, it is not reasoning.

## When to abstain

Return the abstain form when any of these hold:

- The regime read has low confidence and the cross-asset picture is mixed.
- The data you need is stale relative to its own release cadence.
- The move you would forecast is smaller than the asset's normal daily range \
over that horizon -- you would be forecasting noise.
- Several strong signals genuinely conflict and you cannot say which dominates.

Abstaining costs you nothing. It is never scored. Manufacturing a 0.55 call on \
noise dilutes whatever real edge you have and is the single most common way a \
forecasting record becomes worthless.

## How to think about the evidence

The brief gives you macro series with their staleness, a growth/inflation \
regime read with its own confidence, curve structure, credit spreads and \
cross-asset state. Weight them by what they are:

- Weekly and daily series (claims, spreads, VIX, yields) are current. Monthly \
macro (CPI, PCE, payrolls) describes a world 3-7 weeks old. A regime read \
built mostly on stale inputs deserves less weight than its confidence number \
suggests, and the staleness is given to you so you can make that adjustment.
- Second derivatives beat levels. Inflation at 3% falling from 5% and \
inflation at 3% rising from 1% are opposite regimes.
- Credit spreads lead equity drawdowns more reliably than equity momentum does.
- The curve is a slow signal. Inversion leads recession by 6-18 months, which \
is far outside a 1-day or 1-week horizon. Do not use a slow signal to justify \
a fast call.
- Positioning and momentum matter over days; macro matters over months. Match \
the evidence to the horizon you choose.

## Horizon selection

Choose the horizon your evidence actually supports. A curve inversion does not \
justify a 1-day call. A VIX spike does not justify a 3-month one. Mismatching \
these is the most common way a well-reasoned view becomes a losing trade.

Be precise, be brief, and be willing to say you do not know."""


def build_user_prompt(brief: dict[str, Any], *, calibration: dict[str, Any] | None = None) -> str:
    """
    Render the market brief into the user turn.

    Calibration feedback is included when the agent has a track record. This is
    deliberate: telling the model "your last 40 calls at 0.7 confidence came in
    at 0.52" is the most direct correction available for overconfidence, and it
    costs nothing. It is only included once there are enough resolved calls for
    the number to mean something -- feeding back a hit rate computed on four
    samples would inject noise as if it were a signal.
    """
    lines: list[str] = []

    lines.append("# MARKET BRIEF")
    lines.append(f"Generated: {brief.get('generated_at', 'unknown')}")
    data_mode = brief.get("data_mode", "live")
    if data_mode != "live":
        lines.append(
            f"\n**DATA MODE: {data_mode.upper()}** -- these series are synthetic "
            f"fixtures, not real market data. Reason about them as an exercise; "
            f"do not treat the readings as facts about the world."
        )

    # ---- regime ----
    regime = brief.get("regime", {})
    lines.append("\n## REGIME")
    lines.append(
        f"{regime.get('quadrant', 'unknown').upper()} "
        f"({regime.get('probabilities', {}).get(regime.get('quadrant'), 0):.0%} probability) "
        f"-- {regime.get('description', '')}"
    )
    lines.append(
        f"growth score {regime.get('growth_score', 0):+.2f}, "
        f"inflation score {regime.get('inflation_score', 0):+.2f}, "
        f"classifier confidence {regime.get('confidence', 0):.0%}"
    )
    if regime.get("transitioning"):
        lines.append(
            "NOTE: the top two quadrants are within 15 percentage points. "
            "This is a transition, and transitions are when a single label is "
            "most confidently wrong."
        )
    if regime.get("max_staleness_days", 0) > 40:
        lines.append(
            f"NOTE: the oldest input to this classification is "
            f"{regime['max_staleness_days']} days old."
        )
    lines.append(f"Historical tilt in this regime: {regime.get('historical_tilt', 'n/a')}")

    probabilities = regime.get("probabilities", {})
    if probabilities:
        ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
        lines.append(
            "Full distribution: "
            + ", ".join(f"{k} {v:.0%}" for k, v in ranked)
        )

    # ---- curve & rates ----
    curve = brief.get("curve", {})
    if curve:
        lines.append("\n## CURVE AND RATES")
        for key, value in curve.items():
            lines.append(f"- {key}: {value}")

    # ---- cross-asset ----
    cross = brief.get("cross_asset", {})
    if cross:
        lines.append("\n## CROSS-ASSET")
        for key, value in cross.items():
            lines.append(f"- {key}: {value}")

    # ---- series table ----
    series = brief.get("series", [])
    if series:
        lines.append("\n## SERIES")
        lines.append(
            "key | last | change | YoY% | 1y z-score | age | note"
        )
        for row in series:
            age = row.get("staleness_days")
            flag = " STALE" if row.get("stale") else ""
            lines.append(
                f"{row.get('key')} | {_fmt(row.get('last'))} | "
                f"{_fmt(row.get('pct_change_1'))}% | {_fmt(row.get('yoy_pct'))} | "
                f"{_fmt(row.get('zscore_1y'))} | {age}d{flag} | "
                f"{row.get('note', '')[:70]}"
            )

    # ---- signals ----
    signals = brief.get("signals", [])
    if signals:
        lines.append("\n## NAMED SIGNALS")
        for signal in signals:
            lines.append(f"- {signal}")

    # ---- data quality ----
    coverage = brief.get("coverage", {})
    if coverage.get("failed") or coverage.get("stale"):
        lines.append("\n## DATA QUALITY")
        if coverage.get("failed"):
            lines.append(f"Series that failed to load: {', '.join(coverage['failed'])}")
        for row in coverage.get("stale", [])[:6]:
            lines.append(
                f"STALE: {row['key']} last updated {row['as_of']} "
                f"({row['days']}d, limit {row['limit']}d)"
            )

    # ---- calibration feedback ----
    if calibration and calibration.get("scoreable", 0) >= 20:
        lines.append("\n## YOUR TRACK RECORD")
        lines.append(
            f"{calibration['scoreable']} resolved calls. "
            f"Hit rate {calibration.get('hit_rate', 0):.1%}. "
            f"Brier {calibration.get('brier', 0):.4f} against a 0.2500 baseline "
            f"(lower is better). Skill {calibration.get('skill', 0):+.3f}."
        )
        bias = calibration.get("overconfidence")
        if bias is not None and abs(bias) > 0.04:
            direction = "OVERCONFIDENT" if bias > 0 else "UNDERCONFIDENT"
            lines.append(
                f"You are systematically {direction} by "
                f"{abs(bias):.1%}: your average stated probability is "
                f"{calibration.get('mean_probability', 0):.2f} and your realised "
                f"hit rate is {calibration.get('hit_rate', 0):.2f}. Adjust your "
                f"stated probabilities accordingly on this call."
            )
        buckets = calibration.get("reliability", [])
        if buckets:
            lines.append("Reliability by confidence bucket (stated -> realised, n):")
            for bucket in buckets:
                lines.append(
                    f"  {bucket['predicted']:.2f} -> {bucket['observed']:.2f} "
                    f"(n={bucket['count']})"
                )

    lines.append("\n---")
    lines.append(
        "Produce one falsifiable thesis, or abstain. If you abstain, set "
        f"`asset` to the instrument you came closest to calling and put "
        f"'{ABSTAIN_SENTINEL}' as the first word of `reasoning`; the system will "
        "record an abstention rather than a call."
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)
