---
name: tune-gate
description: Diagnose and safely adjust Triangulum's expected-value gate and risk limits. Use when the engine is rejecting everything, accepting too much, when asked to make it more or less aggressive, or to tune thresholds.
---

# Tuning the EV gate

The gate is the only place capital is risked. Changes here move real money, so
diagnose before touching anything.

## Diagnose first

Call `why_no_trades`, then `model_diagnostics`. Map the dominant rejection
reason to its actual cause — they are not interchangeable:

| Dominant reason | What it means | Correct response |
|---|---|---|
| `reject_uncertainty` | Model has too few samples; the confidence bound cannot clear zero | **Wait.** Exploration decays automatically. Check `exploration_rate`. |
| `reject_ev` | EV genuinely below threshold | The opportunities are real but too small. Lowering the threshold does not make them bigger. |
| `reject_edge` | Net edge negative after predicted slippage | The opportunities are illusory. Nothing to tune. |
| `reject_fill_probability` | Model predicts no fill, usually from stale books | Fix the data feed, not the gate. |
| Risk rejections | A hard limit fired | Read which one. Do not raise it to make the message go away. |

If the engine is not even reaching the gate, the constraint is upstream —
the graph, the lot-value screen, or min-notional. `capital_adequacy` will say
which, in basis points.

## Safe adjustments

`tighten_risk` accepts only changes that reduce risk. Use it freely:

- `min_expected_value_bps` **up** — demand more edge before committing
- `max_cycle_notional_pct` **down** — smaller positions
- `max_daily_loss_pct`, `max_drawdown_pct` **down** — tighter stops
- `halt` — always safe

## Adjustments that need a human

Anything that increases risk. Do not attempt these through the MCP tools; they
will be refused. Instead, explain what you would change, why, and what could go
wrong, and let the person edit the config:

- lowering `min_expected_value_bps`
- raising any position or loss limit
- raising `strategy.drag_budget_bps` (admits instruments whose lot rounding
  costs more than the edge)
- lowering `strategy.min_net_edge_bps` below the empirical viability floor
- releasing a halt caused by stranded inventory

## The trap to avoid

If asked to "make it hit the target", say plainly that the target is tracked,
never chased. Every knob that increases trade frequency does so by accepting
worse expected value. Relaxing the gate to reach a return number converts a
positive-expectancy system into a negative one while making the dashboard look
busier — which is precisely the failure mode the gate exists to prevent.

The honest answer when the target is unreachable is that it is unreachable, and
`capital_adequacy` will give the number that proves it.
