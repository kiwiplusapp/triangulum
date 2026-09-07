---
name: arb-report
description: Produce an operator's report on the running Triangulum engine — what it did, why it did or did not trade, and what to change. Use when asked how the bot is doing, why nothing is trading, whether the models are working, or for a performance summary.
---

# Arbitrage engine report

Produce a report a person can act on in two minutes. Lead with the answer.

## Gather

Use the `triangulum` MCP tools, in this order. Stop early if one answers the
question.

1. `engine_status` — is it running, halted, and what is equity doing?
2. `why_no_trades` — the decision funnel and the dominant constraint. This
   answers most questions on its own.
3. `performance` — returns, risk-adjusted ratios, and the sample-adequacy verdict.
4. `model_diagnostics` — only if the question is about the learning layer.
5. `recent_decisions` — only when a specific decision needs explaining.

If the engine is unreachable, say so and stop. Do not report on a snapshot you
could not fetch.

## Report

**Verdict** — one sentence. "Healthy, 47 cycles, +2.1 bps expectancy" or
"Halted since 14:02 on stranded inventory" or "Running but has not traded: the
lot-value screen leaves no cycles at this capital."

**What happened** — equity, cycles by outcome, expectancy per cycle. Quote the
sample-adequacy verdict verbatim; if it says `insufficient`, say the ratios are
not yet evidence and do not repeat them as though they were.

**Where opportunities died** — the funnel, with the biggest drop named and its
cause explained.

**Models** — only if they have samples. Skill above 0 means the fill model
beats the base rate; calibration error below 0.05 means the EV maths is running
on a real probability. Below those, say the gate is not yet trustworthy.

**What to change** — at most three items, each with the specific config key or
command. If nothing needs changing, say so; inventing recommendations to fill
the section is worse than a short report.

## Rules

- Never annualise a sample under a day. The tool reports `annualisable: false`
  for a reason.
- A halt caused by stranded inventory is the highest-severity finding there is.
  Lead with it and say the position must be flattened by a person.
- Do not present a positive return over a handful of cycles as evidence of
  edge. Say how many cycles it took and what the Monte Carlo 5th percentile is.
- If asked whether it is "working", answer about the *pipeline* (is it
  detecting, sizing, deciding, executing) separately from *profitability*.
  A correctly-working engine that finds nothing profitable is the expected
  outcome at small capital, and conflating the two misleads.
