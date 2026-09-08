# Triangulum

Autonomous multi-venue arbitrage engine. Negative-cycle detection over a
currency graph, online-learning execution gating, and a real-time HUD.

**Paper-first.** Live trading requires three independent locks and there is no
bypass.

```bash
pip install -e .

python -m triangulum doctor --capital 100   # what can this capital actually do?
python -m triangulum demo                   # synthetic market + HUD on :8787
python -m triangulum paper                  # live data, simulated fills
```

---

## Start with `doctor`

It answers, in basis points, what edge the market must hand you before your
configuration breaks even. That is a more useful number than any backtest, and
it takes a second.

```
  ── low-lot (TRX/XRP) cycles (one lot = $0.0120) ──────────────
  venue          fees     drag     needs  verdict
  ----------------------------------------------------------------------
  binance       22.5b     1.9b     31.7b  feasible
  mexc          10.0b     1.9b     15.5b  feasible
  kraken       105.0b     1.9b    139.0b  NOT FEASIBLE

  Best case at this size: mexc on low-lot instruments needs the
  market to hand you 15.5 bps before you break even.

  Observed triangular dislocations on a liquid venue run 1-8 bps and
  last 50-300ms. Compare those two numbers honestly.
```

Read [`docs/EXPECTATIONS.md`](docs/EXPECTATIONS.md) before funding anything.

---

## How it works

```
  market data ──► currency graph ──► negative-cycle search ──► opportunity
                                                                    │
       ledger ◄── executor ◄── cycle plan ◄── EV gate ◄── learner ◄──┘
```

**The graph.** Assets are nodes; tradable conversions are edges carrying the
rate you actually receive — after fees, after walking the book. Take logs and
set `w = −ln(rate)`, and a profitable cycle becomes a *negative-weight cycle*,
which Bellman-Ford finds in O(V·E) for any cycle length simultaneously. Cycle
templates are also enumerated once and re-priced every scan, because the topology
changes weekly while prices change thousands of times a second.

**The gate.** The only place capital is risked:

```
EV = P(fill) × net_edge − (1 − P(fill)) × failure_cost
```

Both terms matter. A cycle showing +4 bps with a 60% fill probability and a
12 bps unwind cost has an EV of **−2.4 bps** — a losing trade that looks like a
winner in every naive backtest. `P(fill)` comes from an FTRL-Proximal logistic
regression trained online on every attempt, calibrated with Platt scaling, and
gated on its own uncertainty so the engine is timid when ignorant and confident
when informed.

**The executor.** There is no atomic multi-leg order on any spot venue, so
between legs you hold a directional position nobody asked for. Everything here
is built around that: fail before committing, order legs by unwind cost, run
under a shrinking latency budget, and unwind aggressively when a leg misses.

---

## What the numbers say

Reproducible with `pytest -q` and `python -m triangulum doctor`.

**Quantization drag is the small-account tax.** The same +100 bps cycle:

| Capital | Net captured | Drag | Share of edge kept |
|---|---|---|---|
| $100 | +9.98 bps | 60.0 bps | **14%** |
| $1,000 | +63.93 bps | 6.0 bps | 91% |
| $10,000 | +70.22 bps | 0.6 bps | **99.5%** |

**Instrument choice beats venue choice at small size.** Same venue, same fees:
a BTC-anchored triangle costs **94.7 bps** of lot-rounding drag at $100; a
TRX-anchored one costs **1.9 bps**. Enforced as a hard screen in the graph.

**The gate earns its place.** Five cycles all showing +6 bps, differing only in
book staleness. A naive detector takes all five:

| Book age | P(fill) | EV | Verdict |
|---|---|---|---|
| 20 ms | 0.955 | +3.66 bps | accept |
| 80 ms | 0.872 | +1.17 bps | accept |
| 150 ms | 0.660 | −5.18 bps | reject |
| 250 ms | 0.250 | −17.50 bps | reject |
| 350 ms | 0.055 | −23.36 bps | reject |

---

## Venues

Ten adapters. `python -m triangulum venues` prints the economics of each.

| Venue | Maker/taker | 3-leg cost | Note |
|---|---|---|---|
| **MEXC** | 0 / 5 bps | 15 bps | Best retail fee structure in crypto; thinner books |
| **Binance** | 10 / 10 | 30 bps | Deepest triangular graph, $5 min notional |
| **OKX** | 8 / 10 | 28 bps | Best maker fee; publishes book checksums |
| KuCoin | 10 / 10 | 30 bps | $0.10 min notional — friendliest to tiny capital |
| Kraken | 25 / 40 | 120 bps | Deep fiat books; not a triangular venue |
| Coinbase | 40 / 60 | 180 bps | Structurally cannot support this at retail |
| OANDA (FX) | spread only | — | Crosses are *derived*; the triangle is closed |
| Alpaca (equities) | 0 | — | Star topology; no cycles exist |

The non-crypto adapters are implemented and honest about why they cannot do
triangular arbitrage. See `exchanges/oanda_fx.py` for a live demonstration.

---

## The dashboard

Real-time HUD on `:8787`. Equity against the target trajectory, the live
currency graph with a pulse tracing the active cycle, a decision funnel showing
where opportunities die, the detected-edge distribution split by taken/passed,
fill-model calibration, the bandit leaderboard, risk limits, and a kill switch.

Served by a stdlib-only HTTP + WebSocket server with no dependencies, because
the dashboard is how you find out the engine is in trouble and must come up
when the rest of the stack does not.

---

## Claude Code

`.mcp.json` registers an MCP server exposing engine state and safe controls.

```
engine_status       what it is doing right now
why_no_trades       walks the funnel and names the constraint, with the remedy
performance         metrics + Monte Carlo, with a sample-adequacy verdict
capital_adequacy    what edge your capital needs, per venue
model_diagnostics   is the learned gate trustworthy yet?
halt / release_halt / tighten_risk
```

The mutations are asymmetric on purpose: an assistant can make the engine
*safer* without a human, and needs a human to make it riskier. `tighten_risk`
refuses any value that loosens a limit; `release_halt` refuses while inventory
is stranded.

Two project skills ship in `.claude/skills/`: `arb-report` and `tune-gate`.

---

## Safety

- **Paper by default.** Live needs `mode: live` **and** an environment
  variable **and** a signed acknowledgment file. Three independent accidents
  cannot combine into a live order.
- **Hard guardrails.** Daily loss, drawdown, consecutive losses, order rate,
  cycle notional, per-asset and per-venue exposure, book staleness, error
  bursts. No override parameter exists.
- **Unwinder.** A cycle that fails mid-flight is flattened by escalating IOC
  orders finishing with a market order. A failed unwind trips the kill switch
  and demands a human.
- **Reconciliation.** The double-entry ledger is compared against venue
  balances on a timer; divergence beyond tolerance halts trading.

---

## Layout

```
core/         Decimal math, domain types, clocks, event bus, config
marketdata/   L2 books with checksums + gap detection, normalizer, feeds
exchanges/    Adapter ABC, venue specs, rate limits, 10 venues, paper matching
graph/        Currency graph, Bellman-Ford, cycle enumeration
strategy/     Triangular, polygonal, cross-venue, statistical
execution/    Fees, depth-aware sizing, leg planner, executor, unwinder
learning/     FTRL fill model, online ridge, calibration, bandit, EV gate, drift
risk/         Guardrails, breakers, live-arming triple lock
portfolio/    Double-entry ledger, FIFO lots, P&L attribution
backtest/     Metrics, Monte Carlo, replay
api/ dashboard/  stdlib server + canvas HUD
claude/       MCP server

vault/                The macro agent, built on the same principles
  data/               FRED + market series, point-in-time slicing, caching
  macro/              Growth-inflation regime, curve and cross-asset reads
  signals/            23 named signals; per-signal scoring that gates weight
  nn/                 Pure-Python MLP, purged walk-forward, ensemble
  thesis/             Falsifiable prediction schema + hash-chained journal
  resolve/            Auto-resolution, Brier, Murphy decomposition
  calibration/        The capital gate and recalibration
  hud/                Canvas HUD

web/                  Next.js console: draggable, resizable, near-monochrome
```

~39,000 lines. `CLAUDE.md` documents the constraints that look like style
rules and are not.

---

## Vault: the macro agent

Triangulum hunts arbitrage. Vault does something different with the same
discipline: it makes falsifiable macro calls, grades them automatically, scores
its own calibration, and derives position size from that score. Before there is
a track record, the size is zero — not a small number, zero.

```bash
python3 -m vault --fixtures doctor      # what it can and cannot do right now
python3 -m vault --fixtures signals     # all 23 signals, with provenance
python3 -m vault --fixtures learn       # score signals, fit models, adopt what earns it
python3 -m vault --fixtures simulate    # run agents of known skill through the gate
python3 -m vault --fixtures demo --learn  # seed a record and serve the HUD
```

### Signals earn their weight

Twenty-three signals across ten families — curve, inflation, credit, momentum,
volatility, currency, commodity, labour, liquidity, housing. Each is scored
independently against forward returns, and a signal with no measured edge is
weighted **zero**. Fame is not evidence: the yield curve's recession record is a
fact about the last seventy years, not about whether this implementation of it,
on this data, at this horizon, predicts anything.

Three hurdles, and the first two exist because of bugs found while testing:

- **Both sides.** A signal that read positive 177 times and negative 0 times has
  not been tested as a short signal. The first version of the weighting rule
  tested hit rate against 0.5; equities rose in 84% of the sampled windows, so
  four permanently-bullish signals were credited with an "84% hit rate" for
  demonstrating nothing but the market's own drift.
- **IC significance, not hit rate.** Tested via Fisher's z on the *effective*
  sample size after deflating for window overlap, at a Bonferroni-adjusted
  level. Twenty-three signals tested at the usual 95% will hand you one
  "significant" result from pure noise per twenty tested, every time.
- **Positive spread**, and no silent sign-flipping of anti-correlated signals.

The scorecard also reports what it *cannot* see. With two years of history at a
21-day horizon there are ~34 independent observations and the minimum detectable
IC is about **0.50**, while real macro signals live at 0.02–0.06. The honest
conclusion is not "no signal works" — it is "this sample cannot see a signal of
realistic size", and the remedy is decades of history, not a lower threshold.

### The network

A feed-forward network written from scratch in pure Python — forward pass,
backprop, Adam, dropout, L2, early stopping with weight restoration. About 600
parameters by default, deliberately.

> On frameworks: there is no neural-network library called Obsidian — Obsidian
> is a Markdown note-taking app, and the nearest thing by name is ONNX, a model
> interchange format rather than a training framework. Rather than guess, it is
> implemented directly. That is also the right call here: NumPy is not installed
> in this environment and PyTorch would be a 700MB dependency to train 600
> parameters on a few hundred samples. Everything else in this repo —
> Bellman-Ford, FTRL, RMSProp, Thompson sampling — is implemented the same way.
> If the model ever needs to be bigger, the honest move is to swap in PyTorch
> wholesale, not to grow that file.

Correctness is pinned by XOR, which no linear model can solve, with logistic
regression as the control.

**The network does not get to be in the ensemble because it is a neural
network.** It is fitted alongside logistic regression and the base rate on
identical purged walk-forward folds and judged on the same out-of-sample Brier.
On synthetic data with known structure the harness reaches all three verdicts
correctly: no structure → base rate, linear structure → logistic wins,
interaction structure → network wins. On a few hundred noisy macro samples the
honest expectation is that the linear model wins or ties, and when it does, the
system says so and the weights follow.

Splits are chronological, **purged** (training samples whose label window
overlaps the test block are dropped) and **embargoed**. Scaling is fitted on the
training fold only. Shuffled k-fold on this data would be wrong in the direction
that flatters the model.

### The prior

Signals, model and base rate are blended into one probability, with weights
proportional to how much each beat the base rate out of sample. A source that
did not beat it gets zero, and when nothing beat it the prediction *is* the base
rate at full weight, with the provenance string saying exactly that. Even a
demonstrably skilful blend is capped — it never fully escapes the base rate on a
sample this size.

### Prediction-market arbitrage

`vault arb` scans Kalshi and Polymarket for trades whose payoff is arithmetic
rather than a forecast. This is the one strategy on which $200 is not a
handicap: capacity is measured in hundreds of dollars, so institutions cannot
be bothered and there is no latency race to lose.

The scanner is built around one distinction that decides everything:

|  |  |
|---|---|
| **Mutually exclusive** | at most one outcome resolves YES |
| **Exhaustive** | at least one outcome resolves YES |

Kalshi's `mutually_exclusive` and Polymarket's `negRisk` assert the first.
**Neither asserts the second**, and the two directions of the basket trade
depend on them differently: selling every outcome pays at least n−1 whether or
not the list is complete, while buying every outcome pays **zero** if the real
answer is not on the list.

On a live run over 4,953 contracts it found 8 candidates and locked none of
them. The largest showed +706% on "What will be the 51st state in Trump's
term?" — eight outcomes bought for $23.25 against a "guaranteed" $189. The most
likely outcome is that there is no 51st state, and the basket pays nothing.
A basket priced far below $1 is evidence that outcomes are missing, not a
bargain.

Full results, including two structural facts found by reading live data, are in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

### What 40 years of real data actually says

`vault learn` on real FRED history — 694 non-overlapping 21-day observations
back to 1981 — earns weight for **0 of 23 signals**, and the base rate beats
both models out of sample (0.2392 against 0.2444 for the network and 0.2622 for
logistic regression).

That is a null result, and it is reported as one rather than tuned away. It is
also honest about its own power: the minimum detectable IC at this sample size
is 0.116, while real macro signals live at 0.02–0.06. **A genuine signal at the
top of the published range would still be invisible here.**

### The console

`web/` is a Next.js console on a board where every panel drags, resizes, hides
and restores, with the arrangement persisted. Near-monochrome by design: eleven
greys, one cold accent, two desaturated tints for sign. See `web/README.md`.

---

## License

MIT. No warranty, and specifically no warranty that it will make money.
