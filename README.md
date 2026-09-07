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
```

~21,000 lines of Python. `CLAUDE.md` documents the constraints that look like
style rules and are not.

---

## License

MIT. No warranty, and specifically no warranty that it will make money.
