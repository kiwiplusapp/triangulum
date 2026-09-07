# Triangulum — notes for Claude Code

An autonomous multi-venue arbitrage engine. Read this before changing anything;
several of the constraints below look like arbitrary style rules and are not.

## Run it

```bash
pip install -e ".[dev]"

python -m triangulum doctor --capital 100   # start here: what can this capital do?
python -m triangulum demo                   # synthetic market + HUD on :8787
python -m triangulum paper                  # live data, simulated fills
python -m triangulum backtest --data data/recordings

pytest -q                                   # the test suite
```

`doctor` first, always. It answers in basis points what edge the market must
provide before a given configuration breaks even. That number is more useful
than any backtest and it takes a second to produce.

## The five rules

**1. Money is `Decimal`. Never float.**
`triangulum/core/decimal_math.py` holds the arithmetic. Every rounding is
explicit about direction, and the house rule is *always round against
yourself*: quantities down, fees up, prices adversely. Applied consistently,
projected P&L becomes a lower bound on realized P&L, which is worth more than a
fraction of a basis point of precision. A `float` in a price path is a bug even
when the test passes.

**2. Fees live inside the conversion rate, never subtracted at the end.**
Spot venues charge a BUY fee in the asset received (base) and a SELL fee in the
asset received (quote). Fees also compound: three 10 bps legs cost 29.97 bps,
not 30. `execution/fees.py::net_conversion_rate` is the only correct
formulation; use it.

**3. Nothing bypasses the EV gate or the guardrails.**
`learning/ev_gate.py` is the single point where capital is risked, and
`risk/guardrails.py` holds the hard limits. There is no override parameter and
adding one would be a regression. Live trading needs three independent locks
(`risk/live_arming.py`); they are independent on purpose.

**4. The backtest runs the production code path.**
Backtest, paper and live differ only in the injected clock and adapters. There
is no `if backtest:` branch anywhere and there must never be one — that is the
property that stops a backtest from flattering the strategy.

**5. A silent failure is worse than a loud one.**
When the engine cannot trade it must say why, specifically, with the remedy.
See `strategy/triangular.py::_explain_empty` for the pattern. An engine that
scans an empty template set forever without comment is the most confusing
failure this system can present.

## The numbers that shape the design

These came out of running the code, not from the literature. If a change makes
one of them worse, that is the thing to discuss.

| Finding | Value |
|---|---|
| Quantization drag, $100 account, 3-leg BTC cycle | **94.7 bps** |
| Same cycle at $10,000 | **0.60 bps** |
| Share of a +100 bps edge captured at $100 | **14%** |
| Same at $10,000 | **99.5%** |
| Cheapest venue's break-even at $100, low-lot instruments | **15.5 bps** |
| Typical triangular dislocation, liquid venue | **1-8 bps, 50-300ms** |
| Fraction of valued cycles with positive edge (synthetic, realistic) | **1 in 8,082** |

The first four are one fact: **drag is inversely proportional to notional**, so
a small account pays a tax that no fee tier or execution cleverness recovers.
This is why `exchanges/spec.py` carries a lot-value screen and why the graph
refuses to build edges that fail it.

## Layout

```
core/         Decimal math, domain types, clocks, event bus, config
marketdata/   L2 books (checksums, sequence gaps), normalizer, feeds, recorder
exchanges/    Adapter ABC, venue specs, rate limits, 10 venues, paper matching
graph/        Currency graph, Bellman-Ford negative cycles, cycle enumeration
strategy/     Triangular, polygonal, cross-venue, statistical
execution/    Fees, depth-aware sizing, leg planner, executor, unwinder
learning/     Features, FTRL fill model, online ridge, calibration, bandit,
              EV gate, regime, drift, model store
risk/         Guardrails, circuit breakers, live-arming triple lock
portfolio/    Double-entry ledger, FIFO lots, P&L attribution
backtest/     Metrics, Monte Carlo, replay driver
api/ dashboard/  stdlib HTTP + WebSocket server, canvas HUD
claude/       MCP server exposing state and safe controls
```

## Things that will bite you

- **Reference notional is denominated in the base currency, not each pair's
  quote.** Sizing a BTC-quoted pair at "100" once meant asking for $6M of depth
  and silently marking every such edge THIN. `graph/currency_graph.py` converts
  per-pair through a valuation pass. There is a regression test.
- **Features must be scaled.** They spanned five orders of magnitude once and
  diverged the slippage regression to a 6,637 bps prediction. See
  `learning/features.py::FEATURE_SCALES`.
- **The online models use RMSProp, not AdaGrad.** AdaGrad's step size decays
  monotonically to zero, so the model freezes and stops adapting to regime
  changes — which defeats online learning entirely. It stalled at a coefficient
  of 2.8 against a true 8.0.
- **Never scale a `CyclePlan` by a multiplier.** It takes leg quantities off the
  venue's lot grid and the order is rejected for LOT_SIZE. Re-plan at the
  smaller size instead.
- **The unwinder escalates in basis points, not ticks.** A tick is absolute, so
  "5 ticks through" is 0.016% on ETH and 0.4% on TRX. Escalating by ticks left
  positions stranded on high-priced instruments.
- **A partial fill is not a successful unwind.** Treating it as one abandons the
  remainder and reports an absurd cost. The unwinder loops until flat.

## Working on this

Run `pytest -q` before and after. `tests/unit/test_graph.py` and
`tests/unit/test_learning.py` encode the bugs listed above; if one of them
fails, you have reintroduced a specific, documented mistake.

When adding a venue, prefer `exchanges/generic.py`'s profile table. Write a
dedicated module only when the venue's behaviour genuinely diverges — Binance's
snapshot/diff reconciliation, Kraken's checksum-only integrity, OKX's
three-credential auth.

Do not add a dependency to the control plane (`api/`, `dashboard/`). It is
stdlib-only so that it comes up when the rest of the stack is broken, which is
exactly when you need it.
