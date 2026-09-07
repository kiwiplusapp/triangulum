# Expectations

This document exists because the engine's own numbers say something the brief
did not. It is here in full, with the arithmetic, so the decision is yours and
not mine.

---

## 1. The target, stated plainly

**1% per day, net.** Compounded, that is:

| Horizon | Multiple | Return |
|---|---|---|
| 30 days | 1.35× | +35% |
| 1 year | 37.8× | **+3,678%** |
| 3 years | 54,000× | +5,400,000% |

$100 becomes $5.4 million in three years. There is no mechanism by which that
is sustainable, and the absence is not an opinion — if such a mechanism existed
at retail scale, capital would flow into it until the return fell to the cost
of capital. That process is what "efficient market" describes, and it operates
on a timescale of hours in crypto.

The fallback you offered, **20% per month**, is +792% per year. The same
argument applies with slightly smaller numbers.

For calibration: Renaissance Technologies' Medallion fund, the best-documented
track record in the history of quantitative trading, returned roughly 39% per
year net over three decades — and it closed to outside capital because the
strategies did not scale past a few billion dollars.

None of this means don't build it. It means the target is a target and not a
plan, and the engine is built to tell you the difference every day rather than
to flatter it.

---

## 2. What the code found

Everything below came out of running this repository, not from reading. Each is
reproducible: `pytest -q` and `python -m triangulum doctor`.

### 2.1 Quantization drag is the small-account tax

Every venue quantizes order quantities to a **lot step**. Truncating to it
loses, on average, half a step. The cost in basis points is

```
drag = (lot_step × price ÷ 2) ÷ notional × 10,000
```

The numerator is the **value of one lot**. The denominator is your size. So the
cost is *inversely proportional to capital* — and nobody writing about
triangular arbitrage mentions it, because everyone writing about it assumes
institutional size, where the term vanishes.

The same +100 bps opportunity, on the same venue, at different capital:

| Capital | Net captured | Fees | Drag | Share of edge captured |
|---|---|---|---|---|
| $100 | +9.98 bps | 30.0 | **60.0** | **14.1%** |
| $250 | +45.94 bps | 30.0 | 23.9 | 65.1% |
| $500 | +63.93 bps | 30.0 | 12.0 | 90.6% |
| $1,000 | +63.93 bps | 30.0 | 6.0 | 90.6% |
| $10,000 | +70.22 bps | 30.0 | 0.6 | **99.5%** |

At $100 you keep one seventh of an edge that a $10,000 account keeps almost all
of. Not because of fees — those are identical — but because of rounding.

### 2.2 Instrument choice matters more than venue choice

Drag depends on the *value of one lot*, which varies by fifty times across
instruments on the same venue under the same fee schedule:

| Instrument | Lot step | Price | One lot | Drag per 3-leg cycle at $95/leg |
|---|---|---|---|---|
| BTCUSDT | 0.00001 | $62,000 | $0.620 | **94.7 bps** |
| ETHUSDT | 0.0001 | $3,050 | $0.305 | 47.4 bps |
| SOLUSDT | 0.001 | $148 | $0.148 | 23.7 bps |
| DOGEUSDT | 1 | $0.10 | $0.100 | 15.8 bps |
| XRPUSDT | 0.1 | $0.542 | $0.054 | 8.7 bps |
| ADAUSDT | 0.1 | $0.447 | $0.045 | 7.1 bps |
| TRXUSDT | 0.1 | $0.119 | $0.012 | **1.9 bps** |

**A BTC-anchored triangle at $100 is fifty times more expensive than a
TRX-anchored one.** This is encoded as a hard screen in `exchanges/spec.py` and
enforced in the graph builder: instruments whose lot value exceeds the drag
budget never become edges.

### 2.3 The break-even, per venue

`python -m triangulum doctor --capital 100 --maker-legs 1`:

| Venue | Fees | Drag (low-lot) | Market must give you |
|---|---|---|---|
| **MEXC** | 10.0 | 1.9 | **15.5 bps** |
| Gate.io | 20.2 | 1.9 | 28.8 bps |
| OKX | 22.4 | 1.9 | 31.6 bps |
| Binance | 22.5 | 1.9 | 31.7 bps |
| KuCoin | 24.0 | 1.9 | 33.7 bps |
| Kraken | 105.0 | 1.9 | 139.0 bps |
| Coinbase | 160.0 | 1.9 | 210.5 bps |

**Observed triangular dislocations on a liquid venue: 1–8 bps, lasting
50–300 ms.**

The best case needs 15.5 bps. The market typically offers 1–8. That gap is the
answer to the question, and it does not close by trying harder.

### 2.4 The frequency

Against a synthetic market calibrated to realistic dislocation sizes, over
8,082 valued cycles:

- median cycle edge: **−51 bps**
- 95th percentile: −29.9 bps
- best observed: +3.52 bps
- **positive-edge cycles: 1 in 8,082 (0.01%)**

That distribution is the strategy. Most of what looks like an opportunity is
the bid-ask spread of the mid prices you computed it from.

---

## 3. What was asked for, and what is true

**"Avoid cryptocurrencies."** Understood, and not possible for this strategy.
Triangular arbitrage needs a *cycle* — a path A→B→C→A of independently priced
markets. Outside crypto and the interbank market, that cycle does not exist:

- **Retail FX** brokers quote crosses *derived* from the majors. EUR/JPY is
  computed from EUR/USD and USD/JPY plus a markup. The triangle is closed by
  construction and its product is below 1 by exactly the summed spread.
  `exchanges/oanda_fx.py::demonstrate_closed_triangle` computes this live from
  a broker's own quotes so you can verify it rather than take my word.
- **Equities** form a *star*, not a mesh: AAPL trades against USD, MSFT trades
  against USD, AAPL does not trade against MSFT. Every path back to AAPL
  retraces itself. That is a round trip paying the spread twice, not an
  arbitrage. The graph's `connectivity_report()` detects and reports this shape.
- The real equity arbitrages (ADR/ordinary, ETF/NAV, index/constituent) need
  creation-unit size or multi-market data feeds costing more per month than the
  account holds.

Both adapters are implemented — for statistical mean-reversion, which is a real
edge, and which is a *directional* strategy with drawdowns, not a riskless one.

**"Learn and optimise on the fly."** Built, and it is the part that most
changes the outcome. Not because it finds more opportunities, but because it
*rejects* the ones that look profitable and are not. The measured result: five
cycles all displaying +6 bps, differing only in book staleness — a naive
detector takes all five; the gate takes two, and the three it rejects are the
losing ones.

**"20,000+ lines."** ~21,000 of Python plus the HUD. Every module does work
that changes a decision; none of it is padding.

---

## 4. What a realistic plan looks like

If you want to pursue this, in this order:

**Phase 1 — record, cost nothing (2–4 weeks).**
`python -m triangulum record --minutes 10080`. Collect real order books. This
is the input to everything; there is no downloadable substitute and no way to
skip it.

**Phase 2 — measure, cost nothing (1 week).**
`python -m triangulum backtest --data data/recordings`. Now you have the actual
distribution of opportunities on your venue, at your fee tier, at your size —
not my synthetic estimate. Read `sample_adequacy` before believing any ratio.

**Phase 3 — paper, cost nothing (4+ weeks).**
`python -m triangulum paper`. Real data, simulated fills, real latency. The
models learn. You need a few hundred cycles before the fill model's output
means anything.

**Phase 4 — decide.**
If paper shows positive expectancy over 500+ cycles with a Monte Carlo 5th
percentile above zero, you have something worth risking money on. If it does
not, you have learned that for free, which is the point.

**Phase 5 — live, small.**
The triple lock exists so this cannot happen by accident.

### What actually moves the needle

Ranked by measured effect:

1. **Capital.** $1,000 captures 90% of an edge; $100 captures 14%. This
   single variable dominates everything else in this document.
2. **Fee tier.** MEXC's 0/5 schedule versus Coinbase's 40/60 is a 13× difference
   in break-even. Verify your actual tier on your account's fee page — the
   published entry tier is a guess about you and it is usually wrong.
3. **Maker legs.** One post-only leg saves the maker/taker spread. Costs fill
   risk, which is exactly what the bandit is there to measure.
4. **Instrument selection.** Fifty times, as shown above.
5. Latency. Real, and the last thing to optimise, not the first.

---

## 5. The honest summary

Everything you asked for is built and works. The engine detects arbitrage
correctly (validated to 0.000 bps against hand calculation), sizes it against
real venue rules, decides with a learned expected-value model, executes with an
unwinder for the failure path, and reports honestly.

What it will not do is turn $100 into $130 in a month. Not because it is badly
built, but because at $100 the arithmetic does not permit it: the market offers
1–8 bps, the cheapest configuration needs 15.5, and the gap is structural.

The most valuable thing this repository can do for you is give you that number
before you fund an account, rather than after.

If you have $2,000–5,000 rather than $100, the picture changes materially — the
drag term falls by 20–50× and the break-even drops to roughly the fee bill.
Still not 1% per day. But no longer arithmetically impossible.
