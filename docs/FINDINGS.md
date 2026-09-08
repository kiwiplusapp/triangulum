# Findings

Measured results, with the configuration that produced them. Every number here
is reproducible from the commands shown.

---

## 1. Macro signals on 40 years of real data: nothing earns weight

**Command**

```bash
python3 -m vault learn --target NASDAQ100 --horizon 21 --step 21 --lookback 14600
```

**Data.** 23 series pulled from FRED and Coinbase, longest history available:
`DFF` from 1981 (16,421 points), `NASDAQ100` from 1986 (10,250), `VIXCLS` from
1990 (9,268). Non-overlapping 21-day windows, 694 independent observations.

**Signal scorecard: 0 of 23 signals earning weight.**

| | |
|---|---|
| Observations | 694 (non-overlapping, no deflation needed) |
| Base rate | NASDAQ100 rose in 61.8% of windows |
| Minimum detectable \|IC\| | 0.116 at Bonferroni-adjusted alpha |
| Largest \|IC\| observed | 0.138 (`equity_momentum_fast`, n=166) |
| Signals clearing every hurdle | **0** |

**Walk-forward training: the base rate wins.**

| Model | Out-of-sample Brier |
|---|---|
| **Base rate** | **0.2392** |
| Network | 0.2444 |
| Logistic | 0.2622 |

Neither model beats predicting the historical frequency of an up move. Per the
rule in `vault/nn/train.py`, nothing is used downstream and the predictor keeps
returning the base rate.

### What this does and does not mean

It does **not** mean macro is unpredictable in general. It means *these 23
signals, at these horizons, on this target, over this sample* carry no
detectable edge — and that the sample is only just large enough to see an
effect of realistic size. Published out-of-sample macro ICs cluster around
0.02–0.06; at 694 observations the detection floor is 0.116. **A real signal at
the top of the published range would still be invisible here.**

That is the honest summary: this is a null result at a sample size that could
not have detected a realistic positive one either. Shorter horizons buy more
observations (2,921 at 5 days, floor 0.057) at the cost of trading a noisier
target.

### The one place something passed, and why it was thrown out

Scored against `VIXCLS` at a 5-day horizon, three signals cleared every
hurdle — `vol_regime` at IC +0.200, `vol_of_vol` at +0.166,
`equity_momentum_fast` at +0.143.

`vol_regime` computes a VIX percentile. It was being scored on its ability to
predict VIX. That is the target's own autocorrelation wearing a signal's
clothes — real as a statistical property, worthless as a forecast, and not
information the market has failed to price.

A `self-referential` hurdle was added to `vault/signals/scoring.py` as a
result: any signal whose inputs include the target earns zero and says why.
That removed `vol_regime`. The remaining two read S&P realised vol to predict
implied vol, which is a genuine lead-lag relationship but is still vol
predicting vol — and the tradeable expression of it carries roll costs that
dwarf an IC of 0.17.

The general lesson is worth more than the specific one: **volatility is more
forecastable than direction.** That points at variance strategies rather than
directional macro calls.

---

## 2. Prediction-market arbitrage: no locked trades, and the reason matters

**Command**

```bash
python3 -m vault arb --budget 200
```

**Universe scanned.** 618 event groups, **4,953 contracts** — 600 groups from
Kalshi, 18 from Polymarket with full CLOB depth.

**Result: 8 candidate baskets, 0 of them locked arbitrage.**

| Candidate | Apparent return | Verdict |
|---|---|---|
| "What will be the 51st state in Trump's term?" | +706% | not exhaustive |
| "Who will the next Pope be?" | +240% | not exhaustive |
| "New York Democratic Senate nominee in 2028?" | +7.0% | not exhaustive |
| "Taiwan presidential election winner?" | +4.5% | not exhaustive |
| …4 more | +3% to +21% | not exhaustive |

Every one is the same trade: buy YES on every listed outcome for less than $1
and collect the certain dollar. The 51st-state basket costs $23.25 for a
"guaranteed" $189.

**It is not guaranteed.** Eight outcomes are listed — DC, Puerto Rico, Canada,
Greenland, Venezuela, Guam, Colombia, Cuba — and by far the most likely
outcome is that there is no 51st state at all, in which case the basket pays
zero and the entire $23.25 is lost. The YES bids across those eight sum to
$0.088, which is not a bargain; it is the market pricing in exactly that.

### The distinction the whole scanner is built around

Two structural facts, constantly conflated:

- **Mutually exclusive** — at most one outcome resolves YES.
- **Exhaustive** — at least one outcome resolves YES.

Kalshi's `mutually_exclusive` flag and Polymarket's `negRisk` flag both assert
the first. **Neither asserts the second.** And the two sides of the basket
trade depend on them differently:

| Direction | Condition | Needs exhaustive? |
|---|---|---|
| **Sell all** (buy every NO) | Σ yes_bid > 1 | **No** — pays ≥ n−1 either way |
| **Buy all** (buy every YES) | Σ yes_ask < 1 | **Yes** — pays 0 if none resolves |

The sell side survives an incomplete outcome list, because if nothing resolves
YES every short wins. The buy side does not. That asymmetry is why all eight
candidates are buy-side and all eight are unlocked.

### Two structural facts found by reading live data

**Kalshi YES and NO share one book.** Empirically `no_ask == 1 − yes_bid`
exactly, and the API carries `yes_bid_size_fp` / `yes_ask_size_fp` with no NO
equivalents. Therefore `yes_ask + no_ask == 1 + spread ≥ 1` identically: **the
classic "buy both sides for under $1" arbitrage is arithmetically impossible
on a single Kalshi market.** Polymarket tokenises YES and NO into separate CLOB
books, so there the same check is meaningful — and it is the cheapest real
opportunity the scanner looks for.

**Range markets are exhaustive even when they look cheap.** A Kalshi market on
2034 GDP growth listed 14 buckets from "0.0% or Below" to "6.1% or Above" —
genuinely covering the number line — while its YES bids summed to 0.52, because
a market nine years out is quoted wide and thin. A price-based test reads that
as "outcomes are missing". It is not: it is illiquid, not incomplete.
Structural detection of tiled numeric ranges now runs before any price
heuristic and reclassified 23 groups on the live run.

### What has not been verified

**The fee models.** Kalshi publishes no fee fields on any endpoint used here.
Polymarket reports `takerBaseFee: 1000` with a `feeType` of `politics_fees` or
`sports_fees_v3` and **does not state the scale of 1000** anywhere in the
response. The model assumes 10 basis points. If the true scale is 1%, every net
edge below 2% reported by the scanner is wrong by enough to flip its sign.

Every opportunity therefore reports gross and net side by side with the fee
assumption named, and an unverified model marks the opportunity unlocked. Before
trading: place one small order on each venue and read the actual fee off the
fill.

---

## 3. What this says about the $200 question

The compounding arithmetic is in the README. The measured results above add the
part that is specific to this system:

- The **macro side has no demonstrated edge** and, at this sample size, could
  not have demonstrated a realistic one. It is not ready to size anything, and
  the capital gate correctly refuses to.
- The **arbitrage side found no locked trades** in ~5,000 live contracts. That
  is the expected result: these venues are watched, and a genuinely risk-free
  basket does not sit around waiting.

The honest read is that the scanner's value is in **being run repeatedly**
rather than once. Locked arbitrage appears in bursts — around news, new
listings, and resolution ambiguity — and lasts minutes. A single scan finding
nothing says almost nothing about the strategy; a hundred scans finding nothing
would.
