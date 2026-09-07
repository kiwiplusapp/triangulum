"""
Learning-layer tests.

These encode three bugs that were found by running the models against synthetic
data with a known ground truth, and that would each have silently destroyed the
live system:

1. AdaGrad's monotonically decaying step size froze the slippage regression
   before convergence (recovered coefficient 2.8 against a true 8.0). Fixed by
   switching to RMSProp -- see ``test_ridge_converges_to_known_coefficient``.

2. Unscaled features spanning five orders of magnitude (min_depth_ratio 6000
   alongside spread_to_edge_ratio 0.028) diverged the regression to a 6,637 bps
   prediction. Fixed by scaling and clipping -- see ``test_features_are_bounded``.

3. The feature extractor recomputed book ages from live books instead of using
   the ages recorded at detection, collapsing the single most predictive feature
   to zero whenever detection and evaluation happened in the same tick. See
   ``test_recorded_book_age_is_used``.
"""

from __future__ import annotations

import math
import random

import pytest

from triangulum.core.decimal_math import D
from triangulum.core.types import (
    Asset, AssetClass, CyclePlan, ExecutionMode, Leg, LegPlan, Opportunity,
    OrderType, Side, TimeInForce,
)
from triangulum.learning.bandit import ThompsonBandit
from triangulum.learning.ev_gate import EVGate, GateVerdict
from triangulum.learning.features import FEATURE_CLIP, FeatureExtractor
from triangulum.learning.online_lr import FTRLProximal, OnlineRidge
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.normalizer import SymbolNormalizer


@pytest.fixture
def market():
    norm = SymbolNormalizer()
    books = BookManager(max_age_ns=10**12)
    books.mark_connected("binance")

    def mk(b, q, bid, ask, size="1000"):
        s = norm.register("binance", f"{b}{q}", b, q)
        books.apply_snapshot(s, [(D(bid), D(size))], [(D(ask), D(size))])
        return s

    symbols = [
        mk("BTC", "USDT", "60000", "60001"),
        mk("ETH", "USDT", "3000", "3000.5"),
        mk("ETH", "BTC", "0.04945", "0.0495", "500"),
    ]
    return norm, books, symbols


def _legs(symbols):
    usdt = Asset("USDT", AssetClass.STABLECOIN)
    btc, eth = Asset("BTC"), Asset("ETH")
    return (
        Leg(symbols[0], Side.BUY, usdt, btc),
        Leg(symbols[2], Side.BUY, btc, eth),
        Leg(symbols[1], Side.SELL, eth, usdt),
    ), usdt


def _make(legs, usdt, edge, age_ms, notional="95"):
    opp = Opportunity(
        "o", legs, usdt, D(str(edge)), D("100"), 0,
        book_ages_ns=(int(age_ms * 1e6),) * len(legs), venues=("binance",),
    )
    lps = tuple(
        LegPlan(l, D(notional), D(notional), D("0.001"), D("1"),
                OrderType.LIMIT, TimeInForce.IOC, D("0.01"), usdt)
        for l in legs
    )
    plan = CyclePlan(
        "c", lps, usdt, D(notional), D(notional), ExecutionMode.TTT,
        D(str(edge)), D("30"), D("2"), D(str(edge)),
    )
    return opp, plan


# --------------------------------------------------------------------------


def test_ridge_converges_to_known_coefficient():
    """RMSProp must recover the true coefficient; AdaGrad stalled at ~2.8/8.0."""
    rng = random.Random(7)
    model = OnlineRidge(learning_rate=0.05, bias_indices=frozenset({1}))
    for _ in range(10_000):
        depth = rng.uniform(0.1, 3.0)
        model.update({1: 1.0, 2: 1.0 / depth}, 8.0 / depth + rng.gauss(0, 1.0))

    assert model._w[2] == pytest.approx(8.0, rel=0.05)
    assert abs(model._w.get(1, 0.0)) < 0.6
    # RMSE should approach the irreducible noise level of 1.0.
    assert model.recent_rmse < 1.3


def test_ftrl_beats_the_base_rate():
    rng = random.Random(11)
    model = FTRLProximal(alpha=0.15, l1=0.1)

    def truth(age, edge):
        return 1 / (1 + math.exp(-(2.0 - 0.02 * age + 0.35 * edge)))

    for _ in range(20_000):
        age, edge = rng.uniform(0, 400), rng.uniform(0, 12)
        features = {1: 1.0, 2: age / 100.0, 3: edge}
        model.update(features, 1 if rng.random() < truth(age, edge) else 0)

    assert model.skill > 0.3, "model must beat always-predict-the-base-rate"
    assert model.predict({1: 1.0, 2: 0.1, 3: 8}) > 0.9
    assert model.predict({1: 1.0, 2: 3.5, 3: 8}) < 0.3


def test_features_are_bounded(market):
    """Unscaled features previously spanned 0.001 to 6000 and diverged the model."""
    _norm, books, symbols = market
    legs, usdt = _legs(symbols)
    opp, plan = _make(legs, usdt, 6.0, 300)
    features = FeatureExtractor(books).extract(opp, now_ns=1, plan=plan)
    for name, value in features.values.items():
        assert -FEATURE_CLIP <= value <= FEATURE_CLIP, f"{name} = {value} is unbounded"


def test_recorded_book_age_is_used(market):
    """The age recorded at detection must survive into the feature vector."""
    _norm, books, symbols = market
    legs, usdt = _legs(symbols)
    extractor = FeatureExtractor(books)

    fresh = extractor.extract(_make(legs, usdt, 6.0, 20)[0], now_ns=1)
    stale = extractor.extract(_make(legs, usdt, 6.0, 300)[0], now_ns=1)

    assert stale.get("max_book_age_ms") > fresh.get("max_book_age_ms")
    assert fresh.get("max_book_age_ms") == pytest.approx(20 / 250.0, rel=1e-6)


def test_gate_rejects_stale_opportunities_that_look_profitable(market):
    """
    The core claim: five opportunities all showing +6 bps, differing only in
    book freshness. A naive bot takes all five. The gate must not.
    """
    _norm, books, symbols = market
    legs, usdt = _legs(symbols)

    extractor = FeatureExtractor(books)
    fill = FTRLProximal(alpha=0.3, l1=0.02, l2=0.5)
    slippage = OnlineRidge(learning_rate=0.02)
    gate = EVGate(
        fill, slippage, extractor, min_ev_bps=0.5, min_samples_before_trust=300,
        seed=5, exploration_floor=0.0, default_unwind_cost_bps=8.0,
    )

    rng = random.Random(1)
    for _ in range(6000):
        age, edge = rng.uniform(0, 400), rng.uniform(1, 10)
        opp, plan = _make(legs, usdt, edge, age)
        decision = gate.evaluate(opp, plan, now_ns=1)
        p_true = 1 / (1 + math.exp(-(3.0 - 0.018 * age)))
        filled = rng.random() < p_true
        gate.observe(decision, filled=filled, realized_bps=edge * 0.7 if filled else 0.0)

    assert fill.skill > 0.2
    assert gate.calibrator.calibration_error < 0.10

    verdicts = {}
    for age in (20, 80, 150, 250, 350):
        opp, plan = _make(legs, usdt, 6.0, age)
        verdicts[age] = gate.evaluate(opp, plan, now_ns=1)

    assert verdicts[20].accept, "a fresh +6 bps cycle must be taken"
    assert not verdicts[250].accept, "a 250ms-stale +6 bps cycle must be rejected"
    assert not verdicts[350].accept
    # Fill probability must decrease monotonically with staleness.
    probabilities = [verdicts[a].fill_probability for a in (20, 80, 150, 250, 350)]
    assert probabilities == sorted(probabilities, reverse=True)


def test_gate_records_a_reason_for_every_rejection(market):
    _norm, books, symbols = market
    legs, usdt = _legs(symbols)
    gate = EVGate(
        FTRLProximal(), OnlineRidge(), FeatureExtractor(books),
        exploration_floor=0.0, seed=1,
    )
    opp, plan = _make(legs, usdt, -5.0, 20)
    decision = gate.evaluate(opp, plan, now_ns=1)
    assert not decision.accept
    assert decision.verdict == GateVerdict.REJECT_EDGE
    assert decision.reason
    assert "bps" in decision.explain()


def test_bandit_finds_the_best_arm_and_counts_pulls_once():
    """
    Ground truth: ``mtt`` fills least often but pays far more, so it has the
    highest EV. The bandit must discover that from outcomes alone.

    The pull-count assertion guards a double-counting bug: ``update`` wrote to
    both the global and the regime bucket, which are the same object when no
    regime is supplied, applying the decay twice per observation.
    """
    rng = random.Random(3)
    bandit = ThompsonBandit(seed=11, decay=0.9995)
    truth = {
        "ttt_safe": (0.85, 1.2), "ttt_tight": (0.55, 2.0), "mtt": (0.45, 6.5),
        "tmt": (0.40, 4.0), "ttt_half": (0.88, 0.9), "ttt_small": (0.92, 0.4),
    }
    for _ in range(4000):
        arm = bandit.select()
        p, bps = truth[arm.name]
        filled = rng.random() < p
        bandit.update(
            arm.name, filled=filled,
            realized_bps=rng.gauss(bps, 1.5) if filled else 0.0,
        )

    assert bandit.best_arm() == "mtt"
    assert sum(r["pulls"] for r in bandit.leaderboard()) == 4000
    # It should exploit heavily but never abandon the others entirely.
    assert all(r["pulls"] > 0 for r in bandit.leaderboard())
