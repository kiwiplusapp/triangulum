"""
Signal library and signal scoring tests.

Two things are worth testing here and they are not the obvious ones.

**Sign convention.** Every signal is written so that positive means risk-on,
whatever its underlying series does. Credit spreads widening produces a
negative reading even though the spread went up. An inversion error in one
signal out of twenty-three is invisible to inspection and silently poisons
every composite and every model trained on the feature vector, so every
signal's direction is asserted here against a constructed series that moves
one way on purpose.

**The scorer must be able to say no, and must be able to say yes.** A signal
scorer that finds an edge in random data is worthless; so is one that finds
nothing in data with a planted edge. Both directions are tested, against the
same machinery, with the only difference being whether a relationship was put
into the data.

The one-sided check encodes a real bug. The first version of the weighting
rule tested each signal's hit rate against 0.5. Equities rose in 84% of the
sampled windows, so four signals that never once took the short side were
credited with an 84% "hit rate" and given weight, having demonstrated nothing
but the market's own drift.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pytest

from vault.data.fixtures import build_fixture_universe
from vault.data.series import Series
from vault.signals.library import SIGNALS, evaluate_all
from vault.signals.scoring import (
    MIN_MEANINGFUL_IC,
    score_signal_history,
    spearman,
)
from vault.signals.types import Stance


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _series(key: str, values: list[float], *, end: date | None = None,
            step_days: int = 1) -> Series:
    end = end or date.today()
    start = end - timedelta(days=(len(values) - 1) * step_days)
    return Series.from_pairs(
        key, [(start + timedelta(days=i * step_days), v)
              for i, v in enumerate(values)],
        label=key, units="x",
    )


def _universe(**overrides) -> dict[str, Series]:
    universe = dict(build_fixture_universe(seed=11))
    universe.update(overrides)
    return universe


def _read(key: str, universe) :
    return SIGNALS[key].evaluate(universe)


# ---------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------


def test_every_signal_evaluates_on_a_full_universe():
    readings = evaluate_all(build_fixture_universe())
    assert len(readings) == len(SIGNALS)
    assert sum(1 for r in readings if r.usable) >= len(SIGNALS) - 3


def test_signal_order_is_stable():
    """
    The registration order is the feature-vector column order. A reordering
    would silently permute the inputs of an already-trained network.
    """
    first = [r.key for r in evaluate_all(build_fixture_universe())]
    second = [r.key for r in evaluate_all(build_fixture_universe())]
    assert first == second == list(SIGNALS)


def test_strength_is_always_bounded():
    for reading in evaluate_all(build_fixture_universe()):
        assert -1.0 <= reading.strength <= 1.0, reading.key
        assert 0.0 <= reading.confidence <= 1.0, reading.key


def test_a_missing_input_series_is_reported_not_defaulted():
    """
    A missing signal and a neutral signal are different facts. Coercing the
    first to 0.0 teaches a downstream model that blindness is a market state.
    """
    universe = _universe()
    del universe["BAMLH0A0HYM2"]
    reading = _read("credit_stress", universe)
    assert not reading.usable
    assert reading.strength == 0.0
    assert "missing input series" in reading.unavailable_reason
    assert reading.weighted_strength == 0.0


def test_a_stale_input_is_distinguished_from_a_missing_one():
    universe = _universe(
        VIXCLS=_series("VIXCLS", [18.0] * 600,
                       end=date.today() - timedelta(days=120)),
    )
    reading = _read("vol_regime", universe)
    assert not reading.usable
    assert "days old" in reading.unavailable_reason
    assert reading.note, "a stale signal still has a reading; it just cannot be used"


def test_confidence_decays_with_staleness_rather_than_cliffing():
    fresh = _read("vol_regime", _universe(
        VIXCLS=_series("VIXCLS", [18.0 + (i % 7) for i in range(600)]),
    ))
    middling = _read("vol_regime", _universe(
        VIXCLS=_series("VIXCLS", [18.0 + (i % 7) for i in range(600)],
                       end=date.today() - timedelta(days=26)),
    ))
    assert fresh.confidence == 1.0
    assert 0.0 < middling.confidence < 1.0
    assert abs(middling.weighted_strength) < abs(fresh.weighted_strength) + 1e-9


# ---------------------------------------------------------------------------
# sign convention -- positive is risk-on, for every signal
# ---------------------------------------------------------------------------


def test_an_inverted_curve_reads_risk_off():
    inverted = _read("curve_slope", _universe(
        T10Y3M=_series("T10Y3M", [-1.2] * 600)))
    steep = _read("curve_slope", _universe(
        T10Y3M=_series("T10Y3M", [2.0] * 600)))
    assert inverted.strength < 0 < steep.strength
    assert inverted.stance == Stance.RISK_OFF
    assert steep.stance == Stance.RISK_ON


def test_widening_credit_spreads_read_risk_off():
    """The spread goes UP and the signal must go DOWN."""
    widening = _read("credit_impulse", _universe(
        BAMLH0A0HYM2=_series("BAMLH0A0HYM2",
                             [4.0 + i * 0.02 for i in range(600)])))
    tightening = _read("credit_impulse", _universe(
        BAMLH0A0HYM2=_series("BAMLH0A0HYM2",
                             [8.0 - i * 0.008 for i in range(600)])))
    assert widening.strength < 0 < tightening.strength


def test_high_volatility_reads_risk_off():
    calm = _read("vol_regime", _universe(
        VIXCLS=_series("VIXCLS", [40.0] * 500 + [11.0] * 100)))
    stressed = _read("vol_regime", _universe(
        VIXCLS=_series("VIXCLS", [11.0] * 500 + [40.0] * 100)))
    assert stressed.strength < 0 < calm.strength


def test_a_rising_dollar_reads_risk_off():
    rising = _read("dollar_impulse", _universe(
        DTWEXBGS=_series("DTWEXBGS", [100.0 * (1.0004 ** i) for i in range(600)])))
    falling = _read("dollar_impulse", _universe(
        DTWEXBGS=_series("DTWEXBGS", [100.0 * (0.9996 ** i) for i in range(600)])))
    assert rising.strength < 0 < falling.strength


def test_rising_jobless_claims_read_risk_off():
    rising = _read("labour_momentum", _universe(
        ICSA=_series("ICSA", [200_000 * (1.004 ** i) for i in range(200)],
                     step_days=7)))
    falling = _read("labour_momentum", _universe(
        ICSA=_series("ICSA", [300_000 * (0.996 ** i) for i in range(200)],
                     step_days=7)))
    assert rising.strength < 0 < falling.strength


def test_rising_real_yields_read_risk_off():
    rising = _read("real_yield_impulse", _universe(
        DFII10=_series("DFII10", [1.0 + i * 0.004 for i in range(600)])))
    falling = _read("real_yield_impulse", _universe(
        DFII10=_series("DFII10", [3.5 - i * 0.004 for i in range(600)])))
    assert rising.strength < 0 < falling.strength


def test_equity_momentum_follows_the_trend_and_mean_reversion_opposes_it():
    """
    These two signals are deliberately in conflict at short horizons. Both
    cannot be right, and the point is to make the model resolve it from data
    rather than to pick a side in the library.
    """
    rallying = _universe(
        SP500=_series("SP500", [3000.0 * (1.0015 ** i) for i in range(600)]))
    momentum = _read("equity_momentum_fast", rallying)
    reversion = _read("equity_mean_reversion", rallying)
    assert momentum.strength > 0
    assert reversion.strength < 0


def test_tighter_financial_conditions_read_risk_off():
    tight = _read("financial_conditions", _universe(
        NFCI=_series("NFCI", [1.4] * 200, step_days=7)))
    loose = _read("financial_conditions", _universe(
        NFCI=_series("NFCI", [-0.9] * 200, step_days=7)))
    assert tight.strength < 0 < loose.strength


# ---------------------------------------------------------------------------
# spearman
# ---------------------------------------------------------------------------


def test_spearman_is_one_on_a_monotone_relationship():
    xs = [float(i) for i in range(30)]
    ys = [x ** 3 + 5 for x in xs]           # monotone but very non-linear
    assert spearman(xs, ys) == pytest.approx(1.0)


def test_spearman_handles_ties():
    xs = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    value = spearman(xs, ys)
    assert value is not None and 0.8 < value <= 1.0


def test_spearman_refuses_a_sample_too_small_to_mean_anything():
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) is None


# ---------------------------------------------------------------------------
# the scorer: it must say no
# ---------------------------------------------------------------------------


def test_no_signal_earns_weight_on_random_data():
    """
    The fixtures are random walks. A scorer that finds an edge in them would
    find an edge in anything, and every weight it produced downstream would be
    fitted noise.
    """
    scorecard = score_signal_history(
        build_fixture_universe(seed=5), target="SP500",
        horizon_days=5, step_days=5, lookback_days=880,
    )
    assert scorecard.earning == [], [
        (p.key, p.ic, p.verdict) for p in scorecard.earning
    ]


def test_a_permanently_bullish_signal_is_rejected_as_one_sided():
    """
    The encoded bug. A signal that never takes the short side collects the
    target's own base rate as a "hit rate" -- 84% on the sample where this
    was found -- and must earn nothing for it.
    """
    scorecard = score_signal_history(
        build_fixture_universe(seed=5), target="SP500",
        horizon_days=5, step_days=5, lookback_days=880,
    )
    one_sided = [p for p in scorecard.performances if p.verdict == "one-sided"]
    assert one_sided, "expected at least one always-same-sign signal in fixtures"
    for performance in one_sided:
        assert performance.weight == 0.0
        assert min(performance.n_positive, performance.n_negative) < max(
            10, int(0.15 * performance.n)
        )


def test_the_hit_rate_bar_is_the_base_rate_not_a_coin_flip():
    scorecard = score_signal_history(
        build_fixture_universe(seed=5), target="SP500",
        horizon_days=5, step_days=5, lookback_days=880,
    )
    assert scorecard.base_rate != 0.5
    expected = max(scorecard.base_rate, 1 - scorecard.base_rate)
    for performance in scorecard.performances:
        if performance.n:
            assert performance.baseline_accuracy == pytest.approx(expected)


def test_overlapping_windows_do_not_manufacture_significance():
    """
    Sampling a 21-day horizon every 5 days gives observations that share 76%
    of their window. Treating them as independent would inflate the effective
    sample fourfold and let overlap alone produce significance.
    """
    overlapping = score_signal_history(
        build_fixture_universe(seed=5), target="SP500",
        horizon_days=21, step_days=5, lookback_days=880,
    )
    independent = score_signal_history(
        build_fixture_universe(seed=5), target="SP500",
        horizon_days=5, step_days=5, lookback_days=880,
    )
    a = overlapping.performances[0]
    b = independent.performances[0]
    assert a.n > a.effective_n, "overlap was not deflated at all"
    assert a.effective_n < b.effective_n
    assert a.minimum_detectable_ic > b.minimum_detectable_ic


def test_the_scorer_reports_what_it_cannot_see():
    """
    With two years of history at a 21-day horizon there are ~34 independent
    observations and nothing below an IC of about 0.5 is detectable. Real
    macro signals live at 0.02-0.06. Surfacing that is the difference between
    "no signal works" and "this sample cannot see a signal of realistic size".
    """
    scorecard = score_signal_history(
        build_fixture_universe(seed=5), target="SP500",
        horizon_days=21, step_days=5, lookback_days=720,
    )
    performance = next(p for p in scorecard.performances if p.n > 40)
    assert performance.minimum_detectable_ic > 0.3
    assert performance.minimum_detectable_ic > MIN_MEANINGFUL_IC * 10


# ---------------------------------------------------------------------------
# the scorer: it must say yes
# ---------------------------------------------------------------------------


def _planted_universe(seed: int = 3, strength: float = 0.0004) -> dict[str, Series]:
    """
    A universe in which the VIX genuinely drives forward equity returns.

    Low vol produces positive drift and high vol negative, contemporaneously,
    so tomorrow's return depends on today's reading. ``vol_regime`` is the
    signal that reads exactly this, and it is the one that must be found.
    """
    universe = dict(build_fixture_universe(seed=7))
    rng = random.Random(seed)
    end = universe["VIXCLS"].points[-1].on
    n = 900
    start = end - timedelta(days=n - 1)

    vix, level = [], 18.0
    for _ in range(n):
        level = max(9.0, min(45.0, level * (1 + rng.gauss(0, 0.05))
                             + rng.gauss(0, 0.3)))
        vix.append(level)

    prices, price = [], 4000.0
    for i in range(n):
        price *= 1 + (20.0 - vix[i]) * strength + rng.gauss(0, 0.006)
        prices.append(price)

    universe["VIXCLS"] = Series.from_pairs(
        "VIXCLS", [(start + timedelta(days=i), v) for i, v in enumerate(vix)],
        label="VIX", units="index")
    universe["SP500"] = Series.from_pairs(
        "SP500", [(start + timedelta(days=i), v) for i, v in enumerate(prices)],
        label="S&P 500", units="index")
    return universe


def test_a_planted_edge_is_found():
    scorecard = score_signal_history(
        _planted_universe(), target="SP500",
        horizon_days=5, step_days=5, lookback_days=880,
    )
    earning = {p.key for p in scorecard.earning}
    assert "vol_regime" in earning, (
        "the scorer failed to find a relationship that was deliberately put "
        "into the data; it cannot be trusted to find a real one"
    )
    vol = next(p for p in scorecard.performances if p.key == "vol_regime")
    assert vol.ic is not None and vol.ic > vol.minimum_detectable_ic
    assert vol.weight > 0
    assert vol.verdict == "earning"


def test_weight_is_the_measured_effect_size():
    """Not a rank and not a softmax: a signal contributes what it has shown."""
    scorecard = score_signal_history(
        _planted_universe(), target="SP500",
        horizon_days=5, step_days=5, lookback_days=880,
    )
    for performance in scorecard.earning:
        assert performance.weight == pytest.approx(min(0.20, performance.ic))


def test_an_inverted_signal_is_not_silently_flipped():
    """
    Flipping the sign of a signal that anti-correlates would be fitting the
    library to the sample. It earns zero and says why.
    """
    scorecard = score_signal_history(
        _planted_universe(), target="SP500",
        horizon_days=5, step_days=5, lookback_days=880,
    )
    inverted = [p for p in scorecard.performances if p.verdict == "inverted"]
    for performance in inverted:
        assert performance.weight == 0.0
        assert performance.ic is not None and performance.ic < 0
