"""
Execution-layer tests: fees, sizing, quantization, and the planner.

The sizing tests encode the central finding of this project -- that the same
opportunity is worth wildly different amounts at different capital, because lot
quantization is inversely proportional to notional. If ``test_capital_scaling``
starts passing with flat numbers, the drag model has been broken.
"""

from __future__ import annotations

import pytest

from triangulum.core.clock import SystemClock
from triangulum.core.decimal_math import (
    D, ONE, ZERO, ceil_to_step, floor_to_step, quantization_drag_bps,
    round_price_for_side,
)
from triangulum.core.types import Asset, AssetClass, Leg, Liquidity, Side
from triangulum.exchanges.spec import (
    SymbolSpec, assess_capital_adequacy, max_lot_value_for_budget,
)
from triangulum.execution.fees import FeeEngine
from triangulum.execution.sizing import CycleSizer, SizingFailure
from triangulum.graph.currency_graph import CurrencyGraph
from triangulum.graph.cycle_enum import CycleEnumerator
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.normalizer import SymbolNormalizer


# ── decimal primitives ─────────────────────────────────────────────────────

def test_floor_to_step_never_exceeds_input():
    assert floor_to_step(D("0.123456"), D("0.0001")) == D("0.1234")
    assert floor_to_step(D("4580.5149"), D("0.1")) == D("4580.5")
    assert floor_to_step(D("0.0000049"), D("0.00001")) == ZERO


def test_price_rounding_is_biased_against_us():
    """A marketable order must stay marketable after quantization."""
    # Buying: round UP so the limit is at least as aggressive as intended.
    assert round_price_for_side(D("60000.004"), D("0.01"), "buy") == D("60000.01")
    # Selling: round DOWN.
    assert round_price_for_side(D("60000.006"), D("0.01"), "sell") == D("60000.00")


def test_quantization_drag_is_inverse_to_notional():
    """Halving the notional doubles the drag. This is the whole thesis."""
    small = quantization_drag_bps(D("33"), D("0.00001"), D("60000"))
    large = quantization_drag_bps(D("3300"), D("0.00001"), D("60000"))
    assert float(small) == pytest.approx(90.91, abs=0.01)
    assert float(large) == pytest.approx(0.909, abs=0.001)
    assert float(small / large) == pytest.approx(100.0, rel=1e-6)


def test_lot_value_screen_admits_only_fine_grained_instruments():
    budget = max_lot_value_for_budget(D("95"), D("5"), legs=3)
    assert float(budget) == pytest.approx(0.03167, abs=1e-4)
    assert D("0.1") * D("0.1187") <= budget          # TRX admitted
    assert D("0.00001") * D("62000") > budget        # BTC rejected


# ── fees ───────────────────────────────────────────────────────────────────

def test_cycle_fees_compound_multiplicatively():
    """Three 10 bps legs cost 29.97 bps, not 30.00."""
    fees = FeeEngine()
    total = fees.cycle_fee_bps(["binance"] * 3, [Liquidity.TAKER] * 3)
    assert float(total) == pytest.approx(29.97, abs=0.01)
    assert float(total) < 30.0


def test_fee_is_charged_in_the_received_asset():
    """BUY fees are in base, SELL fees in quote. Reversing this is a classic bug."""
    norm = SymbolNormalizer()
    symbol = norm.register("binance", "BTCUSDT", "BTC", "USDT")
    fees = FeeEngine()

    buy = fees.quote_fee(symbol, Side.BUY, D("1"), D("60000"), Liquidity.TAKER)
    assert buy.asset == symbol.base
    assert float(buy.amount) == pytest.approx(0.001)      # 10 bps of 1 BTC

    sell = fees.quote_fee(symbol, Side.SELL, D("1"), D("60000"), Liquidity.TAKER)
    assert sell.asset == symbol.quote
    assert float(sell.amount) == pytest.approx(60.0)      # 10 bps of 60000 USDT


def test_net_conversion_rate_composes():
    """The multiplicative form is what makes cycle returns compose correctly."""
    norm = SymbolNormalizer()
    symbol = norm.register("binance", "BTCUSDT", "BTC", "USDT")
    fees = FeeEngine()
    buy_rate = fees.net_conversion_rate(symbol, Side.BUY, D("60000"))
    # 1/60000 * 0.999
    assert float(buy_rate) == pytest.approx(0.999 / 60000, rel=1e-9)


def test_mexc_maker_structure_beats_binance_materially():
    fees = FeeEngine()
    mixed = [Liquidity.MAKER, Liquidity.TAKER, Liquidity.TAKER]
    mexc = fees.cycle_fee_bps(["mexc"] * 3, mixed)
    binance = fees.cycle_fee_bps(["binance"] * 3, mixed)
    assert float(mexc) < float(binance) / 2


# ── sizing ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sized_market():
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
    graph = CurrencyGraph(
        books, base_currency="USDT", reference_notional=D("100"),
        taker_fee_bps=D("10"), enforce_lot_value=False, consumption_cap=D("1"),
    )
    graph.build(symbols, SystemClock().wall_ns())

    sizer = CycleSizer(books, FeeEngine(), consumption_cap=D("1"))
    for symbol, lot, tick, minimum in (
        (symbols[0], "0.00001", "0.01", "5"),
        (symbols[1], "0.0001", "0.01", "5"),
        (symbols[2], "0.0001", "0.000001", "0.0001"),
    ):
        sizer.register_spec(symbol, SymbolSpec(
            venue_symbol=symbol.venue_symbol,
            base=symbol.base.code, quote=symbol.quote.code,
            tick_size=D(tick), lot_step=D(lot), min_notional=D(minimum),
        ))

    enumerator = CycleEnumerator(graph, min_length=3, max_length=3, start_assets=("USDT",))
    enumerator.enumerate()
    cycle = [c for c in enumerator.price_all() if c.profitable][0]
    return sizer, cycle, Asset("USDT", AssetClass.STABLECOIN)


def test_capital_scaling(sized_market):
    """
    THE central result. One opportunity, five capital levels.

    At $100 the account keeps 14% of the edge; at $10,000 it keeps 99.5%. The
    difference is entirely lot quantization, and it is why a $100 account cannot
    run this strategy on BTC-anchored cycles.
    """
    sizer, cycle, usdt = sized_market
    results = {
        capital: sizer.size(cycle.to_legs(), usdt, D(capital))
        for capital in ("100", "1000", "10000")
    }
    for result in results.values():
        assert result.ok, result.summary()

    net = {k: float(v.net_edge_bps) for k, v in results.items()}
    drag = {k: float(v.drag_bps) for k, v in results.items()}

    # Drag falls roughly 10x per 10x of capital.
    assert drag["100"] > drag["1000"] * 8
    assert drag["1000"] > drag["10000"] * 8
    # Net edge rises monotonically with capital.
    assert net["100"] < net["1000"] < net["10000"]
    # And the small account keeps a small fraction of the available edge.
    assert net["100"] / net["10000"] < 0.25


def test_sizing_reports_why_it_failed(sized_market):
    sizer, cycle, usdt = sized_market
    result = sizer.size(cycle.to_legs(), usdt, D("0.5"))
    assert not result.ok
    assert result.failure in (
        SizingFailure.MIN_NOTIONAL, SizingFailure.LOT_ZERO,
        SizingFailure.NEGATIVE_EDGE,
    )
    assert result.detail
    assert result.failed_leg >= 0


def test_every_leg_lands_on_the_lot_grid(sized_market):
    """A quantity off the grid is rejected by the venue for LOT_SIZE."""
    sizer, cycle, usdt = sized_market
    result = sizer.size(cycle.to_legs(), usdt, D("5000"))
    assert result.ok
    for leg in result.legs:
        remainder = leg.quantity % leg.spec.lot_step
        assert remainder == 0, f"{leg.quantity} is off the {leg.spec.lot_step} grid"
        assert leg.notional_quote >= leg.spec.min_notional


def test_zero_capital_is_rejected_not_crashed(sized_market):
    sizer, cycle, usdt = sized_market
    result = sizer.size(cycle.to_legs(), usdt, ZERO)
    assert not result.ok
    assert result.failure == SizingFailure.NO_CAPITAL


# ── capital adequacy ───────────────────────────────────────────────────────

def test_adequacy_depends_on_instrument_not_just_venue():
    """
    Same venue, same fees, same capital -- the instrument decides.

    Note what this test does NOT claim: that a low-lot instrument is feasible on
    Binance at $100 with all-taker legs at the entry tier. It is not
    (41.5 bps required against a 35 bps ceiling). Feasibility needs the fee
    schedule to cooperate too -- a maker leg and the token discount -- which is
    exactly the point: drag and fees are separate levers and both must be pulled.
    """
    def report(lot, price, **kwargs):
        return assess_capital_adequacy(
            "binance", D("100"),
            representative_lot_step=lot, representative_price=price, **kwargs,
        )

    btc = report(D("0.00001"), D("62000"))
    trx = report(D("0.1"), D("0.12"))

    # Drag differs by more than 40x on the instrument alone.
    assert float(btc.drag_bps) > float(trx.drag_bps) * 40
    assert not btc.feasible
    assert not trx.feasible, "all-taker at the entry tier is not feasible at $100"

    # Pull the fee lever too and the low-lot cycle clears; BTC still does not.
    btc_cheap = report(D("0.00001"), D("62000"), maker_legs=1, use_discount=True)
    trx_cheap = report(D("0.1"), D("0.12"), maker_legs=1, use_discount=True)
    assert not btc_cheap.feasible, "no fee schedule rescues BTC drag at $100"
    assert trx_cheap.feasible
    assert float(trx_cheap.required_edge_bps) < 35


def test_high_fee_venues_are_flagged_infeasible():
    for venue in ("kraken", "coinbase"):
        report = assess_capital_adequacy(
            venue, D("100"),
            representative_lot_step=D("0.1"), representative_price=D("0.12"),
        )
        assert not report.feasible
        assert report.warnings
