"""
Graph-layer tests.

The centrepiece is ``test_known_arbitrage_is_detected_exactly``: a synthetic
market with a hand-inserted dislocation whose profit is computed independently
by hand. Both cycle-finding algorithms must agree with that number to within a
hundredth of a basis point. It is the test that would have caught the reference-
notional denomination bug (sizing every pair at 100 units of its *own* quote
asset, which asked for $6M of depth on a BTC-quoted pair and silently marked
every such edge THIN).
"""

from __future__ import annotations

import pytest

from triangulum.core.clock import SystemClock
from triangulum.core.decimal_math import D
from triangulum.graph.bellman_ford import find_negative_cycles
from triangulum.graph.currency_graph import CurrencyGraph, EdgeQuality
from triangulum.graph.cycle_enum import CycleEnumerator
from triangulum.marketdata.book_manager import BookManager
from triangulum.marketdata.normalizer import SymbolNormalizer


@pytest.fixture
def market():
    """Three-asset market. ETH/BTC is dislocated: ETH is cheap in BTC terms."""
    norm = SymbolNormalizer()
    books = BookManager(max_age_ns=10**12)
    books.mark_connected("binance")

    def mk(base, quote, bid, ask, size="1000"):
        s = norm.register("binance", f"{base}{quote}", base, quote)
        books.apply_snapshot(s, [(D(bid), D(size))], [(D(ask), D(size))])
        return s

    symbols = [
        mk("BTC", "USDT", "60000", "60001"),
        mk("ETH", "USDT", "3000", "3000.5"),
        mk("ETH", "BTC", "0.04945", "0.0495", size="500"),   # fair would be 0.05
    ]
    return norm, books, symbols


def build_graph(books, symbols, **kwargs):
    defaults = dict(
        base_currency="USDT",
        reference_notional=D("100"),
        taker_fee_bps=D("10"),
        enforce_lot_value=False,
        consumption_cap=D("1"),
    )
    defaults.update(kwargs)
    g = CurrencyGraph(books, **defaults)
    g.build(symbols, SystemClock().wall_ns())
    return g


def test_known_arbitrage_is_detected_exactly(market):
    """
    Hand calculation, 10 bps taker per leg:

        100 USDT / 60001      * 0.999 = 0.0016649723 BTC
        0.0016649723 / 0.0495 * 0.999 = 0.0336021672 ETH
        0.0336021672 * 3000   * 0.999 = 100.705695   USDT

    -> +70.57 bps. Both detectors must reproduce it.
    """
    _norm, books, symbols = market
    graph = build_graph(books, symbols)

    enumerated = CycleEnumerator(graph, min_length=3, max_length=3, start_assets=("USDT",))
    enumerated.enumerate()
    priced = [c for c in enumerated.price_all() if c.profitable]

    assert len(priced) == 1, "exactly one direction of the triangle should pay"
    assert priced[0].path == "USDT -> BTC -> ETH -> USDT"
    assert float(priced[0].gross_edge_bps) == pytest.approx(70.57, abs=0.01)

    cycles = find_negative_cycles(graph, min_length=3, max_length=4, start_assets=("USDT",))
    assert cycles, "Bellman-Ford must find the same cycle"
    assert cycles[0].edge_bps == pytest.approx(70.57, abs=0.01)
    assert cycles[0].path == "USDT -> BTC -> ETH -> USDT"


def test_reverse_direction_is_not_profitable(market):
    """The opposite rotation pays the spread three times; it must be negative."""
    _norm, books, symbols = market
    graph = build_graph(books, symbols)
    enumerated = CycleEnumerator(graph, min_length=3, max_length=3, start_assets=("USDT",))
    enumerated.enumerate()
    reverse = [c for c in enumerated.price_all() if "USDT -> ETH" in c.path]
    for cycle in reverse:
        assert cycle.gross_edge_bps < 0


def test_fair_market_yields_no_cycles(market):
    """With ETH/BTC at its fair 0.05, fees alone must close the triangle."""
    norm, books, symbols = market
    ethbtc = norm.canonical("binance", "ETH/BTC")
    books.apply_snapshot(ethbtc, [(D("0.049995"), D("500"))], [(D("0.050005"), D("500"))])

    graph = build_graph(books, symbols)
    enumerated = CycleEnumerator(graph, min_length=3, max_length=3, start_assets=("USDT",))
    enumerated.enumerate()
    assert not [c for c in enumerated.price_all() if c.profitable]
    assert not find_negative_cycles(graph, min_length=3, max_length=4)


def test_reference_notional_is_converted_per_quote_asset(market):
    """
    The regression test for the denomination bug.

    100 USDT must become ~0.001667 BTC when sizing a BTC-quoted pair, not 100
    BTC. If this breaks, every BTC-quoted edge is marked THIN and the engine
    silently finds nothing forever.
    """
    _norm, books, symbols = market
    graph = build_graph(books, symbols)

    assert graph.notional_in("USDT") == D("100")
    assert float(graph.notional_in("BTC")) == pytest.approx(0.00166665, rel=1e-4)
    assert graph.stats()["edges_usable"] == 6
    assert not graph.stats()["excluded"], "no edge should be excluded in this market"


def test_stale_book_excludes_edges(market):
    """A book past the freshness budget must not produce a tradeable edge."""
    _norm, books, symbols = market
    graph = CurrencyGraph(
        books, base_currency="USDT", reference_notional=D("100"),
        max_book_age_ns=1, enforce_lot_value=False,
    )
    graph.build(symbols, SystemClock().wall_ns() + 10**9)   # one second later
    assert graph.stats()["edges_usable"] == 0
    assert graph.stats()["excluded"].get(EdgeQuality.STALE, 0) > 0


def test_lot_value_screen_rejects_btc_at_small_size(market):
    """
    At $100, BTC's lot grid costs ~31 bps per leg. The screen must reject it,
    which is the entire reason a $100 account cannot trade BTC triangles.
    """
    _norm, books, symbols = market
    graph = CurrencyGraph(
        books, base_currency="USDT", reference_notional=D("100"),
        enforce_lot_value=True, drag_budget_bps=D("5"),
    )
    for s in symbols:
        # Binance's real BTCUSDT grid.
        graph.set_symbol_rules(s, lot_step=D("0.00001"), min_notional=D("5"))
    graph.build(symbols, SystemClock().wall_ns())
    assert graph.stats()["excluded"].get(EdgeQuality.LOT_VALUE, 0) > 0


def test_cycle_rotation_preserves_weight(market):
    _norm, books, symbols = market
    graph = build_graph(books, symbols)
    cycles = find_negative_cycles(graph, min_length=3, max_length=3)
    assert cycles
    original = cycles[0]
    rotated = original.rotate_to(original.edges[1].frm.code)
    assert rotated is not None
    assert rotated.total_weight == pytest.approx(original.total_weight)
    assert rotated.length == original.length


def test_star_topology_is_reported(market):
    """
    An equities-shaped universe (every asset quoted only against USD) has no
    cycles at all. The connectivity report must say so rather than leaving the
    operator wondering why nothing ever fires.
    """
    norm = SymbolNormalizer()
    books = BookManager(max_age_ns=10**12)
    books.mark_connected("alpaca")
    symbols = []
    for ticker, px in (("AAPL", "230"), ("MSFT", "420"), ("NVDA", "900")):
        s = norm.register("alpaca", ticker, ticker, "USD")
        books.apply_snapshot(s, [(D(px), D("1000"))], [(D(px) + D("0.01"), D("1000"))])
        symbols.append(s)

    graph = CurrencyGraph(books, base_currency="USD", reference_notional=D("100"),
                          enforce_lot_value=False)
    graph.build(symbols, SystemClock().wall_ns())
    report = graph.connectivity_report()
    assert report["is_star_topology"] is True
    assert not find_negative_cycles(graph, min_length=3, max_length=4)
