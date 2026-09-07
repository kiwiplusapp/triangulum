"""
End-to-end engine tests.

These run the REAL engine -- the same scan loop, planner, gate, executor and
ledger that paper and live use -- against a synthetic market. Only the clock and
the venue adapter are substituted, which is the property that makes a backtest
number comparable to a live one.

What they assert is deliberately about *behaviour under stress*, not about
returns. A test that asserted "the engine makes money" would be asserting that
the synthetic market is generous, which proves nothing.
"""

from __future__ import annotations

import asyncio

import pytest

from triangulum.core.config import Config, VenueConfig
from triangulum.core.decimal_math import D
from triangulum.core.types import RunMode
from triangulum.simulation import MarketParams
from triangulum.wiring import build_stack


def make_config(capital: float = 10_000.0, **overrides) -> Config:
    config = Config(
        mode="backtest",
        base_currency="USDT",
        initial_capital=capital,
        venues=(VenueConfig(name="binance", quote_assets=("USDT", "BTC", "ETH")),),
    )
    config.dashboard.enabled = False
    config.storage.record_opportunities = False
    config.learning.model_dir = "/tmp/triangulum-test-models"
    config.strategy.max_concurrent_cycles = 1
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        setattr(getattr(config, section), field, value) if field else setattr(config, section, value)
    return config


async def run_engine(config: Config, *, ticks: int, edge_bps: float = 45.0,
                     rate: float = 0.06):
    stack = await build_stack(
        config, mode=RunMode.BACKTEST, demo=True,
        market_params=MarketParams(
            dislocation_bps_mean=edge_bps,
            dislocation_bps_std=edge_bps * 0.4,
            dislocation_probability=rate,
            seed=11,
        ),
    )
    await stack.engine.start()
    try:
        for _ in range(ticks):
            stack.market.step()
            await asyncio.sleep(0)
        # Keep the market ticking while the scan loop drains. Pausing it for
        # even 400ms ages every book past the 250ms freshness budget, and the
        # graph then correctly reports zero usable edges -- the engine working
        # as designed, but not the state under test.
        for _ in range(40):
            stack.market.step()
            await asyncio.sleep(0.01)
        snapshot = stack.engine.snapshot()
    finally:
        await stack.engine.stop()
        await stack.shutdown()
    return stack, snapshot


@pytest.mark.asyncio
async def test_engine_runs_and_reports_a_coherent_snapshot():
    _stack, snapshot = await run_engine(make_config(), ticks=3000)

    assert snapshot["scans"] > 0
    assert snapshot["graph"]["edges_usable"] > 0
    assert snapshot["equity"] > 0
    for section in ("risk", "executor", "ledger", "gate", "books", "venues"):
        assert section in snapshot, f"snapshot is missing {section}"

    # The funnel must be internally consistent: you cannot accept more than you
    # evaluated, nor complete more than you accepted.
    gate = snapshot["gate"]
    executor = snapshot["executor"]
    assert gate["accepts"] <= gate["evaluations"]
    assert executor["completed"] <= executor["attempted"]


@pytest.mark.asyncio
async def test_no_cycle_is_ever_left_stranded():
    """
    The most important guarantee in the system.

    A cycle that commits capital and then fails must be unwound. A stranded
    position is the one state from which the engine cannot recover on its own,
    and it trips the kill switch by design -- so a test that ends with stuck
    cycles is reporting a real regression in the unwinder.
    """
    _stack, snapshot = await run_engine(make_config(), ticks=6000)
    executor = snapshot["executor"]
    assert executor["stuck"] == 0, (
        f"{executor['stuck']} cycles left inventory stranded; "
        f"unwound={executor['unwound']}, attempted={executor['attempted']}"
    )


@pytest.mark.asyncio
async def test_ledger_balances_reconcile_with_the_simulated_venue():
    """The engine's model of its own position must match the venue's."""
    stack, _snapshot = await run_engine(make_config(), ticks=3000)
    ledger = stack.engine.ledger
    adapter = next(iter(stack.adapters.values()))
    venue_balances = {k: v.total for k, v in adapter.all_balances().items()}

    for asset, amount in ledger.balances().items():
        venue_amount = venue_balances.get(asset, D("0"))
        if abs(amount) < D("1e-9"):
            continue
        relative = abs((venue_amount - amount) / amount) if amount else D("0")
        assert relative < D("0.01"), (
            f"{asset}: ledger says {amount}, venue says {venue_amount}"
        )


@pytest.mark.asyncio
async def test_small_capital_is_screened_out_and_explains_itself():
    """
    At $100 the lot-value screen leaves a graph with no cycles. The engine must
    not trade, and must say why rather than scanning an empty set in silence.
    """
    config = make_config(capital=100.0)
    stack, snapshot = await run_engine(config, ticks=800)

    assert snapshot["cycles_executed"] == 0
    excluded = snapshot["graph"]["excluded"]
    assert excluded.get("lot_value", 0) > 0, (
        f"the lot-value screen should be the binding constraint at $100, "
        f"but exclusions were {excluded}"
    )
    # Fewer than three assets can sit inside a cycle, so no 3-cycle exists.
    assert snapshot["graph"]["edges_usable"] < 10


@pytest.mark.asyncio
async def test_kill_switch_stops_all_new_cycles():
    config = make_config()
    stack = await build_stack(
        config, mode=RunMode.BACKTEST, demo=True,
        market_params=MarketParams(dislocation_bps_mean=60.0, dislocation_probability=0.1, seed=3),
    )
    await stack.engine.start()
    try:
        for _ in range(1500):
            stack.market.step()
            await asyncio.sleep(0)
        stack.engine.guardrails.engage_kill_switch("test")
        before = stack.engine.state.cycles_executed
        for _ in range(3000):
            stack.market.step()
            await asyncio.sleep(0)
        await asyncio.sleep(0.3)
        assert stack.engine.state.cycles_executed == before
    finally:
        await stack.engine.stop()
        await stack.shutdown()


@pytest.mark.asyncio
async def test_models_persist_and_reload():
    config = make_config()
    stack, _ = await run_engine(config, ticks=2500)
    samples = stack.gate.fill_model.samples

    reloaded = await build_stack(config, mode=RunMode.BACKTEST, demo=True)
    try:
        # A checkpoint was written on shutdown; the fresh stack must pick it up.
        assert reloaded.gate.fill_model.samples >= samples
    finally:
        await reloaded.shutdown()
