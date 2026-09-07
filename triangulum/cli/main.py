"""
Command-line interface.

    triangulum doctor       capital adequacy and configuration sanity
    triangulum demo         run against a synthetic market with the HUD
    triangulum paper        run against live market data, simulated fills
    triangulum backtest     replay recorded data
    triangulum live         real orders (requires the triple lock)
    triangulum record       record market data for training
    triangulum arm-live     write the live acknowledgment file

``doctor`` is the one to run first. It answers, in basis points, what edge the
market must hand you before your configuration breaks even -- which is a more
useful thing to know than any backtest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from decimal import Decimal
from pathlib import Path

from triangulum.core import constants as C
from triangulum.core.clock import SimulatedClock, SystemClock
from triangulum.core.config import Config, load_config
from triangulum.core.decimal_math import D, ZERO
from triangulum.core.eventbus import EventBus, Topics
from triangulum.core.types import ExecutionMode, RunMode
from triangulum.exchanges.spec import (
    VENUE_SPECS, assess_capital_adequacy, max_lot_value_for_budget,
)
from triangulum.ops.logging_setup import setup_logging
from triangulum.version import __version__

logger = logging.getLogger("triangulum.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triangulum",
        description="Autonomous multi-venue arbitrage engine. Paper-first.",
    )
    parser.add_argument("--version", action="version", version=f"triangulum {__version__}")
    parser.add_argument("-c", "--config", help="path to a YAML config file")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="machine-readable output")

    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="capital adequacy and config sanity")
    doctor.add_argument("--capital", type=float, default=0.0)
    doctor.add_argument("--legs", type=int, default=3)
    doctor.add_argument("--maker-legs", type=int, default=0)

    demo = sub.add_parser("demo", help="run against a synthetic market")
    demo.add_argument("--ticks", type=int, default=0, help="0 = run until stopped")
    demo.add_argument("--speed", type=float, default=1.0)
    demo.add_argument("--no-dashboard", action="store_true")
    demo.add_argument("--port", type=int, default=0)
    demo.add_argument(
        "--edge-bps", type=float, default=9.0,
        help=(
            "mean size of injected dislocations. The realistic value on a liquid "
            "venue is 5-10, which after 30 bps of fees means almost nothing is "
            "tradeable -- that is the true state of the world. Raise it (e.g. 45) "
            "to make the full pipeline visible end to end."
        ),
    )
    demo.add_argument(
        "--rate", type=float, default=0.012,
        help="per-tick probability of injecting a dislocation on each cross pair",
    )

    for name, help_text in (
        ("paper", "live market data, simulated fills"),
        ("live", "real orders (requires the triple lock)"),
    ):
        run = sub.add_parser(name, help=help_text)
        run.add_argument("--no-dashboard", action="store_true")
        run.add_argument("--port", type=int, default=0)

    backtest = sub.add_parser("backtest", help="replay recorded market data")
    backtest.add_argument("--data", default=C.DEFAULT_RECORDING_DIR)
    backtest.add_argument("--speed", type=float, default=0.0)
    backtest.add_argument("--report", default="")

    record = sub.add_parser("record", help="record market data for training")
    record.add_argument("--minutes", type=float, default=60.0)
    record.add_argument("--out", default=C.DEFAULT_RECORDING_DIR)

    sub.add_parser("arm-live", help="write the live-trading acknowledgment file")
    sub.add_parser("venues", help="list known venues and their economics")

    return parser


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def cmd_doctor(args, config: Config, warnings: list[str]) -> int:
    capital = D(str(args.capital)) if args.capital else config.capital
    legs = args.legs
    out: dict = {"capital": str(capital), "legs": legs, "venues": {}}

    print()
    print("=" * 74)
    print(f"  CAPITAL ADEQUACY — {capital} {config.base_currency}, {legs}-leg cycles")
    print("=" * 74)
    # Two scenarios, because the drag term depends entirely on WHICH
    # instruments the cycle passes through -- and that varies by a factor of
    # fifty on the same venue with the same fee schedule.
    scenarios = (
        ("BTC-anchored", D("0.00001"), D("62000")),   # lot value $0.62
        ("low-lot (TRX/XRP)", D("0.1"), D("0.12")),   # lot value $0.012
    )

    for scenario_name, lot_step, price in scenarios:
        print()
        print(f"  ── {scenario_name} cycles "
              f"(one lot = ${float(lot_step * price):.4f}) "
              + "─" * max(0, 34 - len(scenario_name)))
        print(f"  {'venue':<11} {'fees':>7} {'drag':>8} {'needs':>9}  verdict")
        print("  " + "-" * 70)

        for name in ("binance", "mexc", "okx", "kucoin", "bybit", "gateio",
                     "kraken", "coinbase"):
            report = assess_capital_adequacy(
                name, capital, cycle_legs=legs, maker_legs=args.maker_legs,
                use_discount=True,
                representative_price=price,
                representative_lot_step=lot_step,
            )
            verdict = "feasible" if report.feasible else "NOT FEASIBLE"
            print(
                f"  {name:<11} {float(report.fee_bps):>6.1f}b "
                f"{float(report.drag_bps):>7.1f}b "
                f"{float(report.required_edge_bps):>8.1f}b  {verdict}"
            )
            out["venues"].setdefault(name, {})[scenario_name] = {
                "fee_bps": float(report.fee_bps),
                "drag_bps": float(report.drag_bps),
                "required_edge_bps": float(report.required_edge_bps),
                "feasible": report.feasible,
            }

    notional = capital * D("0.95")
    budget_5 = max_lot_value_for_budget(notional, D("5"), legs=legs)
    budget_10 = max_lot_value_for_budget(notional, D("10"), legs=legs)
    out["max_lot_value_5bps"] = str(budget_5)
    out["max_lot_value_10bps"] = str(budget_10)

    print()
    print("  " + "-" * 70)
    print("  INSTRUMENT SCREEN")
    print("  " + "-" * 70)
    print(f"  At this size, a leg may use an instrument whose ONE LOT is worth")
    print(f"  at most ${float(budget_5):.4f} for a 5 bps drag budget "
          f"(${float(budget_10):.4f} at 10 bps).")
    print()
    print(f"  {'instrument':<12} {'lot value':>11}  admitted?")
    print("  " + "-" * 70)
    examples = (
        ("BTCUSDT",  D("0.00001"), D("62000")),
        ("ETHUSDT",  D("0.0001"),  D("3050")),
        ("SOLUSDT",  D("0.001"),   D("148")),
        ("DOGEUSDT", D("1"),       D("0.10")),
        ("ADAUSDT",  D("0.1"),     D("0.447")),
        ("XRPUSDT",  D("0.1"),     D("0.542")),
        ("TRXUSDT",  D("0.1"),     D("0.1187")),
    )
    for label, lot, price in examples:
        value = lot * price
        ok = value <= budget_5
        marginal = not ok and value <= budget_10
        mark = "yes" if ok else ("marginal" if marginal else "no")
        print(f"  {label:<12} {float(value):>10.4f}$  {mark}")

    if warnings:
        print()
        print("  " + "-" * 70)
        print("  CONFIGURATION NOTES")
        print("  " + "-" * 70)
        for w in warnings:
            print(f"  · {w}")

    print()
    print("  " + "-" * 70)
    print("  WHAT THIS MEANS")
    print("  " + "-" * 70)
    best = min(
        (v.get("low-lot (TRX/XRP)", {}).get("required_edge_bps", 1e9), k)
        for k, v in out["venues"].items()
    )
    print(f"  Best case at this size: {best[1]} on low-lot instruments needs the")
    print(f"  market to hand you {best[0]:.1f} bps before you break even.")
    print()
    print("  Observed triangular dislocations on a liquid venue run 1-8 bps and")
    print("  last 50-300ms. Compare those two numbers honestly.")
    print()
    print("  See docs/EXPECTATIONS.md for the full arithmetic, without hedging.")
    print()

    if args.json:
        print(json.dumps(out, indent=2))
    return 0


def cmd_venues(args, config: Config, warnings: list[str]) -> int:
    for name, spec in sorted(VENUE_SPECS.items()):
        print()
        print(f"── {spec.display_name} ({name}) " + "─" * max(0, 50 - len(spec.display_name)))
        print(f"   maker {spec.default_maker_bps}bps / taker {spec.default_taker_bps}bps"
              + (f", {spec.discount_pct}% off with {spec.discount_asset}"
                 if spec.discount_asset else ""))
        print(f"   3-leg all-taker cost: {spec.round_trip_bps(3)} bps"
              f"  ·  1 maker leg: {spec.round_trip_bps(3, maker_legs=1)} bps")
        print(f"   typical min notional: ${spec.typical_min_notional_usd}")
        if spec.notes:
            import textwrap
            for line in textwrap.wrap(spec.notes, 68):
                print(f"   {line}")
    print()
    return 0


def cmd_arm_live(args, config: Config, warnings: list[str]) -> int:
    from triangulum.risk.live_arming import check_live_arming, write_acknowledgment

    print()
    print("You are about to enable real-money trading.")
    print()
    print("  · This engine's expected edge is a few basis points per cycle.")
    print("  · At small capital, quantization drag can exceed the entire edge.")
    print("  · Total loss of the deployed capital is a realistic outcome.")
    print()
    print(f"Writing: {C.LIVE_ACK_FILENAME}")
    path = write_acknowledgment()
    print(f"Wrote {path}")
    print()
    check = check_live_arming(config.run_mode)
    print(check.report())
    print()
    if not check.armed:
        print("Live trading is still NOT armed. Satisfy the remaining locks above.")
    return 0


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


async def _run(args, config: Config, mode: RunMode) -> int:
    from triangulum.wiring import build_stack

    market_params = None
    if args.command == "demo":
        from triangulum.simulation import MarketParams
        market_params = MarketParams(
            dislocation_bps_mean=args.edge_bps,
            dislocation_bps_std=max(1.0, args.edge_bps * 0.5),
            dislocation_probability=args.rate,
        )

    stack = await build_stack(
        config,
        mode=mode,
        demo=(mode is RunMode.BACKTEST and args.command == "demo"),
        market_params=market_params,
    )
    engine = stack.engine

    stop = asyncio.Event()

    def _signal_handler(*_):
        logger.info("shutdown signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:      # pragma: no cover - Windows
            signal.signal(sig, _signal_handler)

    await engine.start()
    if stack.dashboard:
        print(f"\n  ▸ HUD: {stack.dashboard.url}\n")

    try:
        if getattr(args, "ticks", 0):
            for _ in range(args.ticks):
                if stop.is_set():
                    break
                stack.market.step() if stack.market else None
                await asyncio.sleep(0.001)
            await asyncio.sleep(1.0)
        else:
            if stack.market:
                asyncio.create_task(_drive_market(stack.market, stop, args))
            await stop.wait()
    finally:
        await engine.stop()
        await stack.shutdown()

    print()
    print(stack.report())
    return 0


async def _drive_market(market, stop: asyncio.Event, args) -> None:
    interval = 0.02 / max(0.01, getattr(args, "speed", 1.0))
    while not stop.is_set():
        market.step()
        await asyncio.sleep(interval)


def cmd_demo(args, config: Config, warnings: list[str]) -> int:
    if args.port:
        config.dashboard.port = args.port
    config.dashboard.enabled = not args.no_dashboard

    if args.edge_bps > 15:
        print()
        print("  ┌" + "─" * 68 + "┐")
        print("  │ DEMO MODE: dislocations inflated to "
              f"{args.edge_bps:.0f} bps mean.{' ' * (30 - len(f'{args.edge_bps:.0f}'))}│")
        print("  │ Real triangular dislocations on a liquid venue are 1-8 bps and     │")
        print("  │ almost never clear a 30 bps fee bill. This setting exists to make  │")
        print("  │ the pipeline visible, NOT to represent achievable returns.         │")
        print("  └" + "─" * 68 + "┘")
    return asyncio.run(_run(args, config, RunMode.BACKTEST))


def cmd_paper(args, config: Config, warnings: list[str]) -> int:
    if args.port:
        config.dashboard.port = args.port
    config.dashboard.enabled = not args.no_dashboard
    config.mode = RunMode.PAPER.value
    return asyncio.run(_run(args, config, RunMode.PAPER))


def cmd_live(args, config: Config, warnings: list[str]) -> int:
    from triangulum.risk.live_arming import check_live_arming

    config.mode = RunMode.LIVE.value
    check = check_live_arming(RunMode.LIVE)
    print()
    print(check.report())
    print()
    if not check.armed:
        print("Refusing to start. Run `triangulum arm-live` and read the output.")
        return 2
    if args.port:
        config.dashboard.port = args.port
    config.dashboard.enabled = not args.no_dashboard
    return asyncio.run(_run(args, config, RunMode.LIVE))


def cmd_backtest(args, config: Config, warnings: list[str]) -> int:
    from triangulum.backtest.engine import run_backtest

    result = asyncio.run(run_backtest(config, data_dir=args.data, speed=args.speed))
    print(result.metrics.report())
    if args.report:
        Path(args.report).write_text(
            json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        print(f"\nwrote {args.report}")
    return 0


def cmd_record(args, config: Config, warnings: list[str]) -> int:
    from triangulum.wiring import record_market_data

    return asyncio.run(record_market_data(config, minutes=args.minutes, out_dir=args.out))


# --------------------------------------------------------------------------


COMMANDS = {
    "doctor": cmd_doctor,
    "venues": cmd_venues,
    "arm-live": cmd_arm_live,
    "demo": cmd_demo,
    "paper": cmd_paper,
    "live": cmd_live,
    "backtest": cmd_backtest,
    "record": cmd_record,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config, warnings = load_config(args.config)
    except Exception as exc:
        # Fall back to defaults for the commands that do not need a venue.
        if args.command in ("doctor", "venues", "arm-live"):
            config, warnings = Config(), [str(exc)]
        else:
            print(f"configuration error: {exc}", file=sys.stderr)
            return 2

    setup_logging(
        level="DEBUG" if args.verbose else config.ops.log_level,
        json_output=config.ops.log_json,
        log_file=config.ops.log_file,
    )
    for warning in warnings:
        logger.warning("config: %s", warning)

    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args, config, warnings)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
