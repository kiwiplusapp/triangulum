"""
Vault command line.

    vault doctor     what the system can and cannot do right now
    vault scan       SCAN -> MACRO -> FLOW, print the brief, no API call
    vault run        the full cycle, including the thesis (needs credentials)
    vault resolve    grade every due call
    vault score      the scoreboard
    vault journal    inspect and verify the hash chain
    vault serve      the HUD
    vault demo       seed a simulated track record and serve the HUD
    vault simulate   run agents of known skill through the gate

``score`` is the one that matters. Everything else feeds it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from vault.version import __version__

logger = logging.getLogger("vault.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vault",
        description="Autonomous macro-synthesis agent that keeps score on itself.",
    )
    parser.add_argument("--version", action="version", version=f"vault {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--journal", default="data/vault/journal.ndjson")
    parser.add_argument("--cache", default="data/vault/cache")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--fixtures", action="store_true",
                        help="use synthetic data instead of live sources")
    parser.add_argument("--scenario", default="late_cycle",
                        choices=["late_cycle", "reflation", "stagflation"])
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--json", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="what the system can and cannot do right now")
    sub.add_parser("scan", help="scan, classify, and print the brief")
    run = sub.add_parser("run", help="the full cycle including the thesis")
    run.add_argument("--no-commit", action="store_true")
    sub.add_parser("resolve", help="grade every due call")
    sub.add_parser("score", help="the calibration scoreboard")
    journal = sub.add_parser("journal", help="inspect and verify the hash chain")
    journal.add_argument("--full", action="store_true")

    serve = sub.add_parser("serve", help="the HUD")
    serve.add_argument("--port", type=int, default=8899)
    serve.add_argument("--host", default="127.0.0.1")

    demo = sub.add_parser("demo", help="seed a simulated record and serve the HUD")
    demo.add_argument("--port", type=int, default=8899)
    demo.add_argument("--calls", type=int, default=80)
    demo.add_argument("--agent", default="oracle",
                      choices=["oracle", "coinflip", "overconfident", "lucky"])
    demo.add_argument("--no-serve", action="store_true")

    simulate = sub.add_parser("simulate", help="agents of known skill through the gate")
    # 300, not 200: against the always-valid bound a true 65% forecaster needs
    # a median of 94 resolved calls to clear the gate, and the run has to be
    # long enough to show that it eventually does.
    simulate.add_argument("--calls", type=int, default=300)
    simulate.add_argument("--seeds", type=int, default=25,
                          help="one seed proves nothing; this is the sample")
    return parser


def _vault(args, **overrides):
    from vault.pipeline import Vault
    kwargs = dict(
        journal_path=args.journal, cache_dir=args.cache, capital=args.capital,
        model=args.model, effort=args.effort, min_samples=args.min_samples,
        use_fixtures=args.fixtures, fixture_scenario=args.scenario,
    )
    kwargs.update(overrides)
    return Vault(**kwargs)


# --------------------------------------------------------------------------


def cmd_doctor(args) -> int:
    from vault.thesis.engine import ThesisEngine
    from vault.thesis.journal import ThesisJournal
    from vault.data.http import get_transport

    print()
    print("=" * 70)
    print("  VAULT DOCTOR")
    print("=" * 70)

    journal = ThesisJournal(args.journal)
    stats = journal.stats()
    engine = ThesisEngine(journal, model=args.model)

    print()
    print("  CREDENTIALS")
    print("  " + "-" * 66)
    creds = engine.has_credentials
    print(f"  [{'x' if creds else ' '}] Anthropic API reachable "
          f"({'credentials found' if creds else 'set ANTHROPIC_API_KEY or run `ant auth login`'})")
    try:
        import anthropic  # noqa: F401
        print("  [x] anthropic SDK installed")
    except ImportError:
        print("  [ ] anthropic SDK missing -- pip install anthropic")
    try:
        import pydantic  # noqa: F401
        print("  [x] pydantic installed (strict schema validation active)")
    except ImportError:
        print("  [ ] pydantic missing -- schema falls back to raw JSON validation")

    print()
    print("  DATA")
    print("  " + "-" * 66)
    transport = get_transport()
    print(f"  HTTP transport: {transport.name}")
    if args.fixtures:
        print("  Mode: FIXTURES -- synthetic data, nothing here is real")
    else:
        print("  Mode: live (FRED + Coinbase, both keyless)")

    print()
    print("  TRACK RECORD")
    print("  " + "-" * 66)
    print(f"  Total calls:     {stats['total']}")
    print(f"  Resolved:        {stats['resolved']}  ({stats['scoreable']} scoreable)")
    print(f"  Pending:         {stats['pending']}  ({stats['due_now']} due now)")
    print(f"  Chain:           {stats['chain_summary']}")

    if stats["scoreable"]:
        from vault.resolve.scoring import score_records
        from vault.calibration.gate import CapitalGate
        score = score_records(list(journal))
        gate = CapitalGate(min_samples=args.min_samples)
        progress = gate.progress(score)
        print()
        print("  CAPITAL GATE")
        print("  " + "-" * 66)
        for g in progress["gates"]:
            print(f"  [{'x' if g['passed'] else ' '}] {g['gate']:<20} "
                  f"{g['requirement']:<38} now: {g['current']}")
        print(f"\n  {progress['passed']}/{progress['total']} gates passed. "
              f"Capital {'UNLOCKED' if progress['unlocked'] else 'LOCKED'}.")
    else:
        print()
        print("  CAPITAL GATE")
        print("  " + "-" * 66)
        print("  No resolved calls, so no capital may be risked. This is the")
        print("  correct state for a new system, and it is the state the")
        print("  reference terminal was permanently in.")

    print()
    return 0


def cmd_scan(args) -> int:
    from vault.thesis.prompts import build_user_prompt
    vault = _vault(args, dry_run=True)
    vault.scan()
    regime = vault.classify()
    brief = vault.brief(regime)

    if args.json:
        print(json.dumps(brief, indent=2, default=str))
        return 0

    print()
    print(build_user_prompt(brief))
    print()
    return 0


def cmd_run(args) -> int:
    vault = _vault(args)
    result = vault.run(commit=not args.no_commit)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print()
        print(result.summary())
        if result.thesis and result.thesis.thesis:
            t = result.thesis.thesis
            print()
            print(f"  reasoning: {t.reasoning}")
            print(f"  key risk:  {t.key_risk}")
        if result.sizing:
            print()
            print("  " + result.sizing.explain().replace("\n", "\n  "))
        print()
    return 0 if not result.errors else 1


def cmd_resolve(args) -> int:
    vault = _vault(args, dry_run=True)
    vault.scan()
    report = vault.resolve()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return 0
    print()
    print(f"  {report.summary()}")
    for detail in report.details:
        print(f"    {detail['thesis_id']}  {detail['asset']:<10} "
              f"{detail['direction']:<5} p={detail['probability']:.2f}  "
              f"{detail['outcome']:<12} {detail['note']}")
    print()
    return 0


def cmd_score(args) -> int:
    from vault.calibration.gate import CapitalGate
    from vault.calibration.recalibrate import fit_recalibrator
    from vault.resolve.scoring import score_records
    from vault.thesis.journal import ThesisJournal

    journal = ThesisJournal(args.journal)
    records = list(journal)
    score = score_records(records)

    if args.json:
        print(json.dumps(score.to_dict(), indent=2, default=str))
        return 0

    print()
    print(score.report())

    _recalibrator, report = fit_recalibrator(records)
    if report.fitted:
        print()
        print("  RECALIBRATION")
        print("  " + "-" * 62)
        print(f"  {report.summary()}")
        print(f"  {report.reason}")
        if report.helped:
            print("  stated -> corrected:  " + ",  ".join(
                f"{e['stated']:.2f}->{e['recalibrated']:.2f}" for e in report.examples
            ))

    gate = CapitalGate(min_samples=args.min_samples)
    progress = gate.progress(score)
    print()
    print("  CAPITAL GATE")
    print("  " + "-" * 62)
    for g in progress["gates"]:
        print(f"  [{'x' if g['passed'] else ' '}] {g['gate']:<20} "
              f"{g['requirement']:<36} now: {g['current']}")
    print(f"\n  {progress['passed']}/{progress['total']} gates. "
          f"Capital {'UNLOCKED' if progress['unlocked'] else 'LOCKED'}.")
    print()
    return 0


def cmd_journal(args) -> int:
    from vault.thesis.journal import ThesisJournal
    journal = ThesisJournal(args.journal)
    verification = journal.verify_chain()
    stats = journal.stats()

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
        return 0

    print()
    print(f"  {verification.summary()}")
    print(f"  {stats['total']} records, {stats['resolved']} resolved, "
          f"{stats['pending']} pending")
    print()
    if args.full:
        for record in journal:
            print(f"  {record.thesis_id}  {record.content_hash[:12]}  "
                  f"{record.outcome:<12} p={record.thesis.probability:.2f}")
        print()
    return 0 if verification.valid else 2


def cmd_serve(args) -> int:
    from vault.hud.server import VaultServer
    vault = _vault(args, dry_run=False)
    vault.scan()
    server = VaultServer(vault, host=args.host, port=args.port)
    server.start()
    print(f"\n  ▸ Vault HUD: {server.url}\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        server.stop()
    return 0


def cmd_demo(args) -> int:
    """Seed a simulated track record, then serve the HUD against it."""
    from vault.calibration.gate import CapitalGate
    from vault.simulate import AGENT_PROFILES, run_simulation
    from vault.thesis.journal import ThesisJournal

    path = Path(args.journal)
    if path.exists():
        path.unlink()

    profile = next(p for p in AGENT_PROFILES if p.name == args.agent)
    journal = ThesisJournal(path)
    gate = CapitalGate(min_samples=args.min_samples)
    result = run_simulation(
        profile, journal=journal, gate=gate,
        calls=args.calls, capital=args.capital,
    )

    print()
    print(f"  seeded {args.calls} simulated calls from the '{profile.name}' agent")
    print(f"  {profile.description}")
    print(f"  measured hit rate {result.score.hit_rate:.1%}, "
          f"Brier {result.score.brier:.4f}, verdict {result.verdict}")
    print()

    if args.no_serve:
        return 0

    from vault.hud.server import VaultServer
    vault = _vault(args, dry_run=True)
    vault.use_fixtures = True
    vault.scan()
    vault.run(commit=False)
    server = VaultServer(vault, port=args.port)
    server.start()
    print(f"  ▸ Vault HUD: {server.url}\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        server.stop()
    return 0


def cmd_simulate(args) -> int:
    """
    Run forecasters of KNOWN skill through the real gate, over many seeds.

    One seed proves nothing. The first version of this command ran a single
    seed and printed HELD -- while, on that very seed, the no-edge "lucky"
    profile had taken $356 at call 50 and given it back by call 56. A summary
    line that can disagree with the table above it is the exact failure this
    whole system is built to avoid, so the claim is now measured: every profile
    runs `--seeds` times and the number reported is the fraction of runs in
    which the gate released capital at ANY point.
    """
    import statistics
    import tempfile
    from vault.calibration.gate import CapitalGate
    from vault.simulate import AGENT_PROFILES, run_simulation
    from vault.thesis.journal import ThesisJournal

    budget = 0.05          # the gate advertises a 5% error rate; hold it to that

    print()
    print(f"  Does the scoreboard detect real skill? "
          f"({args.calls} calls x {args.seeds} seeds per agent, "
          f"min_samples={args.min_samples})")
    print()
    print(f"  {'agent':<15} {'true':>5} {'stated':>7} {'hit':>7} {'Brier':>8} "
          f"{'funded':>9} {'want':>6} {'max $':>8}")
    print("  " + "-" * 78)

    rows = []
    for profile in AGENT_PROFILES:
        unlocked = 0
        peaks, hits, briers = [], [], []
        for seed in range(args.seeds):
            directory = tempfile.mkdtemp()
            journal = ThesisJournal(Path(directory) / "j.ndjson")
            gate = CapitalGate(min_samples=args.min_samples)
            result = run_simulation(
                profile, journal=journal, gate=gate,
                calls=args.calls, capital=args.capital, seed=seed,
            )
            if result.unlocked_at is not None:
                unlocked += 1
            peaks.append(result.peak_notional)
            hits.append(result.score.hit_rate)
            briers.append(result.score.brier)

        rate = unlocked / args.seeds
        want = "always" if profile.should_unlock else "never"
        ok = rate >= 0.90 if profile.should_unlock else rate <= budget
        rows.append((profile, rate, ok, max(peaks)))
        print(f"  {profile.name:<15} {profile.true_accuracy:>4.0%} "
              f"{profile.stated_probability:>7.0%} "
              f"{statistics.fmean(hits):>6.1%} {statistics.fmean(briers):>8.4f} "
              f"{rate:>8.0%} {want:>6} {'$' + format(max(peaks), ',.0f'):>8}")

    print()
    print("  `funded` is the fraction of runs in which capital was released at")
    print("  ANY point, not at the end -- money released at call 50 and pulled")
    print("  back at call 56 was still money at risk. An agent that should")
    print("  never be funded is held to the 5% error rate the gate advertises.")
    print()

    failures = [(pr, rate) for pr, rate, ok, _ in rows if not ok]
    if not failures:
        print("  -> HELD")
    else:
        print("  -> VIOLATED")
        for profile, rate in failures:
            if profile.should_unlock:
                print(f"     - {profile.name} has a real edge but was funded in "
                      f"only {rate:.0%} of runs")
            else:
                print(f"     - {profile.name} should never be funded but was, in "
                      f"{rate:.0%} of runs (budget {budget:.0%})")
    print()

    residual = [
        (pr.name, rate) for pr, rate, ok, _ in rows
        if not pr.should_unlock and 0 < rate <= budget
    ]
    if residual:
        print("  Residual, within budget and worth stating plainly:")
        for name, rate in residual:
            print(f"     {name} was funded in {rate:.0%} of runs.")
        print("     This does not go to zero and should not be tuned until it")
        print("     does. `lucky` wins its first 12 calls for real, so its")
        print("     realised hit rate genuinely is above 50% -- no honest test")
        print("     can rule out an edge that the data does show. The gate's")
        print("     job is to make that rare and small, not impossible.")
        print()

    return 0 if not failures else 1


COMMANDS = {
    "doctor": cmd_doctor, "scan": cmd_scan, "run": cmd_run,
    "resolve": cmd_resolve, "score": cmd_score, "journal": cmd_journal,
    "serve": cmd_serve, "demo": cmd_demo, "simulate": cmd_simulate,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
