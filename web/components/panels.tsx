"use client";

import { useMemo, useState } from "react";
import {
  money,
  num,
  pct,
  signed,
  toneOf,
  type Snapshot,
  type SignalReading,
} from "@/lib/api";

/**
 * Panel bodies.
 *
 * A rule applied throughout: **never render a number the backend did not
 * supply.** Every panel has an explicit empty state that says what is missing
 * and how to produce it. A dashboard that shows 0.0% for "no data" and 0.0%
 * for "measured zero" is worse than one that shows nothing, because it is
 * confidently wrong rather than obviously incomplete.
 */

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="empty">{children}</div>;
}

// ---------------------------------------------------------------------------

export function TrackRecord({ snapshot }: { snapshot: Snapshot }) {
  const score = snapshot.score;
  if (!score || score.n === 0) {
    return (
      <Empty>
        No resolved calls yet.
        <br />
        A forecasting system that does not resolve its own calls is not a
        forecasting system — this panel is the difference, and right now it is
        honestly empty.
      </Empty>
    );
  }

  const gate = snapshot.gate_progress;
  const sizing = snapshot.last_run?.sizing;
  const ceiling = snapshot.gate?.max_position_fraction;
  const capital = snapshot.capital ?? 0;

  let capitalValue = "$0";
  let capitalNote = "gated on calibration";
  if (sizing?.approved) {
    capitalValue = money(sizing.notional);
    capitalNote = "released for this call";
  } else if (gate?.unlocked) {
    capitalValue = ceiling ? `≤ ${money(capital * ceiling)}` : "—";
    capitalNote = "gate open · no call to size";
  }

  return (
    <>
      <div className="metric-row">
        <div className="metric">
          <span className="metric-label">Resolved calls</span>
          <span className="metric-value">{score.n}</span>
          <span className="metric-sub">
            {score.correct} correct · {score.wrong} wrong · {score.invalidated} stopped
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Hit rate</span>
          <span className="metric-value">{pct(score.hit_rate)}</span>
          <span className="metric-sub">
            95% CI [{pct(score.hit_rate_ci?.[0], 0)}, {pct(score.hit_rate_ci?.[1], 0)}]
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Brier vs 0.2500</span>
          <span className={`metric-value ${score.beats_baseline ? "pos" : "neg"}`}>
            {num(score.brier)}
          </span>
          <span className="metric-sub">
            {score.beats_baseline
              ? "beats the 0.25 baseline"
              : "WORSE than always saying 0.50"}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Capital unlocked</span>
          <span className="metric-value">{capitalValue}</span>
          <span className="metric-sub">{capitalNote}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Anytime lower bound</span>
          <span className="metric-value">
            {pct(score.hit_rate_ci_sequential?.[0])}
          </span>
          <span className="metric-sub">what the gate actually tests</span>
        </div>
      </div>
      <div className="rule" />
      <p className="note">
        Sample adequacy: <b className="accent">{score.adequacy}</b>. The 95% CI is
        the fixed-sample interval and is what gets quoted; the always-valid bound
        is wider and is what the capital gate tests, because the gate is
        consulted after every resolved call and a fixed 95% bound checked
        repeatedly is not a 95% bound.
      </p>
    </>
  );
}

// ---------------------------------------------------------------------------

export function CapitalGate({ snapshot }: { snapshot: Snapshot }) {
  const progress = snapshot.gate_progress;
  if (!progress) return <Empty>Gate state unavailable.</Empty>;

  return (
    <div className="gate">
      {progress.gates.map((row) => (
        <div className="gate-row" key={row.gate} data-passed={String(row.passed)}>
          <div className="gate-line">
            <b>
              {row.passed ? "✓" : "·"} {row.gate}
            </b>
            <span className={row.passed ? "accent" : "dim"}>{row.current}</span>
          </div>
          <div className="gate-track">
            <i style={{ width: `${Math.round(row.progress * 100)}%` }} />
          </div>
          <span className="note">{row.requirement}</span>
        </div>
      ))}
      <div className="rule" />
      <p className="note">
        Position size is a function of demonstrated calibration. Before there is
        a track record, that function returns zero — not a small number, zero.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------

export function PriorPanel({ snapshot }: { snapshot: Snapshot }) {
  const prior = snapshot.prior;
  if (!prior) {
    return (
      <Empty>
        No prior computed. Run a cycle to evaluate the signals.
      </Empty>
    );
  }

  const width = Math.min(100, Math.abs(prior.edge) * 400);

  return (
    <>
      <div className="metric">
        <span className="metric-label">P(up) over the horizon</span>
        <span className="metric-value">{pct(prior.probability, 1)}</span>
        <span className="metric-sub">
          base rate {pct(prior.base_rate, 1)} ·{" "}
          <b className={toneOf(prior.edge)}>{signed(prior.edge)}</b> edge
        </span>
      </div>

      <div className="rule" />

      <div className="bipolar" title="Distance from the base rate">
        <b />
        <i
          style={{
            width: `${width / 2}%`,
            left: prior.edge >= 0 ? "50%" : `${50 - width / 2}%`,
            background: prior.edge >= 0 ? "var(--pos)" : "var(--neg)",
          }}
        />
      </div>

      <div className="rule" />

      <table>
        <tbody>
          <tr>
            <td>signals</td>
            <td className="num">{pct(prior.weights.signals, 0)}</td>
            <td className="num dim">
              {prior.from_signals === null ? "—" : pct(prior.from_signals)}
            </td>
          </tr>
          <tr>
            <td>model</td>
            <td className="num">{pct(prior.weights.model, 0)}</td>
            <td className="num dim">
              {prior.from_model === null ? "—" : pct(prior.from_model)}
            </td>
          </tr>
          <tr>
            <td>base rate</td>
            <td className="num">{pct(prior.weights.base_rate, 0)}</td>
            <td className="num dim">{pct(prior.base_rate)}</td>
          </tr>
        </tbody>
      </table>

      <div className="rule" />
      <p className="note">{prior.provenance}</p>
    </>
  );
}

// ---------------------------------------------------------------------------

type SortKey = "family" | "strength" | "confidence";

export function Signals({ snapshot }: { snapshot: Snapshot }) {
  const [sort, setSort] = useState<SortKey>("strength");
  const [showAll, setShowAll] = useState(false);

  const signals = snapshot.signals ?? [];
  const rows = useMemo(() => {
    const visible = showAll ? signals : signals.filter((s) => s.usable);
    const copy = [...visible];
    copy.sort((a, b) => {
      if (sort === "family") return a.family.localeCompare(b.family) || a.key.localeCompare(b.key);
      if (sort === "confidence") return b.confidence - a.confidence;
      return Math.abs(b.strength) - Math.abs(a.strength);
    });
    return copy;
  }, [signals, sort, showAll]);

  if (signals.length === 0) {
    return <Empty>No signal readings. Run a cycle to evaluate them.</Empty>;
  }

  return (
    <>
      <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
        {(["strength", "family", "confidence"] as SortKey[]).map((key) => (
          <button
            key={key}
            className="btn"
            data-accent={sort === key ? "true" : undefined}
            onClick={() => setSort(key)}
          >
            {key}
          </button>
        ))}
        <button className="btn" onClick={() => setShowAll((value) => !value)}>
          {showAll ? "usable only" : "show unusable"}
        </button>
      </div>

      <table>
        <thead>
          <tr>
            <th>signal</th>
            <th>family</th>
            <th className="num">strength</th>
            <th className="num">conf</th>
            <th style={{ width: "34%" }}>reading</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((signal) => (
            <SignalRow key={signal.key} signal={signal} />
          ))}
        </tbody>
      </table>

      {!showAll && signals.some((s) => !s.usable) ? (
        <p className="note" style={{ marginTop: 10 }}>
          {signals.filter((s) => !s.usable).length} unusable signal(s) hidden.
          A missing signal and a neutral signal are different facts, so they are
          never folded into zero.
        </p>
      ) : null}
    </>
  );
}

function SignalRow({ signal }: { signal: SignalReading }) {
  if (!signal.usable) {
    return (
      <tr>
        <td className="dim">{signal.key}</td>
        <td className="dim">{signal.family}</td>
        <td className="num dim">—</td>
        <td className="num dim">—</td>
        <td className="dim">{signal.unavailable_reason}</td>
      </tr>
    );
  }
  const magnitude = Math.min(100, Math.abs(signal.strength) * 100);
  return (
    <tr title={signal.note}>
      <td>{signal.key}</td>
      <td className="dim">{signal.family}</td>
      <td className={`num ${toneOf(signal.strength)}`}>{signed(signal.strength)}</td>
      <td className="num dim">{signal.confidence.toFixed(2)}</td>
      <td>
        <div className="bipolar">
          <b />
          <i
            style={{
              width: `${magnitude / 2}%`,
              left: signal.strength >= 0 ? "50%" : `${50 - magnitude / 2}%`,
              background: signal.strength >= 0 ? "var(--pos)" : "var(--neg)",
            }}
          />
        </div>
      </td>
    </tr>
  );
}

// ---------------------------------------------------------------------------

export function Regime({ snapshot }: { snapshot: Snapshot }) {
  const regime = snapshot.regime;
  if (!regime) return <Empty>No regime read. Run a cycle.</Empty>;

  const order = ["goldilocks", "reflation", "deflation", "stagflation"];
  const probabilities = regime.probabilities ?? {};

  return (
    <>
      <div className="quad">
        {order.map((name) => (
          <div
            className="quad-cell"
            key={name}
            data-on={String(name === regime.quadrant)}
          >
            <span className="quad-name">{name}</span>
            <span className="quad-value">{pct(probabilities[name], 0)}</span>
          </div>
        ))}
      </div>
      <div className="rule" />
      <table>
        <tbody>
          <tr>
            <td>growth</td>
            <td className={`num ${toneOf(regime.growth_score)}`}>
              {signed(regime.growth_score, 2)}
            </td>
          </tr>
          <tr>
            <td>inflation</td>
            {/* Sign is inverted for tone only: accelerating inflation is the
                unwelcome direction, so it reads as the negative tint while the
                number itself keeps its true sign. */}
            <td className={`num ${toneOf(-regime.inflation_score)}`}>
              {signed(regime.inflation_score, 2)}
            </td>
          </tr>
          <tr>
            <td>confidence</td>
            <td className="num">{pct(regime.confidence, 0)}</td>
          </tr>
          {regime.max_staleness_days !== undefined ? (
            <tr>
              <td>oldest input</td>
              <td className={`num ${regime.max_staleness_days > 45 ? "warn" : "dim"}`}>
                {regime.max_staleness_days}d
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
      {regime.transitioning ? (
        <>
          <div className="rule" />
          <p className="note warn">
            Transitioning. A transition is when a single regime label is most
            confidently wrong.
          </p>
        </>
      ) : null}
      {regime.historical_tilt ? (
        <>
          <div className="rule" />
          <p className="note">
            Historically in this regime: {regime.historical_tilt}.
          </p>
        </>
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------------------

export function Scorecard({ snapshot }: { snapshot: Snapshot }) {
  const learning = snapshot.learning;
  if (!learning) {
    return (
      <Empty>
        Nothing learned yet.
        <br />
        Run <b className="accent">vault learn</b> to score every signal against
        forward returns and find out which of them, if any, has ever predicted
        anything.
      </Empty>
    );
  }

  const card = learning.scorecard;
  const rows = [...card.signals].sort((a, b) => b.weight - a.weight || (b.ic ?? 0) - (a.ic ?? 0));
  const detectable = rows.find((row) => row.n > 0)?.minimum_detectable_ic;

  return (
    <>
      <p className="note" style={{ marginBottom: 10 }}>
        {card.earning} of {card.total} signals earning weight against{" "}
        <b>{card.target}</b> at {card.horizon_days}d. {card.samples} observations;
        the target rose in {pct(card.base_rate, 1)} of windows.
        {detectable !== undefined ? (
          <>
            {" "}
            Nothing below an IC of <b className="warn">{num(detectable, 3)}</b> is
            distinguishable from zero at this sample size.
          </>
        ) : null}
      </p>
      <table>
        <thead>
          <tr>
            <th>signal</th>
            <th className="num">+/−</th>
            <th className="num">hit</th>
            <th className="num">IC</th>
            <th className="num">weight</th>
            <th>verdict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} title={row.reason}>
              <td className={row.weight > 0 ? "accent" : ""}>{row.key}</td>
              <td className="num dim">
                {row.n_positive}/{row.n_negative}
              </td>
              <td className="num dim">{pct(row.hit_rate, 0)}</td>
              <td className={`num ${toneOf(row.ic)}`}>{signed(row.ic)}</td>
              <td className="num">{row.weight > 0 ? num(row.weight, 3) : "—"}</td>
              <td className={row.weight > 0 ? "pos" : "dim"}>{row.verdict}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

// ---------------------------------------------------------------------------

export function Training({ snapshot }: { snapshot: Snapshot }) {
  const training = snapshot.learning?.training;
  if (!training) {
    return (
      <Empty>
        No training run. <b className="accent">vault learn</b> fits the network
        and a linear baseline on purged walk-forward folds and reports which, if
        either, beat the base rate.
      </Empty>
    );
  }

  const best = Math.min(training.brier_base, training.brier_logistic, training.brier_network);
  const cell = (value: number) => (value === best ? "accent" : "dim");

  return (
    <>
      <div className="metric-row" style={{ marginBottom: 12 }}>
        <div className="metric">
          <span className="metric-label">Base rate</span>
          <span className={`metric-value ${cell(training.brier_base)}`} style={{ fontSize: 20 }}>
            {num(training.brier_base)}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Logistic</span>
          <span className={`metric-value ${cell(training.brier_logistic)}`} style={{ fontSize: 20 }}>
            {num(training.brier_logistic)}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">Network</span>
          <span className={`metric-value ${cell(training.brier_network)}`} style={{ fontSize: 20 }}>
            {num(training.brier_network)}
          </span>
        </div>
      </div>

      <table>
        <thead>
          <tr>
            <th className="num">fold</th>
            <th className="num">train</th>
            <th className="num">test</th>
            <th className="num">purged</th>
            <th className="num">base</th>
            <th className="num">logistic</th>
            <th className="num">network</th>
          </tr>
        </thead>
        <tbody>
          {training.folds.map((fold) => (
            <tr key={fold.fold}>
              <td className="num">{fold.fold}</td>
              <td className="num dim">{fold.n_train}</td>
              <td className="num dim">{fold.n_test}</td>
              <td className="num dim">{fold.purged}</td>
              <td className="num">{num(fold.brier_base, 3)}</td>
              <td className="num">{num(fold.brier_logistic, 3)}</td>
              <td className="num">{num(fold.brier_network, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="rule" />
      <p className="note">
        <b className={training.usable ? "pos" : "warn"}>
          {training.winner.replace("_", " ").toUpperCase()}
        </b>{" "}
        — {training.reason}
      </p>
    </>
  );
}

// ---------------------------------------------------------------------------

export function Provenance({ snapshot }: { snapshot: Snapshot }) {
  const importance = snapshot.learning?.training.importance ?? [];
  const predictor = snapshot.predictor;

  return (
    <>
      {predictor ? (
        <>
          <p className="note">{predictor.summary}</p>
          <div className="rule" />
        </>
      ) : null}

      {importance.length === 0 ? (
        <Empty>
          No feature attribution yet. It is computed during{" "}
          <b className="accent">vault learn</b>.
        </Empty>
      ) : (
        <table>
          <thead>
            <tr>
              <th>feature</th>
              <th className="num">Δ brier</th>
            </tr>
          </thead>
          <tbody>
            {importance.slice(0, 12).map((row) => (
              <tr key={row.feature}>
                <td className={row.delta_brier > 0 ? "" : "dim"}>{row.feature}</td>
                <td className={`num ${toneOf(row.delta_brier)}`}>
                  {signed(row.delta_brier, 5)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {importance.some((row) => row.delta_brier <= 0) ? (
        <p className="note" style={{ marginTop: 10 }}>
          Features with a non-positive score are noise to this model — removing
          them would improve it.
        </p>
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------------------

export function OpenCalls({ snapshot }: { snapshot: Snapshot }) {
  const positions = snapshot.open_positions ?? [];
  if (positions.length === 0) return <Empty>No open calls.</Empty>;
  return (
    <table>
      <thead>
        <tr>
          <th>asset</th>
          <th>dir</th>
          <th className="num">P</th>
          <th className="num">entry</th>
          <th className="num">days</th>
        </tr>
      </thead>
      <tbody>
        {positions.map((position) => (
          <tr key={position.thesis_id}>
            <td>{position.asset}</td>
            <td className={position.direction === "up" ? "pos" : "neg"}>
              {position.direction}
            </td>
            <td className="num">{position.probability.toFixed(2)}</td>
            <td className="num dim">{position.entry.toFixed(2)}</td>
            <td className="num dim">{position.days_left}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CallHistory({ snapshot }: { snapshot: Snapshot }) {
  const calls = snapshot.recent_calls ?? [];
  if (calls.length === 0) return <Empty>No calls recorded.</Empty>;
  return (
    <table>
      <thead>
        <tr>
          <th>when</th>
          <th>asset</th>
          <th>dir</th>
          <th className="num">P</th>
          <th>result</th>
          <th className="num">move</th>
          <th className="num">brier</th>
        </tr>
      </thead>
      <tbody>
        {calls.map((call) => (
          <tr key={call.thesis_id}>
            <td className="dim">{call.created_at.slice(5, 10)}</td>
            <td>{call.asset}</td>
            <td className={call.direction === "up" ? "pos" : "neg"}>{call.direction}</td>
            <td className="num">{call.probability.toFixed(2)}</td>
            <td
              className={
                call.outcome === "correct" ? "pos" : call.outcome === "pending" ? "dim" : "neg"
              }
            >
              {call.outcome}
            </td>
            <td className={`num ${toneOf(call.realized_pct)}`}>
              {call.realized_pct === null ? "—" : `${signed(call.realized_pct, 2)}%`}
            </td>
            <td className="num dim">{call.brier === null ? "—" : num(call.brier, 3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
