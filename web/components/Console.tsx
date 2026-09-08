"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Board, { type PanelContent } from "@/components/Board";
import {
  CallHistory,
  CapitalGate,
  OpenCalls,
  PriorPanel,
  Provenance,
  Regime,
  Scorecard,
  Signals,
  TrackRecord,
  Training,
} from "@/components/panels";
import { fetchSnapshot, pct, type Feed } from "@/lib/api";

const POLL_MS = 4000;

export default function Console() {
  const [feed, setFeed] = useState<Feed>({ snapshot: null, error: null, fetchedAt: null });
  const [busy, setBusy] = useState(false);
  const inFlight = useRef<AbortController | null>(null);

  const poll = useCallback(async () => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    try {
      setFeed(await fetchSnapshot(controller.signal));
    } catch {
      /* aborted by the next poll; the newer request owns the state */
    }
  }, []);

  useEffect(() => {
    void poll();
    const timer = setInterval(() => void poll(), POLL_MS);
    return () => {
      clearInterval(timer);
      inFlight.current?.abort();
    };
  }, [poll]);

  const runCycle = useCallback(async () => {
    setBusy(true);
    try {
      await fetch("/api/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: "{}",
      });
    } catch {
      /* the poll below surfaces whatever actually happened */
    } finally {
      setBusy(false);
      void poll();
    }
  }, [poll]);

  const snapshot = feed.snapshot;
  const live = Boolean(snapshot) && !feed.error;

  const panels: PanelContent[] = snapshot
    ? [
        { id: "record", body: <TrackRecord snapshot={snapshot} /> },
        {
          id: "gate",
          badge: snapshot.gate_progress
            ? `${snapshot.gate_progress.passed}/${snapshot.gate_progress.total}`
            : undefined,
          body: <CapitalGate snapshot={snapshot} />,
        },
        {
          id: "prior",
          badge: snapshot.prior?.confidence,
          body: <PriorPanel snapshot={snapshot} />,
        },
        {
          id: "regime",
          badge: snapshot.regime ? pct(snapshot.regime.confidence, 0) : undefined,
          body: <Regime snapshot={snapshot} />,
        },
        {
          id: "signals",
          badge: snapshot.signals
            ? `${snapshot.signals.filter((s) => s.usable).length}/${snapshot.signals.length}`
            : undefined,
          body: <Signals snapshot={snapshot} />,
        },
        {
          id: "scorecard",
          badge: snapshot.learning
            ? `${snapshot.learning.scorecard.earning} earning`
            : undefined,
          body: <Scorecard snapshot={snapshot} />,
        },
        {
          id: "training",
          badge: snapshot.learning?.training.winner,
          body: <Training snapshot={snapshot} />,
        },
        { id: "provenance", body: <Provenance snapshot={snapshot} /> },
        {
          id: "positions",
          badge: snapshot.open_positions?.length
            ? `${snapshot.open_positions.length}`
            : undefined,
          body: <OpenCalls snapshot={snapshot} />,
        },
        { id: "calls", body: <CallHistory snapshot={snapshot} /> },
      ]
    : [];

  return (
    <>
      <header className="chrome">
        <div className="mark">
          <b>Vault</b>
          <span>Console</span>
        </div>

        <div className="chrome-meta">
          {snapshot ? (
            <>
              <span>
                {snapshot.series_loaded ?? 0} series · {snapshot.journal?.total ?? 0} calls
              </span>
              <span className="pill">{snapshot.data_mode ?? "unknown"}</span>
              <span className="pill">{snapshot.stage ?? "idle"}</span>
            </>
          ) : null}
          <span className="pill" data-live={String(live)}>
            <i />
            {live ? "connected" : "offline"}
          </span>
          <button className="btn" data-accent="true" onClick={runCycle} disabled={busy}>
            {busy ? "running…" : "Run cycle"}
          </button>
        </div>
      </header>

      <main>
        {feed.error ? (
          <div style={{ padding: "16px 16px 0" }}>
            <div className="banner">
              <b>Offline.</b>
              <span>{feed.error}</span>
            </div>
          </div>
        ) : null}

        {snapshot?.data_mode === "fixture" ? (
          <div style={{ padding: "16px 16px 0" }}>
            <div className="banner">
              <b>Fixture data.</b>
              <span>
                These series are synthetic, generated locally for demonstration.
                Nothing shown here is a reading about the real world.
              </span>
            </div>
          </div>
        ) : null}

        {snapshot ? (
          <Board panels={panels} />
        ) : (
          <div className="empty" style={{ minHeight: "50vh" }}>
            {feed.fetchedAt ? "Waiting for the Vault API…" : "Connecting…"}
          </div>
        )}
      </main>
    </>
  );
}
