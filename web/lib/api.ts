/**
 * The shape of what the Python side serves, and how to fetch it.
 *
 * Every field is optional. The console has to render honestly against a
 * backend that is starting up, running on fixtures, mid-run, or not there at
 * all — and "not there at all" must look like an empty console that says the
 * backend is unreachable, never like a console reporting zeros as though
 * they were measurements.
 */

export interface CalibrationScore {
  n: number;
  correct: number;
  wrong: number;
  invalidated: number;
  hit_rate: number;
  hit_rate_ci: [number, number];
  hit_rate_ci_sequential: [number, number];
  brier: number;
  brier_baseline: number;
  skill: number;
  overconfidence: number;
  adequacy: string;
  beats_baseline: boolean;
  reliability_buckets?: { predicted: number; observed: number; count: number }[];
}

export interface GateRow {
  gate: string;
  requirement: string;
  current: string;
  passed: boolean;
  progress: number;
}

export interface SignalReading {
  key: string;
  label: string;
  family: string;
  strength: number;
  weighted_strength: number;
  raw: number | null;
  zscore: number | null;
  confidence: number;
  stance: "risk_on" | "risk_off" | "neutral";
  inputs: string[];
  staleness_days: number;
  usable: boolean;
  note: string;
  unavailable_reason: string;
}

export interface Prior {
  probability: number;
  base_rate: number;
  edge: number;
  from_signals: number | null;
  from_model: number | null;
  weights: { signals: number; model: number; base_rate: number };
  shrinkage: number;
  confidence: string;
  provenance: string;
  signals_used: number;
  signals_available: number;
  contributions: {
    signal: string;
    strength: number;
    weight: number;
    contribution: number;
  }[];
}

export interface SignalPerformance {
  key: string;
  label: string;
  family: string;
  n: number;
  n_positive: number;
  n_negative: number;
  hit_rate: number;
  baseline_accuracy: number;
  hit_rate_low_sequential: number;
  ic: number | null;
  effective_n: number;
  minimum_detectable_ic: number;
  ic_significant: boolean;
  spread: number;
  weight: number;
  verdict: string;
  reason: string;
}

export interface TrainingFold {
  fold: number;
  n_train: number;
  n_test: number;
  purged: number;
  brier_base: number;
  brier_logistic: number;
  brier_network: number;
  brier_network_in_sample: number;
  overfit_gap: number;
}

export interface Snapshot {
  generated_at?: string;
  stage?: string;
  runs?: number;
  capital?: number;
  data_mode?: string;
  series_loaded?: number;
  journal?: {
    total: number;
    pending: number;
    resolved: number;
    scoreable: number;
    chain_valid: boolean;
    chain_summary: string;
  };
  score?: CalibrationScore;
  gate?: { max_position_fraction: number; min_samples: number };
  gate_progress?: {
    gates: GateRow[];
    passed: number;
    total: number;
    unlocked: boolean;
    overall_progress: number;
  };
  regime?: {
    quadrant: string;
    description?: string;
    confidence: number;
    probabilities: Record<string, number>;
    growth_score: number;
    inflation_score: number;
    transitioning?: boolean;
    historical_tilt?: string;
    max_staleness_days?: number;
    missing_inputs?: string[];
  };
  signals?: SignalReading[];
  prior?: Prior;
  predictor?: { summary: string; model_kind: string; trained: boolean };
  learning?: {
    dataset: { n: number; n_features: number; base_rate: number; horizon_days: number };
    scorecard: {
      target: string;
      horizon_days: number;
      samples: number;
      base_rate: number;
      tests_run: number;
      earning: number;
      total: number;
      signals: SignalPerformance[];
    };
    training: {
      n_samples: number;
      folds: TrainingFold[];
      brier_base: number;
      brier_logistic: number;
      brier_network: number;
      winner: string;
      usable: boolean;
      reason: string;
      importance: { feature: string; delta_brier: number }[];
    };
  };
  last_run?: { sizing?: { approved: boolean; notional: number; reason: string } | null };
  open_positions?: {
    thesis_id: string;
    asset: string;
    direction: string;
    probability: number;
    entry: number;
    resolve_on: string;
    days_left: number;
  }[];
  recent_calls?: {
    thesis_id: string;
    asset: string;
    direction: string;
    horizon: string;
    probability: number;
    outcome: string;
    realized_pct: number | null;
    brier: number | null;
    created_at: string;
  }[];
}

export interface Feed {
  snapshot: Snapshot | null;
  error: string | null;
  fetchedAt: number | null;
}

export async function fetchSnapshot(signal?: AbortSignal): Promise<Feed> {
  try {
    const response = await fetch("/api/state", { signal, cache: "no-store" });
    if (!response.ok) {
      return {
        snapshot: null,
        error: `backend returned ${response.status}`,
        fetchedAt: Date.now(),
      };
    }
    return { snapshot: (await response.json()) as Snapshot, error: null, fetchedAt: Date.now() };
  } catch (cause) {
    if (signal?.aborted) throw cause;
    return {
      snapshot: null,
      // Say what is actually wrong. "Failed to fetch" on its own sends people
      // looking at their network tab instead of at the process they forgot to
      // start.
      error:
        "cannot reach the Vault API. Start it with `python3 -m vault --fixtures serve` " +
        "(or `demo` for a seeded track record) and it will appear here.",
      fetchedAt: Date.now(),
    };
  }
}

// ---- formatting -----------------------------------------------------------

export const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(digits)}%`;

export const num = (value: number | null | undefined, digits = 4) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : value.toFixed(digits);

export const signed = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;

export const money = (value: number | null | undefined) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

export const toneOf = (value: number | null | undefined) =>
  value === null || value === undefined || Math.abs(value) < 1e-9
    ? "dim"
    : value > 0
      ? "pos"
      : "neg";
