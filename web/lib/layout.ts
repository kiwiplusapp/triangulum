/**
 * Board layout: geometry, persistence, and the rules for moving panels.
 *
 * Implemented directly rather than with react-grid-layout. The whole
 * behaviour is about 150 lines of pointer arithmetic, and the library would
 * be a larger dependency than the rest of the app put together — this
 * project ships a neural network written in pure Python for the same reason.
 */

export type PanelId =
  | "record"
  | "gate"
  | "prior"
  | "signals"
  | "regime"
  | "scorecard"
  | "training"
  | "calls"
  | "positions"
  | "provenance";

export interface Box {
  id: PanelId;
  x: number; // grid columns from the left
  y: number; // grid rows from the top
  w: number; // width in columns
  h: number; // height in rows
}

export const COLUMNS = 12;
export const ROW_HEIGHT = 34;
export const GAP = 10;

/** Minimum sizes, per panel, so nothing can be shrunk into illegibility. */
export const MIN_SIZE: Record<PanelId, { w: number; h: number }> = {
  record: { w: 4, h: 4 },
  gate: { w: 3, h: 5 },
  prior: { w: 3, h: 5 },
  signals: { w: 4, h: 6 },
  regime: { w: 3, h: 5 },
  scorecard: { w: 5, h: 5 },
  training: { w: 4, h: 5 },
  calls: { w: 4, h: 5 },
  positions: { w: 3, h: 4 },
  provenance: { w: 3, h: 4 },
};

export const TITLES: Record<PanelId, string> = {
  record: "Track record",
  gate: "Capital gate",
  prior: "Quantitative prior",
  signals: "Signal readings",
  regime: "Macro regime",
  scorecard: "Signal scorecard",
  training: "Walk-forward training",
  calls: "Call history",
  positions: "Open calls",
  provenance: "Provenance",
};

export const DEFAULT_LAYOUT: Box[] = [
  { id: "record", x: 0, y: 0, w: 12, h: 4 },
  { id: "gate", x: 0, y: 4, w: 3, h: 8 },
  { id: "prior", x: 3, y: 4, w: 3, h: 8 },
  { id: "regime", x: 6, y: 4, w: 3, h: 8 },
  { id: "provenance", x: 9, y: 4, w: 3, h: 8 },
  { id: "signals", x: 0, y: 12, w: 6, h: 11 },
  { id: "scorecard", x: 6, y: 12, w: 6, h: 11 },
  { id: "training", x: 0, y: 23, w: 5, h: 8 },
  { id: "positions", x: 5, y: 23, w: 3, h: 8 },
  { id: "calls", x: 8, y: 23, w: 4, h: 8 },
];

const STORAGE_KEY = "vault.board.v1";

export function loadLayout(): Box[] {
  if (typeof window === "undefined") return DEFAULT_LAYOUT;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_LAYOUT;
    const parsed = JSON.parse(raw) as Box[];
    if (!Array.isArray(parsed) || parsed.length === 0) return DEFAULT_LAYOUT;

    // Merge rather than replace. A stored layout from an older version is
    // missing any panel added since, and dropping those panels silently is
    // how a dashboard quietly loses a feature after an update.
    const known = new Map(parsed.map((box) => [box.id, box]));
    return DEFAULT_LAYOUT.map((fallback) => {
      const stored = known.get(fallback.id);
      return stored ? { ...fallback, ...stored, id: fallback.id } : fallback;
    });
  } catch {
    return DEFAULT_LAYOUT;
  }
}

export function saveLayout(boxes: Box[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(boxes));
  } catch {
    /* private mode, quota, blocked storage — the board still works, it just
       forgets. Never let persistence failure break the render. */
  }
}

export function clearLayout(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* as above */
  }
}

export const clamp = (value: number, low: number, high: number) =>
  Math.max(low, Math.min(high, value));

/** Pixel geometry for a box, given the measured column width. */
export function toPixels(box: Box, columnWidth: number) {
  return {
    left: box.x * columnWidth + GAP / 2,
    top: box.y * ROW_HEIGHT + GAP / 2,
    width: box.w * columnWidth - GAP,
    height: box.h * ROW_HEIGHT - GAP,
  };
}

/** Rows needed to contain every panel, plus breathing room at the bottom. */
export function boardRows(boxes: Box[]): number {
  return boxes.reduce((tallest, box) => Math.max(tallest, box.y + box.h), 0) + 2;
}

export function moveBox(box: Box, x: number, y: number): Box {
  const nextX = clamp(Math.round(x), 0, COLUMNS - box.w);
  return { ...box, x: nextX, y: Math.max(0, Math.round(y)) };
}

export function resizeBox(box: Box, w: number, h: number): Box {
  const min = MIN_SIZE[box.id];
  const nextW = clamp(Math.round(w), min.w, COLUMNS - box.x);
  const nextH = Math.max(min.h, Math.round(h));
  return { ...box, w: nextW, h: nextH };
}

/**
 * Bring a panel to the front of the stacking order.
 *
 * Panels may overlap — this is a board the user arranges, not a packed grid
 * that rearranges itself. Auto-packing looks tidier in a screenshot and is
 * infuriating in use: you drag one panel and three others jump.
 */
export function raise(boxes: Box[], id: PanelId): Box[] {
  const target = boxes.find((box) => box.id === id);
  if (!target) return boxes;
  return [...boxes.filter((box) => box.id !== id), target];
}
