"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  boardRows,
  clearLayout,
  COLUMNS,
  DEFAULT_LAYOUT,
  loadLayout,
  MIN_SIZE,
  moveBox,
  raise,
  resizeBox,
  ROW_HEIGHT,
  saveLayout,
  TITLES,
  toPixels,
  type Box,
  type PanelId,
} from "@/lib/layout";

type Drag =
  | { kind: "move"; id: PanelId; grabX: number; grabY: number; originX: number; originY: number }
  | { kind: "resize"; id: PanelId; startX: number; startY: number; originW: number; originH: number };

export interface PanelContent {
  id: PanelId;
  badge?: ReactNode;
  body: ReactNode;
}

/**
 * The board.
 *
 * Every panel can be dragged by its header and resized from its corner, and
 * the arrangement survives a reload. Panels may overlap; nothing auto-packs.
 * That is a deliberate choice — a grid that reflows while you are dragging
 * makes precise arrangement impossible, and the request was for something
 * organizable, which means the user's arrangement wins.
 */
export default function Board({ panels }: { panels: PanelContent[] }) {
  const [boxes, setBoxes] = useState<Box[]>(DEFAULT_LAYOUT);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [columnWidth, setColumnWidth] = useState(96);
  const [hydrated, setHydrated] = useState(false);
  const surface = useRef<HTMLDivElement>(null);

  // Load the stored arrangement only after mount. Reading localStorage during
  // render would produce different markup on the server and the client, and
  // React would discard the whole tree with a hydration error.
  useEffect(() => {
    setBoxes(loadLayout());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (hydrated) saveLayout(boxes);
  }, [boxes, hydrated]);

  useLayoutEffect(() => {
    const element = surface.current;
    if (!element) return;
    const measure = () => setColumnWidth(element.clientWidth / COLUMNS);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const beginMove = useCallback(
    (event: React.PointerEvent, id: PanelId) => {
      const box = boxes.find((candidate) => candidate.id === id);
      if (!box || event.button !== 0) return;
      event.preventDefault();
      setBoxes((current) => raise(current, id));
      setDrag({
        kind: "move",
        id,
        grabX: event.clientX,
        grabY: event.clientY,
        originX: box.x,
        originY: box.y,
      });
    },
    [boxes],
  );

  const beginResize = useCallback(
    (event: React.PointerEvent, id: PanelId) => {
      const box = boxes.find((candidate) => candidate.id === id);
      if (!box || event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      setBoxes((current) => raise(current, id));
      setDrag({
        kind: "resize",
        id,
        startX: event.clientX,
        startY: event.clientY,
        originW: box.w,
        originH: box.h,
      });
    },
    [boxes],
  );

  // Listeners live on the window, not on the panel. A pointer moving faster
  // than React re-renders will leave a panel-bound listener behind, and the
  // panel sticks to the cursor until the next click.
  useEffect(() => {
    if (!drag) return;

    const onMove = (event: PointerEvent) => {
      setBoxes((current) =>
        current.map((box) => {
          if (box.id !== drag.id) return box;
          if (drag.kind === "move") {
            return moveBox(
              box,
              drag.originX + (event.clientX - drag.grabX) / columnWidth,
              drag.originY + (event.clientY - drag.grabY) / ROW_HEIGHT,
            );
          }
          return resizeBox(
            box,
            drag.originW + (event.clientX - drag.startX) / columnWidth,
            drag.originH + (event.clientY - drag.startY) / ROW_HEIGHT,
          );
        }),
      );
    };

    const onUp = () => setDrag(null);

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [drag, columnWidth]);

  // Keyboard nudging. A board that can only be arranged by dragging cannot be
  // arranged precisely, and cannot be arranged at all without a pointer.
  const nudge = useCallback((id: PanelId, dx: number, dy: number, resize: boolean) => {
    setBoxes((current) =>
      current.map((box) => {
        if (box.id !== id) return box;
        return resize
          ? resizeBox(box, box.w + dx, box.h + dy)
          : moveBox(box, box.x + dx, box.y + dy);
      }),
    );
  }, []);

  const hidden = panels.filter(
    (panel) => !boxes.some((box) => box.id === panel.id),
  );

  const rows = boardRows(boxes);

  return (
    <>
      <div
        className="board"
        ref={surface}
        data-dragging={drag ? "true" : "false"}
        style={
          {
            height: rows * ROW_HEIGHT + 32,
            ["--col-w" as string]: `${columnWidth}px`,
            ["--row-h" as string]: `${ROW_HEIGHT}px`,
          } as React.CSSProperties
        }
      >
        {boxes.map((box) => {
          const panel = panels.find((candidate) => candidate.id === box.id);
          if (!panel) return null;
          const geometry = toPixels(box, columnWidth);
          const active = drag?.id === box.id;
          return (
            <section
              key={box.id}
              className="panel"
              data-active={active ? "true" : "false"}
              style={geometry}
              aria-label={TITLES[box.id]}
            >
              <header
                className="panel-head"
                onPointerDown={(event) => beginMove(event, box.id)}
                tabIndex={0}
                role="button"
                aria-label={`Move ${TITLES[box.id]}. Arrow keys move, shift+arrows resize.`}
                onKeyDown={(event) => {
                  const step = event.shiftKey;
                  const map: Record<string, [number, number]> = {
                    ArrowLeft: [-1, 0],
                    ArrowRight: [1, 0],
                    ArrowUp: [0, -1],
                    ArrowDown: [0, 1],
                  };
                  const delta = map[event.key];
                  if (!delta) return;
                  event.preventDefault();
                  nudge(box.id, delta[0], delta[1], step);
                }}
              >
                <span className="panel-grip" aria-hidden>
                  <i />
                  <i />
                  <i />
                </span>
                <span className="panel-title">{TITLES[box.id]}</span>
                {panel.badge ? <span className="panel-badge">{panel.badge}</span> : null}
              </header>
              <div className="panel-body">{panel.body}</div>
              <div
                className="resize"
                onPointerDown={(event) => beginResize(event, box.id)}
                aria-hidden
              />
            </section>
          );
        })}
      </div>

      <div className="tray">
        <button className="btn" onClick={() => { clearLayout(); setBoxes(DEFAULT_LAYOUT); }}>
          Reset layout
        </button>
        {hidden.map((panel) => (
          <button
            key={panel.id}
            className="chip"
            onClick={() =>
              setBoxes((current) => [
                ...current,
                {
                  id: panel.id,
                  x: 0,
                  y: boardRows(current),
                  w: Math.max(MIN_SIZE[panel.id].w, 4),
                  h: Math.max(MIN_SIZE[panel.id].h, 6),
                },
              ])
            }
          >
            + {TITLES[panel.id]}
          </button>
        ))}
      </div>
    </>
  );
}
