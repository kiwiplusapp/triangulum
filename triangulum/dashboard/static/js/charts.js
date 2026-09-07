/* ==========================================================================
   Triangulum — canvas chart primitives
   --------------------------------------------------------------------------
   Hand-rolled on canvas rather than a charting library, for the same reason
   the server is stdlib-only: the dashboard is how you find out the engine is
   in trouble, so it must render when things are broken — including offline,
   behind a proxy that blocks CDNs, or on a machine where npm never ran.

   Every chart here follows the same rules:
     · thin marks (2px lines, 8px+ markers)
     · recessive grid and axes
     · a hover layer with a crosshair and a tooltip
     · a legend whenever there are two or more series
     · direct labels on the values that matter, never on every point
     · one y-scale per chart, never two
   ========================================================================== */
'use strict';

/* Everything below is scoped to this IIFE. Only `window.TriCharts` escapes.
   Without this, `const Palette` here and the destructuring `const { Palette }`
   in hud.js are two top-level declarations of the same identifier in the same
   global scope, which is a SyntaxError that kills the whole page before a
   single pixel renders — and, because it is a parse-time error, it produces no
   partial render to hint at the cause. */
(function () {

  const CSSVar = (name, fallback) => {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  };

  const Palette = {
    get s1()      { return CSSVar('--series-1', '#3987e5'); },
    get s2()      { return CSSVar('--series-2', '#c98500'); },
    get s3()      { return CSSVar('--series-3', '#199e70'); },
    get s4()      { return CSSVar('--series-4', '#9085e9'); },
    get good()    { return CSSVar('--good', '#0ca30c'); },
    get warning() { return CSSVar('--warning', '#fab219'); },
    get critical(){ return CSSVar('--critical', '#d03b3b'); },
    get ink()     { return CSSVar('--ink', '#ffffff'); },
    get ink2()    { return CSSVar('--ink-2', '#a8b4c4'); },
    get ink3()    { return CSSVar('--ink-3', '#6b7889'); },
    get hairline(){ return CSSVar('--hairline', '#1e2836'); },
    get surface() { return CSSVar('--surface-1', '#0b0f14'); },
  };

  /** Resize a canvas for the device pixel ratio and return a scaled context. */
  function prepare(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  const fmt = {
    num(v, digits = 2) {
      if (!isFinite(v)) return '—';
      return v.toLocaleString(undefined, {
        minimumFractionDigits: digits, maximumFractionDigits: digits,
      });
    },
    bps(v) { return `${v >= 0 ? '+' : ''}${fmt.num(v, 2)} bps`; },
    pct(v, digits = 2) { return `${v >= 0 ? '+' : ''}${fmt.num(v, digits)}%`; },
    time(ns) {
      const d = new Date(ns / 1e6);
      return d.toLocaleTimeString(undefined, { hour12: false });
    },
    compact(v) {
      if (Math.abs(v) >= 1e6) return `${fmt.num(v / 1e6, 1)}M`;
      if (Math.abs(v) >= 1e3) return `${fmt.num(v / 1e3, 1)}k`;
      return fmt.num(v, 0);
    },
  };

  /** Nice axis ticks: 1/2/5 × 10^n covering [lo, hi]. */
  function niceTicks(lo, hi, count = 4) {
    if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
    const span = hi - lo;
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
    const start = Math.ceil(lo / step) * step;
    const out = [];
    for (let v = start; v <= hi + step * 1e-9; v += step) out.push(v);
    return out;
  }

  function drawGrid(ctx, box, ticks, scaleY, labelFn) {
    ctx.save();
    ctx.strokeStyle = Palette.hairline;
    ctx.fillStyle = Palette.ink3;
    ctx.lineWidth = 1;
    ctx.font = '10px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of ticks) {
      const y = Math.round(scaleY(t)) + 0.5;
      if (y < box.top - 1 || y > box.bottom + 1) continue;
      ctx.beginPath();
      ctx.moveTo(box.left, y);
      ctx.lineTo(box.right, y);
      ctx.stroke();
      if (labelFn) ctx.fillText(labelFn(t), box.left - 6, y);
    }
    ctx.restore();
  }

  /* ── Equity curve ─────────────────────────────────────────────────────────
     One series plus a dashed reference line for the target trajectory. Two
     marks on screen, so a legend is mandatory — the caller renders it into
     #legend-equity from EquityChart.legend.
     ------------------------------------------------------------------------ */

  class EquityChart {
    constructor(canvas, tooltip) {
      this.canvas = canvas;
      this.tooltip = tooltip;
      this.points = [];
      this.target = [];
      this.hoverIndex = -1;
      this.box = null;

      canvas.addEventListener('mousemove', (e) => this._onMove(e));
      canvas.addEventListener('mouseleave', () => {
        this.hoverIndex = -1;
        this.tooltip.dataset.visible = 'false';
        this.render();
      });
    }

    static get legend() {
      return [
        { label: 'equity', color: Palette.s1 },
        { label: 'target', color: Palette.s2, dashed: true },
      ];
    }

    setData(points, { startEquity = 0, dailyTargetBps = 0 } = {}) {
      this.points = points || [];
      this.target = [];
      if (this.points.length >= 2 && dailyTargetBps > 0 && startEquity > 0) {
        // The target trajectory compounds at the configured daily rate from the
        // first observation. Drawn so the gap is never in doubt.
        const t0 = this.points[0][0];
        const perNs = dailyTargetBps / 1e4 / (86400 * 1e9);
        this.target = this.points.map(([ts]) => {
          const days = (ts - t0) * perNs;
          return [ts, startEquity * Math.pow(1 + dailyTargetBps / 1e4, days * 1)];
        });
        // Simpler and numerically safer: linear-in-log compounding.
        this.target = this.points.map(([ts]) => {
          const days = (ts - t0) / (86400 * 1e9);
          return [ts, startEquity * Math.pow(1 + dailyTargetBps / 1e4, days)];
        });
      }
      this.render();
    }

    _scales(w, h) {
      const box = { left: 52, right: w - 10, top: 10, bottom: h - 20 };
      const xs = this.points.map(p => p[0]);
      const all = this.points.map(p => p[1]).concat(this.target.map(p => p[1]));
      let lo = Math.min(...all), hi = Math.max(...all);
      if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
      if (lo === hi) { lo -= 1; hi += 1; }
      const pad = (hi - lo) * 0.12;
      lo -= pad; hi += pad;
      const x0 = Math.min(...xs), x1 = Math.max(...xs);
      const scaleX = (v) => box.left + ((v - x0) / Math.max(1, x1 - x0)) * (box.right - box.left);
      const scaleY = (v) => box.bottom - ((v - lo) / (hi - lo)) * (box.bottom - box.top);
      return { box, scaleX, scaleY, lo, hi };
    }

    render() {
      const { ctx, w, h } = prepare(this.canvas);
      if (this.points.length < 2) {
        ctx.fillStyle = Palette.ink3;
        ctx.font = '11px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('awaiting equity samples', w / 2, h / 2);
        return;
      }

      const { box, scaleX, scaleY, lo, hi } = this._scales(w, h);
      this.box = box; this.scaleX = scaleX; this.scaleY = scaleY;

      drawGrid(ctx, box, niceTicks(lo, hi, 4), scaleY, (v) => fmt.num(v, 2));

      // Target trajectory: dashed, recessive, drawn under the equity line.
      if (this.target.length > 1) {
        ctx.save();
        ctx.strokeStyle = Palette.s2;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        this.target.forEach(([ts, v], i) => {
          const x = scaleX(ts), y = scaleY(v);
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.restore();
      }

      // Equity area + line.
      const grad = ctx.createLinearGradient(0, box.top, 0, box.bottom);
      grad.addColorStop(0, `${Palette.s1}44`);
      grad.addColorStop(1, `${Palette.s1}00`);
      ctx.beginPath();
      this.points.forEach(([ts, v], i) => {
        const x = scaleX(ts), y = scaleY(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.lineTo(scaleX(this.points[this.points.length - 1][0]), box.bottom);
      ctx.lineTo(scaleX(this.points[0][0]), box.bottom);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.beginPath();
      this.points.forEach(([ts, v], i) => {
        const x = scaleX(ts), y = scaleY(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = Palette.s1;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.stroke();

      // Direct label on the last value only — never a number on every point.
      const last = this.points[this.points.length - 1];
      const lx = scaleX(last[0]), ly = scaleY(last[1]);
      ctx.beginPath();
      ctx.arc(lx, ly, 4, 0, Math.PI * 2);
      ctx.fillStyle = Palette.s1;
      ctx.fill();
      ctx.strokeStyle = Palette.surface;
      ctx.lineWidth = 2;
      ctx.stroke();

      if (this.hoverIndex >= 0 && this.hoverIndex < this.points.length) {
        const [ts, v] = this.points[this.hoverIndex];
        const x = scaleX(ts), y = scaleY(v);
        ctx.save();
        ctx.strokeStyle = Palette.ink3;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(x, box.top); ctx.lineTo(x, box.bottom);
        ctx.stroke();
        ctx.restore();
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = Palette.s1;
        ctx.fill();
        ctx.strokeStyle = Palette.surface;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    _onMove(event) {
      if (!this.points.length || !this.box) return;
      const rect = this.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      let best = 0, bestDist = Infinity;
      for (let i = 0; i < this.points.length; i++) {
        const d = Math.abs(this.scaleX(this.points[i][0]) - x);
        if (d < bestDist) { bestDist = d; best = i; }
      }
      this.hoverIndex = best;
      const [ts, v] = this.points[best];
      const target = this.target[best];
      this.tooltip.innerHTML =
        `<b>${fmt.num(v, 4)}</b><br><span>${fmt.time(ts)}</span>` +
        (target ? `<br><span>target ${fmt.num(target[1], 4)}</span>` : '');
      this.tooltip.style.left = `${this.scaleX(ts)}px`;
      this.tooltip.style.top = `${this.scaleY(v)}px`;
      this.tooltip.dataset.visible = 'true';
      this.render();
    }
  }

  /* ── Edge histogram ───────────────────────────────────────────────────────
     Distribution of detected gross edges, split by whether the gate took them.
     Two series, so: legend present, 2px surface gap between stacked segments.
     ------------------------------------------------------------------------ */

  class EdgeHistogram {
    constructor(canvas, tooltip) {
      this.canvas = canvas;
      this.tooltip = tooltip;
      this.bins = [];
      this.hover = -1;
      canvas.addEventListener('mousemove', (e) => this._onMove(e));
      canvas.addEventListener('mouseleave', () => {
        this.hover = -1;
        this.tooltip.dataset.visible = 'false';
        this.render();
      });
    }

    static get legend() {
      return [
        { label: 'taken', color: Palette.s3 },
        { label: 'passed over', color: Palette.s4 },
      ];
    }

    setData(edges, decisions) {
      // Bin on gross edge. 12 bins across the observed range keeps each bin wide
      // enough to be meaningful at the sample counts this engine produces.
      const values = (edges || []).map(e => e.edge_bps).filter(v => isFinite(v));
      if (!values.length) { this.bins = []; this.render(); return; }
      const lo = Math.min(0, Math.min(...values));
      const hi = Math.max(...values);
      const n = 12;
      const width = Math.max(1e-9, (hi - lo) / n);
      const taken = new Map();
      for (const d of decisions || []) {
        if (d.accept) taken.set(d.path, (taken.get(d.path) || 0) + 1);
      }
      this.bins = Array.from({ length: n }, (_, i) => ({
        lo: lo + i * width, hi: lo + (i + 1) * width, taken: 0, passed: 0,
      }));
      for (const e of edges || []) {
        const i = Math.min(n - 1, Math.max(0, Math.floor((e.edge_bps - lo) / width)));
        if (taken.has(e.path)) this.bins[i].taken++;
        else this.bins[i].passed++;
      }
      this.render();
    }

    render() {
      const { ctx, w, h } = prepare(this.canvas);
      if (!this.bins.length) {
        ctx.fillStyle = Palette.ink3;
        ctx.font = '11px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('no opportunities detected yet', w / 2, h / 2);
        return;
      }
      const box = { left: 36, right: w - 8, top: 10, bottom: h - 22 };
      const maxCount = Math.max(1, ...this.bins.map(b => b.taken + b.passed));
      const scaleY = (v) => box.bottom - (v / maxCount) * (box.bottom - box.top);
      drawGrid(ctx, box, niceTicks(0, maxCount, 3), scaleY, (v) => fmt.num(v, 0));

      const slot = (box.right - box.left) / this.bins.length;
      const barW = Math.max(2, slot - 3);
      this.box = box; this.slot = slot; this.scaleY = scaleY;

      this.bins.forEach((bin, i) => {
        const x = box.left + i * slot + (slot - barW) / 2;
        const total = bin.taken + bin.passed;
        if (!total) return;
        const dim = this.hover >= 0 && this.hover !== i;
        ctx.globalAlpha = dim ? 0.35 : 1;

        // Passed-over sits below; taken stacks on top with a 2px surface gap.
        let y = box.bottom;
        if (bin.passed) {
          const barH = (bin.passed / maxCount) * (box.bottom - box.top);
          ctx.fillStyle = Palette.s4;
          roundRect(ctx, x, y - barH, barW, barH, bin.taken ? 0 : 4);
          y -= barH;
        }
        if (bin.taken) {
          const gap = bin.passed ? 2 : 0;
          const barH = (bin.taken / maxCount) * (box.bottom - box.top);
          ctx.fillStyle = Palette.s3;
          roundRect(ctx, x, y - barH - gap, barW, barH, 4);
        }
        ctx.globalAlpha = 1;
      });

      ctx.fillStyle = Palette.ink3;
      ctx.font = '9.5px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      [0, Math.floor(this.bins.length / 2), this.bins.length - 1].forEach(i => {
        const b = this.bins[i];
        if (!b) return;
        ctx.fillText(`${fmt.num(b.lo, 1)}`, box.left + i * slot + slot / 2, box.bottom + 6);
      });
      ctx.textAlign = 'right';
      ctx.fillText('bps', box.right, box.bottom + 6);
    }

    _onMove(event) {
      if (!this.bins.length || !this.box) return;
      const rect = this.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const i = Math.floor((x - this.box.left) / this.slot);
      if (i < 0 || i >= this.bins.length) return;
      this.hover = i;
      const b = this.bins[i];
      this.tooltip.innerHTML =
        `<b>${fmt.num(b.lo, 1)}–${fmt.num(b.hi, 1)} bps</b><br>` +
        `<span>taken</span> ${b.taken}<br><span>passed over</span> ${b.passed}`;
      this.tooltip.style.left = `${this.box.left + i * this.slot + this.slot / 2}px`;
      this.tooltip.style.top = `${this.scaleY(b.taken + b.passed)}px`;
      this.tooltip.dataset.visible = 'true';
      this.render();
    }
  }

  /* ── Calibration (reliability) ────────────────────────────────────────────
     Predicted probability against observed frequency, with the y=x reference.
     The single most important model diagnostic in the system: the EV maths
     multiplies by P(fill), so a miscalibrated probability is a wrong decision.
     ------------------------------------------------------------------------ */

  class CalibrationChart {
    constructor(canvas, tooltip) {
      this.canvas = canvas;
      this.tooltip = tooltip;
      this.rows = [];
      canvas.addEventListener('mousemove', (e) => this._onMove(e));
      canvas.addEventListener('mouseleave', () => {
        this.tooltip.dataset.visible = 'false';
      });
    }

    setData(rows) { this.rows = rows || []; this.render(); }

    render() {
      const { ctx, w, h } = prepare(this.canvas);
      const size = Math.min(w - 46, h - 26);
      const box = { left: 38, top: 8, right: 38 + size, bottom: 8 + size };
      this.box = box;

      ctx.save();
      ctx.strokeStyle = Palette.hairline;
      ctx.lineWidth = 1;
      ctx.strokeRect(box.left + 0.5, box.top + 0.5, size, size);
      // Perfect-calibration reference.
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = Palette.ink3;
      ctx.beginPath();
      ctx.moveTo(box.left, box.bottom);
      ctx.lineTo(box.right, box.top);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = Palette.ink3;
      ctx.font = '9.5px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('1.0', box.left - 6, box.top);
      ctx.fillText('0.0', box.left - 6, box.bottom);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('predicted', (box.left + box.right) / 2, box.bottom + 7);

      if (!this.rows.length) {
        ctx.fillStyle = Palette.ink3;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('gathering samples', (box.left + box.right) / 2, (box.top + box.bottom) / 2);
        return;
      }

      const maxCount = Math.max(1, ...this.rows.map(r => r.count));
      this.pts = this.rows.map(r => {
        const x = box.left + r.predicted * size;
        const y = box.bottom - r.observed * size;
        // Marker area encodes sample count; never below the 8px minimum.
        const radius = 4 + 5 * Math.sqrt(r.count / maxCount);
        return { ...r, x, y, radius };
      });

      ctx.beginPath();
      this.pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.strokeStyle = `${Palette.s1}99`;
      ctx.lineWidth = 2;
      ctx.stroke();

      for (const p of this.pts) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = Math.abs(p.error) > 0.15 ? Palette.warning : Palette.s1;
        ctx.fill();
        // 2px surface ring so overlapping markers stay separable.
        ctx.strokeStyle = Palette.surface;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    _onMove(event) {
      if (!this.pts || !this.pts.length) return;
      const rect = this.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left, y = event.clientY - rect.top;
      let best = null, bestDist = 22;
      for (const p of this.pts) {
        const d = Math.hypot(p.x - x, p.y - y);
        if (d < bestDist) { bestDist = d; best = p; }
      }
      if (!best) { this.tooltip.dataset.visible = 'false'; return; }
      this.tooltip.innerHTML =
        `<b>predicted ${fmt.num(best.predicted, 2)}</b><br>` +
        `<span>observed</span> ${fmt.num(best.observed, 3)}<br>` +
        `<span>n</span> ${best.count}`;
      this.tooltip.style.left = `${best.x}px`;
      this.tooltip.style.top = `${best.y}px`;
      this.tooltip.dataset.visible = 'true';
    }
  }

  /* ── Currency graph ───────────────────────────────────────────────────────
     Node-link view of the tradable universe, with a travelling pulse on the
     most recent cycle. Deliberately the one animated element: it shows the
     thing that is otherwise invisible — that a cycle is a closed loop through
     the graph, and which loop just fired.
     ------------------------------------------------------------------------ */

  class CurrencyGraphChart {
    constructor(canvas, tooltip) {
      this.canvas = canvas;
      this.tooltip = tooltip;
      this.nodes = [];
      this.edges = [];
      this.pulse = null;
      this.pulseT = 0;
      this._raf = null;
      canvas.addEventListener('mousemove', (e) => this._onMove(e));
      canvas.addEventListener('mouseleave', () => {
        this.tooltip.dataset.visible = 'false';
        this.hoverNode = null;
      });
    }

    setData(assets, edgeCount, cyclePath) {
      const list = (assets || []).slice(0, 14);
      this.nodes = list.map((a, i) => ({ code: a.code, weight: a.weight || 1, index: i }));
      this.edges = edgeCount || 0;
      if (cyclePath && cyclePath !== this._lastPath) {
        this._lastPath = cyclePath;
        const codes = cyclePath.split('->').map(s => s.trim());
        const indices = codes
          .map(c => this.nodes.findIndex(n => n.code === c))
          .filter(i => i >= 0);
        if (indices.length >= 2) { this.pulse = indices; this.pulseT = 0; }
      }
      this._layout();
      this._start();
    }

    _layout() {
      const rect = this.canvas.getBoundingClientRect();
      const cx = rect.width / 2, cy = rect.height / 2;
      const radius = Math.min(cx, cy) - 34;
      const n = this.nodes.length || 1;
      this.nodes.forEach((node, i) => {
        const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
        node.x = cx + Math.cos(angle) * radius;
        node.y = cy + Math.sin(angle) * radius;
      });
    }

    _start() {
      if (this._raf) return;
      const tick = () => {
        this.render();
        this._raf = requestAnimationFrame(tick);
      };
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        this.render();
        return;
      }
      this._raf = requestAnimationFrame(tick);
    }

    stop() {
      if (this._raf) { cancelAnimationFrame(this._raf); this._raf = null; }
    }

    render() {
      const { ctx, w, h } = prepare(this.canvas);
      if (!this.nodes.length) {
        ctx.fillStyle = Palette.ink3;
        ctx.font = '11px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('graph not built', w / 2, h / 2);
        return;
      }
      this._layout();

      // Chords between every pair, weighted down so they read as substrate.
      ctx.save();
      ctx.strokeStyle = Palette.hairline;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.55;
      for (let i = 0; i < this.nodes.length; i++) {
        for (let j = i + 1; j < this.nodes.length; j++) {
          ctx.beginPath();
          ctx.moveTo(this.nodes[i].x, this.nodes[i].y);
          ctx.lineTo(this.nodes[j].x, this.nodes[j].y);
          ctx.stroke();
        }
      }
      ctx.restore();

      // The active cycle.
      if (this.pulse && this.pulse.length >= 2) {
        const path = this.pulse.concat([this.pulse[0]]);
        ctx.save();
        ctx.strokeStyle = Palette.s3;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        path.forEach((idx, i) => {
          const node = this.nodes[idx];
          if (!node) return;
          i === 0 ? ctx.moveTo(node.x, node.y) : ctx.lineTo(node.x, node.y);
        });
        ctx.stroke();

        this.pulseT = (this.pulseT + 0.006) % 1;
        const segments = path.length - 1;
        const position = this.pulseT * segments;
        const seg = Math.min(segments - 1, Math.floor(position));
        const local = position - seg;
        const a = this.nodes[path[seg]], b = this.nodes[path[seg + 1]];
        if (a && b) {
          const px = a.x + (b.x - a.x) * local;
          const py = a.y + (b.y - a.y) * local;
          const glow = ctx.createRadialGradient(px, py, 0, px, py, 14);
          glow.addColorStop(0, `${Palette.s3}cc`);
          glow.addColorStop(1, `${Palette.s3}00`);
          ctx.fillStyle = glow;
          ctx.beginPath();
          ctx.arc(px, py, 14, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(px, py, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = Palette.s3;
          ctx.fill();
        }
        ctx.restore();
      }

      // Nodes, with their labels — identity is never colour alone here.
      for (const node of this.nodes) {
        const active = this.pulse && this.pulse.includes(node.index);
        const r = 5 + Math.min(5, node.weight * 0.6);
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
        ctx.fillStyle = active ? Palette.s3 : Palette.s1;
        ctx.fill();
        ctx.strokeStyle = Palette.surface;
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = active ? Palette.ink : Palette.ink2;
        ctx.font = active
          ? '600 10.5px ui-monospace, Menlo, monospace'
          : '10px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const cx = this.canvas.getBoundingClientRect().width / 2;
        const cy = this.canvas.getBoundingClientRect().height / 2;
        const dx = node.x - cx, dy = node.y - cy;
        const len = Math.hypot(dx, dy) || 1;
        ctx.fillText(node.code, node.x + (dx / len) * 15, node.y + (dy / len) * 13);
      }
    }

    _onMove(event) {
      const rect = this.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left, y = event.clientY - rect.top;
      let best = null, bestDist = 18;
      for (const node of this.nodes) {
        const d = Math.hypot(node.x - x, node.y - y);
        if (d < bestDist) { bestDist = d; best = node; }
      }
      if (!best) { this.tooltip.dataset.visible = 'false'; return; }
      this.tooltip.innerHTML = `<b>${best.code}</b><br><span>${best.weight} tradable edges</span>`;
      this.tooltip.style.left = `${best.x}px`;
      this.tooltip.style.top = `${best.y}px`;
      this.tooltip.dataset.visible = 'true';
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, w / 2, Math.abs(h));
    ctx.beginPath();
    ctx.moveTo(x, y + h);
    ctx.lineTo(x, y + radius);
    ctx.arcTo(x, y, x + radius, y, radius);
    ctx.lineTo(x + w - radius, y);
    ctx.arcTo(x + w, y, x + w, y + radius, radius);
    ctx.lineTo(x + w, y + h);
    ctx.closePath();
    ctx.fill();
  }

  window.TriCharts = {
    Palette, fmt, prepare, niceTicks,
    EquityChart, EdgeHistogram, CalibrationChart, CurrencyGraphChart,
  };

})();
