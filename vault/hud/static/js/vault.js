/* ==========================================================================
   Vault Engine HUD
   --------------------------------------------------------------------------
   Everything is scoped to one IIFE. Nothing leaks to the global scope.
   Canvas is hand-rolled for the same reason the server is stdlib-only: the
   page that tells you whether your forecaster works has to render when the
   rest of the stack does not.
   ========================================================================== */
'use strict';

(function () {

  const $ = (id) => document.getElementById(id);
  const cssVar = (name, fallback) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

  const C = {
    get s1()   { return cssVar('--series-1', '#3987e5'); },
    get s2()   { return cssVar('--series-2', '#c98500'); },
    get s3()   { return cssVar('--series-3', '#199e70'); },
    get s4()   { return cssVar('--series-4', '#9085e9'); },
    get good() { return cssVar('--good', '#0ca30c'); },
    get warn() { return cssVar('--warning', '#fab219'); },
    get crit() { return cssVar('--critical', '#d03b3b'); },
    get ink()  { return cssVar('--ink', '#fff'); },
    get ink2() { return cssVar('--ink-2', '#b3a9d4'); },
    get ink3() { return cssVar('--ink-3', '#7a6da3'); },
    get line() { return cssVar('--hairline', '#2e2352'); },
    get surf() { return cssVar('--surface-1', '#16102a'); },
  };

  const fmt = {
    n(v, d = 2) {
      if (v === null || v === undefined || !isFinite(v)) return '—';
      return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
    },
    pct(v, d = 1) { return v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`; },
    money(v) {
      if (!v) return '$0';
      return '$' + Math.round(v).toLocaleString();
    },
    date(iso) {
      if (!iso) return '—';
      return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    },
  };

  /* ── Reliability chart ──────────────────────────────────────────────────
     Predicted probability against realised frequency, with the y=x reference.
     The single most important model diagnostic: the capital gate multiplies by
     these numbers, so a point far off the diagonal is a sizing error waiting
     to happen.
     -------------------------------------------------------------------- */

  class ReliabilityChart {
    constructor(canvas, tooltip) {
      this.canvas = canvas;
      this.tooltip = tooltip;
      this.rows = [];
      this.points = [];
      canvas.addEventListener('mousemove', (e) => this.hover(e));
      canvas.addEventListener('mouseleave', () => { this.tooltip.dataset.visible = 'false'; });
    }

    static get legend() {
      return [
        { label: 'observed', color: C.s4 },
        { label: 'perfect calibration', color: C.ink3, dashed: true },
      ];
    }

    setData(rows) { this.rows = rows || []; this.render(); }

    render() {
      const dpr = window.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      if (this.canvas.width !== w * dpr) { this.canvas.width = w * dpr; this.canvas.height = h * dpr; }
      const ctx = this.canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const size = Math.min(w - 52, h - 30);
      const box = { l: 40, t: 8, r: 40 + size, b: 8 + size };
      this.box = box;

      ctx.strokeStyle = C.line;
      ctx.lineWidth = 1;
      ctx.strokeRect(box.l + 0.5, box.t + 0.5, size, size);

      // Perfect-calibration diagonal.
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = C.ink3;
      ctx.beginPath();
      ctx.moveTo(box.l, box.b);
      ctx.lineTo(box.r, box.t);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = C.ink3;
      ctx.font = '9.5px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('1.0', box.l - 6, box.t);
      ctx.fillText('0.5', box.l - 6, (box.t + box.b) / 2);
      ctx.fillText('0.0', box.l - 6, box.b);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('stated probability', (box.l + box.r) / 2, box.b + 9);

      if (!this.rows.length) {
        ctx.fillStyle = C.ink3;
        ctx.textBaseline = 'middle';
        ctx.fillText('no resolved calls yet', (box.l + box.r) / 2, (box.t + box.b) / 2);
        this.points = [];
        return;
      }

      const maxN = Math.max(1, ...this.rows.map(r => r.count));
      this.points = this.rows.map(r => ({
        ...r,
        x: box.l + r.predicted * size,
        y: box.b - r.observed * size,
        radius: 4.5 + 6 * Math.sqrt(r.count / maxN),
      }));

      ctx.beginPath();
      this.points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.strokeStyle = C.s4 + 'aa';
      ctx.lineWidth = 2;
      ctx.stroke();

      for (const p of this.points) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = Math.abs(p.error) > 0.15 ? C.warn : C.s4;
        ctx.fill();
        // 2px surface ring keeps overlapping markers separable.
        ctx.strokeStyle = C.surf;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    hover(event) {
      if (!this.points.length) return;
      const rect = this.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left, y = event.clientY - rect.top;
      let best = null, dist = 24;
      for (const p of this.points) {
        const d = Math.hypot(p.x - x, p.y - y);
        if (d < dist) { dist = d; best = p; }
      }
      if (!best) { this.tooltip.dataset.visible = 'false'; return; }
      this.tooltip.innerHTML =
        `<b>stated ${fmt.n(best.predicted, 2)}</b><br>` +
        `<span>realised</span> ${fmt.n(best.observed, 3)}<br>` +
        `<span>n</span> ${best.count}`;
      this.tooltip.style.left = `${best.x}px`;
      this.tooltip.style.top = `${best.y}px`;
      this.tooltip.dataset.visible = 'true';
    }
  }

  /* ── Rendering ───────────────────────────────────────────────────────── */

  const charts = {};
  const STAGES = ['scan', 'macro', 'flow', 'resolve', 'bias', 'gate'];

  function renderLegend(el, items) {
    el.innerHTML = items.map(i => `
      <span class="legend-item">
        <span class="legend-swatch" style="background:${i.color};color:${i.color}"
              ${i.dashed ? 'data-dash="true"' : ''}></span>${i.label}
      </span>`).join('');
  }

  function setChip(id, state, text) {
    const el = $(id);
    el.dataset.state = state;
    el.querySelector('span').textContent = text;
  }

  function renderScoreboard(s) {
    const score = s.score || {};
    const n = score.n || 0;

    $('stat-n').textContent = n;
    $('stat-n-sub').textContent = n
      ? `${score.correct} correct · ${score.wrong} wrong · ${score.invalidated} stopped`
      : 'nothing scored yet';

    $('stat-hit').textContent = n ? fmt.pct(score.hit_rate) : '—';
    const ci = score.hit_rate_ci || [0, 0];
    $('stat-hit-sub').textContent = n
      ? `95% CI [${fmt.pct(ci[0], 0)}, ${fmt.pct(ci[1], 0)}]`
      : '95% CI';

    const brier = $('stat-brier');
    brier.textContent = n ? fmt.n(score.brier, 4) : '—';
    brier.style.color = !n ? '' : (score.beats_baseline ? C.good : C.crit);
    $('stat-brier-sub').textContent = n
      ? (score.beats_baseline ? 'beats the 0.25 baseline' : 'WORSE than always saying 0.50')
      : 'lower is better';

    // Three states, not two. "$0" beside "gate open" is a contradiction, and
    // it is the exact species of decorative panel this dashboard exists to be
    // the opposite of: the gate passing all four checks while the headline
    // number reads zero, because there simply is no call in front of it to
    // size. Say which of the three it is.
    const sizing = (s.last_run && s.last_run.sizing) || null;
    const unlocked = (s.gate_progress || {}).unlocked;
    const ceiling = (s.gate || {}).max_position_fraction;
    const capital = s.capital || 0;

    if (sizing && sizing.approved) {
      $('stat-capital').textContent = fmt.money(sizing.notional);
      $('stat-capital-sub').textContent = 'released for this call';
    } else if (unlocked) {
      $('stat-capital').textContent = ceiling
        ? `\u2264 ${fmt.money(capital * ceiling)}` : '—';
      $('stat-capital-sub').textContent = 'gate open · no call to size';
    } else {
      $('stat-capital').textContent = '$0';
      $('stat-capital-sub').textContent = 'gated on calibration';
    }

    $('score-adequacy').textContent = n ? `sample: ${score.adequacy}` : 'no sample';

    $('score-caption').textContent = n === 0
      ? 'A forecasting system that does not resolve its own calls is not a forecasting system. This panel is the difference, and right now it is honestly empty.'
      : `${n} resolved calls. Brier ${fmt.n(score.brier, 4)} against the 0.2500 baseline of always saying 0.50. Sample adequacy: ${score.adequacy} — read the confidence interval before believing the hit rate.`;
  }

  function renderPipeline(s) {
    const stage = s.stage || 'idle';
    const index = STAGES.indexOf(stage);
    document.querySelectorAll('.pipeline li').forEach((li, i) => {
      li.dataset.active = String(i === index);
      li.dataset.done = String(index >= 0 && i < index);
    });
    $('pipe-hint').textContent = stage === 'idle'
      ? `${s.runs || 0} run${s.runs === 1 ? '' : 's'}` : stage;
  }

  function renderGate(s) {
    const progress = s.gate_progress || { gates: [] };
    $('gate-hint').textContent =
      `${progress.passed || 0}/${progress.total || 4} gates`;
    $('gate-gates').innerHTML = (progress.gates || []).map(g => {
      const colour = g.passed ? C.good : (g.progress > 0.6 ? C.warn : C.s4);
      return `
        <div class="gate-row">
          <div class="gate-head">
            <span class="gate-mark" data-passed="${g.passed}">${g.passed ? '✓' : '○'}</span>
            <span class="gate-name">${g.gate}</span>
            <span class="gate-current">${g.current}</span>
          </div>
          <div class="gate-req">${g.requirement}</div>
          <div class="gate-track">
            <div class="gate-fill" style="width:${Math.min(100, (g.progress || 0) * 100)}%;background:${colour}"></div>
          </div>
        </div>`;
    }).join('');
  }

  function renderRegime(s) {
    const regime = s.regime;
    if (!regime) {
      $('regime-conf').textContent = 'not classified';
      return;
    }
    const probabilities = regime.probabilities || {};
    document.querySelectorAll('.q').forEach(el => {
      const key = el.dataset.q;
      const p = probabilities[key] || 0;
      el.querySelector('.q-p').textContent = fmt.pct(p, 0);
      el.dataset.active = String(key === regime.quadrant);
    });
    $('regime-conf').textContent = `confidence ${fmt.pct(regime.confidence, 0)}` +
      (regime.transitioning ? ' · transitioning' : '');
    $('regime-growth').textContent = (regime.growth_score >= 0 ? '+' : '') + fmt.n(regime.growth_score, 2);
    $('regime-infl').textContent = (regime.inflation_score >= 0 ? '+' : '') + fmt.n(regime.inflation_score, 2);
    $('regime-tilt').textContent = regime.historical_tilt
      ? `Historically in this regime: ${regime.historical_tilt}.`
      : '';
  }

  function renderThesis(s) {
    const run = s.last_run;
    const card = $('thesis-card');
    const thesis = run && run.thesis;

    if (!thesis) {
      card.className = 'thesis-empty';
      card.textContent = 'No thesis yet. Run a scan.';
      $('thesis-model').textContent = '—';
      return;
    }
    $('thesis-model').textContent = thesis.model || '—';

    if (thesis.abstained) {
      card.className = 'thesis-empty';
      card.innerHTML =
        `<strong style="color:var(--ink)">ABSTAINED.</strong> ${thesis.reason || ''}` +
        `<br><br>Declining to forecast is a first-class outcome here. It is never scored, ` +
        `and manufacturing a 0.55 call on noise is the fastest way to make a track record worthless.`;
      return;
    }
    if (thesis.error) {
      card.className = 'thesis-empty';
      card.innerHTML = `<strong style="color:var(--critical)">ERROR.</strong> ${thesis.error}`;
      return;
    }

    const t = thesis.thesis;
    if (!t) { card.className = 'thesis-empty'; card.textContent = 'No thesis.'; return; }

    const sizing = run.sizing || {};
    card.className = 'thesis-card';
    card.innerHTML = `
      <div class="thesis-headline">
        <span class="thesis-dir" data-dir="${t.direction}">${t.asset} ${t.direction.toUpperCase()}</span>
        <span class="thesis-p">p=${fmt.n(t.probability, 2)}</span>
        <span class="thesis-meta">over ${t.horizon} · expect ${fmt.n(t.magnitude_pct, 1)}% · invalidate at ${fmt.n(t.invalidation_pct, 1)}%</span>
      </div>
      <div class="thesis-fields">
        <div class="tf"><div class="tf-label">Capital authorised</div>
          <div class="tf-value" style="color:${sizing.approved ? C.good : C.ink3}">
            ${sizing.approved ? fmt.money(sizing.notional) : '$0'}</div></div>
        <div class="tf"><div class="tf-label">Gate verdict</div>
          <div class="tf-value" style="font-size:12px">${(sizing.verdict || 'n/a').replace(/_/g, ' ')}</div></div>
        <div class="tf"><div class="tf-label">Evidence</div>
          <div class="tf-value" style="font-size:12px">${(t.primary_evidence || []).join(', ')}</div></div>
      </div>
      <div class="thesis-prose"><b>Reasoning.</b> ${escapeHtml(t.reasoning || '')}</div>
      <div class="thesis-risk"><b>Key risk:</b> ${escapeHtml(t.key_risk || '')}</div>
      ${sizing.approved ? '' :
        `<div class="thesis-risk" style="border-color:var(--ink-3)"><b>Why no capital:</b> ${escapeHtml(sizing.reason || '')}</div>`}
    `;
  }

  function renderOpen(s) {
    const rows = s.open_positions || [];
    $('open-count').textContent = `${rows.length} open`;
    const body = $('table-open').querySelector('tbody');
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="6" class="muted">no open calls</td></tr>';
      return;
    }
    body.innerHTML = rows.map(r => `
      <tr>
        <td>${r.asset}</td>
        <td style="color:${r.direction === 'up' ? C.s3 : C.warn}">${r.direction}</td>
        <td class="num">${fmt.n(r.probability, 2)}</td>
        <td class="num">${fmt.n(r.entry, 2)}</td>
        <td class="num">${fmt.n(r.invalidation, 2)}</td>
        <td class="num">${r.days_left}</td>
      </tr>`).join('');
  }

  function renderSignals(s) {
    const signals = (s.brief && s.brief.signals) || [];
    $('signals').innerHTML = signals.length
      ? signals.map(x => `<li>${escapeHtml(x)}</li>`).join('')
      : '<li class="muted">no named signals</li>';
  }

  function renderHistory(s) {
    const rows = s.recent_calls || [];
    const journal = s.journal || {};
    $('chain-status').textContent = journal.chain_valid
      ? `chain intact · ${journal.total || 0} records`
      : 'CHAIN BROKEN';
    $('chain-status').style.color = journal.chain_valid ? '' : C.crit;

    const body = $('table-history').querySelector('tbody');
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="9" class="muted">no calls yet</td></tr>';
      return;
    }
    body.innerHTML = rows.map(r => `
      <tr>
        <td>${fmt.date(r.created_at)}</td>
        <td>${r.asset}</td>
        <td style="color:${r.direction === 'up' ? C.s3 : C.warn}">${r.direction}</td>
        <td>${r.horizon}</td>
        <td class="num">${fmt.n(r.probability, 2)}</td>
        <td><span class="tag" data-k="${r.outcome}">${r.outcome}</span></td>
        <td class="num">${r.realized_pct === null || r.realized_pct === undefined ? '—' : fmt.n(r.realized_pct, 2) + '%'}</td>
        <td class="num">${r.brier === null || r.brier === undefined ? '—' : fmt.n(r.brier, 3)}</td>
        <td>${r.regime || '—'}</td>
      </tr>`).join('');
  }

  function renderStatus(s) {
    const mode = s.data_mode || 'unknown';
    setChip('chip-mode', 'ok', 'paper');
    setChip('chip-data', mode === 'live' ? 'ok' : 'warn', mode);
    $('fixture-banner').hidden = mode === 'live';
    $('brand-sub').textContent =
      `${s.series_loaded || 0} series · ${(s.journal || {}).total || 0} calls · ${mode}`;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /* ── Data ────────────────────────────────────────────────────────────── */

  async function fetchState() {
    try {
      const response = await fetch('/api/state', { cache: 'no-store' });
      if (!response.ok) throw new Error(String(response.status));
      const state = await response.json();
      setChip('chip-link', 'ok', 'live');
      render(state);
    } catch (err) {
      setChip('chip-link', 'down', 'disconnected');
    }
  }

  function render(s) {
    try {
      renderStatus(s);
      renderScoreboard(s);
      renderPipeline(s);
      renderGate(s);
      renderRegime(s);
      renderThesis(s);
      renderOpen(s);
      renderSignals(s);
      renderHistory(s);
      charts.reliability.setData((s.score || {}).reliability_buckets || []);
    } catch (err) {
      console.error('render failed', err);
    }
  }

  async function runScan() {
    const button = $('run-btn');
    button.disabled = true;
    button.dataset.busy = 'true';
    $('run-label').textContent = 'RUNNING…';
    try {
      const response = await fetch('/api/run', { method: 'POST' });
      const result = await response.json();
      if (result.error) console.error(result.error);
    } catch (err) {
      console.error('run failed', err);
    } finally {
      button.disabled = false;
      button.dataset.busy = 'false';
      $('run-label').textContent = 'RUN SCAN';
      fetchState();
    }
  }

  function boot() {
    charts.reliability = new ReliabilityChart($('chart-reliability'), $('tip-reliability'));
    renderLegend($('legend-rel'), ReliabilityChart.legend);

    $('run-btn').addEventListener('click', runScan);

    fetchState();
    setInterval(fetchState, 4000);

    const redraw = () => Object.values(charts).forEach(c => c.render && c.render());
    window.addEventListener('resize', redraw);
    if (window.ResizeObserver) {
      const observer = new ResizeObserver(redraw);
      document.querySelectorAll('.canvas-wrap').forEach(el => observer.observe(el));
    }
  }

  document.readyState === 'loading'
    ? document.addEventListener('DOMContentLoaded', boot)
    : boot();

})();
