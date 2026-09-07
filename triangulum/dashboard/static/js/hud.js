/* ==========================================================================
   Triangulum HUD — application shell
   --------------------------------------------------------------------------
   Connects to the engine's WebSocket, renders state, and sends the two
   commands the operator actually needs at 3am: halt, and release.

   Reconnection is aggressive and visible. A dashboard that silently stops
   updating is worse than one that says "disconnected", because a frozen
   equity curve looks exactly like a flat one.
   ========================================================================== */
'use strict';

const { Palette, fmt, EquityChart, EdgeHistogram, CalibrationChart, CurrencyGraphChart } =
  window.TriCharts;

const $ = (id) => document.getElementById(id);

/* ── Connection ─────────────────────────────────────────────────────────── */

class Feed {
  constructor(onSnapshot, onLog, onStatus) {
    this.onSnapshot = onSnapshot;
    this.onLog = onLog;
    this.onStatus = onStatus;
    this.ws = null;
    this.attempt = 0;
    this.token = new URLSearchParams(location.search).get('token') || '';
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const query = this.token ? `?token=${encodeURIComponent(this.token)}` : '';
    this.onStatus('connecting');
    try {
      this.ws = new WebSocket(`${proto}://${location.host}/ws${query}`);
    } catch (err) {
      return this.scheduleReconnect();
    }

    this.ws.onopen = () => {
      this.attempt = 0;
      this.onStatus('ok');
    };
    this.ws.onmessage = (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.type === 'snapshot') this.onSnapshot(message.data);
      else if (message.type === 'log') this.onLog(message.data);
      else if (message.type === 'error') this.onLog({
        level: 'ERROR', message: String(message.data), ts: Date.now() * 1e6,
      });
    };
    this.ws.onclose = () => { this.onStatus('down'); this.scheduleReconnect(); };
    this.ws.onerror = () => { this.onStatus('down'); };
  }

  scheduleReconnect() {
    this.attempt += 1;
    // Full jitter, capped at 10s. Same reasoning as the venue reconnect: when
    // the engine restarts, every open dashboard tab retries at once.
    const ceiling = Math.min(10000, 500 * Math.pow(2, this.attempt));
    setTimeout(() => this.connect(), Math.random() * ceiling);
  }

  send(payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
      return true;
    }
    return false;
  }
}

/* ── Rendering ──────────────────────────────────────────────────────────── */

const charts = {};
let lastCyclePath = null;

function renderLegend(container, items) {
  container.innerHTML = items.map(item => `
    <span class="legend-item">
      <span class="legend-swatch" style="background:${item.color};color:${item.color}"
            ${item.dashed ? 'data-dash="true"' : ''}></span>${item.label}
    </span>`).join('');
}

function setPill(id, state, text) {
  const pill = $(id);
  pill.dataset.state = state;
  pill.querySelector('span').textContent = text;
}

function renderTiles(s) {
  const equity = Number(s.equity || 0);
  const start = Number(s.starting_equity || 0);
  const pnlPct = start > 0 ? (equity / start - 1) * 100 : 0;

  $('stat-equity').textContent = `${fmt.num(equity, 4)}`;
  const delta = $('stat-equity-delta');
  delta.textContent = `${fmt.pct(pnlPct, 3)} · ${s.base_currency || ''}`;
  delta.dataset.dir = pnlPct > 0 ? 'up' : pnlPct < 0 ? 'down' : '';

  // Target attainment. The engine never chases this; it only reports it.
  const risk = s.risk || {};
  const daily = Number(risk.daily_pnl_pct || 0) * 100;   // -> bps
  const targetBps = Number(s.daily_target_bps || 100);
  const attainment = targetBps ? daily / targetBps : 0;
  $('stat-attainment').textContent = `${fmt.num(attainment * 100, 0)}%`;
  $('stat-attainment-sub').textContent =
    `${fmt.bps(daily)} today vs ${fmt.num(targetBps, 0)} bps target`;
  const meter = $('meter-target');
  meter.style.width = `${Math.max(0, Math.min(100, attainment * 100))}%`;
  meter.style.background = attainment >= 1 ? Palette.good
    : attainment >= 0 ? Palette.s1 : Palette.critical;

  const returns = s.cycle_returns_bps || [];
  const expectancy = returns.length
    ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
  const exp = $('stat-expectancy');
  exp.textContent = returns.length ? fmt.bps(expectancy) : '—';
  exp.style.color = expectancy > 0 ? Palette.good
    : expectancy < 0 ? Palette.critical : Palette.ink;

  const ex = s.executor || {};
  $('stat-cycles').textContent = fmt.compact(ex.attempted || 0);
  $('stat-cycles-sub').textContent =
    `${ex.completed || 0} done · ${ex.aborted_pre_trade || 0} aborted · ${ex.stuck || 0} stuck`;

  const gate = s.gate || {};
  const model = gate.fill_model || {};
  const skill = Number(model.skill || 0);
  $('stat-skill').textContent = model.samples ? fmt.num(skill, 3) : '—';
  $('stat-skill-sub').textContent = model.samples
    ? `${fmt.compact(model.samples)} samples · ${gate.trusts_model ? 'trusted' : 'warming up'}`
    : 'no samples yet';

  // Quantization drag as a share of the gross edge — the small-account tax.
  const planned = s.cycles_planned || 0;
  const decisions = s.recent_decisions || [];
  const withEdge = decisions.filter(d => d.net_edge_bps);
  const dragShare = withEdge.length
    ? withEdge.reduce((acc, d) => {
        const gross = Math.abs(d.net_edge_bps) + Math.abs(d.predicted_slippage_bps || 0);
        return acc + (gross > 0 ? Math.abs(d.predicted_slippage_bps || 0) / gross : 0);
      }, 0) / withEdge.length
    : 0;
  $('stat-drag').textContent = withEdge.length ? `${fmt.num(dragShare * 100, 0)}%` : '—';
  $('tile-drag').style.display = withEdge.length ? '' : 'none';
}

function renderFunnel(s) {
  const gate = s.gate || {};
  const rejects = gate.rejects_by_reason || {};
  const stages = [
    { label: 'detected', value: s.opportunities_seen || 0, color: Palette.s1 },
    { label: 'sized', value: s.cycles_planned || 0, color: Palette.s4 },
    { label: 'risk ok', value: gate.evaluations || 0, color: Palette.s2 },
    { label: 'ev accept', value: gate.accepts || 0, color: Palette.s3 },
    { label: 'completed', value: (s.executor || {}).completed || 0, color: Palette.good },
  ];
  const max = Math.max(1, ...stages.map(x => x.value));
  $('funnel').innerHTML = stages.map((stage, i) => {
    const previous = i > 0 ? stages[i - 1].value : stage.value;
    const dropped = previous - stage.value;
    const pct = (stage.value / max) * 100;
    return `
      <div class="funnel-row">
        <span class="funnel-label">${stage.label}</span>
        <div class="funnel-track">
          <div class="funnel-bar" style="width:${pct}%;background:${stage.color}"></div>
        </div>
        <span class="funnel-value">${fmt.compact(stage.value)}${
          i > 0 && dropped > 0
            ? `<br><span class="funnel-drop">−${fmt.compact(dropped)}</span>`
            : ''
        }</span>
      </div>`;
  }).join('');
}

function renderBandit(s) {
  const rows = ((s.bandit || {}).leaderboard) || [];
  const body = $('table-bandit').querySelector('tbody');
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="5" style="color:var(--ink-3)">bandit disabled or no pulls yet</td></tr>`;
    return;
  }
  const best = rows[0] && rows[0].arm;
  body.innerHTML = rows.map(r => `
    <tr>
      <td>${r.arm === best ? '▸ ' : '&nbsp;&nbsp;'}${r.arm}</td>
      <td class="num">${fmt.compact(r.pulls)}</td>
      <td class="num">${fmt.num(r.fill_rate * 100, 0)}%</td>
      <td class="num">${fmt.num(r.mean_bps, 2)}</td>
      <td class="num" style="color:${r.expected_value_bps > 0 ? Palette.good : Palette.ink3}">
        ${fmt.num(r.expected_value_bps, 2)}
      </td>
    </tr>`).join('');
}

const VERDICT_KIND = {
  accept: 'accept',
  accept_exploration: 'explore',
  reject_ev: 'reject',
  reject_uncertainty: 'reject',
  reject_edge: 'reject',
  reject_fill_probability: 'reject',
  reject_stale: 'reject',
};

function renderDecisions(s) {
  const rows = (s.recent_decisions || []).slice(-40).reverse();
  const body = $('table-opps').querySelector('tbody');
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="6" style="color:var(--ink-3)">no decisions yet</td></tr>`;
    return;
  }
  const opps = new Map((s.recent_opportunities || []).map(o => [o.path, o]));
  body.innerHTML = rows.map(d => {
    const opp = opps.get(d.path) || {};
    const kind = VERDICT_KIND[d.verdict] || 'reject';
    return `
      <tr>
        <td class="path" title="${d.path}">${d.path}</td>
        <td class="num">${fmt.num(d.net_edge_bps, 2)}</td>
        <td class="num">${opp.age_ms != null ? `${fmt.num(opp.age_ms, 0)}ms` : '—'}</td>
        <td class="num">${fmt.num(d.p_fill, 3)}</td>
        <td class="num" style="color:${d.ev_bps > 0 ? Palette.good : Palette.ink3}">
          ${fmt.num(d.ev_bps, 2)}
        </td>
        <td><span class="tag" data-kind="${kind}">${d.verdict.replace(/_/g, ' ')}</span></td>
      </tr>`;
  }).join('');
}

function renderVenues(s) {
  const venues = s.venues || {};
  const books = s.books || {};
  const connected = new Set(books.connected_venues || []);
  const entries = Object.entries(venues);
  if (!entries.length) { $('venues').innerHTML = '<p class="caption">no venues</p>'; return; }

  $('venues').innerHTML = entries.map(([name, v]) => {
    const stats = v.market_data || v;
    const sim = v.simulator || {};
    const isUp = connected.has(name) || stats.connected;
    const fillRate = Number(sim.fill_rate ?? stats.fill_rate ?? 0);
    const latency = Number(sim.submit_latency_ms ?? stats.submit_latency_ms ?? 0);
    return `
      <div class="bar-row">
        <div class="bar-head">
          <span class="name">${isUp ? '●' : '○'} ${name}
            <span style="color:var(--ink-3)">${isUp ? 'up' : 'down'}</span></span>
          <span class="val">${fmt.num(latency, 0)}ms · ${fmt.num(fillRate * 100, 0)}% fill</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${Math.min(100, fillRate * 100)}%;
               background:${isUp ? Palette.s3 : Palette.critical}"></div>
        </div>
      </div>`;
  }).join('');
}

function renderRisk(s) {
  const risk = s.risk || {};
  const limits = risk.limits || {};
  const bars = [
    {
      name: 'daily loss', value: Math.max(0, -Number(risk.daily_pnl_pct || 0)),
      limit: Number(limits.max_daily_loss_pct || 3), unit: '%',
    },
    {
      name: 'drawdown', value: Number(risk.drawdown_pct || 0),
      limit: Number(limits.max_drawdown_pct || 10), unit: '%',
    },
    {
      name: 'consecutive losses', value: Number(risk.consecutive_losses || 0),
      limit: Number(limits.max_consecutive_losses || 8), unit: '',
    },
    {
      name: 'orders / min', value: Number(risk.orders_last_minute || 0),
      limit: 120, unit: '',
    },
  ];
  $('risk-bars').innerHTML = bars.map(b => {
    const ratio = b.limit > 0 ? Math.min(1, b.value / b.limit) : 0;
    const color = ratio > 0.8 ? Palette.critical : ratio > 0.5 ? Palette.warning : Palette.s3;
    return `
      <div class="bar-row">
        <div class="bar-head">
          <span class="name">${b.name}</span>
          <span class="val">${fmt.num(b.value, b.unit ? 2 : 0)}${b.unit} / ${b.limit}${b.unit}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${ratio * 100}%;background:${color}"></div>
        </div>
      </div>`;
  }).join('');
}

function renderCharts(s) {
  charts.equity.setData(s.equity_curve || [], {
    startEquity: Number(s.starting_equity || 0),
    dailyTargetBps: Number(s.daily_target_bps || 100),
  });

  const days = (s.equity_curve || []).length > 1
    ? ((s.equity_curve[s.equity_curve.length - 1][0] - s.equity_curve[0][0]) / 864e11)
    : 0;
  $('caption-equity').textContent = days > 0
    ? `${fmt.num(days, 2)} days observed. The dashed line is the configured target trajectory — tracked, never chased: relaxing the gate to hit a number is how a positive-expectancy system becomes a negative one.`
    : 'Target trajectory appears once two equity samples exist.';

  charts.edges.setData(s.recent_opportunities || [], s.recent_decisions || []);
  const opps = s.recent_opportunities || [];
  $('caption-edges').textContent = opps.length
    ? `${opps.length} recent detections. Most cluster near zero — that is the market working correctly, not the detector failing.`
    : 'Detections appear here once the graph finds cycles above the screening threshold.';

  const calibration = ((s.gate || {}).calibration || {}).reliability || [];
  charts.calibration.setData(calibration);
  const ece = ((s.gate || {}).calibration || {}).expected_calibration_error;
  $('calib-hint').textContent = ece != null ? `ECE ${fmt.num(ece, 4)}` : 'no samples';

  // Currency graph: node weight is the asset's tradable degree.
  const graph = s.graph || {};
  const byVenue = graph.edges_by_venue || {};
  const totalEdges = graph.edges_usable || 0;
  const assets = [];
  for (const opp of (s.recent_opportunities || []).slice(-40)) {
    for (const code of opp.path.split('->').map(x => x.trim())) {
      const existing = assets.find(a => a.code === code);
      if (existing) existing.weight += 1;
      else assets.push({ code, weight: 1 });
    }
  }
  assets.sort((a, b) => b.weight - a.weight);
  const newest = (s.recent_opportunities || []).slice(-1)[0];
  charts.graph.setData(assets, totalEdges, newest ? newest.path : null);
  $('graph-hint').textContent =
    `${graph.nodes || 0} assets · ${totalEdges} edges · ${Object.keys(byVenue).length} venues`;
}

function renderStatus(s) {
  const mode = String(s.mode || 'paper');
  setPill('pill-mode', mode === 'live' ? 'live' : 'ok', mode);
  setPill('pill-regime', 'ok', s.regime || 'unknown');

  const risk = s.risk || {};
  const button = $('killswitch');
  button.dataset.engaged = String(Boolean(risk.kill_switch));
  button.querySelector('.ks-label').textContent = risk.kill_switch ? 'HALTED' : 'HALT';
  button.title = risk.kill_switch_reason || 'Stop opening new cycles';

  $('brand-sub').textContent = `${mode} · ${(s.venues && Object.keys(s.venues).join(', ')) || 'no venues'}`;
  $('footer-note').textContent = mode === 'live'
    ? '⚠ LIVE MODE — real capital at risk'
    : 'Paper mode. No capital at risk.';
  $('footer-note').style.color = mode === 'live' ? Palette.critical : '';
}

/* ── Log ────────────────────────────────────────────────────────────────── */

const logLines = [];

function appendLog(entry) {
  logLines.push(entry);
  if (logLines.length > 400) logLines.shift();
  const box = $('log');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  const div = document.createElement('div');
  div.className = 'log-line';
  div.dataset.level = entry.level || 'INFO';
  div.innerHTML =
    `<span class="log-time">${fmt.time(entry.ts || Date.now() * 1e6)}</span>` +
    `<span class="log-level">${(entry.level || 'INFO').slice(0, 8)}</span>` +
    `<span class="log-msg"></span>`;
  div.querySelector('.log-msg').textContent = entry.message || '';
  box.appendChild(div);
  while (box.children.length > 400) box.removeChild(box.firstChild);
  if (atBottom) box.scrollTop = box.scrollHeight;
}

/* ── Boot ───────────────────────────────────────────────────────────────── */

function boot() {
  charts.equity = new EquityChart($('chart-equity'), $('tip-equity'));
  charts.edges = new EdgeHistogram($('chart-edges'), $('tip-edges'));
  charts.calibration = new CalibrationChart($('chart-calibration'), $('tip-calibration'));
  charts.graph = new CurrencyGraphChart($('chart-graph'), $('tip-graph'));

  renderLegend($('legend-equity'), EquityChart.legend);
  renderLegend($('legend-edges'), EdgeHistogram.legend);

  const feed = new Feed(
    (snapshot) => {
      try {
        renderStatus(snapshot);
        renderTiles(snapshot);
        renderFunnel(snapshot);
        renderBandit(snapshot);
        renderDecisions(snapshot);
        renderVenues(snapshot);
        renderRisk(snapshot);
        renderCharts(snapshot);
      } catch (err) {
        console.error('render failed', err);
      }
    },
    appendLog,
    (state) => {
      const text = { ok: 'live', connecting: 'connecting', down: 'disconnected' }[state] || state;
      setPill('pill-link', state === 'ok' ? 'ok' : state === 'down' ? 'down' : 'warn', text);
    },
  );
  feed.connect();

  $('killswitch').addEventListener('click', () => {
    const engaged = $('killswitch').dataset.engaged === 'true';
    const command = engaged ? 'release_kill_switch' : 'engage_kill_switch';
    if (!engaged && !confirm('Halt the engine? No new cycles will be opened.')) return;
    if (engaged && !confirm('Release the halt and resume trading?')) return;
    if (!feed.send({ command, reason: 'dashboard' })) {
      appendLog({ level: 'ERROR', message: 'not connected — command not sent' });
    }
  });

  $('log-clear').addEventListener('click', () => {
    logLines.length = 0;
    $('log').innerHTML = '';
  });

  // Charts are canvas-based, so they must be told when their box changes.
  const redraw = () => Object.values(charts).forEach(c => c.render && c.render());
  window.addEventListener('resize', redraw);
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(redraw);
    document.querySelectorAll('.canvas-wrap').forEach(el => observer.observe(el));
  }

  appendLog({ level: 'INFO', message: 'HUD initialised — awaiting engine state' });
}

document.readyState === 'loading'
  ? document.addEventListener('DOMContentLoaded', boot)
  : boot();
