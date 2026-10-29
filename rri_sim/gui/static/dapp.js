// DAPP results viewer: GET /api/dapp (runs), /api/dapp/<name> (results.json of scripts/run_dapp.py)
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const SCOL = { 無対策: '#64748b', 早期集中投資: '#dc2626', 段階的投資: '#0e7490', 自然基盤重視: '#16a34a', 構造物重視: '#7c4a1e', 状況適応型: '#f59e0b' };
const scol = (n, i) => SCOL[n] || ['#7c3aed', '#db2777', '#0891b2'][i % 3];
const CLIM = { current: '現在気候', '2C': '2℃上昇', '4C': '4℃上昇' };
let D = null;
const charts = {};
function chart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); return charts[id]; }
const rec = (s, c) => D.records[`${s}|${c}`];

async function init() {
  const runs = await (await fetch('/api/dapp')).json();
  if (!runs.length) { $('summary').innerHTML = '<p>DAPP の結果がありません。<code>python scripts/run_dapp.py --quick</code> を実行してください（docs/11_dapp.md）。</p>'; return; }
  $('run').innerHTML = runs.map((r) => `<option>${r.name}</option>`).join('');
  $('run').onchange = () => load($('run').value);
  await load(runs[0].name);
}
async function load(name) {
  D = await (await fetch(`/api/dapp/${encodeURIComponent(name)}`)).json();
  $('clim').innerHTML = D.climates.map((c) => `<option value="${c}">${CLIM[c] || c}</option>`).join('');
  $('clim').value = D.climates[D.climates.length - 1];
  $('tl-st').innerHTML = D.strategies.map((s) => `<option>${s}</option>`).join('');
  $('tl-st').value = D.strategies.includes('状況適応型') ? '状況適応型' : D.strategies[0];
  $('rg-st').innerHTML = Object.keys(D.stakeholders).map((s) => `<option>${s}</option>`).join('');
  $('ts-m').innerHTML = [['event_damage', '代表洪水の被害額 [億円]'], ['pop_hazard', '浸水人口（0.5 m以上）[人]'], ['damage_ratio', '被害額 ÷ 対策なし現在気候'],
    ['cum_damage', '累積期待被害（PV）[億円]'], ['cum_cost', '累積事業費（PV）[億円]'], ['livability', 'Livability'], ['ecology', 'Ecology'], ['rain_factor', '降雨倍率']]
    .map(([k, l]) => `<option value="${k}">${l}</option>`).join('');
  $('clim').onchange = drawAll; $('tl-st').onchange = drawTimeline; $('ts-m').onchange = drawSeries; $('rg-st').onchange = drawRegret;
  $('meta').textContent = `${D.config.basin} ${D.config.res} m・作成 ${D.created}・計算 ${fmt(D.elapsed_s / 60, 1)} 分・RRI ${D.n_simulations ?? '—'} 回`;
  drawAll();
}
function drawAll() { drawSummary(); drawMap(); drawTimeline(); drawSeries(); drawATP(); drawTsens(); drawScore(); drawRegret(); drawWsens(); drawStATP(); }

function drawSummary() {
  const c = D.config, ov = D.strategies[D.overall_minimax];
  const recs = Object.entries(D.stakeholders).map(([n, s]) => `<li><b>${n}</b>: ${D.strategies[s.minimax]}（最大 Regret ${fmt(s.max_regret[s.minimax], 3)}）</li>`).join('');
  const pth = Object.entries(c.climate_paths).map(([k, p]) => `${CLIM[k] || k}: ${p.points.map((q) => `${q[0]}年 ×${q[1]}・+${q[2]} m`).join(' → ')}（${p.status}）`).join('<br>');
  $('summary').innerHTML = `<div class="rec-box">🏅 <b>Minimax Regret 推奨戦略（全意思決定者・全気候経路）: ${ov}</b><ul style="margin:6px 0 0">${recs}</ul></div>
    <p class="hint" style="margin:8px 0 0">⚠️ <b>近似</b>: ${D.approximation}（代表洪水の年超過確率 ${c.representative_event.annual_probability} は${c.representative_event.status}: ${c.representative_event.basis}）</p>
    <p class="hint" style="margin:4px 0 0">期間 ${c.years.start}–${c.years.end}（${c.years.step}年ごとに意思決定）・予算 総額 ${fmt(c.budget.total_oku)} 億円／10年 ${fmt(c.budget.per_decade_oku)} 億円（初期費用）・割引率 ${c.discount_rate}・
    ATP 閾値: 被害額・浸水人口が対策なし現在気候の ${c.atp.damage_ratio}・${c.atp.pop_ratio} 倍<br>${pth}</p>`;
}

function segsOf(s, c) { return D.pathways[`${s}|${c}`]; }
function drawMap() {
  const c = $('clim').value, cv = $('pmap'), ctx = cv.getContext('2d'), dpr = devicePixelRatio || 1;
  const rows = [];
  D.strategies.forEach((s) => segsOf(s, c).segments.forEach((g) => { const k = `${g.var}: ${g.option}`; if (!rows.includes(k)) rows.push(k); }));
  rows.unshift('（現況・施策なし）');
  const W = cv.clientWidth, H = Math.max(260, 34 * rows.length + 60);
  cv.style.height = `${H}px`; cv.width = W * dpr; cv.height = H * dpr; ctx.scale(dpr, dpr);
  const L = 220, R = 30, T = 24, y0 = D.config.years.start, y1 = D.config.years.end;
  const X = (y) => L + (y - y0) / (y1 - y0) * (W - L - R), Y = (i) => T + i * 34 + 12;
  ctx.font = '12px "Hiragino Sans", sans-serif'; ctx.textBaseline = 'middle';
  for (let yr = y0; yr <= y1; yr += 10) { ctx.strokeStyle = '#e2e8f0'; ctx.beginPath(); ctx.moveTo(X(yr), T - 10); ctx.lineTo(X(yr), H - 20); ctx.stroke(); ctx.fillStyle = '#64748b'; ctx.fillText(yr, X(yr) - 14, H - 8); }
  rows.forEach((r, i) => { ctx.fillStyle = '#334155'; ctx.fillText(r.length > 26 ? r.slice(0, 25) + '…' : r, 6, Y(i)); ctx.strokeStyle = '#cbd5e1'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(L, Y(i)); ctx.lineTo(W - R, Y(i)); ctx.stroke(); ctx.setLineDash([]); });
  D.strategies.forEach((s, si) => {
    const P = segsOf(s, c), col = scol(s, si), off = (si - (D.strategies.length - 1) / 2) * 3;
    const ev = P.segments.map((g) => ({ year: g.start, done: g.complete, row: rows.indexOf(`${g.var}: ${g.option}`) })).sort((a, b) => a.year - b.year);
    let row = 0, yr = y0;
    ctx.strokeStyle = col; ctx.lineWidth = 3.5;
    const path = [[X(y0), Y(0) + off]];
    for (const e of ev) { path.push([X(e.year), Y(row) + off]); path.push([X(e.year), Y(e.row) + off]); row = e.row; yr = e.year; }
    path.push([X(y1), Y(row) + off]);
    ctx.beginPath(); path.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    for (const e of ev) {
      ctx.fillStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(e.year), Y(e.row) + off, 5, 0, 7); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(e.done), Y(e.row) + off - 7); ctx.lineTo(X(e.done), Y(e.row) + off + 7); ctx.stroke();
    }
    const atp = Math.min(...[P.atp.damage, P.atp.pop].filter((v) => v != null));
    if (Number.isFinite(atp)) {
      const at = [...ev].reverse().find((e) => e.year <= atp); const rr = at ? at.row : 0;
      ctx.strokeStyle = '#dc2626'; ctx.lineWidth = 3; const x = X(atp + 5), y = Y(rr) + off;
      ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); ctx.stroke();
    }
    ctx.lineWidth = 1;
  });
  $('pmap-legend').innerHTML = D.strategies.map((s, i) => `<span><i style="background:${scol(s, i)}"></i>${s}${Number.isFinite(Math.min(...[segsOf(s, c).atp.damage, segsOf(s, c).atp.pop].filter((v) => v != null))) ? '（ATP あり）' : ''}</span>`).join('');
}

function drawTimeline() {
  const s = $('tl-st').value, c = $('clim').value, P = segsOf(s, c), cv = $('timeline'), ctx = cv.getContext('2d'), dpr = devicePixelRatio || 1;
  const segs = P.segments, W = cv.clientWidth, H = Math.max(140, 30 * segs.length + 50);
  cv.style.height = `${H}px`; cv.width = W * dpr; cv.height = H * dpr; ctx.scale(dpr, dpr);
  const L = 200, R = 20, y0 = D.config.years.start, y1 = D.config.years.end, X = (y) => L + (y - y0) / (y1 - y0) * (W - L - R);
  const TC = { levee: '#7c4a1e', retention: '#db2777', paddy_dam: '#06b6d4', forest: '#15803d', relocate: '#8b5cf6', raise: '#f59e0b' };
  ctx.font = '12px "Hiragino Sans", sans-serif'; ctx.textBaseline = 'middle';
  for (let yr = y0; yr <= y1; yr += 10) { ctx.strokeStyle = '#e2e8f0'; ctx.beginPath(); ctx.moveTo(X(yr), 4); ctx.lineTo(X(yr), H - 18); ctx.stroke(); ctx.fillStyle = '#64748b'; ctx.fillText(yr, X(yr) - 14, H - 7); }
  if (!segs.length) { ctx.fillStyle = '#64748b'; ctx.fillText('施策なし', L, 30); }
  segs.forEach((g, i) => {
    const y = 16 + i * 30, col = TC[g.kind] || '#475569';
    ctx.fillStyle = '#334155'; ctx.fillText(`${g.var}: ${g.option}`.slice(0, 24), 6, y);
    ctx.fillStyle = col + '55'; ctx.fillRect(X(g.start), y - 9, Math.max(X(g.complete) - X(g.start), 2), 18);
    if (g.kind === 'forest') { const gr = ctx.createLinearGradient(X(g.complete), 0, X(Math.min(g.complete + D.config.measures.forest.ramp_years, g.end)), 0); gr.addColorStop(0, col + '55'); gr.addColorStop(1, col); ctx.fillStyle = gr; }
    else ctx.fillStyle = col;
    ctx.fillRect(X(g.complete), y - 9, Math.max(X(g.end) - X(g.complete), 1), 18);
  });
  ctx.fillStyle = '#dc2626';
  P.triggers.forEach((t) => { const x = X(t.year); ctx.beginPath(); ctx.moveTo(x - 6, 2); ctx.lineTo(x + 6, 2); ctx.lineTo(x, 12); ctx.fill(); });
  ctx.strokeStyle = '#475569';
  P.deferred.forEach((t) => { const x = X(t.year) + 4; ctx.beginPath(); ctx.moveTo(x, H - 30); ctx.lineTo(x + 5, H - 25); ctx.lineTo(x, H - 20); ctx.lineTo(x - 5, H - 25); ctx.closePath(); ctx.stroke(); });
}

function drawSeries() {
  const c = $('clim').value, m = $('ts-m').value;
  const val = (r) => { let cd = 0, cc = 0; return r.periods.map((p) => { cd += p.exp_damage_pv; cc += p.capital_pv + p.maintenance_pv;
    return m === 'cum_damage' ? cd : m === 'cum_cost' ? cc : p[m]; }); };
  const ds = D.strategies.map((s, i) => ({ label: s, data: val(rec(s, c)), borderColor: scol(s, i), backgroundColor: scol(s, i), pointRadius: 3, tension: 0 }));
  if (m === 'damage_ratio') ds.push({ label: `ATP 閾値 ${D.config.atp.damage_ratio}`, data: D.years.map(() => D.config.atp.damage_ratio), borderColor: '#dc2626', borderDash: [5, 4], pointRadius: 0 });
  chart('series', { type: 'line', data: { labels: D.years.map((y) => `${y}s`), datasets: ds }, options: { animation: false, plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
}

function drawATP() {
  const yrs = (a) => [a.damage != null ? `被害 ${a.damage}` : null, a.pop != null ? `人口 ${a.pop}` : null, a.budget != null ? `予算 ${a.budget}` : null].filter(Boolean).join('・') || '到達なし';
  $('atp').innerHTML = `<tr><th>戦略</th>${D.climates.map((c) => `<th>${CLIM[c] || c}</th>`).join('')}</tr>`
    + D.strategies.map((s) => `<tr><td>${s}</td>${D.climates.map((c) => { const a = rec(s, c).atp; const any = a.damage != null || a.pop != null;
      return `<td style="background:${any ? '#fee2e2' : a.budget != null ? '#fef3c7' : '#dcfce7'}">${yrs(a)}</td>`; }).join('')}</tr>`).join('');
  $('atp-note').textContent = `被害・人口＝代表洪水の被害額・浸水人口が対策なし現在気候の ${D.config.atp.damage_ratio}・${D.config.atp.pop_ratio} 倍を超える最初の10年。予算＝予算不足で施策が繰り越された最初の年。`;
}
function drawTsens() {
  const c = $('clim').value, T = D.threshold_sensitivity, end = D.config.years.end;
  chart('tsens', { type: 'line', data: { labels: T.grid, datasets: D.strategies.map((s, i) => ({ label: s, data: T.atp_year[c][s].map((v) => v ?? end), borderColor: scol(s, i), stepped: true, pointRadius: 2 })) },
    options: { animation: false, scales: { x: { title: { display: true, text: '許容閾値（対策なし現在気候の何倍まで許容するか）' } }, y: { min: D.config.years.start, max: end, title: { display: true, text: 'ATP 到達年（上端＝到達なし）' } } } } });
}
function drawScore() {
  const c = $('clim').value, st = Object.keys(D.stakeholders);
  const cols = [['cum_damage_pv', '累積被害 PV', 1], ['cum_cost_pv', '累積費用 PV', 1], ['tot', '被害＋費用', 1], ['mean_livability', 'Livability', -1], ['mean_ecology', 'Ecology', -1], ['mean_pop_hazard', '浸水人口（平均）', 1]];
  const v = (s, k) => (k === 'tot' ? rec(s, c).cum_damage_pv + rec(s, c).cum_cost_pv : rec(s, c)[k]);
  const color = (k, sgn, x) => { const xs = D.strategies.map((s) => v(s, k)), lo = Math.min(...xs), hi = Math.max(...xs); const t = hi > lo ? (x - lo) / (hi - lo) : 0.5; const g = sgn > 0 ? 1 - t : t; return `rgba(${Math.round(239 - 200 * g)},${Math.round(68 + 120 * g)},${Math.round(68 + 20 * g)},0.25)`; };
  $('score').innerHTML = `<tr><th>戦略（${CLIM[c] || c}）</th>${cols.map((k) => `<th>${k[1]}</th>`).join('')}<th>ATP</th>${st.map((n) => `<th>${n}<br>最大Regret / 達成率</th>`).join('')}</tr>`
    + D.strategies.map((s, i) => `<tr class="${i === D.overall_minimax ? 'rec' : ''}"><td>${s}${i === D.overall_minimax ? ' 🏅' : ''}</td>`
      + cols.map(([k, , sg]) => `<td style="background:${color(k, sg, v(s, k))}">${fmt(v(s, k), k.includes('liv') || k.includes('eco') ? 1 : 0)}</td>`).join('')
      + `<td>${(() => { const a = rec(s, c).atp; return a.damage ?? a.pop ?? (a.budget != null ? `予算 ${a.budget}` : '—'); })()}</td>`
      + st.map((n) => { const S = D.stakeholders[n]; return `<td style="${S.minimax === i ? 'font-weight:700;color:#b45309' : ''}">${fmt(S.max_regret[i], 3)} / ${fmt(S.satisficing[i] * 100)}%</td>`; }).join('') + '</tr>').join('');
}
function drawRegret() {
  const n = $('rg-st').value, S = D.stakeholders[n];
  chart('regret', { type: 'bar', data: { labels: D.strategies, datasets: [
    ...D.climates.map((c, j) => ({ label: `Regret（${CLIM[c] || c}）`, data: S.regret.map((r) => r[j]), backgroundColor: ['#93c5fd', '#fbbf24', '#f87171'][j % 3] })),
    { type: 'line', label: '最大 Regret', data: S.max_regret, borderColor: '#111', pointRadius: 4, showLine: false }] },
    options: { animation: false, plugins: { title: { display: true, text: `${n}（Minimax: ${D.strategies[S.minimax]}、許容閾値 ${D.config.regret_threshold}、ATP 閾値 ×${S.atp_ratio}）` } } } });
}
function drawWsens() {
  const st = Object.keys(D.weight_sensitivity);
  chart('wsens', { type: 'bar', data: { labels: st, datasets: D.strategies.map((s, i) => ({ label: s, data: st.map((n) => D.weight_sensitivity[n][i] * 100), backgroundColor: scol(s, i) })) },
    options: { animation: false, scales: { x: { stacked: true }, y: { stacked: true, max: 100, title: { display: true, text: '%' } } } } });
}
function drawStATP() {
  const st = Object.keys(D.stakeholders);
  $('statp').innerHTML = `<tr><th>戦略</th>${st.map((n) => `<th colspan="${D.climates.length}">${n}（×${D.stakeholders[n].atp_ratio}）</th>`).join('')}</tr>`
    + `<tr><th></th>${st.map(() => D.climates.map((c) => `<th>${CLIM[c] || c}</th>`).join('')).join('')}</tr>`
    + D.strategies.map((s, i) => `<tr><td>${s}</td>${st.map((n) => D.stakeholders[n].atp[i].map((a) => `<td style="background:${a == null ? '#dcfce7' : '#fee2e2'}">${a ?? '—'}</td>`).join('')).join('')}</tr>`).join('');
}
init();
