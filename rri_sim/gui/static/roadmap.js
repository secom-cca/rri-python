// Roadmap results viewer: GET /api/roadmap (runs), /api/roadmap/<name> (results.json of scripts/run_roadmap.py),
// /api/roadmap-defaults (settings for the form), POST /api/roadmap/simulate (one world with changed settings)
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const PCOL = { P0: '#64748b', P1: '#f59e0b', P2: '#0e7490', P3: '#7c3aed' };
const pcol = (p) => PCOL[p] || '#db2777';
const KCOL = { levee: '#7c4a1e', retention: '#db2777', paddy_dam: '#06b6d4', forest: '#15803d', relocate: '#8b5cf6', raise: '#f59e0b',
  pump_urban: '#0f766e', pump_agri: '#14b8a6', storage_public: '#2563eb', storage_private: '#60a5fa',
  zoning: '#a16207', induce: '#ca8a04', floodproof: '#9333ea', evacuation: '#e11d48',
  housing: '#8b5cf6', pump: '#0f766e', storage: '#2563eb', landuse: '#a16207' };
const GCOL = { climate: '#2563eb', landuse: '#a16207', effects: '#16a34a', delays: '#db2777', social: '#f59e0b', regime: '#7c3aed' };
const KLAB = { climate: '気候', effect: '施策効果', landuse: '土地利用', criteria: '評価基準', delay: '時間遅れ' };
const MODE = { early: '早期参加', staged: '段階参加', late: '事後参加' };
let D = null, SIM = null, DEF = null;
const charts = {};
function chart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); return charts[id]; }
const P = () => D.summary ? D.summary.policies : Object.keys(D.config.policies);
const plabel = (p) => (D.config.policies[p] || {}).label || p;
const world = () => ($('world').value === 'sim' ? SIM : D.worlds[+$('world').value]);
const vname = (i) => D.variables[i].name;
const lvl = (a) => { const v = D.variables[a.var]; return v.continuous ? `${Math.round(a.level * 100)}%` : v.options[Math.round(a.level)]; };

async function init() {
  const runs = await (await fetch('/api/roadmap')).json();
  DEF = await (await fetch('/api/roadmap-defaults')).json().catch(() => null);
  buildForm();
  if (!runs.length) { $('summary').innerHTML = '<p>結果がありません。<code>python scripts/run_roadmap.py --quick</code> を実行してください（docs/13_roadmap.md）。</p>'; return; }
  $('run').innerHTML = runs.map((r) => `<option>${r.name}</option>`).join('');
  $('run').onchange = () => load($('run').value);
  await load(runs[0].name);
}
async function load(name) {
  D = await (await fetch(`/api/roadmap/${encodeURIComponent(name)}`)).json();
  worldOptions();
  const pol = P();
  for (const id of ['tl-pol', 'pp-pol', 'pt-pol', 'sc-pol']) $(id).innerHTML = pol.map((p) => `<option value="${p}">${plabel(p)}</option>`).join('');
  $('tl-pol').value = pol.includes('P3') ? 'P3' : pol[0]; $('pp-pol').value = $('tl-pol').value; $('pt-pol').value = pol.includes('P2') ? 'P2' : pol[0]; $('sc-pol').value = $('pt-pol').value;
  $('ts-m').innerHTML = [['damage_ratio', '被害額 ÷ 対策なし現在気候'], ['pop_ratio', '浸水人口 ÷ 対策なし現在気候'], ['ecology', 'Ecology'], ['agriculture', '農地の維持 [%]'],
    ['livability', 'Livability'], ['rain', '降雨倍率'], ['slr', '海面上昇 [m]'], ['asset', '資産係数'], ['pop', '人口係数']].map(([k, l]) => `<option value="${k}">${l}</option>`).join('');
  if (D.summary) {
    $('sob-t').innerHTML = Object.keys(D.summary.explain.sobol).map((k) => `<option>${k}</option>`).join('');
    $('pd-pair').innerHTML = Object.keys(D.summary.explain.switch).map((k) => `<option>${k}</option>`).join('');
  }
  if (D.morris) $('mo-t').innerHTML = Object.keys(D.morris.indices).map((k) => `<option>${k}</option>`).join('');
  if (D.complementarity) $('sh-m').innerHTML = [['U', '効用'], ['dr_late', '被害比の低減（2050年以降）'], ['dr_early', '被害比の低減（2050年まで）'], ['pr_late', '浸水人口比の低減（2050年以降）'],
    ['violation_share', '許容比超過の回避']].map(([k, l]) => `<option value="${k}">${l}</option>`).join('');
  $('world').onchange = drawWorld; $('tl-pol').onchange = drawTimeline; $('pp-pol').onchange = drawEng; $('ts-m').onchange = drawSeries;
  $('sob-t').onchange = drawSobol; $('pd-pair').onchange = drawPD; $('mo-t').onchange = drawMorris; $('sh-m').onchange = drawShap; $('pt-pol').onchange = drawPart; $('sc-pol').onchange = drawScope;
  $('meta').textContent = `作成 ${D.created}・計算 ${fmt(D.elapsed_s / 60, 1)} 分・世界 ${D.summary ? D.summary.n_worlds : '—'}`;
  drawSummary(); drawKPI(); drawRegret(); drawFOC(); drawUpdate(); drawSobol(); drawSwitch(); drawPD(); drawMorris();
  drawDelay(); drawShap(); drawInt(); drawPart(); drawScope(); drawWorld();
}
function worldOptions() {
  const opts = (D.worlds || []).map((w, i) => {
    const r = w.world.rain_pts[w.world.rain_pts.length - 1][1], reg = w.regimes.length > 1 ? `→${w.regimes[1][1]}(${w.regimes[1][0]})` : '';
    return `<option value="${i}">${i === 0 ? '名目 ' : ''}#${w.i} 降雨×${fmt(r, 2)} ${reg}</option>`;
  });
  if (SIM) opts.unshift('<option value="sim">対話実行</option>');
  $('world').innerHTML = opts.join('');
}
function drawWorld() { drawMap(); drawTimeline(); drawEng(); drawSeries(); drawReasons(); }

// ------------------------------------------------------------------ summary and KPIs
function drawSummary() {
  const c = D.config, ph = D.physics || {}, v = ph.validation || {};
  const road = (D.roadmap.actions || []).map((a) => `${a.year} ${vname(a.var)} ${lvl(a)}`).join('、') || '（施策なし）';
  const S = D.summary;
  const rec = S ? `<div class="rec-box">🏅 <b>Minimax Regret: ${plabel(S.robustness.minimax)}</b>・平均 Regret 最小: ${plabel(S.robustness.best_mean)}</div>` : '';
  $('summary').innerHTML = `${rec}
    <div class="layer"><b>物理モデル（RRI）</b>: 探索（docs/12）で RRI ${''}から学習・検証した代理モデル（${ph.explore_run || '—'}; 検証 被害額 R² ${fmt(v.damage && v.damage.r2, 3)}・
      RRI 再計算 R² ${fmt(ph.verification && ph.verification.r2, 3)}）。施策の水準（実施率・成熟度）と気候から、代表洪水（${c.representative_event.annual_probability} /年, ${c.representative_event.status}）の被害額・浸水人口・Ecology・Livability。
      ${D.verification_rri ? `本計算の状態の RRI 再計算: R² ${fmt(D.verification_rri.r2, 3)}・相対 RMSE ${fmt(100 * D.verification_rri.rel_rmse, 1)}%` : ''}</div>
    <div class="layer soc"><b>社会・制度モデル</b>: 合意形成 → 施工 → 普及 → 成熟の時間遅れ、ステークホルダーの関与（参加形態: ${Object.entries(c.participation.mode).map(([s, m]) => `${s} ${MODE[m]}`).join('・')}）による
      合意期間・失敗確率・実施率の上限、調整費用、予算（総額 ${fmt(c.budget.total_oku)} 億円・10年 ${fmt(c.budget.per_decade_oku)} 億円）、評価基準のレジーム（${Object.keys(c.regimes).join(' / ')}）。</div>
    <p class="hint" style="margin:6px 0 0">初期ロードマップ（${c.years.start}年に計画者の想定で最適化、P0〜P3 共通）: ${road}<br>
      期間 ${c.years.start}–${c.years.end}・意思決定 ${c.years.decision_step} 年ごと・評価 ${c.years.eval_step} 年ごと・割引率 ${c.discount_rate}・世界の因子 ${Object.keys(D.factors).length} 個（LHS）。</p>`;
}
function drawKPI() {
  const S = D.summary; if (!S) { $('kpi').innerHTML = ''; return; }
  const pol = S.policies, R = S.robustness, U = S.updateability, X = S.explain.indices, T = Object.keys(S.coverage[pol[0]]);
  const det = (p, k) => { const b = U[p].by_kind[k]; return b && b.detected > 0 ? `${fmt(b.delay, 0)} 年（${fmt(100 * b.detected)}%）` : '—'; };
  const rows = [
    ['平均 Regret', (p, j) => fmt(R.mean_regret[j], 3), 'min'], ['最大 Regret', (p, j) => fmt(R.max_regret[j], 3), 'min'], ['90% Regret', (p, j) => fmt(R.p90_regret[j], 3), 'min'],
    ['達成率（Regret ≤ 閾値・超過なし）', (p, j) => `${fmt(100 * R.satisficing[j])}%`, 'max'], ['平均効用（実際の基準）', (p, j) => fmt(R.mean_U[j], 3), 'max'],
    ['検知遅れ: 気候', (p) => det(p, 'climate')], ['検知遅れ: 施策効果', (p) => det(p, 'effect')], ['検知遅れ: 土地利用', (p) => det(p, 'landuse')], ['検知遅れ: 評価基準', (p) => det(p, 'criteria')],
    ['再計画の回数', (p) => fmt(U[p].n_replans, 1)], ['誤検知（回/世界）', (p) => fmt(U[p].false_alarms, 2)], ['追加費用（vs P1, 億円 PV）', (p) => fmt(U[p].extra_cost_vs_P1, 1)],
    ['変更費用（億円 PV）', (p) => fmt(U[p].change_cost_pv, 2)], ['更新後 Regret', (p) => fmt(U[p].regret_post, 3)],
    ...T.map((t) => [`FOC ${t}`, (p) => `${fmt(100 * S.coverage[p][t])}%`]), ...T.map((t) => [`Lock-in ${t}`, (p) => `${fmt(100 * S.lockin[p][t], 1)}%`]),
    ['重要な仮定の数', (p) => X[p].n_key_assumptions], ['更新理由の正しさ', (p) => (X[p].attribution == null ? '—' : `${fmt(100 * X[p].attribution)}%`)],
    ['1回の更新での変更数', (p) => fmt(X[p].churn, 1)], ['累積期待被害（億円 PV）', (p) => fmt(S.table[p].ead_pv)], ['事業費＋維持＋調整（億円 PV）', (p) => fmt(S.table[p].cost_pv)],
    ['調整費用（億円 PV）', (p) => fmt(S.table[p].coord_pv, 1)], ['合意の失敗（回）', (p) => fmt(S.table[p].failures, 2)], ['許容比を超えた期間の割合', (p) => `${fmt(100 * S.table[p].violation_share)}%`],
  ];
  $('kpi').innerHTML = `<tr><th>指標</th>${pol.map((p) => `<th><span class="pol" style="background:${pcol(p)}"></span>${plabel(p)}</th>`).join('')}</tr>`
    + rows.map(([l, f]) => `<tr><td>${l}</td>${pol.map((p, j) => `<td>${f(p, j)}</td>`).join('')}</tr>`).join('');
}
function drawRegret() {
  const S = D.summary; if (!S) return;
  const R = S.robustness;
  chart('c-regret', { type: 'bar', data: { labels: S.policies.map(plabel), datasets: [
    { label: '平均', data: R.mean_regret, backgroundColor: '#93c5fd' }, { label: '90%', data: R.p90_regret, backgroundColor: '#fbbf24' }, { label: '最大', data: R.max_regret, backgroundColor: '#f87171' },
    { label: '達成率', data: R.satisficing, type: 'line', yAxisID: 'y2', borderColor: '#15803d', backgroundColor: '#15803d' }] },
    options: { animation: false, scales: { y: { title: { display: true, text: 'Regret（効用）' } }, y2: { position: 'right', min: 0, max: 1, grid: { display: false }, title: { display: true, text: '達成率' } } } } });
}
function drawFOC() {
  const S = D.summary; if (!S) return;
  const T = Object.keys(S.coverage[S.policies[0]]);
  const ds = [];
  T.forEach((t, k) => ds.push({ label: `FOC ${t}`, data: S.policies.map((p) => S.coverage[p][t]), backgroundColor: ['#0e7490', '#67e8f9'][k % 2] }));
  T.forEach((t, k) => ds.push({ label: `Lock-in ${t}`, data: S.policies.map((p) => S.lockin[p][t]), backgroundColor: ['#b91c1c', '#fca5a5'][k % 2] }));
  T.forEach((t) => ds.push({ label: `白紙の状態の FOC ${t}`, data: S.policies.map(() => S.blank_coverage[t]), type: 'line', borderDash: [5, 4], borderColor: '#475569', pointRadius: 0 }));
  chart('c-foc', { type: 'bar', data: { labels: S.policies.map(plabel), datasets: ds }, options: { animation: false, scales: { y: { min: 0, max: 1 } }, plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
}
function drawUpdate() {
  const S = D.summary; if (!S) return;
  const kinds = [...new Set(S.policies.flatMap((p) => Object.keys(S.updateability[p].by_kind)))];
  const pol = S.policies.filter((p) => (D.config.policies[p] || {}).update_model || (D.config.policies[p] || {}).update_criteria);
  chart('c-update', { type: 'bar', data: { labels: kinds.map((k) => KLAB[k] || k), datasets: pol.map((p) => ({ label: plabel(p), backgroundColor: pcol(p),
    data: kinds.map((k) => { const b = S.updateability[p].by_kind[k]; return b && b.detected > 0 ? b.delay : null; }),
    detected: kinds.map((k) => { const b = S.updateability[p].by_kind[k]; return b ? b.detected : 0; }) })) },
    options: { animation: false, scales: { y: { title: { display: true, text: '年' } } },
      plugins: { tooltip: { callbacks: { afterLabel: (c) => `検知率 ${fmt(100 * c.dataset.detected[c.dataIndex])}%` } } } } });
}
function drawSobol() {
  const S = D.summary; if (!S) return;
  const s = S.explain.sobol[$('sob-t').value]; if (!s) return;
  const names = Object.keys(D.factors), order = names.map((n, i) => [n, s.ST[i]]).sort((a, b) => b[1] - a[1]).slice(0, 12);
  chart('c-sobol', { type: 'bar', data: { labels: order.map(([n]) => D.factors[n].label.slice(0, 18)), datasets: [{ label: 'S_T', data: order.map(([, v]) => v),
    backgroundColor: order.map(([n]) => GCOL[D.factors[n].group]) }] },
    options: { indexAxis: 'y', animation: false, plugins: { legend: { display: false }, title: { display: true, text: `重要な仮定の数 ${s.n_key}・群: ${Object.entries(s.groups).sort((a, b) => b[1] - a[1]).map(([g, v]) => `${({ climate: '気候', landuse: '土地利用', effects: '効果', delays: '遅れ', social: '社会', regime: '基準' })[g]} ${fmt(v, 2)}`).join(' ')}`, font: { size: 11 } } } } });
}

// ------------------------------------------------------------------ one world
function regimeAt(w, y) { let n = w.regimes[0][1]; for (const [yy, r] of w.regimes) if (y >= yy) n = r; return D.config.regimes[n]; }
function drawMap() {
  const w = world(); if (!w) return;
  const pol = Object.keys(w.records), cv = $('pmap'), ctx = cv.getContext('2d'), dpr = devicePixelRatio || 1;
  const rows = ['（現況・施策なし）'];
  pol.forEach((p) => w.records[p].actions.forEach((a) => { const k = `${vname(a.var)}: ${lvl(a)}`; if (!rows.includes(k)) rows.push(k); }));
  const W = cv.clientWidth, H = Math.max(240, 30 * rows.length + 60);
  cv.style.height = `${H}px`; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const L = 230, R = 24, T = 26, y0 = D.config.years.start, y1 = D.config.years.end;
  const X = (y) => L + (y - y0) / (y1 - y0) * (W - L - R), Y = (i) => T + i * 30 + 10;
  ctx.font = '12px "Hiragino Sans", sans-serif'; ctx.textBaseline = 'middle';
  for (let yr = y0; yr <= y1; yr += 10) { ctx.strokeStyle = '#e2e8f0'; ctx.beginPath(); ctx.moveTo(X(yr), T - 12); ctx.lineTo(X(yr), H - 20); ctx.stroke(); ctx.fillStyle = '#64748b'; ctx.fillText(yr, X(yr) - 14, H - 8); }
  rows.forEach((r, i) => { ctx.fillStyle = '#334155'; ctx.fillText(r.length > 28 ? r.slice(0, 27) + '…' : r, 6, Y(i)); ctx.strokeStyle = '#cbd5e1'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(L, Y(i)); ctx.lineTo(W - R, Y(i)); ctx.stroke(); ctx.setLineDash([]); });
  pol.forEach((p, si) => {
    const r = w.records[p], col = pcol(p), off = (si - (pol.length - 1) / 2) * 3;
    const ev = r.actions.map((a) => ({ ...a, row: rows.indexOf(`${vname(a.var)}: ${lvl(a)}`) })).sort((a, b) => a.decide - b.decide);
    let row = 0;
    const path = [[X(y0), Y(0) + off]];
    for (const e of ev) { if (e.failed) continue; path.push([X(e.decide), Y(row) + off]); path.push([X(e.decide), Y(e.row) + off]); row = e.row; }
    path.push([X(y1), Y(row) + off]);
    ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.beginPath(); path.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    ctx.lineWidth = 2;
    for (const e of ev) {
      const y = Y(e.row) + off;
      if (e.failed) { ctx.strokeStyle = '#dc2626'; const x = X(e.cons_end); ctx.beginPath(); ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5); ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5); ctx.stroke(); ctx.strokeStyle = col; continue; }
      ctx.strokeStyle = col; ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X(e.decide), y, 4.5, 0, 7); ctx.fill(); ctx.stroke();
      if (e.complete <= y1) { ctx.beginPath(); ctx.moveTo(X(e.complete), y - 6); ctx.lineTo(X(e.complete), y + 6); ctx.stroke(); }
    }
    ctx.fillStyle = col; ctx.font = '13px sans-serif';
    (r.replans || []).filter((q) => !q.skipped).forEach((q) => ctx.fillText('⟳', X(q.year) - 5 + off, T - 16 + si * 0));
    (r.triggers || []).forEach((q) => { const x = X(q.year) + off; ctx.beginPath(); ctx.moveTo(x - 5, T - 8); ctx.lineTo(x + 5, T - 8); ctx.lineTo(x, T); ctx.fill(); });
    if (r.atp != null) { ctx.fillStyle = col; const x = X(r.atp + 2.5), y = H - 30 - si * 7; ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x + 5, y); ctx.lineTo(x, y + 5); ctx.lineTo(x - 5, y); ctx.fill(); }
    ctx.font = '12px "Hiragino Sans", sans-serif'; ctx.lineWidth = 1;
  });
  $('pmap-legend').innerHTML = pol.map((p) => { const r = w.records[p]; return `<span><i style="background:${pcol(p)}"></i>${plabel(p)}（U ${fmt(r.U, 3)}・Regret ${fmt(r.regret, 3)}・ATP ${r.atp ?? 'なし'}・再計画 ${(r.replans || []).filter((q) => !q.skipped).length}）</span>`; }).join('')
    + (w.regimes && w.regimes.length > 1 ? `<span>評価基準: ${w.regimes.map(([y, n]) => `${y} ${n}`).join(' → ')}</span>` : '');
}
function drawTimeline() {
  const w = world(); if (!w) return;
  const r = w.records[$('tl-pol').value]; if (!r) return;
  const acts = [...r.actions].sort((a, b) => a.decide - b.decide || a.var - b.var);
  const cv = $('timeline'), ctx = cv.getContext('2d'), dpr = devicePixelRatio || 1, W = cv.clientWidth, H = Math.max(140, 26 * acts.length + 50);
  cv.style.height = `${H}px`; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const L = 250, R = 20, y0 = D.config.years.start, y1 = D.config.years.end, X = (y) => L + (Math.min(Math.max(y, y0), y1) - y0) / (y1 - y0) * (W - L - R);
  ctx.font = '12px "Hiragino Sans", sans-serif'; ctx.textBaseline = 'middle';
  for (let yr = y0; yr <= y1; yr += 10) { ctx.strokeStyle = '#e2e8f0'; ctx.beginPath(); ctx.moveTo(X(yr), 14); ctx.lineTo(X(yr), H - 18); ctx.stroke(); ctx.fillStyle = '#64748b'; ctx.fillText(yr, X(yr) - 14, H - 7); }
  if (!acts.length) { ctx.fillStyle = '#64748b'; ctx.fillText('施策なし', L, 30); }
  acts.forEach((a, i) => {
    const y = 26 + i * 26, col = KCOL[a.kind] || '#475569';
    const next = acts.find((b) => b.var === a.var && b.decide > a.decide && !b.failed);
    const end = next ? Math.max(next.complete, a.complete) : y1;
    ctx.fillStyle = '#334155';
    const ad = ['paddy_dam', 'forest', 'raise', 'relocate'].includes(a.kind) ? `・上限 ${fmt(100 * a.adopt)}%` : '';
    ctx.fillText(`${vname(a.var)} ${lvl(a)}${ad}${a.source !== 'plan' ? `（${({ trigger: 'トリガー', replan: '再計画', retry: '再挑戦' })[a.source] || a.source}）` : ''}`.slice(0, 34), 6, y);
    ctx.fillStyle = '#cbd5e1'; ctx.fillRect(X(a.decide), y - 8, Math.max(X(a.cons_end) - X(a.decide), 2), 16);
    if (a.failed) { ctx.strokeStyle = '#dc2626'; ctx.lineWidth = 2; const x = X(a.cons_end); ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); ctx.stroke(); ctx.lineWidth = 1; return; }
    const c0 = a.cons_end;
    if (a.kind === 'forest') {
      ctx.fillStyle = col + '44'; ctx.fillRect(X(c0), y - 8, X(c0 + a.build) - X(c0), 16);
      const m0 = c0 + a.build / 2, m1 = m0 + a.mature, gr = ctx.createLinearGradient(X(m0), 0, X(m1), 0); gr.addColorStop(0, col + '33'); gr.addColorStop(1, col);
      ctx.fillStyle = gr; ctx.fillRect(X(m0), y - 8, Math.max(X(Math.min(m1, end)) - X(m0), 1), 16);
      if (m1 < end) { ctx.fillStyle = col; ctx.fillRect(X(m1), y - 8, X(end) - X(m1), 16); }
    } else if (a.diffuse > 0) {
      const d1 = c0 + a.diffuse;
      ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(X(c0), y + 8); ctx.lineTo(X(Math.min(d1, y1)), y + 8 - 16 * Math.min(1, (Math.min(d1, y1) - c0) / a.diffuse)); ctx.lineTo(X(Math.min(d1, y1)), y + 8); ctx.closePath(); ctx.fill();
      if (d1 < end) ctx.fillRect(X(d1), y - 8, X(end) - X(d1), 16);
    } else {
      const b1 = c0 + a.build;
      ctx.fillStyle = col + '55'; ctx.fillRect(X(c0), y - 8, Math.max(X(b1) - X(c0), 1), 16);
      if (b1 < end) { ctx.fillStyle = col; ctx.fillRect(X(b1), y - 8, X(end) - X(b1), 16); }
    }
  });
  ctx.fillStyle = pcol($('tl-pol').value); ctx.font = '13px sans-serif';
  (r.replans || []).filter((q) => !q.skipped).forEach((q) => ctx.fillText('⟳', X(q.year) - 5, 8));
  (r.triggers || []).forEach((q) => { const x = X(q.year); ctx.beginPath(); ctx.moveTo(x - 5, 2); ctx.lineTo(x + 5, 2); ctx.lineTo(x, 11); ctx.fill(); });
}
function drawEng() {
  const w = world(); if (!w) return;
  const r = w.records[$('pp-pol').value]; if (!r) return;
  const years = [...D.years, D.config.years.end], S = Object.keys(r.engagement), cols = ['#2563eb', '#0e7490', '#16a34a', '#a16207', '#db2777'];
  chart('c-eng', { type: 'line', data: { labels: years, datasets: S.map((s, k) => ({ label: `${s}（${MODE[r.modes[s]] || r.modes[s]}）`, data: r.engagement[s], borderColor: cols[k % 5], backgroundColor: cols[k % 5], stepped: false, pointRadius: 2 })) },
    options: { animation: false, scales: { y: { min: 0, max: 1, title: { display: true, text: '関与の度合い e（信頼・合意の蓄積）' } } }, plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
  const fails = r.actions.filter((a) => a.failed).map((a) => `${Math.round(a.cons_end)} ${vname(a.var)}`);
  const cons = r.actions.length ? r.actions.reduce((s, a) => s + a.cons_end - a.decide, 0) / r.actions.length : null;
  $('eng-note').innerHTML = `関与の開始: ${S.map((s) => `${s} ${r.since[s] != null ? Math.round(r.since[s]) : '—'}`).join('・')}。平均の合意期間 ${fmt(cons, 1)} 年・合意の失敗 ${fails.length ? fails.join('、') : 'なし'}・調整費用 ${fmt(r.coord_pv, 1)} 億円（PV）。`;
}
function drawSeries() {
  const w = world(); if (!w) return;
  const m = $('ts-m').value, pol = Object.keys(w.records), ts = w.records[pol[0]].periods.map((p) => p.t);
  const ds = pol.map((p) => ({ label: plabel(p), data: w.records[p].periods.map((q) => q[m]), borderColor: pcol(p), backgroundColor: pcol(p), pointRadius: 2 }));
  if (m === 'damage_ratio' || m === 'pop_ratio') ds.push({ label: '許容比（その時点の評価基準）', data: ts.map((t) => regimeAt(w, t).thresholds[m]), borderColor: '#dc2626', borderDash: [5, 4], pointRadius: 0, stepped: true });
  chart('c-series', { type: 'line', data: { labels: ts.map((t) => `${t}–`), datasets: ds }, options: { animation: false, plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
}
function drawReasons() {
  const w = world(); if (!w) return;
  const out = [];
  if (w.events && w.events.length) out.push(`<p class="hint">この世界の真の変化（計画時の想定との違い）: ${w.events.map((e) => `${KLAB[e.kind] || e.kind}（${Math.round(e.year)}年〜, ${e.what ?? ''} ${typeof e.magnitude === 'number' ? (e.magnitude > 0 ? '+' : '') + fmt(e.magnitude, 2) : e.magnitude}）`).join('、')}</p>`);
  for (const p of Object.keys(w.records)) {
    const r = w.records[p], items = [];
    (r.replans || []).forEach((q) => items.push(`<li>⟳ ${q.text}</li>`));
    (r.triggers || []).forEach((q) => items.push(`<li>▼ ${q.year}年: 観測した危険度（許容比に対する比）${fmt(q.ratio, 2)}${q.reason === 'anticipated' ? `・予測 ${fmt(q.projected, 2)}（リードタイム ${fmt(q.lead, 0)} 年）` : ''} → ${q.actions.map((a) => `${vname(a.var)} ${lvl(a)}`).join('、')} に着手</li>`));
    r.actions.filter((a) => a.failed).forEach((a) => items.push(`<li>✕ ${fmt(a.cons_end, 0)}年: ${vname(a.var)} ${lvl(a)} の合意形成が失敗${p === 'P0' ? '（P0 は再挑戦しない）' : '（次の意思決定で再挑戦）'}</li>`));
    out.push(`<h3 style="font-size:13px;margin:8px 0 2px"><span class="pol" style="background:${pcol(p)}"></span>${plabel(p)}</h3><ul class="reasons">${items.length ? items.sort().join('') : '<li class="hint">更新・トリガーなし（計画どおり）</li>'}</ul>`);
  }
  $('reasons').innerHTML = out.join('');
}

// ------------------------------------------------------------------ explainability
function boxText(b) {
  if (!b || !b.box.length) return b && b.n_cases === b.n ? '常に成り立つ' : b && b.n_cases === 0 ? '該当なし' : '特定の条件なし（全域）';
  return b.box.map((d) => `${d.label} ${fmt(d.low, 2)}〜${fmt(d.high, 2)}`).join(' かつ ') + `（密度 ${fmt(100 * b.density)}%・被覆率 ${fmt(100 * b.coverage)}%）`;
}
function drawSwitch() {
  const S = D.summary; if (!S) return;
  const E = S.explain;
  const rows = Object.entries(E.switch).map(([k, s]) => { const [b, a] = k.split('>'); return `<tr><td>${plabel(b)} が ${plabel(a)} より良い</td><td>${fmt(100 * s.share)}%</td><td>${fmt(s.mean_dU, 4)}</td><td style="text-align:left">${boxText(s.box)}</td>
    <td style="text-align:left">${s.partial_dependence.map((d) => `${d.label}${d.crossing.length ? ` = ${d.crossing.map((x) => fmt(x, 2)).join(', ')}` : '（切替なし）'}`).join('; ')}</td></tr>`; });
  const vul = S.policies.map((p) => `<tr><td>${plabel(p)} の Regret &gt; ${D.config.experiment.regret_threshold}</td><td>${fmt(100 * E.vulnerability[p].n_cases / E.vulnerability[p].n)}%</td><td></td><td style="text-align:left">${boxText(E.vulnerability[p])}</td><td></td></tr>`);
  const best = S.policies.map((p) => `<tr><td>${plabel(p)} が最良</td><td>${fmt(100 * E.best_box[p].n_cases / E.best_box[p].n)}%</td><td></td><td style="text-align:left">${boxText(E.best_box[p])}</td><td></td></tr>`);
  $('switch').innerHTML = `<table class="tbl"><tr><th>事象</th><th>世界の割合</th><th>平均の効用差</th><th style="text-align:left">PRIM の箱（条件）</th><th style="text-align:left">切替点（部分依存が 0 を横切る値）</th></tr>${rows.join('')}${best.join('')}${vul.join('')}</table>`;
}
function drawPD() {
  const S = D.summary; if (!S) return;
  const s = S.explain.switch[$('pd-pair').value]; if (!s) return;
  const cols = ['#2563eb', '#db2777', '#16a34a'];
  chart('c-pd', { type: 'line', data: { datasets: s.partial_dependence.map((d, k) => ({ label: d.label, data: d.unit.map((u, i) => ({ x: u, y: d.pd[i] })), borderColor: cols[k], backgroundColor: cols[k], pointRadius: 0 })) },
    options: { animation: false, parsing: false, scales: { x: { type: 'linear', min: 0, max: 1, title: { display: true, text: '因子の値（範囲の下限 0 → 上限 1）' } }, y: { title: { display: true, text: '効用差（＞0 で後の方策が良い）' } } },
      plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
}
function drawMorris() {
  if (!D.morris) return;
  const m = D.morris.indices[$('mo-t').value]; if (!m) return;
  const names = D.morris.factors, order = names.map((n, i) => [n, m.mu_star[i], m.sigma[i]]).sort((a, b) => b[1] - a[1]).slice(0, 12);
  chart('c-morris', { type: 'bar', data: { labels: order.map(([n]) => D.factors[n].label.slice(0, 18)), datasets: [
    { label: 'μ*', data: order.map(([, v]) => v), backgroundColor: order.map(([n]) => GCOL[D.factors[n].group]) }, { label: 'σ', data: order.map(([, , s]) => s), backgroundColor: '#cbd5e1' }] },
    options: { indexAxis: 'y', animation: false, plugins: { title: { display: true, text: `${D.morris.n_trajectories} 軌道`, font: { size: 11 } } } } });
}

// ------------------------------------------------------------------ experiments
function drawDelay() {
  if (!D.delay) return;
  const sc = Object.keys(D.delay.scales), keys = Object.keys(D.delay.scales[sc[0]].postpone).filter((k) => !k.endsWith('@' + D.config.years.start));
  const cols = ['#0e7490', '#64748b', '#7c3aed', '#f59e0b'];
  chart('c-delay', { type: 'bar', data: { labels: sc.map((s) => `遅れ ×${s}`), datasets: [
    ...keys.map((k, i) => ({ label: `${k.replace('@', ' を ')} 年まで延期: 価値`, data: sc.map((s) => D.delay.scales[s].postpone[k].value_of_early), backgroundColor: cols[i % 4] })),
    ...keys.map((k, i) => ({ label: `${k.replace('@', '@')} 計画の価値に占める割合`, data: sc.map((s) => D.delay.scales[s].postpone[k].share_of_plan_value), type: 'line', yAxisID: 'y2', borderColor: cols[i % 4], borderDash: [4, 3] }))] },
    options: { animation: false, scales: { y: { title: { display: true, text: '早期意思決定の価値（効用）' } }, y2: { position: 'right', min: 0, grid: { display: false }, title: { display: true, text: '割合' } } },
      plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
  chart('c-delay2', { type: 'bar', data: { labels: sc.map((s) => `遅れ ×${s}`), datasets: keys.map((k, i) => ({ label: k, data: sc.map((s) => D.delay.scales[s].postpone[k].extra_violation_years), backgroundColor: cols[i % 4] })) },
    options: { animation: false, scales: { y: { title: { display: true, text: '許容比を超える年数の増加' } } } } });
}
const TLAB = { paddy_dam: '田んぼダム', forest: '森林化', relocate: '住宅対策', housing: '住宅対策', retention: '遊水地', levee: '堤防',
  pump: '排水ポンプ', storage: '雨水貯留浸透', landuse: '土地利用規制・誘導', floodproof: '耐水化', evacuation: '避難' };
function drawShap() {
  const C = D.complementarity; if (!C) return;
  const k = $('sh-m').value, s = C.shapley[k];
  chart('c-shap', { type: 'bar', data: { labels: C.types.map((t) => TLAB[t] || t), datasets: [{ label: 'Shapley 値', data: C.types.map((t) => s[t]), backgroundColor: C.types.map((t) => KCOL[t]) }] },
    options: { animation: false, plugins: { legend: { display: false }, title: { display: true, text: `${C.n_worlds} 世界の平均`, font: { size: 11 } } } } });
}
function drawInt() {
  const C = D.complementarity; if (!C) return;
  const ms = [['U', '効用'], ['dr_early', '被害比〜2050'], ['dr_late', '被害比2050〜'], ['pr_late', '人口比2050〜'], ['violation_share', '超過回避']];
  const pairs = Object.keys(C.shapley_interaction.U);
  const cell = (v, k) => { const mx = Math.max(...pairs.map((p) => Math.abs(C.shapley_interaction[k][p]))) || 1, t = v / mx; return `background:${t > 0 ? `rgba(22,163,74,${0.6 * t})` : `rgba(220,38,38,${-0.6 * t})`}`; };
  $('t-int').innerHTML = `<tr><th>組合せ</th>${ms.map(([, l]) => `<th>${l}</th>`).join('')}</tr>`
    + pairs.map((p) => `<tr><td>${p.split('+').map((t) => TLAB[t] || t).join('＋')}</td>${ms.map(([k]) => `<td style="${cell(C.shapley_interaction[k][p], k)}">${fmt(C.shapley_interaction[k][p], 4)}</td>`).join('')}</tr>`).join('');
}
function drawPart() {
  const Pp = D.participation; if (!Pp) return;
  const v = Object.keys(Pp.results), p = $('pt-pol').value;
  const cols = [['U', '効用', 3], ['regret', 'Regret', 3], ['coverage', 'FOC', 2], ['failures', '合意失敗', 2], ['consensus_years', '合意期間 [年]', 1], ['adoption', '実施率の上限', 2],
    ['coord_pv', '調整費用', 1], ['cost_pv', '費用計', 0], ['response', '変化→対応完了 [年]', 0]];
  $('t-part').innerHTML = `<tr><th>参加形態（${plabel(p)}）</th>${cols.map(([, l]) => `<th>${l}</th>`).join('')}</tr>`
    + v.map((x) => `<tr><td>${x}</td>${cols.map(([k, , d]) => `<td>${fmt(Pp.results[x][p][k], d)}</td>`).join('')}</tr>`).join('');
  chart('c-part', { type: 'bar', data: { labels: v, datasets: Pp.policies.map((q) => ({ label: plabel(q), data: v.map((x) => Pp.results[x][q].regret), backgroundColor: pcol(q) })) },
    options: { animation: false, scales: { x: { ticks: { font: { size: 10 } } } }, plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
  chart('c-part2', { type: 'bar', data: { labels: v, datasets: [{ label: `FOC（${plabel(p)}）`, data: v.map((x) => Pp.results[x][p].coverage), backgroundColor: '#0e7490' },
    { label: '合意失敗（回）', data: v.map((x) => Pp.results[x][p].failures), backgroundColor: '#f87171', yAxisID: 'y2' }] },
    options: { animation: false, scales: { y: { min: 0, max: 1 }, y2: { position: 'right', min: 0, grid: { display: false } }, x: { ticks: { font: { size: 10 } } } } } });
}

const TIERCOL = { 国: '#1d4ed8', 自治体: '#0e7490', 地域: '#16a34a' };
function drawScope() {
  const Q = D.scope; if (!Q) { $('t-scope').innerHTML = '<tr><td class="hint">この結果には実施主体の範囲の比較がありません（scripts/run_roadmap.py で再計算）。</td></tr>'; return; }
  const p = $('sc-pol').value, sc = Q.scopes;
  const cols = [['U', '効用', 3], ['regret', 'Regret', 3], ['ead_pv', '累積期待被害 PV', 0], ['cost_pv', '費用 PV', 0], ['violation_share', '許容比超過の割合', 2],
    ['damage_2100', '被害比 2100', 2], ['pop_2100', '人的リスク比 2100', 2], ['failures', '合意失敗', 2]];
  $('t-scope').innerHTML = `<tr><th>範囲（${plabel(p)}）</th><th>使える変数</th>${cols.map(([, l]) => `<th>${l}</th>`).join('')}</tr>`
    + sc.map((x) => `<tr><td>${x}</td><td>${(Q.variables || {})[x] ? Q.variables[x].length : '—'}</td>${cols.map(([k, , d]) => `<td>${fmt(Q.results[x][p][k], d)}</td>`).join('')}</tr>`).join('');
  $('scope-plans').innerHTML = sc.map((x) => `<b>${x}</b> の初期ロードマップ: ${Q.roadmaps[x].map((a) => `<span style="color:${TIERCOL[a.tier] || '#333'}">${a.year} ${a.name}</span>`).join('、') || 'なし'}`).join('<br>');
  chart('c-scope', { type: 'bar', data: { labels: sc, datasets: Q.policies.map((q) => ({ label: plabel(q), data: sc.map((x) => Q.results[x][q].regret), backgroundColor: pcol(q) })) },
    options: { animation: false, scales: { y: { title: { display: true, text: 'Regret' } }, x: { ticks: { font: { size: 10 } } } }, plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
  const tiers = ['国', '自治体', '地域'];
  chart('c-scope2', { type: 'bar', data: { labels: sc, datasets: tiers.map((t) => ({ label: t, data: sc.map((x) => Q.results[x][p].cost_by_tier[t] || 0), backgroundColor: TIERCOL[t] })) },
    options: { animation: false, scales: { x: { stacked: true, ticks: { font: { size: 10 } } }, y: { stacked: true } } } });
}

// ------------------------------------------------------------------ form: one world with changed settings
function buildForm() {
  if (!DEF) { $('form').innerHTML = '<p class="hint">設定を取得できません。</p>'; $('go').disabled = true; return; }
  const c = DEF.config, F = DEF.factors, M = c.measures, Pt = c.participation;
  const num = (id, v, step = 0.1) => `<input id="${id}" type="number" step="${step}" value="${v}">`;
  const meas = Object.entries(M).map(([k, m]) => {
    const ph = ['build', 'diffuse', 'mature'].find((q) => typeof m[q] === 'object');
    return `<div class="frow"><span>${m.label}: 合意 ${num(`m-${k}-consensus`, m.consensus, 0.5).replace('type="number"', 'type="number" style="width:46px"')}・${({ build: '施工', diffuse: '普及', mature: '成熟' })[ph]}</span>
      ${num(`m-${k}-${ph}-nominal`, m[ph].nominal, 1)}${num(`m-${k}-${ph}-low`, m[ph].low, 1)}${num(`m-${k}-${ph}-high`, m[ph].high, 1)}</div>`;
  }).join('');
  const modes = Object.keys(Pt.stakeholders).map((s) => `<div class="frow"><span>${s}（${Pt.stakeholders[s].role}）</span><select id="pm-${s}" style="grid-column: span 3">
    ${Object.entries(MODE).map(([k, l]) => `<option value="${k}" ${Pt.mode[s] === k ? 'selected' : ''}>${l}</option>`).join('')}</select></div>`).join('');
  const sp = [['engagement_tau_years', '関係構築の年数 τ'], ['consensus_reduction', '合意期間の短縮率 a'], ['fail_scale', '合意失敗確率の倍率'], ['adoption_shift', '参加率の増減'], ['coord_cost_scale', '調整費用の倍率']]
    .map(([k, l]) => `<div class="frow"><span>${l}</span>${num(`ps-${k}-nominal`, Pt[k].nominal)}${num(`ps-${k}-low`, Pt[k].low)}${num(`ps-${k}-high`, Pt[k].high)}</div>`).join('');
  const late = [['consensus_factor', '事後参加: 合意期間 ×'], ['trust', '事後参加: 信頼 ×'], ['fail_factor', '事後参加: 失敗確率 ×'], ['conflict_cost_factor', '事後参加: 調整費用 ×']]
    .map(([k, l]) => `<div class="frow"><span>${l}</span>${num(`pl-${k}`, Pt.late[k])}</div>`).join('');
  const groups = {};
  Object.entries(F).forEach(([n, f]) => { (groups[f.group] = groups[f.group] || []).push([n, f]); });
  const GL = { climate: '気候', landuse: '土地利用', effects: '施策効果（真値）', delays: '時間遅れ（真値, 範囲内の位置）', social: '社会・参加（真値）', regime: '評価基準の変化' };
  const sliders = Object.entries(groups).map(([g, fs]) => `<fieldset><legend>世界の因子: ${GL[g] || g}</legend>${fs.map(([n, f]) => {
    const u = (f.nominal - f.low) / ((f.high - f.low) || 1);
    return `<div class="slider"><span title="${f.label}">${f.label.slice(0, 22)}</span><input id="wf-${n}" type="range" min="0" max="1" step="0.01" value="${u}"><span id="wv-${n}"></span></div>`;
  }).join('')}</fieldset>`).join('');
  $('form').innerHTML = `<fieldset><legend>施策の時間遅れ [年]（名目 / 下限 / 上限）</legend><div class="frow h"><span></span><span>名目</span><span>下限</span><span>上限</span></div>${meas}</fieldset>
    <fieldset><legend>ステークホルダーの参加形態</legend>${modes}</fieldset>
    <fieldset><legend>社会・参加の不確実性（名目 / 下限 / 上限）</legend><div class="frow h"><span></span><span>名目</span><span>下限</span><span>上限</span></div>${sp}${late}</fieldset>${sliders}`;
  const showV = () => Object.entries(F).forEach(([n, f]) => {
    const u = +$(`wf-${n}`).value;
    $(`wv-${n}`).textContent = f.categories ? (u < c.regime_change.p_none ? 'なし' : f.categories[1 + Math.min(Math.floor((u - c.regime_change.p_none) / (1 - c.regime_change.p_none) * (f.categories.length - 1)), f.categories.length - 2)])
      : fmt(f.low + u * (f.high - f.low), 2);
  });
  Object.keys(F).forEach((n) => { $(`wf-${n}`).oninput = showV; });
  showV();
  $('go').onclick = simulate;
}
async function simulate() {
  const c = DEF.config, F = DEF.factors, M = c.measures, Pt = c.participation;
  const over = { measures: {}, participation: { late: {} } };
  for (const [k, m] of Object.entries(M)) {
    const ph = ['build', 'diffuse', 'mature'].find((q) => typeof m[q] === 'object');
    over.measures[k] = { consensus: +$(`m-${k}-consensus`).value, [ph]: { nominal: +$(`m-${k}-${ph}-nominal`).value, low: +$(`m-${k}-${ph}-low`).value, high: +$(`m-${k}-${ph}-high`).value } };
  }
  for (const k of ['engagement_tau_years', 'consensus_reduction', 'fail_scale', 'adoption_shift', 'coord_cost_scale'])
    over.participation[k] = { nominal: +$(`ps-${k}-nominal`).value, low: +$(`ps-${k}-low`).value, high: +$(`ps-${k}-high`).value };
  for (const k of Object.keys(Pt.late)) over.participation.late[k] = +$(`pl-${k}`).value;
  const modes = Object.fromEntries(Object.keys(Pt.stakeholders).map((s) => [s, $(`pm-${s}`).value]));
  over.participation.mode = modes;
  const unit = Object.fromEntries(Object.keys(F).map((n) => [n, +$(`wf-${n}`).value]));
  $('go').disabled = true; $('go-msg').textContent = '計算中…（初期ロードマップの最適化と P0〜P3）';
  const t0 = performance.now();
  try {
    const r = await fetch('/api/roadmap/simulate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ config: over, unit, modes }) });
    if (!r.ok) throw new Error(await r.text());
    SIM = await r.json();
    if (!D) { D = { config: SIM.config, variables: SIM.variables, years: SIM.years, factors: F, worlds: [], roadmap: SIM.roadmap }; }
    worldOptions(); $('world').value = 'sim'; drawWorld();
    $('go-msg').textContent = `完了（${fmt((performance.now() - t0) / 1000, 1)} s）。初期ロードマップ: ${SIM.roadmap.actions.map((a) => `${a.year} ${SIM.variables[a.var].name} ${lvl(a)}`).join('、') || 'なし'}`;
  } catch (e) { $('go-msg').textContent = `エラー: ${e.message}`; }
  $('go').disabled = false;
}
init();
