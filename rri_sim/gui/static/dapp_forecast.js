// Adaptive DAPP with forecasts and learning (results.json of `scripts/run_forecast_dapp.py`, docs/11 11.12): Pareto fronts
// (Coverage, mean cumulative cost) of the decision rules per initial action, rule details, timing of actions, seed convergence.
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : `${(100 * v).toFixed(d)}%`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const PAL = ['#0e7490', '#dc2626', '#16a34a', '#7c3aed', '#d97706', '#db2777', '#2563eb', '#65a30d', '#9333ea', '#0891b2', '#b45309', '#be123c', '#4f46e5', '#15803d', '#c2410c', '#475569'];
const col = (i) => PAL[i % PAL.length];
const charts = {};
function chart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); return charts[id]; }
let D = null, SI = 0, SR = 0;
const rule = (p) => `ρ ${fmt(p.rho, 1)}・先読み ${p.lookahead} 年・k ${p.k}・${p.wait ? '待つ判断あり' : '待たない'}`;
const shortLab = (r) => (r.label.length > 46 ? r.label.slice(0, 45) + '…' : r.label);

export function render(data) {
  D = data; SI = 0; SR = 0;
  const F = D.config.forecast;
  $('meta').textContent = `作成 ${D.created}・計算 ${fmt(D.elapsed_s / 60, 0)} 分・代理モデル ${F.explore_run}・評価シナリオ ${F.scenarios.n_eval}（探索 ${F.scenarios.n_search} × seed ${F.scenarios.seeds.length}）`;
  drawSummary(); drawFronts(); drawInit(); select(0, 0); drawCands();
}

function drawSummary() {
  const F = D.config.forecast, P = F.policy, T = D.targets, fq = F.frequency;
  const nRules = D.policies.length;
  const best = D.table.map((r) => Math.max(...r.front.map((f) => f.coverage)));
  const fq_txt = fq ? fq.T.map((t, i) => `1/${t} ×${fmt(fq.ratio[i], 2)}`).join('・') : '';
  $('f-summary').innerHTML = `<h2>Adaptive DAPP（予測と学習; docs/11 11.12）</h2>
    <div class="kpis">
      <div class="kpi"><div class="l">初期行動</div><div class="v">${D.table.length}</div></div>
      <div class="kpi"><div class="l">ルール（全探索）</div><div class="v">${nRules}</div></div>
      <div class="kpi"><div class="l">最大の Coverage（初期行動ごとの範囲）</div><div class="v">${pct(Math.min(...best))}〜${pct(Math.max(...best))}</div></div>
      <div class="kpi"><div class="l">行動の選び方</div><div class="v" style="font-size:14px">${P.choice === 'net_benefit' ? '純便益が最大' : '目標を回復する行動のうち純便益が最大'}</div></div>
    </div>
    <p class="hint" style="margin:8px 0 0">物理: 1K-RRI の代理モデル（${esc(F.explore_run)}、効果係数なし）。Target: 代表洪水（2019年台風19号, 1/100 とみなす）の被害額 ≤ ${T.damage_ratio} 倍・浸水人口 ≤ ${T.pop_ratio} 倍（対策なし・現在気候）、${T.from_year} 年から毎評価期。
      Coverage＝評価シナリオ（LHS の標本、発生確率ではない）のうち Target を全期間満たした割合。</p>
    <p class="hint" style="margin:4px 0 0">ルール: 監視は将来予測だけ（2100 年の降雨・海面の予測分布で期待した、t＋先読み年の比）。予測した比 &gt; ρ で行動。
      次の行動は 純便益＝PV(年平均被害額の減少) − PV(費用) で選ぶ（罰則項なし、リードタイムは便益の遅れとして入る）。待つ判断＝10 年後の予測で判断し直す価値（事前事後解析）と比べる。
      格子: ρ ${P.rho_grid.join('/')}・先読み ${P.lookahead_grid.join('/')} 年・k ${P.k_grid.join('/')}・待つ ${P.wait_grid.join('/')}。</p>
    ${fq ? `<p class="hint" style="margin:4px 0 0">年平均被害額（治水経済調査マニュアル）: 確率規模 ${fq_txt}（代表洪水に対する雨量の比; アメダスの年最大日降水量の Gumbel 分布）× 予測した気候の降雨倍率。</p>` : ''}`;
}

function drawFronts() {
  const ds = D.table.map((r, i) => ({ label: `${i}: ${shortLab(r)}`, data: r.front.map((f, j) => ({ x: f.mean_cost, y: 100 * f.coverage, i, j })),
    showLine: true, borderColor: col(i), backgroundColor: col(i), pointRadius: (c) => (c.raw && c.raw.i === SI && c.raw.j === SR ? 7 : 3), borderWidth: 1.5 }));
  chart('f-fronts', { type: 'scatter', data: { datasets: ds }, options: { maintainAspectRatio: false, animation: false,
    scales: { x: { title: { display: true, text: '平均累積費用（現在価値）[億円]' } }, y: { title: { display: true, text: 'Coverage [%]' }, min: 0, max: 100 } },
    plugins: { legend: { position: 'right', labels: { boxWidth: 10, font: { size: 10 } } },
      tooltip: { callbacks: { label: (c) => `${c.dataset.label.split(':')[0]}: ${rule(D.table[c.raw.i].front[c.raw.j].policy)}（Coverage ${fmt(c.raw.y, 0)}%・費用 ${fmt(c.raw.x)} 億円）` } } },
    onClick: (e, el) => { if (el.length) { const p = ds[el[0].datasetIndex].data[el[0].index]; select(p.i, p.j); } } } });
}

function drawInit() {
  const head = '<tr><th>#</th><th>初期行動</th><th>初期費用</th><th>年平均被害額 PV（標準値）</th><th>front のルール数</th><th>Coverage</th><th>平均累積費用</th><th>HV 比（最小）</th><th>重なり</th></tr>';
  $('f-init').innerHTML = head + D.table.map((r, i) => {
    const cov = r.front.map((f) => f.coverage), cost = r.front.map((f) => f.mean_cost), c = r.convergence;
    return `<tr data-i="${i}" class="${i === SI ? 'rec' : ''}"><td><i style="display:inline-block;width:10px;height:10px;background:${col(i)};border-radius:2px"></i> ${i}</td>
      <td title="${esc(r.label)}">${esc(shortLab(r))}</td><td>${fmt(r.initial_cost)}</td><td>${fmt(r.nominal?.ead_pv)}</td><td>${r.front.length}</td>
      <td>${pct(Math.min(...cov))}〜${pct(Math.max(...cov))}</td><td>${fmt(Math.min(...cost))}〜${fmt(Math.max(...cost))}</td>
      <td>${fmt(c.hv_ratio_min, 2)}</td><td>${fmt(c.jaccard, 2)}</td></tr>`;
  }).join('');
  $('f-init').querySelectorAll('tr[data-i]').forEach((tr) => { tr.onclick = () => select(+tr.dataset.i, 0); });
}

function select(i, j) {
  SI = i; SR = j;
  const r = D.table[i];
  $('f-sel').textContent = `初期行動 ${i}: ${r.label}（初期費用 ${fmt(r.initial_cost)} 億円）`;
  const seeds = Object.keys(r.convergence.hv_eval);
  $('f-rules').innerHTML = `<tr><th>ルール</th><th>Coverage（評価）</th><th>平均累積費用</th>${seeds.map((s) => `<th>探索 seed ${s}</th>`).join('')}<th>行動の回数</th><th>待った回数</th></tr>` +
    r.front.map((f, k) => `<tr data-k="${k}" class="${k === SR ? 'rec' : ''}"><td>${rule(f.policy)}</td><td>${pct(f.coverage)}</td><td>${fmt(f.mean_cost)}</td>
      ${seeds.map((s) => `<td>${f.search?.[s] ? `${pct(f.search[s][0])}・${fmt(f.search[s][1])}` : '—'}</td>`).join('')}<td>${fmt(f.n_acts, 1)}</td><td>${fmt(f.n_wait, 2)}</td></tr>`).join('');
  $('f-rules').querySelectorAll('tr[data-k]').forEach((tr) => { tr.onclick = () => select(SI, +tr.dataset.k); });
  drawInit(); drawRule(); drawSeeds();
  if (charts['f-fronts']) charts['f-fronts'].update();
}

function drawRule() {
  const r = D.table[SI], f = r.front[SR];
  $('f-rsel').textContent = rule(f.policy);
  chart('f-time', { type: 'line', data: { labels: D.years.map((y) => y + 10), datasets: [
    { label: 'Target を満たし続ける割合 [%]', data: f.met_so_far.map((v) => 100 * v), borderColor: '#0e7490', yAxisID: 'y' },
    { label: '平均累積費用 [億円]', data: f.cost_path, borderColor: '#d97706', yAxisID: 'y1' }] },
    options: { maintainAspectRatio: false, animation: false, scales: { y: { min: 0, max: 100, title: { display: true, text: '%' } },
      y1: { position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: '億円' } } } } });
  // actions by measure and decade
  const V = D.variables, dec = D.years, cnt = {}, waits = {};
  for (const evs of f.events) for (const e of evs) {
    if (e.kind === 'act') { const k = V[e.var].name; cnt[k] = cnt[k] || {}; cnt[k][e.year] = (cnt[k][e.year] || 0) + 1; }
    if (e.kind === 'wait') waits[e.year] = (waits[e.year] || 0) + 1;
  }
  const names = Object.keys(cnt).sort((a, b) => Object.values(cnt[b]).reduce((x, y) => x + y, 0) - Object.values(cnt[a]).reduce((x, y) => x + y, 0));
  const max = Math.max(1, ...names.flatMap((n) => Object.values(cnt[n])));
  $('f-acts').innerHTML = `<tr><th>施策</th>${dec.map((y) => `<th>${y}</th>`).join('')}</tr>` + names.map((n) => `<tr><td>${esc(n)}</td>${dec.map((y) => {
    const v = cnt[n][y] || 0; return `<td style="background:${v ? `rgba(14,116,144,${0.12 + 0.6 * v / max})` : 'transparent'}">${v || ''}</td>`; }).join('')}</tr>`).join('')
    + `<tr><td><i>待つ判断</i></td>${dec.map((y) => `<td>${waits[y] || ''}</td>`).join('')}</tr>`;
  const fv = f.first_violation.filter((x) => x != null);
  const byDec = {}; fv.forEach((y) => { const d = Math.floor(y / 10) * 10; byDec[d] = (byDec[d] || 0) + 1; });
  $('f-acts-note').textContent = `評価シナリオ ${f.first_violation.length}・Target を最初に超えた年代: ${Object.entries(byDec).map(([d, n]) => `${d}年代 ${n}`).join('・') || 'なし'}`;
}

function drawSeeds() {
  const r = D.table[SI], seeds = Object.keys(r.all_search);
  chart('f-seeds', { type: 'scatter', data: { datasets: seeds.map((s, k) => ({ label: `seed ${s}`, data: r.all_search[s].map(([p, cov, cost]) => ({ x: cost, y: 100 * cov, p })),
    backgroundColor: ['#0e749088', '#dc262688', '#16a34a88', '#7c3aed88'][k % 4], pointRadius: 3 })) },
    options: { maintainAspectRatio: false, animation: false, scales: { x: { title: { display: true, text: '平均累積費用 [億円]' } }, y: { min: 0, max: 100, title: { display: true, text: 'Coverage [%]' } } },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${rule(c.raw.p)}（${fmt(c.raw.y)}%・${fmt(c.raw.x)} 億円）` } } } } });
}

function drawCands() {
  const fr = D.initial_front || [];
  const chosen = new Set(D.table.map((r) => r.levels.join(',')));
  chart('f-cands', { type: 'scatter', data: { datasets: [
    { label: 'Pareto（標準値の世界）', data: fr.filter((c) => !chosen.has(c.g.join(','))).map((c) => ({ x: c.initial_cost, y: c.ead_pv })), backgroundColor: '#94a3b8', pointRadius: 2.5 },
    { label: '初期行動の候補', data: D.table.map((r) => ({ x: r.initial_cost, y: r.nominal?.ead_pv })), backgroundColor: '#dc2626', pointRadius: 4 }] },
    options: { maintainAspectRatio: false, animation: false, scales: { x: { title: { display: true, text: '初期費用 [億円]' } }, y: { title: { display: true, text: '年平均被害額の現在価値 [億円]' } } } } });
}
