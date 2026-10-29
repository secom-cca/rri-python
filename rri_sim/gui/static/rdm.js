// RDM results viewer: GET /api/rdm (runs), /api/rdm/<name> (results.json of scripts/run_rdm.py)
const $ = (id) => document.getElementById(id);
const TYPE = { levee: ['堤防', '#7c4a1e'], retention: ['遊水地', '#db2777'], paddy_dam: ['田んぼダム', '#06b6d4'],
  forest: ['森林化', '#15803d'], relocate: ['住宅対策', '#8b5cf6'] };
const CLIM = { current: ['現在気候', '#2563eb'], '2C': ['2℃上昇', '#f59e0b'], '4C': ['4℃上昇', '#dc2626'] };
const ZONE_TINT = ['#dbeafe', '#dcfce7', '#fef3c7', '#fce7f3', '#e0e7ff'];
const fmt = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const dec = (b, T) => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new T(u.buffer); };
let R = null, sel = null;
const charts = {};
function chart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); return charts[id]; }

async function init() {
  const runs = await (await fetch('/api/rdm')).json();
  if (!runs.length) { $('summary').innerHTML = '<p>RDM の結果がありません。<code>python scripts/run_rdm.py --quick</code> を実行してください（docs/09_rdm.md）。</p>'; return; }
  $('run').innerHTML = runs.map((r) => `<option>${r.name}</option>`).join('');
  $('run').onchange = () => load($('run').value);
  await load(runs[0].name);
}
async function load(name) {
  R = await (await fetch(`/api/rdm/${encodeURIComponent(name)}`)).json();
  const stn = Object.keys(R.config.stakeholders);
  $('rg-st').innerHTML = stn.map((s) => `<option>${s}</option>`).join('');
  $('pf-st').innerHTML = '<option value="">全体の Pareto</option>' + stn.map((s) => `<option>${s}</option>`).join('');
  $('sb-target').innerHTML = Object.keys(R.sobol).map((s) => `<option>${s}</option>`).join('');
  $('rg-st').onchange = () => { drawRegret(); drawRobust(); };
  $('pf-color').onchange = drawPareto; $('pf-st').onchange = drawPareto; $('sb-target').onchange = drawSobol;
  if (R.map) { R.map._mask = dec(R.map.mask, Uint8Array); R.map._zone = dec(R.map.zone_id, Uint8Array); }
  sel = R.policies[R.robustness.overall.minimax].archive;
  $('meta').textContent = `${R.config.basin} ${R.config.res} m・${R.config.event}・作成 ${R.created}・計算 ${fmt(R.elapsed_s / 60, 1)} 分・RRI ${R.n_simulations ?? '—'} 回`;
  drawSummary(); drawPareto(); drawMap(); drawParallel(); drawRegret(); drawRobust(); drawSobol(); drawScenarios();
  await loadSA(name);
}
const policyOf = (ai) => R.policies.findIndex((p) => p.archive === ai);
function descOf(x) { return R.variables.map((v, i) => (x[i] ? `${v.name}: ${v.options[x[i]]}` : null)).filter(Boolean); }

function drawSummary() {
  const rob = R.robustness, ov = rob.overall.minimax, P = R.policies, a0 = R.archive[P[0].archive];
  const recs = Object.entries(rob.stakeholders).map(([n, s]) => `<li><b>${n}</b>: ${P[s.minimax].label}（最大 Regret ${fmt(s.summary.max[s.minimax], 3)}、許容閾値達成率 ${fmt(s.summary.satisficing[s.minimax] * 100)}%）</li>`).join('');
  const ar = R.archive[P[ov].archive];
  $('summary').innerHTML = `<div class="kpis">
    <div class="kpi"><div class="l">予算（現在価値）</div><div class="v">${fmt(R.config.budget_oku)} 億円</div></div>
    <div class="kpi"><div class="l">候補施策（変数）</div><div class="v">${R.variables.length}</div></div>
    <div class="kpi"><div class="l">評価した組合せ</div><div class="v">${R.archive.length}</div></div>
    <div class="kpi"><div class="l">Pareto 最適解</div><div class="v">${R.pareto.length}</div></div>
    <div class="kpi"><div class="l">共通シナリオ</div><div class="v">${R.scenarios.length}</div></div>
    <div class="kpi"><div class="l">対策なしの被害額（名目）</div><div class="v">${fmt(a0.damage)} 億円</div></div></div>
    <div class="rec-box">🏅 <b>Minimax Regret 推奨案（全意思決定者の最大 Regret ÷ 許容閾値 が最小）: ${P[ov].label}</b>
      — 事業費 ${fmt(ar.cost)} 億円、名目被害額 ${fmt(ar.damage)} 億円（対策なし ${fmt(a0.damage)}）<br>${P[ov].desc.join('、')}
      <ul style="margin:6px 0 0">${recs}</ul></div>`;
}

function ramp(t) { t = Math.max(0, Math.min(1, t)); const a = [239, 68, 68], b = [250, 204, 21], c = [22, 163, 74];
  const m = t < 0.5 ? a.map((x, i) => x + (b[i] - x) * t * 2) : b.map((x, i) => x + (c[i] - x) * (t - 0.5) * 2); return `rgb(${m.map(Math.round)})`; }
function drawPareto() {
  const key = $('pf-color').value, st = $('pf-st').value;
  const A = R.archive, feas = A.map((a, i) => i).filter((i) => A[i].feasible);
  const hi = st ? R.stakeholder_pareto[st] : R.pareto;
  const vals = hi.map((i) => A[i][key]), lo = Math.min(...vals), up = Math.max(...vals);
  const good = (v) => (up > lo ? (key === 'households' ? (up - v) : (v - lo)) / (up - lo) : 0.5);
  const pt = (i) => ({ x: A[i].cost, y: A[i].damage, i });
  chart('pareto', { type: 'scatter', data: { datasets: [
    { label: '評価済み（予算内）', data: feas.filter((i) => !hi.includes(i)).map(pt), backgroundColor: '#cbd5e1', pointRadius: 3 },
    { label: st ? `${st} の Pareto` : 'Pareto 最適解', data: hi.map(pt), backgroundColor: hi.map((i) => ramp(good(A[i][key]))), borderColor: '#334155', pointRadius: 6 },
    { label: '頑健性評価の対象', data: R.policies.map((p) => pt(p.archive)), pointStyle: 'rectRot', pointRadius: 8, backgroundColor: 'transparent', borderColor: '#0e7490', borderWidth: 2 },
    { label: '推奨案', data: [pt(R.policies[R.robustness.overall.minimax].archive)], pointStyle: 'star', pointRadius: 14, borderColor: '#f59e0b', borderWidth: 3 }] },
    options: { animation: false, onClick: (e, els) => { if (els.length) { sel = charts.pareto.data.datasets[els[0].datasetIndex].data[els[0].index].i; drawMap(); } },
      plugins: { tooltip: { callbacks: { label: (c) => { const a = A[c.raw.i]; return [`事業費 ${fmt(a.cost)} 億円・被害額 ${fmt(a.damage)} 億円`, `床上 ${fmt(a.households)} 世帯・L ${fmt(a.livability, 1)}・E ${fmt(a.ecology, 1)}`, ...descOf(a.x)]; } } } },
      scales: { x: { title: { display: true, text: '事業費（現在価値, 億円）' } }, y: { title: { display: true, text: '被害額（億円）' } } } } });
}

function drawMap() {
  const M = R.map, cv = $('map'), a = R.archive[sel], x = a.x;
  const p = policyOf(sel);
  $('map-title').textContent = `${p >= 0 ? R.policies[p].label + '・' : ''}事業費 ${fmt(a.cost)} 億円、被害額 ${fmt(a.damage)} 億円`;
  $('policy-desc').innerHTML = (descOf(x).length ? descOf(x) : ['施策なし']).map((t) => `<li>${t}</li>`).join('');
  if (!M) { cv.getContext('2d').clearRect(0, 0, cv.width, cv.height); return; }
  const W = cv.clientWidth, H = cv.clientHeight, dpr = devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  const s = Math.min(W * dpr / M.nx, H * dpr / M.ny), ox = (W * dpr - M.nx * s) / 2, oy = (H * dpr - M.ny * s) / 2, ctx = cv.getContext('2d');
  const cell = (c, col) => { ctx.fillStyle = col; ctx.fillRect(ox + (c % M.nx) * s, oy + Math.floor(c / M.nx) * s, s + 0.5, s + 0.5); };
  for (let c = 0; c < M.nx * M.ny; c++) if (M._mask[c]) cell(c, ZONE_TINT[(M._zone[c] - 1) % ZONE_TINT.length] || '#eee');
  R.variables.forEach((v, i) => { if (v.type !== 'levee') for (const c of v.cells) cell(c, x[i] ? TYPE[v.type][1] : TYPE[v.type][1] + '33'); });
  const r = M.river, lev = new Map();
  R.variables.forEach((v, i) => { if (v.type === 'levee') for (const c of v.cells) lev.set(c, x[i]); });
  ctx.lineCap = 'round';
  for (let k = 0; k < r.ri.length; k++) {
    const d = r.dn[k]; if (d < 0 || r.area[k] < 50) continue;
    const c = r.ri[k] * M.nx + r.rj[k], L = lev.get(c);
    ctx.strokeStyle = L ? TYPE.levee[1] : '#2563eb'; ctx.lineWidth = (L ? 2.5 + 1.5 * L : 1 + 2 * Math.sqrt(r.area[k] / 5000)) * dpr;
    ctx.beginPath(); ctx.moveTo(ox + (r.rj[k] + 0.5) * s, oy + (r.ri[k] + 0.5) * s); ctx.lineTo(ox + (r.rj[d] + 0.5) * s, oy + (r.ri[d] + 0.5) * s); ctx.stroke();
  }
  ctx.font = `${11 * dpr}px sans-serif`; ctx.fillStyle = '#0b2e3c';
  for (const [g, k] of Object.entries(M.gauges)) { const X = ox + (r.rj[k] + 0.5) * s, Y = oy + (r.ri[k] + 0.5) * s; ctx.beginPath(); ctx.arc(X, Y, 3 * dpr, 0, 7); ctx.fill(); ctx.fillText(g, X + 5 * dpr, Y - 3 * dpr); }
  $('map-legend').innerHTML = M.zones.map((z, i) => `<span><i style="background:${ZONE_TINT[i]}"></i>${z}</span>`).join('')
    + Object.values(TYPE).map(([n, c]) => `<span><i style="background:${c}"></i>${n}（薄色＝候補地）</span>`).join('');
}

function drawParallel() {
  const A = R.archive, keys = [['damage', '被害額', 1], ['cost', '事業費', 1], ['households', '床上浸水', 1], ['livability', 'Livability', -1], ['ecology', 'Ecology', -1]];
  const idx = R.pareto, rec = R.policies[R.robustness.overall.minimax].archive;
  const rng = keys.map(([k]) => [Math.min(...idx.map((i) => A[i][k])), Math.max(...idx.map((i) => A[i][k]))]);
  const norm = (i) => keys.map(([k, , sgn], j) => { const [lo, hi] = rng[j]; const t = hi > lo ? (A[i][k] - lo) / (hi - lo) : 0.5; return sgn > 0 ? t : 1 - t; });
  chart('parallel', { type: 'line', data: { labels: keys.map((k) => k[1]), datasets: idx.map((i) => ({
    label: policyOf(i) >= 0 ? R.policies[policyOf(i)].label : `#${i}`, data: norm(i), borderColor: i === rec ? '#f59e0b' : 'rgba(14,116,144,.45)',
    borderWidth: i === rec ? 4 : 1.5, pointRadius: 2 })) },
    options: { animation: false, plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1, title: { display: true, text: '0＝最良 … 1＝最悪' } } } } });
}

function drawRegret() {
  const st = $('rg-st').value, S = R.robustness.stakeholders[st], F = R.robustness.feasible, thr = R.config.stakeholders[st].regret_threshold;
  const ds = Object.entries(CLIM).map(([c, [lab, col]]) => ({ label: lab, backgroundColor: col + 'aa', pointRadius: 3,
    data: S.regret.flatMap((row, p) => row.map((v, s) => (R.sims[R.scenarios[s].sim].climate === c && F[p][s] ? { x: p + (s % 9 - 4) * 0.04, y: v } : null)).filter(Boolean)) }));
  ds.push({ label: '予算超過', pointStyle: 'crossRot', borderColor: '#64748b', pointRadius: 5,
    data: S.regret.flatMap((row, p) => row.map((v, s) => (!F[p][s] ? { x: p, y: v } : null)).filter(Boolean)) });
  ds.push({ label: '最大', pointStyle: 'triangle', pointRadius: 8, backgroundColor: '#111', data: S.summary.max.map((v, p) => ({ x: p, y: v })) });
  ds.push({ label: '平均', pointStyle: 'rect', pointRadius: 6, backgroundColor: '#475569', data: S.summary.mean.map((v, p) => ({ x: p, y: v })) });
  ds.push({ type: 'line', label: `許容閾値 ${thr}`, data: [{ x: -0.5, y: thr }, { x: R.policies.length - 0.5, y: thr }], borderColor: '#16a34a', borderDash: [5, 4], pointRadius: 0 });
  chart('regret', { type: 'scatter', data: { datasets: ds }, options: { animation: false,
    scales: { x: { min: -0.5, max: R.policies.length - 0.5, afterBuildTicks: (ax) => { ax.ticks = R.policies.map((_, i) => ({ value: i })); },
      ticks: { autoSkip: false, callback: (v) => R.policies[v]?.label ?? '' } },
      y: { title: { display: true, text: `Regret（${st} の効用）` } } } } });
}

function drawRobust() {
  const st = $('rg-st').value, rob = R.robustness, S = rob.stakeholders[st];
  chart('robust', { type: 'bar', data: { labels: R.policies.map((p) => p.label), datasets: [
    { label: '最大 Regret', data: S.summary.max, backgroundColor: '#0e7490', yAxisID: 'y' },
    { label: '平均 Regret', data: S.summary.mean, backgroundColor: '#67e8f9', yAxisID: 'y' },
    { label: '許容閾値達成率 [%]', data: S.summary.satisficing.map((v) => v * 100), backgroundColor: '#86efac', yAxisID: 'y2' }] },
    options: { animation: false, plugins: { title: { display: true, text: `${st}（Minimax Regret: ${R.policies[S.minimax].label}）` } },
      scales: { y: { title: { display: true, text: 'Regret' } }, y2: { position: 'right', min: 0, max: 100, grid: { display: false }, title: { display: true, text: '%' } } } } });
  const stn = Object.keys(rob.stakeholders);
  $('rob-table').innerHTML = `<tr><th>案</th><th>事業費</th><th>名目被害額</th>${stn.map((n) => `<th>${n}<br>最大 / 平均 / 達成率</th>`).join('')}<th>最大相対Regret</th></tr>`
    + R.policies.map((p, i) => { const a = R.archive[p.archive];
      return `<tr class="${i === rob.overall.minimax ? 'rec' : ''}" data-a="${p.archive}"><td>${p.label}${i === rob.overall.minimax ? ' 🏅' : ''}</td><td>${fmt(a.cost)}</td><td>${fmt(a.damage)}</td>`
        + stn.map((n) => { const s = rob.stakeholders[n]; const b = s.minimax === i ? 'font-weight:700;color:#b45309' : '';
          return `<td style="${b}">${fmt(s.summary.max[i], 3)} / ${fmt(s.summary.mean[i], 3)} / ${fmt(s.summary.satisficing[i] * 100)}%</td>`; }).join('')
        + `<td>${fmt(rob.overall.max_relative_regret[i], 2)}</td></tr>`; }).join('');
  $('rob-table').querySelectorAll('tr[data-a]').forEach((tr) => tr.onclick = () => { sel = +tr.dataset.a; drawMap(); });
}

function drawSobol() {
  const t = $('sb-target').value, S = R.sobol[t], f = Object.keys(S);
  chart('sobol', { type: 'bar', data: { labels: f.map((k) => R.sobol_factors[k] || k), datasets: [
    { label: '一次 S1', data: f.map((k) => S[k].S1), backgroundColor: '#0e7490' }, { label: '全 ST', data: f.map((k) => S[k].ST), backgroundColor: '#f59e0b' }] },
    options: { indexAxis: 'y', animation: false, scales: { x: { min: 0, max: 1 } } } });
  const n = R.config.scenarios.n_per_climate;
  $('sb-note').textContent = `代理モデル（シミュレーションした ${R.sims.length} シナリオへのリッジ回帰）上の Saltelli 法（N=${R.config.sobol.n_base}）。`
    + (n <= 1 ? '気候以外のシミュレーション係数は気候ごとに1点（名目値）しか計算していないため、その指数は推定できず0に近く出る。--quick ではなく n_per_climate ≥ 3 で実行すること。' : '');
}

function drawScenarios() {
  const U = R.config.uncertainties;
  const simk = Object.keys(U).filter((k) => U[k].kind === 'sim'), postk = Object.keys(U).filter((k) => U[k].kind === 'post');
  $('scen').innerHTML = `<tr><th>シミュレーションしたシナリオ</th>${simk.map((k) => `<th>${k}</th>`).join('')}</tr>`
    + R.sims.map((s) => `<tr><td>${CLIM[s.climate]?.[0] ?? s.climate}</td>${simk.map((k) => `<td>${s[k]}</td>`).join('')}</tr>`).join('')
    + `<tr><th>事後係数（各シナリオに ${R.posts.length} 通り）</th>${postk.map((k) => `<th>${k}</th>`).join('')}</tr>`
    + `<tr><td>範囲</td>${postk.map((k) => `<td>${U[k].low}–${U[k].high}</td>`).join('')}</tr>`
    + Object.entries(U).map(([k, u]) => `<tr><td colspan="${Math.max(simk.length, postk.length) + 1}" style="text-align:left">${k}: ${u.label}（${u.low}–${u.high}、名目 ${u.nominal}）</td></tr>`).join('');
}

// ------------------------------------------------------------------ sensitivity analysis (sensitivity.json)
let SA = null;
const PCOL = ['#64748b', '#f59e0b', '#0e7490', '#db2777', '#16a34a', '#7c3aed', '#dc2626', '#0891b2', '#a16207', '#4f46e5'];
const pcol = (i) => PCOL[i % PCOL.length];
const OUTJ = (o) => (o.startsWith('regret:') ? `Regret（${o.slice(7)}）` : (SA.morris.outputs[o] || o));
const plab = (n) => SA.parameters[n]?.label || n;
async function loadSA(name) {
  const r = await fetch(`/api/rdm/${encodeURIComponent(name)}/sensitivity`);
  document.querySelectorAll('.sa').forEach((e) => { e.style.display = r.ok ? '' : 'none'; });
  if (!r.ok) { SA = null; $('sa-summary').innerHTML = `<span class="hint">この実行の感度分析はまだありません: <code>python scripts/run_sensitivity.py ${name} --quick</code>（docs/10_sensitivity.md）</span>`; return; }
  SA = await r.json();
  const opt = (id, list, lab) => { $(id).innerHTML = list.map((k) => `<option value="${k}">${lab ? lab(k) : k}</option>`).join(''); };
  opt('mo-out', Object.keys(SA.morris.outputs), (k) => SA.morris.outputs[k]);
  opt('so-out', Object.keys(SA.sobol.results), OUTJ); opt('to-out', Object.keys(SA.tornado), OUTJ);
  opt('re-par', SA.relation.sel, plab); opt('re-out', ['damage', 'pop_hazard', 'livability', 'ecology'], (k) => SA.morris.outputs[k]);
  opt('va-out', ['damage', 'pop_hazard', 'livability', 'ecology'], (k) => SA.morris.outputs[k]);
  opt('bb-st', Object.keys(SA.decision)); opt('bb-par', SA.sobol.inputs, plab); opt('pw-st', Object.keys(SA.preference.stakeholders));
  ['mo-out'].forEach((i) => { $(i).onchange = drawMorris; }); $('so-out').onchange = drawSobol2; $('to-out').onchange = drawTornado;
  $('re-par').onchange = $('re-out').onchange = drawRelation; $('va-out').onchange = drawValid;
  $('bb-st').onchange = $('bb-par').onchange = drawBestBin; $('pw-st').onchange = drawSweep;
  const sel = SA.morris.selected.map(plab).join('、');
  $('sa-summary').innerHTML = `<div class="hint">作成 ${SA.created}・計算 ${fmt(SA.elapsed_s / 60, 1)} 分・RRI ${SA.n_simulations ?? '—'} 回・対象案 ${SA.policies.join(' / ')}（推奨案 ${SA.policies[SA.recommended]}）</div>
    <div>Morris で選んだ計算パラメータ: <b>${sel}</b>（＋事後係数 ${SA.post_params.map(plab).join('、')}）。
    代理モデル: ${SA.surrogate.kind}、学習 ${SA.surrogate.n_train} 点・検証 ${SA.surrogate.n_val} 点、<b>検証の相対 RMSE（被害額・浸水人口の最大）${fmt(SA.surrogate.worst_rel_rmse * 100, 1)}%</b>。
    Sobol: ${SA.sobol.design}。${SA.surrogate.note}</div>`;
  $('sa-params').innerHTML = '<tr><th>パラメータ</th><th>範囲</th><th>名目</th><th>種類</th><th>区分</th><th>根拠・仮定の内容</th><th>Morris 相対μ*</th></tr>'
    + SA.names.map((n) => { const p = SA.parameters[n];
      return `<tr><td>${p.label}<br><span class="hint">${n}</span></td><td>${p.low}–${p.high}</td><td>${p.nominal}</td><td>${p.kind === 'sim' ? 'RRI 計算' : '事後'}</td>`
        + `<td><span class="tag ${p.status === '根拠' ? 'ok' : 'as'}">${p.status}</span></td><td style="text-align:left">${p.basis}</td>`
        + `<td>${fmt(SA.morris.score[n], 2)}${SA.morris.selected.includes(n) ? ' ●' : ''}</td></tr>`; }).join('');
  drawMorris(); drawSobol2(); drawTornado(); drawRelation(); drawValid(); drawSelFreq(); drawBestBin(); drawParChg(); drawSweep(); drawPrefFreq(); drawConflict();
}
function drawMorris() {
  const o = $('mo-out').value, M = SA.morris.indices[o];
  chart('morris', { type: 'scatter', data: { datasets: SA.names.map((n, i) => ({ label: plab(n), data: [{ x: M.rel_mu_star[i], y: M.rel_sigma[i] }],
    backgroundColor: pcol(i), pointRadius: SA.morris.selected.includes(n) ? 8 : 5, pointStyle: SA.parameters[n].kind === 'post' ? 'rect' : 'circle' })) },
    options: { animation: false, plugins: { legend: { labels: { boxWidth: 8, font: { size: 10 } } } },
      scales: { x: { min: 0, title: { display: true, text: 'μ*（÷ 出力の標準偏差）' } }, y: { min: 0, title: { display: true, text: 'σ（÷ 出力の標準偏差）' } } } } });
}
function drawSobol2() {
  const o = $('so-out').value, S = SA.sobol.results[o], ins = SA.sobol.inputs;
  chart('sobol2', { type: 'bar', data: { labels: ins.map(plab), datasets: [
    { label: '一次 S1', data: S.S1, backgroundColor: '#0e7490' }, { label: 'S1 95%区間', data: S.S1_ci, backgroundColor: 'rgba(14,116,144,.25)', grouped: false, barPercentage: 0.25 },
    { label: '全効果 ST', data: S.ST, backgroundColor: '#f59e0b' }, { label: 'ST 95%区間', data: S.ST_ci, backgroundColor: 'rgba(245,158,11,.3)', grouped: false, barPercentage: 0.25 }] },
    options: { indexAxis: 'y', animation: false, scales: { x: { min: Math.min(0, ...S.S1_ci.map((c) => c[0])), max: 1 } }, plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
  $('so-note').textContent = `${OUTJ(o)}・分散 ${fmt(S.var, 2)}・N=${S.n}（${SA.sobol.design}）。区間はブートストラップ。代理モデル上の推定。`;
}
function drawTornado() {
  const o = $('to-out').value, T = SA.tornado[o];
  chart('tornado', { type: 'bar', data: { labels: T.rows.map((r) => plab(r.param)), datasets: [
    { label: '下限', data: T.rows.map((r) => [T.base, r.low]), backgroundColor: '#60a5fa' },
    { label: '上限', data: T.rows.map((r) => [T.base, r.high]), backgroundColor: '#f87171' }] },
    options: { indexAxis: 'y', animation: false, plugins: { title: { display: true, text: `${OUTJ(o)}（名目 ${fmt(T.base, 3)}）` } }, scales: { y: { stacked: true }, x: { stacked: false } } } });
}
function drawRelation() {
  const d = $('re-par').value, o = $('re-out').value, i = SA.relation.sel.indexOf(d), p = SA.parameters[d];
  const val = (u) => p.low + u * (p.high - p.low), g = [...Array(21).keys()].map((j) => val(j / 20));
  const ds = [];
  SA.policies.forEach((lab, k) => {
    const pts = SA.relation.points[k];
    ds.push({ type: 'scatter', label: `${lab}（RRI）`, data: pts.u.map((u, j) => ({ x: val(u[i]), y: pts.y[o][j] })), backgroundColor: pcol(k), pointRadius: 3 });
    ds.push({ type: 'line', label: `${lab}（代理モデル）`, data: g.map((x, j) => ({ x, y: SA.relation.curves[d][o][k][j] })), borderColor: pcol(k), pointRadius: 0, borderWidth: 1.5 });
  });
  chart('relation', { type: 'scatter', data: { datasets: ds }, options: { animation: false, plugins: { legend: { labels: { boxWidth: 8, font: { size: 9 } } } },
    scales: { x: { title: { display: true, text: plab(d) } }, y: { title: { display: true, text: SA.morris.outputs[o] } } } } });
}
function drawValid() {
  const o = $('va-out').value, V = SA.surrogate.validation;
  const ds = SA.policies.map((lab, k) => ({ label: lab, data: V[lab][o].y_val.map((y, j) => ({ x: y, y: V[lab][o].yhat_val[j] })), backgroundColor: pcol(k), pointRadius: 4 }));
  const all = SA.policies.flatMap((lab) => V[lab][o].y_val.concat(V[lab][o].yhat_val)), lo = Math.min(...all), hi = Math.max(...all);
  ds.push({ type: 'line', label: '1:1', data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: '#94a3b8', borderDash: [4, 3], pointRadius: 0 });
  chart('valid', { type: 'scatter', data: { datasets: ds }, options: { animation: false, plugins: { legend: { labels: { boxWidth: 8, font: { size: 9 } } } },
    scales: { x: { title: { display: true, text: 'RRI（検証点）' } }, y: { title: { display: true, text: '代理モデルの予測' } } } } });
  $('va-table').innerHTML = '<tr><th>案</th><th>採用モデル</th><th>採用 R²</th><th>採用 相対RMSE</th><th>GP R² / 相対RMSE</th><th>線形 R² / 相対RMSE</th></tr>' + SA.policies.map((lab) => {
    const e = V[lab][o], s = e.selected || e.gp;
    return `<tr><td>${lab}</td><td>${e.chosen === 'linear' ? '線形' : 'GP'}</td><td>${fmt(s.r2, 3)}</td><td>${fmt(s.rel_rmse * 100, 1)}%</td>`
      + `<td>${fmt(e.gp.r2, 2)} / ${fmt(e.gp.rel_rmse * 100, 1)}%</td><td>${fmt(e.linear.r2, 2)} / ${fmt(e.linear.rel_rmse * 100, 1)}%</td></tr>`; }).join('')
    + '<tr><td colspan="6" class="hint" style="text-align:left">採用モデルは学習点の1個抜き交差検証で選択（検証データは選択に使っていない）。値は独立な検証 LHS 点での誤差。</td></tr>';
}
function drawSelFreq() {
  const D = SA.decision, st = Object.keys(D);
  chart('selfreq', { type: 'bar', data: { labels: st, datasets: SA.policies.map((lab, k) => ({ label: lab, data: st.map((n) => D[n].best_frequency[k] * 100), backgroundColor: pcol(k) })) },
    options: { animation: false, scales: { x: { stacked: true }, y: { stacked: true, max: 100, title: { display: true, text: '最良案になる割合 [%]' } } } } });
  $('dec-table').innerHTML = '<tr><th>意思決定者</th><th>名目の最良案</th><th>順位逆転率</th><th>Kendall τ 平均 / 10%値</th><th>推奨案の Regret 5–50–95%</th></tr>'
    + st.map((n) => { const d = D[n], q = d.regret_quantiles;
      return `<tr><td>${n}</td><td>${SA.policies[d.nominal_best]}</td><td>${fmt(d.rank_reversal_rate * 100)}%</td><td>${fmt(d.kendall_tau_mean, 2)} / ${fmt(d.kendall_tau_q10, 2)}</td><td>${fmt(q[0], 3)} – ${fmt(q[2], 3)} – ${fmt(q[4], 3)}</td></tr>`; }).join('');
}
function drawBestBin() {
  const st = $('bb-st').value, d = $('bb-par').value, C = SA.decision[st].best_by_param_bin[d], p = SA.parameters[d];
  const labs = [0, 1, 2, 3, 4].map((b) => `${fmt(p.low + (p.high - p.low) * b / 5, 2)}–${fmt(p.low + (p.high - p.low) * (b + 1) / 5, 2)}`);
  chart('bestbin', { type: 'bar', data: { labels: labs, datasets: SA.policies.map((lab, k) => ({ label: lab, data: C.map((f) => f[k] * 100), backgroundColor: pcol(k) })) },
    options: { animation: false, scales: { x: { stacked: true, title: { display: true, text: plab(d) } }, y: { stacked: true, max: 100, title: { display: true, text: '%' } } } } });
}
function drawParChg() {
  const P = SA.pareto_change, cl = Object.keys(P.by_climate);
  chart('parchg', { type: 'bar', data: { labels: SA.pareto_policies, datasets: cl.map((c, i) => ({ label: CLIM[c]?.[0] || c, data: P.by_climate[c].map((v) => v * 100), backgroundColor: (CLIM[c]?.[1] || pcol(i)) + 'cc' })) },
    options: { animation: false, scales: { y: { max: 100, title: { display: true, text: 'Pareto 最適になる割合 [%]' } } } } });
}
function drawSweep() {
  const n = $('pw-st').value, S = SA.preference.stakeholders[n], labs = SA.pareto_policies;
  chart('sweep', { type: 'line', data: { labels: S.sweep_values.map((v) => fmt(v, 2)), datasets: S.active.map((k, i) => ({ label: `${k} の重み（元 ${fmt(S.base_weights[k], 2)}）`, data: S.sweeps[k], borderColor: pcol(i + 1), stepped: true, pointRadius: 2 })) },
    options: { animation: false, scales: { x: { title: { display: true, text: '重み' } }, y: { min: -0.5, max: labs.length - 0.5, afterBuildTicks: (ax) => { ax.ticks = labs.map((_, i) => ({ value: i })); },
      ticks: { autoSkip: false, callback: (v) => labs[v] ?? '' }, title: { display: true, text: '推奨案' } } } } });
}
function drawPrefFreq() {
  const S = SA.preference.stakeholders, st = Object.keys(S), labs = SA.pareto_policies;
  chart('prefreq', { type: 'bar', data: { labels: st, datasets: labs.map((lab, k) => ({ label: lab, data: st.map((n) => S[n].frequency[k] * 100), backgroundColor: pcol(k) })).filter((d) => d.data.some((v) => v > 0)) },
    options: { animation: false, plugins: { legend: { labels: { boxWidth: 8, font: { size: 9 } } } }, scales: { x: { stacked: true }, y: { stacked: true, max: 100, title: { display: true, text: '推奨される割合 [%]（重みは各自の指標上で一様）' } } } } });
}
function drawConflict() {
  const C = SA.preference.conflict, cm = SA.preference.common, labs = SA.pareto_policies, mx = Math.max(...C.cross_max_regret.flat());
  $('conflict').className = 'tbl heat';
  $('conflict').innerHTML = `<tr><th>自分＼採用する推奨案</th>${C.names.map((k) => `<th>${k}の案<br>（${labs[C.base_recommendation[k]]}）</th>`).join('')}</tr>`
    + C.names.map((j, a) => `<tr><td>${j}</td>${C.cross_max_regret[a].map((v) => `<td style="background:rgba(220,38,38,${(v / (mx || 1)) * 0.6})">${fmt(v, 3)}</td>`).join('')}</tr>`).join('');
  chart('common', { type: 'bar', data: { labels: labs, datasets: [
    { label: '全員の許容閾値内', data: cm.acceptable_to_all.map((v) => v * 100), backgroundColor: '#16a34a' },
    { label: '全員の上位3案', data: cm.top3_of_all.map((v) => v * 100), backgroundColor: '#86efac' }] },
    options: { animation: false, scales: { y: { max: 100, title: { display: true, text: '%（重みの同時ランダム抽出）' } } } } });
  $('common-note').textContent = `各意思決定者の重みを独立に ${cm.n_joint} 回抽出: 全員の推奨案が一致した割合 ${fmt(cm.agreement_rate * 100, 1)}%。`
    + '対角は自分の推奨案（Regret 最小）。行の値が大きい列ほど、その意思決定者にとって受け入れにくい。';
}
init();
