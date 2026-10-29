// Exploration without stakeholder weights (rri_sim/rdm/explore.py): raw metrics of the Pareto policies,
// the damage target / required share / rain range chosen here and re-aggregated from the stored samples.
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !isFinite(v)) ? '–' : Number(v).toLocaleString('ja-JP', { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (v) => v == null ? '–' : `${Math.round(100 * v)}%`;
let R = null, sel = 0, sortKey = 'cost', sortDir = 1;
const charts = {};

function chart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), cfg);
  return charts[id];
}

async function init() {
  const runs = await (await fetch('/api/explore')).json();
  if (!runs.length) { $('summary').innerHTML = '<p>結果がありません。<code>python scripts/run_explore.py --quick</code> を実行してください。</p>'; return; }
  $('run').innerHTML = runs.map((r) => `<option>${r.name}</option>`).join('');
  $('run').onchange = () => load($('run').value);
  for (const id of ['tgt', 'alpha', 'rain']) $(id).addEventListener('input', refresh);
  load(runs[0].name);
}

async function load(name) {
  R = await (await fetch(`/api/explore/${encodeURIComponent(name)}`)).json();
  sel = 0;
  $('tgt').innerHTML = R.target_curve.map((t) => `<option value="${t}" ${t === R.reference.target_ratio ? 'selected' : ''}>${t.toFixed(1)}</option>`).join('');
  const e = R.rain_edges;
  $('rain').innerHTML = '<option value="all">全範囲</option>' + e.slice(0, -1).map((a, i) => `<option value="${i}">${a.toFixed(2)}–${e[i + 1].toFixed(2)}</option>`).join('');
  $('meta').textContent = `${R.created}・RRI ${R.n_simulations} 回・${Math.round(R.elapsed_s / 60)} 分`;
  summary(); refresh(); validation(); uncertainties();
}

// share of samples meeting the target (ratio <= t) within the selected rain range
function mask() {
  const v = $('rain').value, rf = R.ensemble.rain_factor, e = R.rain_edges;
  if (v === 'all') return rf.map(() => true);
  const b = +v;
  return rf.map((x) => x >= e[b] && (x < e[b + 1] || b === e.length - 2));
}
function rel(p, t, m) {
  let n = 0, k = 0;
  p.ratio_samples.forEach((r, i) => { if (m[i]) { n++; if (r <= t + 1e-9) k++; } });
  return n ? k / n : null;
}

function summary() {
  const v = R.surrogate.validation, ver = R.verification, ref = R.reference;
  $('summary').innerHTML = `<div class="kpis">
    <div class="kpi"><div class="l">基準：現在気候・対策なしの被害額</div><div class="v">${fmt(ref.damage)} 億円</div></div>
    <div class="kpi"><div class="l">施策の変数</div><div class="v">${R.variables.length}</div></div>
    <div class="kpi"><div class="l">不確実性</div><div class="v">${Object.keys(R.uncertainties).length}</div></div>
    <div class="kpi"><div class="l">RRI 学習点</div><div class="v">${R.design.n_train}</div></div>
    <div class="kpi"><div class="l">代理モデル 被害額（検証）</div><div class="v">R² ${fmt(v.damage.r2, 2)}・${pct(v.damage.rel_rmse)}</div></div>
    <div class="kpi"><div class="l">目標判定の一致率（検証／RRI 確認）</div><div class="v">${pct(v.target_agreement)}／${pct(ver.agreement)}</div></div>
    <div class="kpi"><div class="l">Pareto 解</div><div class="v">${R.policies.length}</div></div></div>
    <p class="hint">田んぼダム・森林化は地域ごとの実施率（0〜100%、対象セルを${R.config.explore.cell_order === 'random' ? '無作為な順' : '設定の順'}に選ぶ）、堤防・遊水地・住宅対策は選択肢。
    予算の制約は置かず、費用は結果として示す。主体の重みは用いず、地域別の被害額・費用・指標をそのまま示す。
    達成率・分布は代理モデル上の ${R.ensemble.rain_factor.length} サンプルによる（RRI での確認は下段）。</p>`;
}

function refresh() {
  const t = +$('tgt').value, a = +$('alpha').value / 100, m = mask();
  $('alpha-v').textContent = `${Math.round(a * 100)}%`;
  R.policies.forEach((p) => { p._rel = rel(p, t, m); });
  const ok = R.policies.map((p, i) => [p, i]).filter(([p]) => p._rel >= a - 1e-9).sort((x, y) => x[0].cost - y[0].cost);
  $('choice').innerHTML = ok.length
    ? `<div class="rec-box">条件を満たす案 <b>${ok.length}</b> / ${R.policies.length}。最も安い案：<b>案${ok[0][1]}</b>（費用 ${fmt(ok[0][0].cost)} 億円、達成率 ${pct(ok[0][0]._rel)}）
       <button id="pick">この案を表示</button></div>`
    : `<div class="rec-box">条件を満たす案はありません（最大の達成率 ${pct(Math.max(...R.policies.map((p) => p._rel)))}）。目標を緩めるか達成率を下げてください。</div>`;
  if (ok.length) $('pick').onclick = () => select(ok[0][1]);
  scatter(a); table(); select(sel);
}

function scatter(a) {
  const P = R.policies, eco = P.map((p) => p.ecology), lo = Math.min(...eco), hi = Math.max(...eco);
  const col = (e) => { const f = hi > lo ? (e - lo) / (hi - lo) : 0.5; return `rgba(${Math.round(200 - 160 * f)},${Math.round(220 - 100 * f)},${Math.round(240 - 30 * f)},0.95)`; };
  chart('scatter', {
    type: 'scatter',
    data: { datasets: [
      { label: 'Pareto 解', data: P.map((p, i) => ({ x: p.cost, y: 100 * p._rel, i })), pointBackgroundColor: P.map((p) => col(p.ecology)),
        pointBorderColor: P.map((p, i) => i === sel ? '#0b0b0b' : '#2a78d6'), pointBorderWidth: P.map((p, i) => i === sel ? 2.5 : 1), pointRadius: P.map((p, i) => i === sel ? 7 : 4.5) },
      { label: `必要な達成率 ${Math.round(a * 100)}%`, type: 'line', data: [{ x: 0, y: 100 * a }, { x: Math.max(...P.map((p) => p.cost)), y: 100 * a }],
        borderColor: '#eb6834', borderDash: [5, 4], pointRadius: 0, borderWidth: 1.5 }] },
    options: { animation: false, onClick: (e, els) => { const el = els.find((x) => x.datasetIndex === 0); if (el) select(el.index); },
      plugins: { tooltip: { callbacks: { label: (c) => c.datasetIndex ? '' : `案${c.raw.i}: ${fmt(c.raw.x)} 億円, 達成率 ${fmt(c.raw.y)}%` } } },
      scales: { x: { title: { display: true, text: '費用 [億円, 現在価値]' } }, y: { min: 0, max: 100, title: { display: true, text: '達成率 [%]' } } } },
  });
}

function select(i) {
  sel = i;
  const p = R.policies[i], p0 = R.policies[0], t = +$('tgt').value;
  $('sel-name').textContent = i === 0 ? '施策なし' : `案${i}`;
  const zr = (z) => R.reference.damage_zone[z];
  $('detail').innerHTML = `<ul class="desc">${p.desc.map((d) => `<li>${d}</li>`).join('')}</ul>
    <table class="tbl"><tr><th>地域</th><th>費用 [億円]</th><th>被害 平均</th><th>被害 90%値</th><th>基準（現在・対策なし）</th><th>地域の達成率*</th></tr>
    ${R.zones.map((z) => `<tr><td>${z}</td><td>${fmt(p.cost_zone[z] || 0)}</td><td>${fmt(p.damage_zone[z].mean)}</td><td>${fmt(p.damage_zone[z].p90)}</td><td>${fmt(zr(z))}</td><td>${pct(p.reliability_zone[z])}</td></tr>`).join('')}
    <tr style="font-weight:600"><td>全域</td><td>${fmt(p.cost)}</td><td>${fmt(p.damage.mean)}</td><td>${fmt(p.damage.p90)}</td><td>${fmt(R.reference.damage)}</td><td>${pct(p._rel)}</td></tr></table>
    <table class="tbl"><tr><th>指標</th><th>この案</th><th>施策なし</th></tr>
    <tr><td>費用の幅（事業費係数）</td><td>${fmt(p.cost_range[0])}–${fmt(p.cost_range[1])}</td><td>0</td></tr>
    <tr><td>浸水人口（0.5 m以上） 平均／90%値</td><td>${fmt(p.pop_hazard.mean)}／${fmt(p.pop_hazard.p90)}</td><td>${fmt(p0.pop_hazard.mean)}／${fmt(p0.pop_hazard.p90)}</td></tr>
    <tr><td>床上浸水世帯 平均</td><td>${fmt(p.households)}</td><td>${fmt(p0.households)}</td></tr>
    <tr><td>Livability</td><td>${fmt(p.livability, 2)}</td><td>${fmt(p0.livability, 2)}</td></tr>
    <tr><td>Ecology</td><td>${fmt(p.ecology, 2)}</td><td>${fmt(p0.ecology, 2)}</td></tr></table>
    <div class="hint">被害額は資産額係数込み。*地域の達成率は目標倍率 ${R.reference.target_ratio}（設定値）で、その地域の基準と比べた割合。</div>`;
  const e = R.rain_edges, labels = e.slice(0, -1).map((a, b) => `${a.toFixed(2)}–${e[b + 1].toFixed(2)}`);
  const byBin = (q) => labels.map((_, b) => { const m = R.ensemble.rain_factor.map((x) => x >= e[b] && (x < e[b + 1] || b === e.length - 2)); return 100 * rel(q, t, m); });
  chart('byrain', { type: 'line', data: { labels, datasets: [
    { label: i === 0 ? '施策なし' : `案${i}`, data: byBin(p), borderColor: '#2a78d6', backgroundColor: '#2a78d6', borderWidth: 2 },
    { label: '施策なし', data: byBin(p0), borderColor: '#a3a29b', backgroundColor: '#a3a29b', borderWidth: 2 }].slice(0, i ? 2 : 1) },
    options: { animation: false, scales: { y: { min: 0, max: 100, title: { display: true, text: '達成率 [%]' } }, x: { title: { display: true, text: '降雨倍率' } } } } });
  const edges = Array.from({ length: 25 }, (_, k) => 0.2 + 0.1 * k);
  const h = (q) => edges.map((a) => q.ratio_samples.filter((r) => r >= a && r < a + 0.1).length);
  chart('hist', { type: 'bar', data: { labels: edges.map((a) => a.toFixed(1)), datasets: [
    { label: i === 0 ? '施策なし' : `案${i}`, data: h(p), backgroundColor: '#2a78d6' },
    { label: '施策なし', data: h(p0), backgroundColor: 'rgba(163,162,155,0.6)' }].slice(0, i ? 2 : 1) },
    options: { animation: false, scales: { x: { title: { display: true, text: `被害額 ÷ 基準（目標 ${t}）` } }, y: { title: { display: true, text: 'サンプル数' } } } } });
  if (charts.scatter) scatter(+$('alpha').value / 100);
  document.querySelectorAll('#ptab tr[data-i]').forEach((tr) => tr.classList.toggle('rec', +tr.dataset.i === sel));
}

function table() {
  const cols = [['cost', '費用', (p) => p.cost], ['_rel', '達成率', (p) => p._rel, pct], ['dmean', '被害 平均', (p) => p.damage.mean],
    ['dp90', '被害 90%', (p) => p.damage.p90], ['pop', '浸水人口', (p) => p.pop_hazard.mean], ['eco', 'Ecology', (p) => p.ecology, (v) => fmt(v, 2)],
    ['liv', 'Livability', (p) => p.livability, (v) => fmt(v, 2)], ...R.zones.map((z) => [`c_${z}`, `費用・${z}`, (p) => p.cost_zone[z] || 0])];
  const get = Object.fromEntries(cols.map((c) => [c[0], c[2]]));
  const rows = R.policies.map((p, i) => [p, i]).sort((a, b) => sortDir * (get[sortKey](a[0]) - get[sortKey](b[0])));
  $('ptab').innerHTML = `<tr><th>案</th>${cols.map((c) => `<th data-k="${c[0]}" style="cursor:pointer">${c[1]}${c[0] === sortKey ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}<th style="text-align:left">施策</th></tr>`
    + rows.map(([p, i]) => `<tr data-i="${i}" class="${i === sel ? 'rec' : ''}"><td>${i ? `案${i}` : '施策なし'}</td>${cols.map((c) => `<td>${(c[3] || fmt)(c[2](p))}</td>`).join('')}
      <td style="text-align:left;font-size:11px">${p.desc.join('、')}</td></tr>`).join('');
  $('ptab').querySelectorAll('th[data-k]').forEach((th) => th.onclick = () => { sortDir = th.dataset.k === sortKey ? -sortDir : 1; sortKey = th.dataset.k; table(); });
  $('ptab').querySelectorAll('tr[data-i]').forEach((tr) => tr.onclick = () => select(+tr.dataset.i));
}

function validation() {
  const v = R.surrogate.validation, ver = R.verification;
  const vo = ver.policies.flatMap((q) => q.obs.map((o, k) => ({ x: o, y: q.pred[k] })));
  const all = v.damage.y_val.concat(vo.map((p) => p.x)), hi = Math.max(...all) * 1.05, lo = Math.min(...all) * 0.95;
  chart('val', { type: 'scatter', data: { datasets: [
    { label: '検証 LHS', data: v.damage.y_val.map((y, k) => ({ x: y, y: v.damage.yhat_val[k] })), backgroundColor: '#2a78d6' },
    { label: 'Pareto 解の確認', data: vo, backgroundColor: '#eb6834' },
    { label: '1:1', type: 'line', data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: '#a3a29b', borderDash: [4, 4], pointRadius: 0 },
    { label: '目標', type: 'line', data: [{ x: R.reference.target, y: lo }, { x: R.reference.target, y: hi }], borderColor: '#e34948', pointRadius: 0, borderWidth: 1 }] },
    options: { animation: false, scales: { x: { title: { display: true, text: 'RRI 被害額 [億円]' } }, y: { title: { display: true, text: '代理モデル [億円]' } } } } });
  const keys = Object.keys(v).filter((k) => v[k] && v[k].chosen);
  $('valtab').innerHTML = '<tr><th>出力</th><th>モデル</th><th>R²</th><th>相対 RMSE</th><th>最大誤差</th></tr>'
    + keys.map((k) => `<tr><td>${k.replace('dz:', '被害・')}</td><td>${v[k].chosen}</td><td>${fmt(v[k].r2, 2)}</td><td>${pct(v[k].rel_rmse)}</td><td>${fmt(v[k].max_abs, 1)}</td></tr>`).join('');
  $('vertab').innerHTML = '<tr><th>案</th><th>費用</th><th>達成率 RRI</th><th>達成率 代理</th><th>一致率</th><th>相対 RMSE</th></tr>'
    + ver.policies.map((q) => `<tr data-i="${q.policy}"><td>${q.policy ? `案${q.policy}` : '施策なし'}</td><td>${fmt(R.policies[q.policy].cost)}</td><td>${pct(q.reliability_rri)}</td><td>${pct(q.reliability_sur)}</td><td>${pct(q.agreement)}</td><td>${pct(q.rel_rmse)}</td></tr>`).join('')
    + `<tr style="font-weight:600"><td>全体</td><td></td><td></td><td></td><td>${pct(ver.agreement)}</td><td>${pct(ver.rel_rmse)}</td></tr>`;
  $('vertab').querySelectorAll('tr[data-i]').forEach((tr) => tr.onclick = () => select(+tr.dataset.i));
}

function uncertainties() {
  $('unc').innerHTML = '<tr><th>パラメータ</th><th>範囲</th><th>種類</th><th>区分</th><th style="text-align:left">根拠・仮定</th></tr>'
    + Object.entries(R.uncertainties).map(([k, u]) => `<tr><td>${u.label || k}</td><td>${u.low}–${u.high}</td><td>${u.kind === 'sim' ? 'RRI' : '事後'}</td>
      <td><span class="tag ${u.status === '根拠' ? 'ok' : 'as'}">${u.status}</span></td><td style="text-align:left">${u.basis || ''}</td></tr>`).join('');
}

init();
