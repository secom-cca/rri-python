// Exploration without stakeholder weights (rri_sim/rdm/explore.py): raw metrics of the Pareto policies,
// the damage target / required share / rain range chosen here and re-aggregated from the stored samples.
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !isFinite(v)) ? '–' : Number(v).toLocaleString('ja-JP', { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (v) => v == null ? '–' : `${Math.round(100 * v)}%`;
let R = null, sel = 0, sortKey = 'cost', sortDir = 1;
let SC = null, REL = null, selPt = null, USED = null, offMask = 0;   // USED: bit i = point uses variable i           // trade spaces by actor (scopes.json) and the selected point
const shown = new Set();
const PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#9b59b6', '#d4a017', '#e34948', '#4a5568', '#0f9fb5'];
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
  for (const id of ['tgt', 'alpha', 'rain', 'ax-x', 'ax-y', 'show-all', 'show-pol']) $(id).addEventListener('input', refresh);
  const opt = (sel) => Object.entries(AX).map(([k, a]) => `<option value="${k}" ${k === sel ? 'selected' : ''}>${a.short}</option>`).join('');
  $('ax-x').innerHTML = opt('cost'); $('ax-y').innerHTML = opt('rel');
  load(runs[0].name);
}

async function load(name) {
  R = await (await fetch(`/api/explore/${encodeURIComponent(name)}`)).json();
  sel = 0; selPt = null;
  const rs = await fetch(`/api/explore/${encodeURIComponent(name)}/scopes`);
  SC = rs.ok ? await rs.json() : null;
  REL = SC ? Uint8Array.from(atob(SC.points.rel), (c) => c.charCodeAt(0)) : null;
  USED = SC ? Int32Array.from(SC.points.x, (x) => x.reduce((m, a, i) => (a > 0 ? m | (1 << i) : m), 0)) : null;
  offMask = 0;
  chips(); mchips();
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
  if (ok.length) $('pick').onclick = () => { selPt = null; select(ok[0][1]); };
  scopeTable(); table();
  if (selPt != null) selectPoint(selPt); else select(sel);
}

// ---- trade spaces: axes, actor combinations, 2-D Pareto front
// sub-indicators (results of older runs lack them: NaN)
const SUB = {
  relocated_households: ['移転世帯', '移転世帯 平均', -1], relocated_population: ['移転人口', '移転人口 平均 [人]', -1],
  service_access: ['サービスアクセス', 'サービスアクセス（4サービス平均, 0–1）', 1],
  natural_habitat_km2: ['自然生息地', '自然生息地面積 [km²]', 1], habitat_connectivity: ['生息地の連結性', '生息地の連結性（陸域, 0–1）', 1],
};
const subRows = (p, p0) => (p.sub && Object.keys(p.sub).length ? Object.entries(SUB).filter(([k]) => p.sub[k] != null)
  .map(([k, [, label]]) => `<tr><td>${label}</td><td>${fmt(p.sub[k], 2)}</td><td>${fmt(p0.sub?.[k], 2)}</td></tr>`).join('') : '');
const AX = {
  cost: { short: '費用', label: '費用 [億円, 現在価値]', dir: -1, pol: (p) => p.cost, pt: (j) => SC.points.cost[j] },
  rel: { short: '達成率', label: '達成率 [%]', dir: 1, pol: (p) => 100 * p._rel, pt: (j) => 100 * relAt(j) },
  dmean: { short: '被害額 平均', label: '被害額 平均 [億円]', dir: -1, pol: (p) => p.damage.mean, pt: (j) => SC.points.damage_mean[j] },
  dp90: { short: '被害額 90%値', label: '被害額 90%値 [億円]', dir: -1, pol: (p) => p.damage.p90, pt: (j) => SC.points.damage_p90[j] },
  pop: { short: '浸水人口', label: '浸水人口（0.5 m以上）平均 [人]', dir: -1, pol: (p) => p.pop_hazard.mean, pt: (j) => SC.points.pop[j] },
  ...Object.fromEntries(Object.entries(SUB).map(([k, [short, label, dir]]) => [k, { short, label, dir,
    pol: (p) => p.sub?.[k] ?? NaN, pt: (j) => SC.points[k]?.[j] ?? NaN }])),
  liv: { short: 'Livability（補助）', label: 'Livability（補助指標）', dir: 1, pol: (p) => p.livability, pt: (j) => SC.points.livability[j] },
  eco: { short: 'Ecology（補助）', label: 'Ecology（補助指標）', dir: 1, pol: (p) => p.ecology, pt: (j) => SC.points.ecology[j] },
};
function relIdx() {
  const t = +$('tgt').value, ti = SC.target_curve.findIndex((x) => Math.abs(x - t) < 1e-9), v = $('rain').value;
  return [Math.max(ti, 0), v === 'all' ? 0 : +v + 1];
}
function relAt(j, ri = relIdx()) {
  const [nt, ng] = SC.points.rel_shape;
  return REL[(j * nt + ri[0]) * ng + ri[1]] / SC.points.rel_scale;
}
const scopeLabel = (s) => (s.mask === (1 << SC.actors.length) - 1 ? `全主体（${s.label}）` : s.label);
const inScope = (j, m, off = offMask) => (SC.points.need[j] & ~m) === 0 && (USED[j] & off) === 0;
const kindOf = (v) => v.name.replace(/・(上流|中流|下流)$/, '').replace(/\d+$/, '');
function groups() {
  // measure kinds (堤防, 遊水地, 田んぼダム, ...) in variable order: [label, actor, bit mask of their variables]
  const g = new Map();
  R.variables.forEach((v, i) => { const k = kindOf(v); if (!g.has(k)) g.set(k, [k, SC.actor_of[i].split('・')[0], 0]); g.get(k)[2] |= 1 << i; });
  return [...g.values()];
}
function mchips() {
  if (!SC) { $('ms-chips').innerHTML = ''; return; }
  const A = SC.actors, box = (i) => `<label style="margin-right:6px;white-space:nowrap"><input type="checkbox" data-v="${i}" ${offMask & (1 << i) ? '' : 'checked'}> ${R.variables[i].name}</label>`;
  $('ms-chips').innerHTML = `<div>施策の有無: <button id="ms-all">すべて使う</button>${offMask ? ` <b style="color:#e34948">${R.variables.filter((v, i) => offMask & (1 << i)).length} 施策を除外中（点線＝除外なしの Pareto 線）</b>` : ''}</div>`
    + A.map((a) => `<div style="margin:2px 0 2px 12px"><b>${a}</b> <button data-a="${a}" data-on="1">全部</button> <button data-a="${a}" data-on="0">なし</button>：
      ${R.variables.map((v, i) => i).filter((i) => SC.actor_of[i] === a).map(box).join('')}</div>`).join('');
  $('ms-chips').querySelectorAll('input[data-v]').forEach((el) => el.onchange = () => { offMask ^= 1 << +el.dataset.v; mchips(); refresh(); });
  $('ms-chips').querySelectorAll('button[data-a]').forEach((el) => el.onclick = () => {
    SC.actor_of.forEach((a, i) => { if (a === el.dataset.a) offMask = el.dataset.on === '1' ? offMask & ~(1 << i) : offMask | (1 << i); });
    mchips(); refresh(); });
  $('ms-all').onclick = () => { offMask = 0; mchips(); refresh(); };
}
function chips() {
  shown.clear();
  if (!SC) { $('sc-chips').innerHTML = '<span class="hint">実施主体別のトレードスペースは未計算です（<code>python scripts/run_explore.py --scopes 実行名</code>）。探索の Pareto 解のみ表示します。</span>'; $('show-pol').checked = true; return; }
  const A = SC.actors, bit = (a) => 1 << A.indexOf(a), loc = A.filter((a) => a.includes('自治体'));
  const kuni = A.includes('国') ? bit('国') : 0;
  [kuni, ...loc.map((a) => kuni | bit(a)), kuni | loc.reduce((m, a) => m | bit(a), 0), (1 << A.length) - 1]
    .filter((m) => m && SC.scopes.some((s) => s.mask === m)).forEach((m) => shown.add(m));
  $('sc-chips').innerHTML = '表示する範囲: ' + SC.scopes.map((s) => `<label style="margin-right:8px;white-space:nowrap"><input type="checkbox" data-m="${s.mask}" ${shown.has(s.mask) ? 'checked' : ''}> ${scopeLabel(s)}</label>`).join('');
  $('sc-chips').querySelectorAll('input').forEach((el) => el.onchange = () => { el.checked ? shown.add(+el.dataset.m) : shown.delete(+el.dataset.m); refresh(); });
  $('show-pol').checked = false;
}
function front(js, X, Y) {
  // non-dominated on (X, Y) with their directions; returned in order of X
  const s = js.slice().sort((a, b) => X.dir * (X.pt(b) - X.pt(a)) || Y.dir * (Y.pt(b) - Y.pt(a)));
  const out = []; let best = -Infinity;
  for (const j of s) { const y = Y.dir * Y.pt(j); if (y > best + 1e-9) { out.push(j); best = y; } }
  return out.sort((a, b) => X.pt(a) - X.pt(b));
}
function scatter(a) {
  const X = AX[$('ax-x').value], Y = AX[$('ax-y').value], ds = [];
  const scopes = SC ? SC.scopes.filter((s) => shown.has(s.mask)).sort((p, q) => p.n_points - q.n_points) : [];
  const all = $('show-all').checked;
  scopes.forEach((s) => {
    const c = PAL[SC.scopes.indexOf(s) % PAL.length], js = [];
    for (let j = 0; j < SC.points.n; j++) if (inScope(j, s.mask)) js.push(j);
    const F = front(js, X, Y);
    if (offMask) {
      const js0 = [];
      for (let j = 0; j < SC.points.n; j++) if (inScope(j, s.mask, 0)) js0.push(j);
      const F0 = front(js0, X, Y);
      ds.push({ label: `${scopeLabel(s)}（除外なし）`, kind: 'pt', type: 'line', data: F0.map((j) => ({ x: X.pt(j), y: Y.pt(j), j })), borderColor: c + '90',
        borderDash: [5, 4], borderWidth: 1.2, pointRadius: 0, pointHitRadius: 4, showLine: true, order: 2 });
    }
    if (all) ds.push({ label: `${scopeLabel(s)}（全案）`, kind: 'pt', data: js.map((j) => ({ x: X.pt(j), y: Y.pt(j), j })), backgroundColor: c + '50',
      borderWidth: 0, pointRadius: 2, pointHoverRadius: 4, order: 3 });
    ds.push({ label: scopeLabel(s), kind: 'pt', type: 'line', data: F.map((j) => ({ x: X.pt(j), y: Y.pt(j), j })), borderColor: c, backgroundColor: c,
      pointRadius: F.map((j) => (j === selPt ? 7 : 3)), pointBorderColor: F.map((j) => (j === selPt ? '#0b0b0b' : c)), pointBorderWidth: F.map((j) => (j === selPt ? 2.5 : 1)),
      borderWidth: 1.8, stepped: false, showLine: true, order: 1 });
  });
  if ($('show-pol').checked) {
    const P = R.policies, eco = P.map((p) => p.ecology), lo = Math.min(...eco), hi = Math.max(...eco);
    const col = (e) => { const f = hi > lo ? (e - lo) / (hi - lo) : 0.5; return `rgba(${Math.round(200 - 160 * f)},${Math.round(220 - 100 * f)},${Math.round(240 - 30 * f)},0.95)`; };
    ds.push({ label: '探索の Pareto 解', kind: 'pol', data: P.map((p, i) => ({ x: X.pol(p), y: Y.pol(p), i })), pointBackgroundColor: P.map((p) => col(p.ecology)),
      pointBorderColor: P.map((p, i) => i === sel && selPt == null ? '#0b0b0b' : '#2a78d6'), pointBorderWidth: P.map((p, i) => i === sel && selPt == null ? 2.5 : 1),
      pointRadius: P.map((p, i) => i === sel && selPt == null ? 7 : 4), order: 2 });
  }
  const xs = ds.flatMap((d) => d.data.map((p) => p.x)), xmin = Math.min(...xs), xmax = Math.max(...xs);
  if ($('ax-y').value === 'rel') ds.push({ label: `必要な達成率 ${Math.round(a * 100)}%`, kind: 'ref', type: 'line', data: [{ x: xmin, y: 100 * a }, { x: xmax, y: 100 * a }],
    borderColor: '#eb6834', borderDash: [5, 4], pointRadius: 0, borderWidth: 1.5, order: 0 });
  chart('scatter', {
    type: 'scatter', data: { datasets: ds },
    options: { animation: false, onClick: (e, els) => {
        const el = els.find((x) => ['pt', 'pol'].includes(ds[x.datasetIndex].kind));
        if (!el) return;
        const d = ds[el.datasetIndex], p = d.data[el.index];
        if (d.kind === 'pol') { selPt = null; select(p.i); } else selectPoint(p.j);
      },
      plugins: { legend: { labels: { filter: (it) => !/（(全案|除外なし)）$/.test(it.text) } },
        tooltip: { callbacks: { label: (c) => { const d = ds[c.datasetIndex], p = c.raw;
          if (d.kind === 'ref') return '';
          return `${d.kind === 'pol' ? `案${p.i}` : d.label}: ${X.short} ${fmt(p.x, 2)}, ${Y.short} ${fmt(p.y, 2)}`; } } } },
      scales: { x: { title: { display: true, text: X.label } }, y: { title: { display: true, text: Y.label }, ...($('ax-y').value === 'rel' ? { min: 0, max: 100 } : {}) } } },
  });
  $('sc-note').textContent = SC ? `範囲ごとに代理モデル上の NSGA-II で探索（${SC.n_evaluated} 案を評価、${SC.points.n} 案を表示用に保存）。`
    + `その範囲の主体が実施できる施策だけを使う案。区分: ${SC.actors.map((a) => `${a}＝${[...new Set(R.variables.filter((v, i) => SC.actor_of[i] === a).map((v) => v.name.replace(/・(上流|中流|下流)$/, '')))].join('・')}`).join('／')}。` : '';
}

const descOf = (x) => R.variables.map((v, i) => [v, x[i]]).filter(([v, xi]) => xi > 0)
  .map(([v, xi]) => (v.continuous ? `${v.name}: ${Math.round(100 * xi)}%` : `${v.name}: ${v.options[xi]}`));
function scopeTable() {
  if (!SC) { $('sc-tab').innerHTML = ''; $('sc-goal').textContent = ''; return; }
  const a = +$('alpha').value / 100, ri = relIdx(), A = SC.actors;
  $('sc-goal').textContent = `（被害額 ≤ ${(+$('tgt').value).toFixed(1)} × 基準、達成率 ≥ ${Math.round(a * 100)}%、降雨倍率 ${$('rain').selectedOptions[0].text}）`;
  const best = (m) => { let b = -1, top = 0;
    for (let j = 0; j < SC.points.n; j++) if (inScope(j, m)) { const r = relAt(j, ri); top = Math.max(top, r); if (r >= a - 1e-9 && (b < 0 || SC.points.cost[j] < SC.points.cost[b])) b = j; }
    return [b, top]; };
  const full = best((1 << A.length) - 1)[0];
  const rows = SC.scopes.filter((s) => shown.has(s.mask)).map((s) => [s, ...best(s.mask)]);
  $('sc-tab').innerHTML = `<tr><th>範囲</th><th>最小の費用 [億円]</th><th>全主体比</th>${A.map((x) => `<th>うち ${x}</th>`).join('')}<th>達成率</th><th>最大の達成率</th><th>被害額 平均</th><th>Livability</th><th>Ecology</th><th style="text-align:left">施策</th></tr>`
    + rows.map(([s, j, top]) => j < 0
      ? `<tr><td>${scopeLabel(s)}</td><td colspan="${A.length + 3}" style="text-align:left">この範囲の施策では達成できない</td><td>${pct(top)}</td><td colspan="4"></td></tr>`
      : `<tr data-j="${j}" class="${j === selPt ? 'rec' : ''}" style="cursor:pointer"><td>${scopeLabel(s)}</td><td><b>${fmt(SC.points.cost[j])}</b></td>
         <td>${full >= 0 ? `${fmt(SC.points.cost[j] / SC.points.cost[full], 2)}×` : '–'}</td>${A.map((x, k) => `<td>${fmt(SC.points.cost_actor[j][k])}</td>`).join('')}
         <td>${pct(relAt(j, ri))}</td><td>${pct(top)}</td><td>${fmt(SC.points.damage_mean[j])}</td><td>${fmt(SC.points.livability[j], 2)}</td><td>${fmt(SC.points.ecology[j], 2)}</td>
         <td style="text-align:left;font-size:11px">${descOf(SC.points.x[j]).join('、') || '施策なし'}</td></tr>`).join('');
  $('sc-tab').querySelectorAll('tr[data-j]').forEach((tr) => tr.onclick = () => selectPoint(+tr.dataset.j));
  measureTable();
}

function measureTable() {
  // effect of each measure kind within the widest displayed actor combination
  const a = +$('alpha').value / 100, ri = relIdx();
  const scs = SC.scopes.filter((s) => shown.has(s.mask));
  if (!scs.length) { $('ms-tab').innerHTML = ''; $('ms-goal').textContent = ''; return; }
  const s = scs.reduce((p, q) => (q.actors.length > p.actors.length ? q : p));
  const allBits = (1 << R.variables.length) - 1;
  const best = (off) => { let b = -1, top = 0;
    for (let j = 0; j < SC.points.n; j++) if (inScope(j, s.mask, off)) { const r = relAt(j, ri); top = Math.max(top, r); if (r >= a - 1e-9 && (b < 0 || SC.points.cost[j] < SC.points.cost[b])) b = j; }
    return [b, top]; };
  const [b0, t0] = best(offMask);
  const c0 = b0 >= 0 ? SC.points.cost[b0] : null;
  $('ms-goal').textContent = `（範囲: ${scopeLabel(s)}、除外中の施策を除いた最小の費用 ${c0 == null ? '達成できない' : `${fmt(c0)} 億円`}、最大の達成率 ${pct(t0)}）`;
  const cell = (b, top, ref) => b < 0 ? `<td>達成できない</td><td></td><td>${pct(top)}</td>`
    : `<td>${fmt(SC.points.cost[b])}</td><td>${ref == null ? '' : `${SC.points.cost[b] - ref >= 0 ? '+' : ''}${fmt(SC.points.cost[b] - ref)}`}</td><td>${pct(top)}</td>`;
  // stored policies that use the measure's actor but not the measure: few -> the exclusion also drops that actor's other measures
  const actBits = (act) => SC.actor_of.reduce((m, x, i) => (x.split('・')[0] === act ? m | (1 << i) : m), 0);
  const thin = (act, bits) => { let n = 0, k = 0;
    for (let j = 0; j < SC.points.n; j++) if (inScope(j, s.mask) && (USED[j] & actBits(act) & ~bits)) { n++; if (!(USED[j] & bits)) k++; }
    return n > 0 && k / n < 0.05; };
  let flagged = 0;
  const rows = groups().filter(([, act]) => s.actors.some((x) => x.split('・')[0] === act)).map(([k, act, bits]) => {
    const used = (bits & ~offMask) !== 0, weak = used && thin(act, bits);
    if (weak) flagged++;
    const [bw, tw] = best(offMask | bits), [bo, to] = best(offMask | (allBits & ~bits));
    return `<tr${used ? (weak ? ' style="font-style:italic;color:#8a6d00"' : '') : ' style="color:#a3a29b"'}><td style="text-align:left">${k}${used ? (weak ? ' ※' : '') : '（除外中）'}</td><td>${act}</td>${cell(bw, tw, c0)}<td>${fmt(100 * (t0 - tw), 0)} pt</td>${used ? cell(bo, to, null) : '<td></td><td></td><td></td>'}</tr>`;
  });
  $('ms-tab').innerHTML = `<tr><th style="text-align:left" rowspan="2">施策</th><th rowspan="2">主体</th><th colspan="4">除いたとき</th><th colspan="3">それだけ</th></tr>
    <tr><th>最小の費用</th><th>費用の増加</th><th>最大の達成率</th><th>達成率の低下</th><th>最小の費用</th><th></th><th>最大の達成率</th></tr>` + rows.join('')
    + (flagged ? `<tr><td colspan="9" style="text-align:left;color:#8a6d00">※ 保存した案では、この施策の主体の他の施策を使う案のほとんどがこの施策も使っている（実施率がほぼ 0 にならない）ため、
      「除いたとき」は同じ主体の他の施策もほぼ除いた結果になり、効果を過大に見積もる。正確な比較には、その施策を除いた最適化（代理モデルのみ）が必要。</td></tr>` : '');
}

function selectPoint(j) {
  selPt = j;
  const P = SC.points, p0 = R.policies[0], A = SC.actors, ri = relIdx();
  const need = A.filter((a, k) => P.need[j] & (1 << k));
  $('sel-name').textContent = `実施主体別の案（必要な主体: ${need.join('・') || 'なし'}）`;
  $('detail').innerHTML = `<ul class="desc">${(descOf(P.x[j]).length ? descOf(P.x[j]) : ['施策なし']).map((d) => `<li>${d}</li>`).join('')}</ul>
    <table class="tbl"><tr><th>実施主体</th>${A.map((a) => `<th>${a}</th>`).join('')}<th>計</th></tr>
    <tr><td>費用 [億円]</td>${A.map((a, k) => `<td>${fmt(P.cost_actor[j][k])}</td>`).join('')}<td><b>${fmt(P.cost[j])}</b></td></tr></table>
    <table class="tbl"><tr><th>指標</th><th>この案</th><th>施策なし</th></tr>
    <tr><td>達成率（選んだ目標・降雨倍率）</td><td>${pct(relAt(j, ri))}</td><td>${pct(p0._rel)}</td></tr>
    <tr><td>被害額 平均／90%値 [億円]</td><td>${fmt(P.damage_mean[j])}／${fmt(P.damage_p90[j])}</td><td>${fmt(p0.damage.mean)}／${fmt(p0.damage.p90)}</td></tr>
    <tr><td>浸水人口（0.5 m以上） 平均</td><td>${fmt(P.pop[j])}</td><td>${fmt(p0.pop_hazard.mean)}</td></tr>
    <tr><td>床上浸水世帯 平均</td><td>${fmt(P.households[j])}</td><td>${fmt(p0.households)}</td></tr>
    <tr><td>Livability</td><td>${fmt(P.livability[j], 2)}</td><td>${fmt(p0.livability, 2)}</td></tr>
    <tr><td>Ecology</td><td>${fmt(P.ecology[j], 2)}</td><td>${fmt(p0.ecology, 2)}</td></tr></table>
    <div class="hint">地域別の被害額・被害額の分布は探索の Pareto 解のみ（トレードスペースで「探索の Pareto 解」を表示してクリック）。</div>`;
  const e = R.rain_edges, labels = e.slice(0, -1).map((x, b) => `${x.toFixed(2)}–${e[b + 1].toFixed(2)}`);
  const byBin = (q) => labels.map((_, b) => { const m = R.ensemble.rain_factor.map((x) => x >= e[b] && (x < e[b + 1] || b === e.length - 2)); return 100 * rel(q, +$('tgt').value, m); });
  chart('byrain', { type: 'line', data: { labels, datasets: [
    { label: 'この案', data: labels.map((_, b) => 100 * relAt(j, [ri[0], b + 1])), borderColor: '#2a78d6', backgroundColor: '#2a78d6', borderWidth: 2 },
    { label: '施策なし', data: byBin(p0), borderColor: '#a3a29b', backgroundColor: '#a3a29b', borderWidth: 2 }] },
    options: { animation: false, scales: { y: { min: 0, max: 100, title: { display: true, text: '達成率 [%]' } }, x: { title: { display: true, text: '降雨倍率' } } } } });
  scatter(+$('alpha').value / 100);
  document.querySelectorAll('#ptab tr[data-i]').forEach((tr) => tr.classList.remove('rec'));
  $('sc-tab').querySelectorAll('tr[data-j]').forEach((tr) => tr.classList.toggle('rec', +tr.dataset.j === selPt));
}

function select(i) {
  sel = i; selPt = null;
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
    ${subRows(p, p0)}
    <tr><td>Livability（補助指標）</td><td>${fmt(p.livability, 2)}</td><td>${fmt(p0.livability, 2)}</td></tr>
    <tr><td>Ecology（補助指標）</td><td>${fmt(p.ecology, 2)}</td><td>${fmt(p0.ecology, 2)}</td></tr></table>
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
  scatter(+$('alpha').value / 100);
  document.querySelectorAll('#ptab tr[data-i]').forEach((tr) => tr.classList.toggle('rec', +tr.dataset.i === sel));
  if (SC) $('sc-tab').querySelectorAll('tr[data-j]').forEach((tr) => tr.classList.remove('rec'));
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
