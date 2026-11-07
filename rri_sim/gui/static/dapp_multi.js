// Multi-Actor Robust DAPP viewer (results.json of `scripts/run_dapp.py --multi`, docs/11 11.10): Individual / Joint / Single
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : `${(100 * v).toFixed(d)}%`);
const sgn = (v, d = 0, u = '') => (v == null || !Number.isFinite(v) ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(d)}${u}`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const charts = {};
function chart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); return charts[id]; }
const metCol = (m) => `hsl(${Math.round(120 * m)}, 65%, ${m > 0.5 ? 38 : 48}%)`;
const ACOL = ['#0e7490', '#16a34a', '#b45309', '#7c3aed', '#db2777'];
const REASON = { observed: '観測した指標', projected: '予測（観測した傾向・リードタイム）', switch: '経路 B への切替' };
const CASE = { individual: 'Individual', joint: 'Joint', single: 'Single' };
const CASE_JA = { individual: '各主体が自分の目的で探索', joint: '主体別の予算・権限のまま流域全体で同時に探索', single: '全施策と期別容量を1主体に統合して探索' };
const DASH = { individual: [], joint: [6, 4], single: [2, 3] };
let D = null, A = [], acol = {}, SELC = 'individual', NETC = 'individual';

export function render(data) {
  D = data; A = D.actors.map((a) => a.name); acol = Object.fromEntries(A.map((n, i) => [n, ACOL[i % ACOL.length]]));
  acol[D.single_actor.name] = '#334155';
  $('meta').textContent = `作成 ${D.created}・計算 ${fmt(D.elapsed_s / 60, 1)} 分・代理モデル ${D.physics.explore_run}・主体 ${A.length}・評価シナリオ ${D.worlds.length}（探索 ${D.n_search_scenarios}）`;
  const opt = (sel, items, cb) => { $(sel).innerHTML = items.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join(''); $(sel).onchange = cb; };
  opt('m-actor', D.actors.map((a) => [a.name, a.label]), drawMap);
  opt('m-conf', [['individual', 'Individual（個別最適）'], ['joint', 'Joint（流域全体で同時探索）']], (e) => { SELC = e.target.value; drawMap(); });
  opt('m-net-conf', Object.entries(CASE), (e) => { NETC = e.target.value; drawNet(); });
  opt('m-bud-case', Object.entries(CASE), drawBudget);
  opt('m-dep-actor', D.actors.map((a) => [a.name, a.label]), drawDep);
  opt('m-front-actor', [...D.actors.map((a) => [a.name, `Individual: ${a.label}`]), ['joint', 'Joint（全主体の戦略）'], ['single', 'Single（統合主体）']], drawFront);
  const rain = D.worlds.map((w) => w.rain_2100), ord = rain.map((r, s) => s).sort((a, b) => rain[a] - rain[b]);
  opt('m-scen', ord.map((s) => [s, `シナリオ ${s}（2100年 降雨 ×${fmt(rain[s], 3)}・海面 ${fmt(D.worlds[s].slr_2100, 2)} m）`]), drawSeries);
  $('m-scen').value = ord[Math.floor(ord.length / 2)];
  opt('m-unr', Object.keys(CASE).flatMap((c) => ['all', ...A].map((k) => [`${c}|${k}`, `${CASE[c]}: ${k === 'all' ? '全主体の目標' : `${k}の目標`}`])), drawUnr);
  drawInputs(); drawHistory(); drawSummary(); drawCases(); drawFrontier(); drawBudget(); drawMap(); drawSingleMap(); drawNet(); drawComp(); drawDep(); drawSpill(); drawSignal(); drawFront(); drawSeries(); drawUnr(); drawVer();
}
const actor = (n) => D.actors.find((a) => a.name === n) || (D.single_actor.name === n ? D.single_actor : null);
const C = (k) => D.comparison[k];
const metricText = (a) => (a.metric === 'basin' ? '流域の被害額・浸水人口' : `${a.metric}の被害額`);

function stratHTML(s, a, pooled = false) {
  const li = (xs) => (xs.length ? xs.map(esc).join(' → ') : '<span class="hint">なし</span>');
  const x = actor(a), b = pooled ? D.single_actor.budget : x.budget;
  return `<div style="margin:2px 0 6px"><b style="color:${acol[a]}">${esc(pooled ? `${a}の施策群（Single が決定）` : x.label)}</b>${s.rule ? `（ルール: ${esc(s.rule)}）` : ''}
    <span class="hint">容量 ${fmt(b.per_decade_oku)} 億円/10年${pooled ? '（全施策で共有）' : ''}${b.total_oku != null ? `・総額 ${fmt(b.total_oku)} 億円` : ''}</span>
    <div>今実施すべき初期施策: <b>${s.init.length ? s.init.map(esc).join('、') : '施策なし（様子見）'}</b></div>
    <div>経路 A（基本）: ${li(s.A)}</div>
    <div>経路 B（厳しい気候）: ${li(s.B)}${s.switch != null ? `<span class="hint">（観測した降雨の傾向から予測した2100年の降雨倍率が ×${fmt(s.switch, 2)} 以上で切替）</span>` : '<span class="hint">（切り替えない）</span>'}</div>
    <div class="hint">トリガー: ${metricText(x)} が 目標 × ${fmt(s.rho, 2)} を超えたら次の行動（${s.anticipate ? '観測、または観測した傾向からリードタイム＋1期 後の予測' : '観測のみ'}）</div></div>`;
}

// ---------------------------------------------------------------- conditions (inputs) -> Recalculate
const WKEYS = { damage: '被害', pop: '浸水人口', cost: '費用', ecology: 'Ecology', livability: 'Livability', agriculture: '農業' };
const SIZE = { fast: '速い（評価 32・探索 16, 約1–2分）', standard: '標準（評価 64・探索 32, 約4分）', full: '本計算（評価 64・探索 48, 約11分）' };
const base = () => (D.settings.multi.gui_inputs || {}).base || D.name;
function drawInputs() {
  const num = (id, v, step, attrs = '') => `<input type="number" id="${id}" value="${v ?? ''}" step="${step}" ${attrs} style="width:72px">`;
  const rows = D.actors.map((a, i) => `<tr><td style="color:${acol[a.name]}"><b>${esc(a.name)}</b><br><span class="hint">${esc(metricText(a))}</span></td>
    <td>${num(`in-b-${i}`, a.budget.per_decade_oku, 10, 'min="1"')}</td><td>${num(`in-d-${i}`, a.targets.damage_ratio, 0.05, 'min="0.3" max="5"')}</td>
    <td>${a.metric === 'basin' ? num(`in-p-${i}`, a.targets.pop_ratio, 0.05, 'min="0.3" max="5"') : '<span class="hint">—（地域別なし）</span>'}</td></tr>`).join('');
  const wrow = (label, id, w) => `<tr><td>${label}</td>${Object.keys(WKEYS).map((k) => `<td>${num(`${id}-${k}`, w[k] ?? 0, 0.05, 'min="0"')}</td>`).join('')}</tr>`;
  $('m-inputs').innerHTML = `<table class="tbl" id="m-in-tbl"><tr><th>主体</th><th>per_decade_oku<br><span class="hint">億円/10年</span></th><th>目標: 被害額の比<br><span class="hint">÷ 対策なし現在気候</span></th><th>目標: 浸水人口の比</th></tr>${rows}
    <tr><td><b>Single</b><br><span class="hint">3主体の合計（自動）</span></td><td id="in-single"></td><td colspan="2" class="hint">目標は各主体のもの（全主体の目標を同時に満たす割合で評価）</td></tr></table>
    <details style="margin-top:6px"><summary>詳細設定: Preference Weights（補助指標の Utility / Regret 用。Coverage には影響しない）</summary>
      <table class="tbl"><tr><th></th>${Object.values(WKEYS).map((l) => `<th>${l}</th>`).join('')}</tr>
      ${D.actors.map((a, i) => wrow(esc(a.name), `in-w-${i}`, a.weights)).join('')}${wrow('流域（全体）', 'in-w-basin', D.settings.robust.weights)}</table>
      <div class="hint">重みは Regret と、探索の目的の1つ（平均効用）に使う。主体の重みは自分の効用、流域の重みは流域の効用（Joint・Single の探索と流域の Regret）。</div></details>
    <div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">計算の規模 <select id="in-size">${Object.entries(SIZE).map(([k, l]) => `<option value="${k}">${l}</option>`).join('')}</select>
      <button id="in-go" style="font-weight:700;padding:4px 14px">Recalculate</button><button id="in-reset">表示中の条件に戻す</button></div>
    <div id="in-note" class="hint" style="margin-top:4px"></div>
    <div id="in-prog" style="display:none;margin-top:6px"><div style="height:8px;background:#e2e8f0;border-radius:4px"><div id="in-bar" style="height:8px;width:0;background:#0e7490;border-radius:4px"></div></div>
      <div id="in-msg" class="hint"></div></div>`;
  $('in-size').value = (D.settings.multi.gui_inputs || {}).size === 'fast' ? 'fast' : 'standard';
  $('m-inputs').querySelectorAll('input').forEach((el) => { el.oninput = markChanges; });
  $('in-reset').onclick = drawInputs; $('in-go').onclick = recalculate;
  markChanges();
}
function readInputs() {
  const v = (id) => { const el = $(id); return el && el.value !== '' ? +el.value : null; };
  const w = (id) => Object.fromEntries(Object.keys(WKEYS).map((k) => [k, v(`${id}-${k}`) ?? 0]).filter(([, x]) => x > 0));
  return { actors: Object.fromEntries(D.actors.map((a, i) => [a.name, { per_decade_oku: v(`in-b-${i}`),
    targets: { damage_ratio: v(`in-d-${i}`), ...(a.metric === 'basin' ? { pop_ratio: v(`in-p-${i}`) } : {}) }, weights: w(`in-w-${i}`) }])),
  basin_weights: w('in-w-basin') };
}
function markChanges() {          // inputs that differ from the displayed result: yellow, and the Single capacity (sum)
  const I = readInputs(), same = (a, b) => Math.abs((a ?? 0) - (b ?? 0)) < 1e-9;
  let n = 0;
  const mark = (id, changed) => { const el = $(id); if (el) { el.style.background = changed ? '#fef08a' : ''; n += changed; } };
  D.actors.forEach((a, i) => {
    const x = I.actors[a.name];
    mark(`in-b-${i}`, !same(x.per_decade_oku, a.budget.per_decade_oku)); mark(`in-d-${i}`, !same(x.targets.damage_ratio, a.targets.damage_ratio));
    if (a.metric === 'basin') mark(`in-p-${i}`, !same(x.targets.pop_ratio, a.targets.pop_ratio));
    Object.keys(WKEYS).forEach((k) => mark(`in-w-${i}-${k}`, !same(x.weights[k], a.weights[k])));
  });
  Object.keys(WKEYS).forEach((k) => mark(`in-w-basin-${k}`, !same(I.basin_weights[k], D.settings.robust.weights[k])));
  const tot = Object.values(I.actors).reduce((t, x) => t + (x.per_decade_oku || 0), 0);
  $('in-single').innerHTML = `<b>${fmt(tot)}</b> 億円/10年${same(tot, D.single_actor.budget.per_decade_oku) ? '' : ` <span style="background:#fef08a">（表示中 ${fmt(D.single_actor.budget.per_decade_oku)}）</span>`}`;
  $('in-note').innerHTML = n ? `<b style="color:#b45309">未計算の変更 ${n} 件</b>（黄色）→ Recalculate で3ケースを再計算する。` : '表示中の結果の条件。値を変えて Recalculate。';
}
async function recalculate() {
  const body = { name: D.name, size: $('in-size').value, ...readInputs() };
  $('in-go').disabled = true; $('in-prog').style.display = ''; $('in-msg').textContent = '送信中'; $('in-bar').style.width = '0';
  try {
    const r = await fetch('/api/dapp/multi/recalculate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    if (j.job) {
      for (;;) {
        await new Promise((ok) => setTimeout(ok, 1500));
        const st = await (await fetch(`/api/job/${j.job}`)).json();
        $('in-bar').style.width = `${Math.round(100 * st.progress)}%`;
        $('in-msg').textContent = `${Math.round(100 * st.progress)}%・${Math.round(st.elapsed_s)} 秒・${st.message}`;
        if (st.status === 'done') break;
        if (st.status === 'error') throw new Error(st.message.split('\n')[0]);
      }
    } else { $('in-msg').textContent = '同じ条件の結果があるため、それを表示する。'; }
    await window.dappReload(j.name);
  } catch (e) {
    $('in-msg').innerHTML = `<span style="color:#b91c1c">エラー: ${esc(e.message)}</span>`; $('in-go').disabled = false;
  }
}
async function drawHistory() {
  const rows = await (await fetch(`/api/dapp/multi/summaries/${encodeURIComponent(base())}`)).json();
  const cur = D.name, A0 = D.actors;
  const t = (x, a) => `${fmt(x.targets.damage_ratio, 2)}${x.targets.pop_ratio != null ? `/${fmt(x.targets.pop_ratio, 2)}` : ''}`;
  const diff = (r, a, f) => (f(r.actors[a.name] || {}) !== f(D.actors.find((x) => x.name === a.name)) ? ' style="background:#fef9c3"' : '');
  const gp = (g) => `<b style="color:${g.coverage >= 0 ? '#15803d' : '#b91c1c'}">${sgn(100 * g.coverage, 0, ' pt')}</b><br><span class="hint">${sgn(100 * g.ci90[0], 0)}〜${sgn(100 * g.ci90[1], 0)}</span>`;
  $('m-history').innerHTML = `<tr><th rowspan="2">実行（作成）</th><th rowspan="2">規模</th><th colspan="${A0.length + 1}">per_decade_oku（億円/10年）</th><th colspan="${A0.length}">目標（被害/人口）</th>
    <th colspan="3">Coverage</th><th rowspan="2">Coordination Gap<br>Joint − Individual</th><th rowspan="2">Institutional Fragmentation Gap<br>Single − Joint</th><th rowspan="2">繰越<br>I / J / S</th></tr>
    <tr>${A0.map((a) => `<th>${esc(a.name)}</th>`).join('')}<th>Single</th>${A0.map((a) => `<th>${esc(a.name)}</th>`).join('')}<th>Individual</th><th>Joint</th><th>Single</th></tr>`
    + rows.map((r) => `<tr data-n="${esc(r.name)}" class="${r.name === cur ? 'rec' : ''}"><td>${esc(r.name === r.base ? '基準（CLI）' : r.name.slice(r.base.length + 1))}<br><span class="hint">${esc(r.created)}</span></td>
      <td>${esc(r.size)}<br><span class="hint">${r.n_eval}/${r.n_search}</span></td>
      ${A0.map((a) => `<td${diff(r, a, (x) => x.per_decade_oku ?? x.budget?.per_decade_oku)}>${fmt(r.actors[a.name]?.per_decade_oku)}</td>`).join('')}<td>${fmt(r.single_per_decade_oku)}</td>
      ${A0.map((a) => `<td${diff(r, a, (x) => JSON.stringify(x.targets))}>${r.actors[a.name] ? t(r.actors[a.name], a) : '—'}</td>`).join('')}
      ${['individual', 'joint', 'single'].map((k) => `<td style="color:${metCol(r.coverage[k])};font-weight:700">${pct(r.coverage[k])}</td>`).join('')}
      <td>${gp(r.gap.coordination_gap)}</td><td>${gp(r.gap.institutional_fragmentation_gap)}</td>
      <td>${['individual', 'joint', 'single'].map((k) => (r.deferred[k] ? '有' : '—')).join(' / ')}</td></tr>`).join('');
  $('m-history').querySelectorAll('tr[data-n]').forEach((tr) => { tr.onclick = () => { if (tr.dataset.n !== cur) window.dappReload(tr.dataset.n); }; });
}

// ---------------------------------------------------------------- the three cases
function drawSummary() {
  const cv = (k) => C(k).basin.coverage, g = D.gap, cg = g.coordination_gap, fg = g.institutional_fragmentation_gap;
  const box = (k) => `<div class="kpi" style="min-width:150px;flex:1;text-align:center"><div class="l"><b>${CASE[k]}</b><br>${CASE_JA[k]}</div>
    <div class="l">Coverage</div><div class="v" style="font-size:32px;color:${metCol(cv(k))}">${pct(cv(k))}</div>
    <div class="l">総費用 ${fmt(C(k).basin.mean_cost)} 億円<br>最大 Regret ${fmt(C(k).basin.max_regret, 3)}</div></div>`;
  const arrow = (gp, name) => `<div style="text-align:center;min-width:120px"><div style="font-size:24px">→</div>
    <div style="font-size:18px;font-weight:700;color:${gp.coverage >= 0 ? '#15803d' : '#b91c1c'}">${sgn(100 * gp.coverage, 0, ' pt')}</div>
    <div class="hint">90%区間 ${sgn(100 * gp.coverage_ci90[0], 0)}〜${sgn(100 * gp.coverage_ci90[1], 0, ' pt')}</div><div class="l hint">${name}</div>
    <div class="hint">費用 ${sgn(gp.mean_cost, 0, ' 億円')}</div></div>`;
  const M = D.settings.multi, gi = M.gui_inputs || {};
  const cond = D.actors.map((a) => `${esc(a.name)} ${fmt(a.budget.per_decade_oku)}・目標 被害 ${fmt(a.targets.damage_ratio, 2)}${a.targets.pop_ratio != null ? `/人口 ${fmt(a.targets.pop_ratio, 2)}` : ''}`).join('　');
  $('m-summary').innerHTML = `<h3 style="font-size:14px;margin:0 0 6px">Coverage（全主体の目標を満たす割合）: 意思決定構造を統合すると robustness はどう変わるか</h3>
    <div class="hint" style="margin-bottom:6px">この結果の条件（億円/10年）: ${cond}　Single ${fmt(D.single_actor.budget.per_decade_oku)}・規模 ${esc(gi.size || 'full')}（評価 ${D.worlds.length}・探索 ${D.n_search_scenarios} シナリオ）</div>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">${box('individual')}${arrow(cg, 'coordination gap<br>（Joint − Individual）')}${box('joint')}${arrow(fg, 'institutional fragmentation gap<br>（Single − Joint）')}${box('single')}</div>
    <p class="hint">${pct(cv('individual'))} → ${pct(cv('joint'))} → ${pct(cv('single'))}。3ケースとも同じ評価シナリオ・同じ目標・同じ期別の財政容量の合計（${fmt(D.single_actor.budget.per_decade_oku)} 億円/10年）。
    総額の上限は置かず（per_decade_oku が制約）、ケースの違いは「誰が・どの予算で・何を決めるか」だけ。Single は Joint と同じ目的で、各主体の施策群の戦略（経路・分岐・トリガー）を
    1主体が同時に決め、期別の容量を全施策で共有する（Joint の戦略はすべて Single でも選べる）。</p>
    <details><summary>各ケースの推奨戦略（今実施すべき初期施策・分岐）</summary>
      ${Object.keys(CASE).map((k) => `<h3 style="font-size:13px;margin:8px 0 2px">${CASE[k]}</h3>${Object.entries(C(k).strategies).map(([n, s]) => stratHTML(s, n, k === 'single')).join('')}`).join('')}</details>`;
}
function drawCases() {
  const K = Object.keys(CASE), g = D.gap;
  const row = (label, f, d = 0, isPct = false, gapF = null) => {
    const v = K.map((k) => f(C(k)));
    const show = (x) => (isPct ? pct(x) : fmt(x, d));
    const gp = (a, b) => (a == null || b == null ? '—' : isPct ? sgn(100 * (b - a), 0, ' pt') : sgn(b - a, d));
    return `<tr><td>${label}</td>${v.map((x) => `<td${isPct && x != null ? ` style="color:${metCol(x)}"` : ''}>${show(x)}</td>`).join('')}
      <td><b>${gapF ? gapF('coordination_gap') : gp(v[0], v[1])}</b></td><td><b>${gapF ? gapF('institutional_fragmentation_gap') : gp(v[1], v[2])}</b></td></tr>`;
  };
  $('m-cases').innerHTML = `<tr><th>指標</th>${K.map((k) => `<th>${CASE[k]}</th>`).join('')}<th>coordination gap<br>（Joint − Individual）</th><th>institutional fragmentation gap<br>（Single − Joint）</th></tr>`
    + row('<b>全主体の目標を満たす割合（流域の robustness）</b>', (c) => c.basin.coverage, 0, true,
      (k) => `${sgn(100 * g[k].coverage, 0, ' pt')}<br><span class="hint">90%区間 ${sgn(100 * g[k].coverage_ci90[0], 0)}〜${sgn(100 * g[k].coverage_ci90[1], 0)}・+${g[k].scenarios_gained}/−${g[k].scenarios_lost} シナリオ</span>`)
    + A.map((n) => row(`<span style="color:${acol[n]}">${esc(n)}</span>の目標`, (c) => c.actors[n].coverage, 0, true)).join('')
    + row('流域の被害・浸水人口の目標（国の目標と同じ）', (c) => c.basin.coverage_basin_metric, 0, true)
    + row('総費用 PV（シナリオ平均, 億円）', (c) => c.basin.mean_cost, 0)
    + A.map((n) => row(`　うち ${esc(n)}の負担`, (c) => c.actors[n].mean_cost, 0)).join('')
    + row('流域の最大 Regret', (c) => c.basin.max_regret, 3)
    + row('流域の平均 Regret', (c) => c.basin.mean_regret, 3)
    + A.map((n) => row(`${esc(n)}の最大 Regret`, (c) => c.actors[n].max_regret, 3)).join('')
    + row('Lock-in（2060年, 意思決定する主体の平均）', (c) => c.basin.lockin, 0, true);
}

function drawFrontier() {
  const F = D.frontier, col = { individual: '#64748b', joint: '#0e7490', single: '#b45309' };
  chart('m-frontier', { type: 'scatter', data: { datasets: [
    ...Object.keys(CASE).map((k) => ({ label: `${CASE[k]}${k === 'individual' ? '（推奨の組合せ）' : '（探索の前線）'}`, data: F[k].map((r) => ({ x: r.mean_cost, y: r.coverage, r })),
      showLine: k !== 'individual', borderColor: col[k], backgroundColor: col[k], pointRadius: F[k].map((r) => (r.chosen ? 9 : 4)),
      pointStyle: F[k].map((r) => (r.chosen ? 'star' : 'circle')) })),
    { label: 'いずれかのケースで達成できる割合（上限の目安）', data: [{ x: 0, y: F.any_case_coverage }, { x: Math.max(...['joint', 'single'].flatMap((k) => F[k].map((r) => r.mean_cost))) * 1.05, y: F.any_case_coverage }],
      showLine: true, borderColor: '#94a3b8', borderDash: [4, 4], pointRadius: 0 }] },
  options: { animation: false, maintainAspectRatio: false, plugins: { tooltip: { callbacks: { label: (c) => (c.raw.r ? [`${c.dataset.label}: カバー率 ${pct(c.raw.r.coverage)}・費用 ${fmt(c.raw.r.mean_cost)} 億円${c.raw.r.chosen ? '（推奨）' : ''}`,
      c.raw.r.search_coverage != null ? `探索シナリオでは ${pct(c.raw.r.search_coverage)}・${fmt(c.raw.r.search_cost)} 億円` : ''] : c.dataset.label) } } },
    scales: { x: { min: 0, title: { display: true, text: '総費用 PV（シナリオ平均, 億円）' } }, y: { min: 0, max: 1, title: { display: true, text: '全主体の目標を満たす割合' } } } } });
  const best = (k, c) => Math.max(...F[k].filter((r) => r.mean_cost <= c + 1e-9).map((r) => r.coverage), 0);
  const cj = C('joint').basin.mean_cost, ci = C('individual').basin.mean_cost;
  $('m-frontier-note').innerHTML = `★＝各ケースの推奨（探索シナリオで最高カバー率から許容幅内で最も安い戦略）。前線＝探索シナリオで カバー率・費用 が非劣な戦略を評価シナリオで再評価したもの。
    同じ費用で比べると: Individual の費用（${fmt(ci)} 億円）以下で Joint ${pct(best('joint', ci))}・Single ${pct(best('single', ci))}、
    Joint の推奨の費用（${fmt(cj)} 億円）以下で Single ${pct(best('single', cj))}（Joint ${pct(C('joint').basin.coverage)}）。点線＝3ケースのいずれかで目標を満たせた割合。`;
}

function drawBudget() {
  const k = $('m-bud-case').value || 'individual', B = D.budget[k], years = D.years;
  $('m-budget').innerHTML = Object.keys(B).map((a, i) => `<div><div style="font-size:12px;font-weight:600;color:${acol[a]}">${esc(a)}（${fmt(B[a].cap)} 億円/10年）</div><div style="position:relative;height:220px;max-width:${Object.keys(B).length === 1 ? '720px' : 'none'}"><canvas id="m-bud-${i}"></canvas></div></div>`).join('');
  Object.entries(B).forEach(([a, b], i) => {
    const top = Math.max(b.cap, ...b.max) * 1.15;
    chart(`m-bud-${i}`, { type: 'bar', data: { labels: years.map((y) => `${y}年代`), datasets: [
      { label: '着手した初期費用（平均）', data: b.mean, backgroundColor: acol[a] },
      { type: 'line', label: '90%値', data: b.p90, borderColor: acol[a], borderWidth: 1, borderDash: [3, 3], pointRadius: 2 },
      { type: 'line', label: '容量 per_decade_oku', data: years.map(() => b.cap), borderColor: '#dc2626', borderWidth: 1.5, pointRadius: 0 },
      { type: 'line', label: '繰越が生じた割合', data: b.deferred_share.map((x) => (x > 0 ? top * 0.96 : null)), showLine: false, pointStyle: 'triangle',
        pointRadius: b.deferred_share.map((x) => 3 + 8 * x), backgroundColor: '#f59e0b', borderColor: '#b45309' }] },
    options: { animation: false, maintainAspectRatio: false, plugins: { legend: { display: i === 0, labels: { boxWidth: 10, font: { size: 10 } } },
      tooltip: { callbacks: { afterBody: (it) => `繰越が生じたシナリオ ${pct(b.deferred_share[it[0].dataIndex])}・最大 ${fmt(b.max[it[0].dataIndex])} 億円` } } },
    scales: { y: { min: 0, max: top, title: { display: true, text: '億円' } } } } });
  });
  const tot = Object.values(B).map((b) => b.mean.reduce((x, y) => x + y, 0)).reduce((x, y) => x + y, 0);
  $('m-budget-note').textContent = `${CASE[k]}: 2020〜2100年に着手した初期費用の合計 平均 ${fmt(tot)} 億円（割引前）。容量は着手した年の初期費用（変更費用を含む）に掛かり、超える要求は翌期に繰り越す（1件で容量を超える事業は、その期に他の着手がなければ着手できる）。総額の上限は置かない。`;
}

// ---------------------------------------------------------------- pathways maps (actors: Individual / Joint; Single: the integrated actor)
function drawMap() {
  const a = $('m-actor').value, s = C(SELC).strategies[a];
  $('m-strat').innerHTML = stratHTML(s, a);
  drawTree('m-map', 'm-branch', D.trees[SELC][a], actor(a), () => s.switch);
}
function drawSingleMap() {
  const S = C('single').strategies;
  $('m-sstrat').innerHTML = Object.entries(S).map(([n, s]) => stratHTML(s, n, true)).join('');
  drawTree('m-smap', 'm-sbranch', D.trees.single[D.single_actor.name], D.single_actor, (b) => S[b.group]?.switch);
}
function drawTree(canvasId, tableId, tree, act, switchOf) {
  const tot = tree[0].n;
  const minN = Math.max(1, Math.round(0.03 * tot));
  const show = new Set(tree.filter((b) => b.n >= minN).map((b) => b.id));
  const T = tree.filter((b) => show.has(b.id)), byId = Object.fromEntries(tree.map((b) => [b.id, b]));
  const rowKey = (b) => (b.kind === 'act' ? `a${b.a}` : b.kind === 'initial' ? 'init' : rowKey(byId[b.parent]));
  const first = {};                    // rows in the order the actions first appear (median year), then by the catalog
  T.filter((b) => b.kind === 'act').forEach((b) => { first[b.a] = Math.min(first[b.a] ?? 9999, b.year); });
  const rows = ['init', ...Object.keys(first).map(Number).sort((x, y) => first[x] - first[y] || x - y).map((x) => `a${x}`)];
  const rlab = (k) => (k === 'init' ? `初期: ${(tree[0].acts || []).map((c) => act.catalog[c]).join('・') || '施策なし'}` : act.catalog[+k.slice(1)]);
  const cv = $(canvasId), ctx = cv.getContext('2d'), dpr = devicePixelRatio || 1;
  const W = cv.clientWidth, H = Math.max(200, 30 * rows.length + 60);
  cv.style.height = `${H}px`; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const L = 250, R = 80, Tp = 18, y0 = D.years[0], y1 = D.years[D.years.length - 1] + (D.years[1] - D.years[0]);
  const X = (y) => L + (y - y0) / (y1 - y0) * (W - L - R), Y = (k) => Tp + rows.indexOf(k) * 30 + 12;
  ctx.font = '12px "Hiragino Sans", sans-serif'; ctx.textBaseline = 'middle';
  for (let yr = y0; yr <= y1; yr += 10) { ctx.strokeStyle = '#e2e8f0'; ctx.beginPath(); ctx.moveTo(X(yr), Tp - 8); ctx.lineTo(X(yr), H - 22); ctx.stroke(); ctx.fillStyle = '#64748b'; ctx.fillText(yr, X(yr) - 14, H - 9); }
  rows.forEach((k) => { ctx.fillStyle = '#334155'; const t = rlab(k); ctx.fillText(t.length > 30 ? `${t.slice(0, 29)}…` : t, 6, Y(k)); ctx.strokeStyle = '#f1f5f9'; ctx.beginPath(); ctx.moveTo(L, Y(k)); ctx.lineTo(W - R, Y(k)); ctx.stroke(); });
  const off = {};
  T.forEach((b) => { const k = rowKey(b); off[b.id] = b.kind === 'switch' ? off[b.parent] ?? 0 : (((off[k] = (off[k] ?? -1) + 1) % 5) - 2) * 2.5; });
  const yy = (b) => Y(rowKey(b)) + (off[b.id] || 0), lw = (b) => 1.5 + 9 * b.n / tot;
  const kids = (b) => T.filter((c) => c.parent === b.id);
  T.forEach((b) => {
    const ch = kids(b), end = b.end || !ch.length ? y1 : Math.max(...ch.map((c) => c.year));
    ctx.strokeStyle = metCol(b.met_share); ctx.lineWidth = lw(b);
    ctx.beginPath(); ctx.moveTo(X(b.year), yy(b)); ctx.lineTo(X(end), yy(b)); ctx.stroke();
    ch.forEach((c) => {
      ctx.fillStyle = 'rgba(100,116,139,.18)';
      ctx.fillRect(X(c.year_range[0]), yy(c) - 7, Math.max(X(c.year_range[1]) - X(c.year_range[0]), 2), 14);
      ctx.strokeStyle = metCol(c.met_share); ctx.lineWidth = lw(c); ctx.beginPath(); ctx.moveTo(X(c.year), yy(b)); ctx.lineTo(X(c.year), yy(c)); ctx.stroke();
    });
    ctx.lineWidth = 1.5; ctx.strokeStyle = '#0f172a';
    if (b.kind === 'switch') {             // branch point: the actor moves to the severe list B
      const x = X(b.year), y = yy(b); ctx.fillStyle = '#f59e0b'; ctx.beginPath(); ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.closePath(); ctx.fill(); ctx.stroke();
    } else { ctx.fillStyle = b.branch === 1 ? '#fde68a' : '#fff'; ctx.beginPath(); ctx.arc(X(b.year), yy(b), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
    if (b.end) {
      const fail = b.end - b.end_met, same = T.filter((x) => x.end && rowKey(x) === rowKey(b));
      ctx.fillStyle = fail ? '#dc2626' : '#15803d';
      ctx.fillText(fail ? `✕ ${fail}/${b.end}` : `✓ ${b.end}`, X(y1) + 4 + 38 * Math.floor(same.indexOf(b) / 2), Y(rowKey(b)) + (same.indexOf(b) % 2) * 11 - 4);
    }
  });
  const hidden = tree.length - T.length;
  ctx.fillStyle = '#64748b'; if (hidden) ctx.fillText(`（${minN} シナリオ未満の枝 ${hidden} は省略）`, L, H - 30);
  const path = (b) => { const out = []; for (let x = byId[b.parent]; x && x.parent != null; x = byId[x.parent]) out.unshift(x.label); return out.join(' → ') || '初期施策'; };
  $(tableId).innerHTML = '<tr><th>年（中央値・範囲）</th><th>直前までの経路 → 分岐</th><th>シナリオ</th><th>条件（トリガー）</th><th>観測した指標 ÷ 目標</th><th>予測した2100年の降雨倍率</th><th>リードタイム</th><th>目標の達成</th></tr>'
    + tree.filter((b) => b.parent != null && show.has(b.id)).sort((x, y) => x.year - y.year || y.n - x.n).map((b) => `<tr>
      <td>${b.year}<br><span class="hint">${b.year_range[0] === b.year_range[1] ? '' : `${b.year_range[0]}–${b.year_range[1]}`}</span></td>
      <td><span class="hint">${esc(path(b))}</span> → <b>${esc(b.label)}</b>${b.branch === 1 && b.kind === 'act' ? ' <span class="tag as">経路 B</span>' : ''}</td>
      <td>${b.n}（${pct(b.n / tot)}）</td>
      <td>${b.kind === 'switch' ? `予測した2100年の降雨 ≥ ×${fmt(switchOf(b), 2)}` : Object.entries(b.reasons).map(([k, v]) => `${REASON[k] || k} ${v}`).join('・')}</td>
      <td>${b.obs_ratio ? `${fmt(b.obs_ratio[0], 2)}–${fmt(b.obs_ratio[1], 2)}` : '—'}</td><td>${b.rain2100 ? `${fmt(b.rain2100[0], 3)}–${fmt(b.rain2100[1], 3)}` : '—'}</td>
      <td>${b.lead != null ? `${fmt(b.lead, 0)} 年` : '—'}</td><td style="color:${metCol(b.met_share)}">${pct(b.met_share)}</td></tr>`).join('');
}

// ---------------------------------------------------------------- joint pathways network
let NETPOS = [];
function drawNet() {
  const N = D.networks[NETC], tot = D.worlds.length;
  const cv = $('m-net'), ctx = cv.getContext('2d'), dpr = devicePixelRatio || 1;
  const byYear = {}; N.nodes.forEach((n) => { (byYear[n.year] = byYear[n.year] || []).push(n); });
  Object.values(byYear).forEach((xs) => xs.sort((a, b) => (a.levels == null) - (b.levels == null) || b.n - a.n));
  const maxK = Math.max(...Object.values(byYear).map((x) => x.length));
  const W = cv.clientWidth, H = Math.max(260, 64 * maxK + 50);
  cv.style.height = `${H}px`; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const L = 40, R = 40, years = N.years, X = (y) => L + (years.indexOf(y) + 0.5) * (W - L - R) / years.length;
  const pos = {}; Object.entries(byYear).forEach(([y, xs]) => xs.forEach((n, k) => { pos[n.id] = [X(+y), 26 + k * 64 + 20]; }));
  ctx.font = '11px "Hiragino Sans", sans-serif'; ctx.textBaseline = 'middle';
  years.forEach((y) => { ctx.fillStyle = '#64748b'; ctx.fillText(y, X(y) - 14, 10); });
  const ecol = (e) => { const as = [...new Set((e.changes || []).map((c) => c.actor))]; return as.length === 1 ? acol[as[0]] : as.length ? '#334155' : '#cbd5e1'; };
  N.edges.forEach((e) => {
    const [x1, y1] = pos[e.from], [x2, y2] = pos[e.to];
    ctx.strokeStyle = ecol(e); ctx.globalAlpha = (e.changes || []).length ? 0.85 : 0.35; ctx.lineWidth = 1 + 10 * e.n / tot;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.bezierCurveTo((x1 + x2) / 2, y1, (x1 + x2) / 2, y2, x2, y2); ctx.stroke();
  });
  ctx.globalAlpha = 1;
  NETPOS = N.nodes.map((n) => ({ n, x: pos[n.id][0], y: pos[n.id][1], r: 4 + 10 * Math.sqrt(n.n / tot) }));
  NETPOS.forEach(({ n, x, y, r }) => {
    ctx.fillStyle = n.levels == null ? '#e2e8f0' : metCol(n.met_share); ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#334155'; ctx.fillText(`${n.n}`, x + r + 2, y - 6);
    if (n.by_actor) A.forEach((a, k) => { const c = n.by_actor[a].length; if (c) { ctx.fillStyle = acol[a]; ctx.fillRect(x - r + k * 6, y + r + 3, 5, Math.min(3 * c, 14)); } });
  });
  $('m-net-legend').innerHTML = A.map((a) => `<span><i style="background:${acol[a]}"></i>${esc(a)}の施策を変更</span>`).join('') + '<span><i style="background:#334155"></i>複数の主体が変更</span><span><i style="background:#cbd5e1"></i>変更なし</span>'
    + '<span class="hint">ノード下の縦棒＝主体ごとの実施中の施策数</span>';
  $('m-net-note').innerHTML = 'ノードをクリックすると、その状態（各主体の施策）と、そこへ至る遷移を表示する。';
  cv.onclick = (ev) => {
    const b = cv.getBoundingClientRect(), mx = ev.clientX - b.left, my = ev.clientY - b.top;
    const hit = NETPOS.find((p) => (p.x - mx) ** 2 + (p.y - my) ** 2 <= (p.r + 3) ** 2);
    if (!hit) return;
    const n = hit.n, ins = N.edges.filter((e) => e.to === n.id && (e.changes || []).length).sort((a, b) => b.n - a.n);
    $('m-net-note').innerHTML = `<b>${n.year}年</b>・${n.n} シナリオ・全主体の目標の達成 ${pct(n.met_share)}<br>`
      + (n.by_actor ? A.map((a) => `<span style="color:${acol[a]}">${esc(a)}</span>: ${n.by_actor[a].map(esc).join('、') || '—'}`).join('<br>') : 'その他の状態（表示を省略した状態の集まり）')
      + (ins.length ? `<br>至る遷移: ${ins.slice(0, 4).map((e) => `${e.n} シナリオ: ${e.changes.map((c) => `${esc(c.actor)} ${esc(c.var)} ${esc(c.from)}→${esc(c.to)}`).join('、')}`).join(' ／ ')}` : '');
  };
}

function drawComp() {
  const lab = (k) => (CASE[k] ? `<b>${CASE[k]}</b>` : `ルール: ${esc(k.slice(6))}`);
  const rows = Object.entries(D.comparison).sort(([a, x], [b, y]) => (b in CASE) - (a in CASE) || y.basin.coverage - x.basin.coverage);
  const head = `<tr><th rowspan="2">構成</th><th colspan="3">流域（全主体）</th>${A.map((n) => `<th colspan="5" style="color:${acol[n]}">${esc(n)}</th>`).join('')}</tr>
    <tr><th>カバー率</th><th>総費用</th><th>最大 Regret</th>${A.map(() => '<th>カバー率</th><th>最大 Regret</th><th>費用</th><th>Lock-in</th><th>選択肢</th>').join('')}</tr>`;
  $('m-comp').innerHTML = head + rows.map(([k, r]) => `<tr class="${CASE[k] ? 'rec' : ''}">
    <td>${lab(k)}</td><td style="color:${metCol(r.basin.coverage)}">${pct(r.basin.coverage)}</td><td>${fmt(r.basin.mean_cost)}</td><td>${fmt(r.basin.max_regret, 3)}</td>
    ${A.map((n) => { const x = r.actors[n]; return `<td style="color:${metCol(x.coverage)}">${pct(x.coverage)}</td><td>${fmt(x.max_regret, 3)}</td><td>${fmt(x.mean_cost)}</td><td>${pct(x.lockin)}</td><td>${pct(x.options)}</td>`; }).join('')}</tr>`).join('');
}

// ---------------------------------------------------------------- dependence, spill-over, search
function drawDep() {
  const a = $('m-dep-actor').value, rows = D.dependence[a], others = A.filter((x) => x !== a);
  $('m-dep').innerHTML = `<tr>${others.map((o) => `<th>${esc(o)}</th>`).join('')}<th>${esc(a)}のカバー率</th><th>費用</th><th>初回の着手（中央値）</th><th>着手した割合</th></tr>`
    + rows.map((r) => `<tr>${others.map((o) => `<td>${esc(r.others[o])}</td>`).join('')}<td style="background:${metCol(r.coverage)};color:#fff">${pct(r.coverage)}</td>
      <td>${fmt(r.mean_cost)}</td><td>${r.first_act_median ?? '—'}</td><td>${pct(r.share_acting)}</td></tr>`).join('');
}
function drawSpill() {
  const S = D.spillover, clims = Object.keys(S.ratio_to_target), ps = Object.keys(S.portfolios);
  const cell = (v) => `<td style="background:${v > 1 ? '#fee2e2' : '#dcfce7'}">${fmt(v, 2)}</td>`;
  $('m-spill').innerHTML = `<tr><th rowspan="2">実施する施策</th>${clims.map((c) => `<th colspan="${A.length}">${esc(c)}</th>`).join('')}</tr>
    <tr>${clims.map(() => A.map((a) => `<th>${esc(a)}</th>`).join('')).join('')}</tr>`
    + ps.map((p) => `<tr><td title="${esc(S.portfolios[p].join('、'))}">${esc(p === '施策なし' || p === '全主体' ? p : `${p}の施策のみ`)}</td>${clims.map((c) => A.map((a) => cell(S.ratio_to_target[c][p][a])).join('')).join('')}</tr>`).join('');
}
function drawFront() {
  const k = $('m-front-actor').value, F = D.search[k].front;
  const cs = F.map((r) => r.mean_regret), lo = Math.min(...cs), hi = Math.max(...cs);
  const col = (v) => `hsl(${Math.round(120 * (1 - (v - lo) / ((hi - lo) || 1)))}, 65%, 45%)`;
  const label = (r) => (r.strategy ? [`初期: ${r.strategy.init.join('・') || 'なし'}`, `A: ${r.strategy.A.join('→') || 'なし'}`, `B: ${r.strategy.B.join('→') || 'なし'}`, `ρ ${r.strategy.rho}`]
    : Object.entries(r.strategies).map(([a, s]) => `${a}: ${s.init.join('・') || 'なし'}`));
  chart('m-front', { type: 'scatter', data: { datasets: [{ label: '非劣な戦略', data: F.map((r) => ({ x: r.mean_cost, y: r.coverage, r })),
    pointBackgroundColor: F.map((r) => col(r.mean_regret)), pointRadius: F.map((r) => (r.chosen ? 9 : 4)), pointStyle: F.map((r) => (r.chosen ? 'star' : 'circle')) }] },
  options: { animation: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => [`カバー率 ${pct(c.raw.r.coverage)}・費用 ${fmt(c.raw.r.mean_cost)}・平均 Regret ${fmt(c.raw.r.mean_regret, 3)}`, ...label(c.raw.r)] } } },
    scales: { x: { title: { display: true, text: '平均累積費用 PV [億円]' } }, y: { min: 0, max: 1, title: { display: true, text: 'カバー率（探索シナリオ）' } } } } });
}

function drawSeries() {
  const s = +$('m-scen').value, Z = D.zones;
  const val = (row, a) => { const x = actor(a); return x.metric === 'basin' ? Math.max(row[1] / x.targets.damage_ratio, row[2] / x.targets.pop_ratio) : row[3 + Z.indexOf(x.metric)] / x.targets.damage_ratio; };
  const ds = [];
  Object.keys(CASE).forEach((c) => {
    const r = D.records[c].find((x) => x.s === s);
    A.forEach((a) => ds.push({ label: `${a}（${CASE[c]}${r.actors[a].met ? '' : ' ✕'}）`, data: r.series.map((row) => ({ x: row[0] + 2.5, y: val(row, a) })), borderColor: acol[a], borderDash: DASH[c], pointRadius: 0, borderWidth: 2 }));
  });
  ds.push({ label: '目標', data: [{ x: D.years[0], y: 1 }, { x: 2100, y: 1 }], borderColor: '#0f172a', borderDash: [2, 3], pointRadius: 0, borderWidth: 1 });
  chart('m-series', { type: 'line', data: { datasets: ds }, options: { animation: false, parsing: false, scales: { x: { type: 'linear', min: D.years[0], max: 2100, ticks: { callback: (v) => String(v) } }, y: { title: { display: true, text: '指標 ÷ 目標' } } } } });
}

function drawUnr() {
  const [c, k] = $('m-unr').value.split('|'), b = D.unmet[c][k];
  $('m-prim').innerHTML = `<p>目標を満たせないシナリオ: <b>${b.n_cases}</b> / ${b.n}</p>`
    + (b.box.length ? `<table class="tbl"><tr><th>条件（PRIM の範囲）</th><th>範囲</th></tr>${b.box.map((d) => `<tr><td>${esc(d.label)}</td><td>${fmt(d.low, 3)} – ${fmt(d.high, 3)}</td></tr>`).join('')}</table>
      <p class="hint">密度 ${pct(b.density)}・カバー ${pct(b.coverage)}・範囲内のシナリオ ${pct(b.mass)}</p>` : '<p class="hint">該当シナリオがない（または全シナリオ）ため範囲は求めない。</p>');
}
function drawVer() {
  const V = D.verification_rri;
  if (!V) { $('m-ver').innerHTML = `<tr><td class="hint">未実施。<code>python scripts/run_dapp.py --multi --verify 4</code> で、Individual の経路の状態を RRI で再計算し、代理モデルと比べる。</td></tr>`; return; }
  $('m-ver').innerHTML = '<tr><th>シナリオ</th><th>期</th><th>RRI 被害額</th><th>代理モデル</th><th>差</th></tr>'
    + V.states.map((r) => `<tr><td>${r.s}</td><td>${r.t}</td><td>${fmt(r.rri)}</td><td>${fmt(r.surrogate)}</td><td>${pct((r.surrogate - r.rri) / Math.max(r.rri, 1e-9), 1)}</td></tr>`).join('')
    + (V.r2 != null ? `<tr><td colspan="5" class="hint">R² ${fmt(V.r2, 3)}・相対 RMSE ${pct(V.rel_rmse, 1)}（被害額, 資産係数を掛ける前）</td></tr>` : '');
}

function drawSignal() {
  const Q = D.signal_quality || [], sf = D.settings.multi.strategy.switch_from_year;
  chart('m-signal', { type: 'bar', data: { labels: Q.map((q) => q.year), datasets: [
    { type: 'line', label: '真の2100年値との相関（右軸）', data: Q.map((q) => q.corr), borderColor: '#0e7490', yAxisID: 'c', pointRadius: 3 },
    { label: '予測した2100年の降雨倍率（10–90%）', data: Q.map((q) => [q.q10, q.q90]), backgroundColor: Q.map((q) => (q.year >= sf ? '#fbbf24' : '#e2e8f0')) }] },
  options: { animation: false, scales: { y: { title: { display: true, text: '降雨倍率' } }, c: { position: 'right', min: -0.2, max: 1, grid: { display: false }, title: { display: true, text: '相関' } } },
    plugins: { tooltip: { callbacks: { label: (c) => (c.dataset.yAxisID === 'c' ? `相関 ${fmt(c.raw, 2)}` : `${fmt(c.raw[0], 3)}–${fmt(c.raw[1], 3)}（中央値 ${fmt(Q[c.dataIndex].median, 3)}）`) } } } } });
}
