// Multi-Actor Robust DAPP viewer (results.json of `scripts/run_dapp.py --multi [--single]`, docs/11 11.10)
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
const CONF = { individual: '個別最適の組合せ', basin: '流域全体の最適' };
let D = null, A = [], acol = {}, SELC = 'individual';

export function render(data) {
  D = data; A = D.actors.map((a) => a.name); acol = Object.fromEntries(A.map((n, i) => [n, ACOL[i % ACOL.length]]));
  $('meta').textContent = `作成 ${D.created}・計算 ${fmt(D.elapsed_s / 60, 1)} 分・代理モデル ${D.physics.explore_run}・主体 ${A.length}・評価シナリオ ${D.worlds.length}`;
  const opt = (sel, items, cb) => { $(sel).innerHTML = items.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join(''); $(sel).onchange = cb; };
  opt('m-actor', D.actors.map((a) => [a.name, a.label]), drawMap);
  opt('m-conf', Object.entries(CONF).filter(([k]) => D.trees[k]), (e) => { SELC = e.target.value; drawSel(); });
  opt('m-dep-actor', D.actors.map((a) => [a.name, a.label]), drawDep);
  opt('m-front-actor', [...D.actors.map((a) => [a.name, a.label]), ...(D.search.basin ? [['basin', '流域全体（全主体の戦略）']] : [])], drawFront);
  const rain = D.worlds.map((w) => w.rain_2100), ord = rain.map((r, s) => s).sort((a, b) => rain[a] - rain[b]);
  opt('m-scen', ord.map((s) => [s, `シナリオ ${s}（2100年 降雨 ×${fmt(rain[s], 3)}・海面 ${fmt(D.worlds[s].slr_2100, 2)} m）`]), drawSeries);
  $('m-scen').value = ord[Math.floor(ord.length / 2)];
  opt('m-unr', Object.keys(CONF).filter((k) => D.unmet[k]).flatMap((c) => ['all', ...A].map((k) => [`${c}|${k}`, `${CONF[c]}: ${k === 'all' ? '全主体の目標' : `${k}の目標`}`])), drawUnr);
  ['m-gap-card'].forEach((id) => { $(id).style.display = D.single ? 'none' : ''; });
  drawSummary(); drawComp(); drawGap(); drawSel(); drawDep(); drawSpill(); drawSignal(); drawFront(); drawUnr(); drawVer();
}
function drawSel() { $('m-conf').value = SELC; drawComp(); drawMap(); drawNet(); drawSeries(); }
const actor = (n) => D.actors.find((a) => a.name === n);
const C = (k) => D.comparison[k];

function stratHTML(s, a) {
  const li = (xs) => (xs.length ? xs.map(esc).join(' → ') : '<span class="hint">なし</span>');
  return `<div style="margin:2px 0 6px"><b style="color:${acol[a]}">${esc(actor(a).label)}</b>${s.rule ? `（ルール: ${esc(s.rule)}）` : ''}
    <div>今実施すべき初期施策: <b>${s.init.length ? s.init.map(esc).join('、') : '施策なし（様子見）'}</b></div>
    <div>経路 A（基本）: ${li(s.A)}</div>
    <div>経路 B（厳しい気候）: ${li(s.B)}${s.switch != null ? `<span class="hint">（観測した降雨の傾向から予測した2100年の降雨倍率が ×${fmt(s.switch, 2)} 以上で切替）</span>` : '<span class="hint">（切り替えない）</span>'}</div>
    <div class="hint">トリガー: ${actor(a).metric === 'basin' ? '流域の被害額・浸水人口' : `${actor(a).metric}の被害額`} が 目標 × ${fmt(s.rho, 2)} を超えたら次の行動（${s.anticipate ? '観測、または観測した傾向からリードタイム＋1期 後の予測' : '観測のみ'}）</div></div>`;
}

function drawSummary() {
  const ind = C('individual'), bas = C('basin'), g = D.gap;
  const unm = D.unmet.individual.all.n_cases;
  $('m-summary').innerHTML = `<h2>${D.single ? 'Robust DAPP（単一主体: 個別施策の条件付き適応戦略）' : 'Multi-Actor Robust DAPP（国・上流自治体・下流自治体）'}</h2>
    <div class="kpis">
      <div class="kpi"><div class="l">全主体の目標を満たす割合（個別最適の組合せ）</div><div class="v">${pct(ind.basin.coverage)}</div></div>
      ${D.single ? '' : `<div class="kpi"><div class="l">同（流域全体の最適）</div><div class="v">${pct(bas.basin.coverage)}</div></div>
      <div class="kpi"><div class="l">Robustness Gap（流域最適 − 個別最適）</div><div class="v">${sgn(100 * g.basin_coverage, 0, ' pt')}</div></div>
      <div class="kpi"><div class="l">総費用の差（流域最適 − 個別最適, PV）</div><div class="v">${sgn(g.basin_cost, 0, ' 億円')}</div></div>`}
      <div class="kpi"><div class="l">目標を満たせない将来（個別最適）</div><div class="v">${unm} / ${D.worlds.length}</div></div>
    </div>
    <div style="margin-top:8px">${A.map((n) => stratHTML(ind.strategies[n], n)).join('')}</div>
    <p class="hint">施策の組合せ（Pareto 解）は戦略の探索の初期値にだけ使い、経路の状態にはしない。状態＝各個別施策の実施水準・施工中の事業・主体ごとの予算・土地利用（2030年以降の各期の代表洪水で評価）。
    各主体の戦略は、他主体の行動（積極投資・段階投資・投資抑制）を不確実性として、経路全体の カバー率・費用・効用・Lock-in で NSGA-II により探索した（重みは Regret の効用だけに使う）。</p>`;
}

function drawComp() {
  const rows = Object.entries(D.comparison).sort(([a, x], [b, y]) => (a in CONF ? -1 : 0) - (b in CONF ? -1 : 0) || y.basin.coverage - x.basin.coverage);
  const head = `<tr><th rowspan="2">構成</th><th colspan="3">流域（全主体）</th>${A.map((n) => `<th colspan="5" style="color:${acol[n]}">${esc(n)}</th>`).join('')}</tr>
    <tr><th>カバー率</th><th>総費用</th><th>最大 Regret</th>${A.map(() => '<th>カバー率</th><th>最大 Regret</th><th>費用</th><th>Lock-in</th><th>選択肢</th>').join('')}</tr>`;
  $('m-comp').innerHTML = head + rows.map(([k, r]) => `<tr data-k="${esc(k)}" class="${k === SELC ? 'rec' : ''}">
    <td>${esc(CONF[k] || `ルール: ${k.slice(6)}`)}</td><td style="color:${metCol(r.basin.coverage)}">${pct(r.basin.coverage)}</td><td>${fmt(r.basin.mean_cost)}</td><td>${fmt(r.basin.max_regret, 3)}</td>
    ${A.map((n) => { const x = r.actors[n]; return `<td style="color:${metCol(x.coverage)}">${pct(x.coverage)}</td><td>${fmt(x.max_regret, 3)}</td><td>${fmt(x.mean_cost)}</td><td>${pct(x.lockin)}</td><td>${pct(x.options)}</td>`; }).join('')}</tr>`).join('');
  $('m-comp').querySelectorAll('tr[data-k]').forEach((tr) => { tr.onclick = () => { if (tr.dataset.k in CONF && D.trees[tr.dataset.k]) { SELC = tr.dataset.k; drawSel(); } }; });
}

function drawGap() {
  if (D.single) return;
  const ind = C('individual'), bas = C('basin'), g = D.gap;
  $('m-gap').innerHTML = `<table class="tbl"><tr><th></th><th>カバー率 個別最適</th><th>流域最適</th><th>個別最適 − 流域最適</th><th>費用 個別最適</th><th>流域最適</th><th>平均 Regret の差（流域 − 個別）</th></tr>
    <tr><td><b>流域（全主体の目標）</b></td><td>${pct(ind.basin.coverage)}</td><td>${pct(bas.basin.coverage)}</td><td>${sgn(-100 * g.basin_coverage, 0, ' pt')}</td><td>${fmt(ind.basin.mean_cost)}</td><td>${fmt(bas.basin.mean_cost)}</td><td></td></tr>
    ${A.map((n) => `<tr><td style="color:${acol[n]}">${esc(n)}</td><td>${pct(ind.actors[n].coverage)}</td><td>${pct(bas.actors[n].coverage)}</td><td>${sgn(100 * g.actors[n].coverage, 0, ' pt')}</td>
      <td>${fmt(ind.actors[n].mean_cost)}</td><td>${fmt(bas.actors[n].mean_cost)}</td><td>${sgn(g.actors[n].regret, 3)}</td></tr>`).join('')}</table>
    <p class="hint">Robustness Gap＝流域全体の最適（全主体の戦略を同時に探索）と、各主体が自分の目標で選んだ戦略の組合せ（個別最適）の、全主体の目標を満たす割合の差。
    主体の行の「個別最適 − 流域最適」が正なら、その主体は流域全体の最適のために自分のカバー率を下げる（費用を負担する）ことになる。</p>
    <div>${A.map((n) => stratHTML(bas.strategies[n], n)).join('')}</div>`;
}

// ---------------------------------------------------------------- per-actor pathways map
function drawMap() {
  const a = $('m-actor').value, act = actor(a), tree = D.trees[SELC][a], tot = tree[0].n;
  $('m-strat').innerHTML = stratHTML(C(SELC).strategies[a], a);
  const minN = Math.max(1, Math.round(0.03 * tot));
  const show = new Set(tree.filter((b) => b.n >= minN).map((b) => b.id));
  const T = tree.filter((b) => show.has(b.id)), byId = Object.fromEntries(tree.map((b) => [b.id, b]));
  const rowKey = (b) => (b.kind === 'act' ? `a${b.a}` : b.kind === 'initial' ? 'init' : rowKey(byId[b.parent]));
  const first = {};                    // rows in the order the actions first appear (median year), then by the catalog
  T.filter((b) => b.kind === 'act').forEach((b) => { first[b.a] = Math.min(first[b.a] ?? 9999, b.year); });
  const rows = ['init', ...Object.keys(first).map(Number).sort((x, y) => first[x] - first[y] || x - y).map((x) => `a${x}`)];
  const rlab = (k) => (k === 'init' ? `初期: ${(tree[0].acts || []).map((c) => act.catalog[c]).join('・') || '施策なし'}` : act.catalog[+k.slice(1)]);
  const cv = $('m-map'), ctx = cv.getContext('2d'), dpr = devicePixelRatio || 1;
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
  drawBranch(a, tree, show);
}
function drawBranch(a, tree, show) {
  const tot = tree[0].n, act = actor(a), byId = Object.fromEntries(tree.map((b) => [b.id, b]));
  const sw = C(SELC).strategies[a].switch;
  const path = (b) => { const out = []; for (let x = byId[b.parent]; x && x.parent != null; x = byId[x.parent]) out.unshift(x.label); return out.join(' → ') || '初期施策'; };
  $('m-branch').innerHTML = '<tr><th>年（中央値・範囲）</th><th>直前までの経路 → 分岐</th><th>シナリオ</th><th>条件（トリガー）</th><th>観測した指標 ÷ 目標</th><th>予測した2100年の降雨倍率</th><th>リードタイム</th><th>目標の達成</th></tr>'
    + tree.filter((b) => b.parent != null && show.has(b.id)).sort((x, y) => x.year - y.year || y.n - x.n).map((b) => `<tr>
      <td>${b.year}<br><span class="hint">${b.year_range[0] === b.year_range[1] ? '' : `${b.year_range[0]}–${b.year_range[1]}`}</span></td>
      <td><span class="hint">${esc(path(b))}</span> → <b>${esc(b.label)}</b>${b.branch === 1 && b.kind === 'act' ? ' <span class="tag as">経路 B</span>' : ''}</td>
      <td>${b.n}（${pct(b.n / tot)}）</td>
      <td>${b.kind === 'switch' ? `予測した2100年の降雨 ≥ ×${fmt(sw, 2)}` : Object.entries(b.reasons).map(([k, v]) => `${REASON[k] || k} ${v}`).join('・')}</td>
      <td>${b.obs_ratio ? `${fmt(b.obs_ratio[0], 2)}–${fmt(b.obs_ratio[1], 2)}` : '—'}</td><td>${b.rain2100 ? `${fmt(b.rain2100[0], 3)}–${fmt(b.rain2100[1], 3)}` : '—'}</td>
      <td>${b.lead != null ? `${fmt(b.lead, 0)} 年` : '—'}</td><td style="color:${metCol(b.met_share)}">${pct(b.met_share)}</td></tr>`).join('');
  void act;
}

// ---------------------------------------------------------------- joint pathways network
let NETPOS = [];
function drawNet() {
  const N = D.networks[SELC], tot = D.worlds.length;
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
  $('m-net-legend').innerHTML = A.map((a) => `<span><i style="background:${acol[a]}"></i>${esc(a)}が変更</span>`).join('') + '<span><i style="background:#334155"></i>複数の主体が変更</span><span><i style="background:#cbd5e1"></i>変更なし</span>'
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
    : A.map((a) => `${a}: ${r.strategies[a].init.join('・') || 'なし'}`));
  chart('m-front', { type: 'scatter', data: { datasets: [{ label: '非劣な戦略', data: F.map((r) => ({ x: r.mean_cost, y: r.coverage, r })),
    pointBackgroundColor: F.map((r) => col(r.mean_regret)), pointRadius: F.map((r) => (r.chosen ? 9 : 4)), pointStyle: F.map((r) => (r.chosen ? 'star' : 'circle')) }] },
  options: { animation: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => [`カバー率 ${pct(c.raw.r.coverage)}・費用 ${fmt(c.raw.r.mean_cost)}・平均 Regret ${fmt(c.raw.r.mean_regret, 3)}`, ...label(c.raw.r)] } } },
    scales: { x: { title: { display: true, text: '平均累積費用 PV [億円]' } }, y: { min: 0, max: 1, title: { display: true, text: 'カバー率（探索シナリオ）' } } } } });
}

function drawSeries() {
  const s = +$('m-scen').value, Z = D.zones;
  const val = (row, a) => { const x = actor(a); return x.metric === 'basin' ? Math.max(row[1] / x.targets.damage_ratio, row[2] / x.targets.pop_ratio) : row[3 + Z.indexOf(x.metric)] / x.targets.damage_ratio; };
  const ds = [];
  Object.keys(CONF).filter((c) => D.records[c]).forEach((c, ci) => {
    const r = D.records[c].find((x) => x.s === s);
    A.forEach((a) => ds.push({ label: `${a}（${CONF[c]}）`, data: r.series.map((row) => ({ x: row[0] + 2.5, y: val(row, a) })), borderColor: acol[a], borderDash: ci ? [6, 4] : [], pointRadius: 0, borderWidth: 2 }));
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
  if (!V) { $('m-ver').innerHTML = `<tr><td class="hint">未実施。<code>python scripts/run_dapp.py --multi${D.single ? ' --single' : ''} --verify 4</code> で、個別最適の経路の状態を RRI で再計算し、代理モデルと比べる。</td></tr>`; return; }
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
