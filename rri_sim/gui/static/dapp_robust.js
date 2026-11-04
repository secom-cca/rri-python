// Robust DAPP viewer (results.json of `scripts/run_dapp.py --robust`, docs/11 11.9); trigger panel: POST /api/dapp/robust/simulate
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : `${(100 * v).toFixed(d)}%`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const charts = {};
function chart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); return charts[id]; }
const metCol = (m) => `hsl(${Math.round(120 * m)}, 65%, ${m > 0.5 ? 38 : 48}%)`;
const REASON = { observed: '観測した被害', projected: '予測（観測した傾向）', initial: '初期' };
let D = null, NAME = null, E = null, NODES = null, EDGES = null, ATP = null, SEL = null, TRIG = null, TGT = null;

export function render(data, name) {
  D = data; NAME = name;
  E = D.evaluation; NODES = D.network.nodes; EDGES = D.network.edges; ATP = D.atp;
  TRIG = { ...D.config.robust.trigger }; TGT = { ...D.config.robust.targets };
  const pr = E.table.filter((r) => r.promising && r.node !== 0).sort((a, b) => b.coverage - a.coverage || a.max_regret - b.max_regret);
  SEL = pr.length ? pr[0].node : E.inits[0];
  $('meta').textContent = `作成 ${D.created}・計算 ${fmt(D.elapsed_s / 60, 1)} 分・代理モデル ${D.physics.explore_run}`;
  drawSummary(); drawPanel(); drawAll();
}
function drawAll() { drawInit(); drawTrade(); select(SEL); drawUnr(); drawVer(); }
function select(n) { SEL = n; drawInit(); drawDetail(); drawMap(); drawBranches(); drawATP(); drawScen(); drawEdges(); drawLand(); }
const node = (i) => NODES[String(i)] || { label: `組合せ ${i}`, implementation: [], landuse: {} };
const short = (i) => { const n = node(i); return i === 0 ? '施策なし' : `#${i} ${n.implementation.length} 施策・${fmt(n.initial)} 億円`; };
const row = (i) => E.table.find((r) => r.node === i);

function drawSummary() {
  const R = D.config.robust, c = D.network, t = R.targets, pr = E.promising.filter((n) => n !== 0);
  const unr = E.uncovered.adaptive.reduce((a, b) => a + b, 0), unp = E.uncovered.perfect.reduce((a, b) => a + b, 0);
  const out = Object.entries(D.physics.outside_range || {}).filter(([, v]) => v > 0).map(([k, v]) => `${k} ${pct(v, 1)}`).join('・');
  $('r-summary').innerHTML = `<div class="rec-box">🏅 <b>今実施すべき初期施策の候補（有望: カバー率・最大 Regret・平均費用・Lock-in で非劣）</b>
    <ul style="margin:6px 0 0">${pr.map((n) => { const r = row(n); return `<li><a href="#" data-n="${n}">${esc(short(n))}</a>: カバー率 ${pct(r.coverage)}（完全情報 ${pct(r.coverage_perfect)}）・最大 Regret ${fmt(r.max_regret, 3)}・平均費用 ${fmt(r.mean_cost)} 億円・Lock-in ${pct(r.lockin.options)}<br><span class="hint">${esc(node(n).label)}</span></li>`; }).join('') || '<li>施策なし以外に非劣な初期施策がない</li>'}</ul>
    <div style="margin-top:6px">対応できない将来: 観測に基づくどの経路でも目標を満たせないシナリオ <b>${unr}</b> / ${E.scenarios.length}（完全情報でも ${unp}）— 下の「対応できない将来」。</div></div>
    <p class="hint" style="margin:8px 0 0">目標（${t.status}）: ${t.from_year}年以降の各期の代表洪水で 被害額 ≤ ${t.damage_ratio} × 対策なし現在気候、浸水人口 ≤ ${t.pop_ratio} ×。${esc(t.basis)}<br>
    Pathways Network: 組合せ ${c.n_nodes}（探索の非劣解・和集合・施策なし）、移行 ${fmt(c.n_edges)}（条件付き ${fmt(c.n_conditional)}: 森林の農地への再転換）。
    候補の探索 ${fmt(D.candidates.n_evaluated)} 案（非劣 ${fmt(D.candidates.n_nondominated)}）。シナリオ ${E.scenarios.length}（${D.config.robust.scenarios.groups.join('・')} を LHS で変化: ${esc(D.config.robust.scenarios.basis)}）。<br>
    代理モデル ${D.physics.explore_run}（docs/12）の学習範囲外の値は範囲に切り詰める（外挿しない）${out ? `: 範囲外の割合 ${out}` : '（範囲外なし）'}。
    社会・参加の過程（合意の失敗・参加率・調整費用, docs/13）は扱わず、合意期間は実施期間に含める。代表洪水の近似は 11.2 と同じ。</p>`;
  $('r-summary').querySelectorAll('a[data-n]').forEach((a) => { a.onclick = (e) => { e.preventDefault(); select(+a.dataset.n); }; });
}

function drawPanel() {
  const f = (k, lab, v, step, obj) => `<label>${lab} <input type="number" data-o="${obj}" data-k="${k}" value="${v}" step="${step}" style="width:64px"></label>`;
  const b = (k, lab, v) => `<label><input type="checkbox" data-o="trig" data-k="${k}" ${v ? 'checked' : ''}> ${lab}</label>`;
  const nS = D.Z.length;
  $('r-panel').innerHTML = `<b>トリガー</b> ${f('ratio_of_threshold', '閾値に対する比 ρ', TRIG.ratio_of_threshold, 0.05, 'trig')}
    ${b('use_observed', '観測した被害が ρ×目標 を超えたら', TRIG.use_observed)} ${b('use_projection', '次の移行の完成前に超えると予測したら', TRIG.use_projection)}
    ${f('horizon_years', '移行先の見通し [年]', TRIG.horizon_years, 5, 'trig')} ${f('safety', '移行先の余裕（≤1 で厳しく）', TRIG.safety, 0.05, 'trig')}
    ／ <b>目標</b> ${f('damage_ratio', '被害額 ≤', TGT.damage_ratio, 0.05, 'tgt')} ${f('pop_ratio', '浸水人口 ≤', TGT.pop_ratio, 0.05, 'tgt')} ${f('from_year', '適用開始年', TGT.from_year, 10, 'tgt')}
    ／ シナリオ数 <select id="r-ns">${[32, 64, nS].filter((v, i, a) => v <= nS && a.indexOf(v) === i).map((v) => `<option ${v === nS ? 'selected' : ''}>${v}</option>`).join('')}</select>
    <button id="r-go">経路を再計算</button> <button id="r-reset">保存した結果に戻す</button>`;
  $('r-panel-note').textContent = '施策候補（ステップ1）・移行可能性（2）・組合せごとの ATP の表（3）は保存したものを使い、経路の生成と初期施策の評価（4–5）だけを代理モデルで再計算する。RRI は使わない。';
  $('r-go').onclick = resim;
  $('r-reset').onclick = () => { E = D.evaluation; NODES = D.network.nodes; EDGES = D.network.edges; ATP = D.atp; TRIG = { ...D.config.robust.trigger }; TGT = { ...D.config.robust.targets }; drawPanel(); drawAll(); };
}
async function resim() {
  const trig = {}, tgt = {};
  $('r-panel').querySelectorAll('input').forEach((el) => { const o = el.dataset.o === 'trig' ? trig : tgt; o[el.dataset.k] = el.type === 'checkbox' ? el.checked : +el.value; });
  $('r-go').disabled = true; $('r-panel-note').textContent = '再計算中…（代理モデル上で全初期施策 × シナリオの経路を生成）';
  const t0 = performance.now();
  try {
    const r = await fetch('/api/dapp/robust/simulate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: NAME, trigger: trig, targets: tgt, n_scenarios: +$('r-ns').value }) });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    E = j.evaluation; NODES = { ...D.network.nodes, ...j.nodes }; EDGES = j.edges; ATP = { ...D.atp, ...j.atp }; TRIG = j.trigger; TGT = j.targets;
    if (!E.inits.includes(SEL)) SEL = E.inits[0];
    drawAll();
    $('r-panel-note').textContent = `再計算しました（${((performance.now() - t0) / 1000).toFixed(1)} 秒, シナリオ ${E.scenarios.length}）。「保存した結果に戻す」で元の設定の結果に戻る。`;
  } catch (e) { $('r-panel-note').textContent = `エラー: ${e.message}`; }
  $('r-go').disabled = false;
}

function drawInit() {
  const cols = [['coverage', 'カバー率（観測に基づく経路）', pct], ['coverage_perfect', 'カバー率（完全情報）', pct], ['max_regret', '最大 Regret', (v) => fmt(v, 3)],
    ['mean_regret', '平均 Regret', (v) => fmt(v, 3)], ['satisficing', '許容 Regret 内で達成', pct], ['mean_cost', '平均累積費用（PV）', (v) => fmt(v)],
    ['p90_cost', '費用 90%値', (v) => fmt(v)], ['initial_cost', '初期費用', (v) => fmt(v)], ['n_moves', '移行の回数（平均）', (v) => fmt(v, 1)],
    ['atp_median', '初期施策だけの ATP（中央値）', (v) => (v >= D.config.robust.model.years.end ? 'なし' : fmt(v))]];
  const lk = [['options', 'Lock-in 選択肢', pct], ['scenarios', 'Lock-in シナリオ', pct], ['sunk_oku', '不可逆な初期費用', (v) => fmt(v)],
    ['forest_km2', '森林化 km²', (v) => fmt(v, 1)], ['retention_km2', '遊水地 km²', (v) => fmt(v, 1)], ['relocated_km2', '移転跡地 km²', (v) => fmt(v, 1)]];
  const best = (k, hi) => { const v = E.table.map((r) => r[k]); return hi ? Math.max(...v) : Math.min(...v); };
  $('r-init').innerHTML = `<tr><th>初期施策</th>${cols.map((c) => `<th>${c[1]}</th>`).join('')}${lk.map((c) => `<th>${c[1]}</th>`).join('')}</tr>`
    + E.table.map((r) => `<tr data-n="${r.node}" style="cursor:pointer;${r.node === SEL ? 'background:#fef3c7' : ''}"><td>${r.promising ? '★ ' : ''}${esc(short(r.node))}</td>`
      + cols.map(([k, , f]) => `<td style="${r[k] === best(k, k.startsWith('coverage') || k === 'satisficing') ? 'font-weight:600' : ''}">${f(r[k])}</td>`).join('')
      + lk.map(([k, , f]) => `<td>${f(r.lockin?.[k])}</td>`).join('') + '</tr>').join('');
  $('r-init').querySelectorAll('tr[data-n]').forEach((tr) => { tr.onclick = () => select(+tr.dataset.n); });
}

function drawTrade() {
  const X = { mean_cost: '平均累積費用（PV）[億円]', max_regret: '最大 Regret', lock: 'Lock-in（選択肢）', initial_cost: '初期費用 [億円]' };
  if (!$('r-tx').options.length) { $('r-tx').innerHTML = Object.entries(X).map(([k, l]) => `<option value="${k}">${l}</option>`).join(''); $('r-tx').onchange = drawTrade; }
  const k = $('r-tx').value, val = (r) => (k === 'lock' ? r.lockin.options : r[k]);
  const mr = Math.max(...E.table.map((r) => r.max_regret), 1e-9);
  chart('r-trade', { type: 'scatter', data: { datasets: [{ data: E.table.map((r) => ({ x: val(r), y: r.coverage, n: r.node })),
    pointRadius: E.table.map((r) => 4 + 14 * (r.lockin.options || 0)), pointBackgroundColor: E.table.map((r) => `hsla(${Math.round(120 * (1 - r.max_regret / mr))},70%,45%,.75)`),
    pointBorderColor: E.table.map((r) => (r.node === SEL ? '#111' : (r.promising ? '#b45309' : '#fff'))), pointBorderWidth: E.table.map((r) => (r.node === SEL || r.promising ? 2 : 1)) }] },
  options: { animation: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${short(c.raw.n)}: カバー率 ${pct(c.raw.y)}` } } },
    scales: { x: { title: { display: true, text: X[k] } }, y: { min: 0, max: 1, title: { display: true, text: 'カバー率' } } },
    onClick: (e, els) => { if (els.length) select(E.table[els[0].index].node); } } });
}

function drawDetail() {
  const r = row(SEL), n = node(SEL), succ = D.network.successors[String(SEL)];
  $('r-sel').textContent = short(SEL);
  const lk = r.lockin || {};
  $('r-detail').innerHTML = `<div><b>${esc(n.label)}</b></div>
    <table class="tbl" style="margin-top:6px"><tr><th>施策</th><th>水準</th><th>種類</th></tr>${n.implementation.map((d) => `<tr><td>${esc(d.name)}</td><td>${esc(d.label)}</td><td>${esc(d.kind)}</td></tr>`).join('') || '<tr><td colspan="3">なし</td></tr>'}</table>
    <p class="hint">初期費用 ${fmt(n.initial)} 億円・維持費 ${fmt(n.annual, 1)} 億円/年。目標を満たせなくなる降雨倍率（名目の仮定）: ${ATP[String(SEL)]?.tipping_rain ? '×' + fmt(ATP[String(SEL)].tipping_rain, 3) : '学習範囲内（×1.3 まで）で到達しない'}。
    移行先 ${succ ? `${succ.n}（条件付き ${succ.n_conditional}）・不可逆で戻れない組合せ ${succ.blocked.irreversible}・土地が合わない ${succ.blocked.land}` : '—'}。
    Lock-in: 有用な組合せ ${lk.n_useful ?? '—'} のうち到達できなくなる ${lk.n_lost ?? '—'}。</p>`;
}

// ---------------------------------------------------------------- branching pathways map
function drawMap() {
  const tree = E.trees[String(SEL)];
  const cv = $('r-map'), ctx = cv.getContext('2d'), dpr = devicePixelRatio || 1;
  const used = [...new Set(tree.map((b) => b.node))].sort((a, b) => (D.network.cost_pv[a] - D.network.cost_pv[b]));
  const W = cv.clientWidth, H = Math.max(240, 30 * used.length + 70);
  cv.style.height = `${H}px`; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const L = 230, R = 70, T = 20, y0 = D.config.robust.model.years.start, y1 = D.config.robust.model.years.end;
  const X = (y) => L + (y - y0) / (y1 - y0) * (W - L - R), Y = (n) => T + used.indexOf(n) * 30 + 12;
  ctx.font = '12px "Hiragino Sans", sans-serif'; ctx.textBaseline = 'middle';
  for (let yr = y0; yr <= y1; yr += 10) { ctx.strokeStyle = '#e2e8f0'; ctx.beginPath(); ctx.moveTo(X(yr), T - 8); ctx.lineTo(X(yr), H - 22); ctx.stroke(); ctx.fillStyle = '#64748b'; ctx.fillText(yr, X(yr) - 14, H - 9); }
  used.forEach((n) => { ctx.fillStyle = n === SEL ? '#b45309' : '#334155'; ctx.fillText(short(n), 6, Y(n)); ctx.strokeStyle = '#f1f5f9'; ctx.beginPath(); ctx.moveTo(L, Y(n)); ctx.lineTo(W - R, Y(n)); ctx.stroke(); });
  const tot = tree[0].n, kids = (b) => tree.filter((c) => c.parent === b.id);
  const lw = (b) => 1.5 + 9 * b.n / tot;
  const off = {};                     // small vertical offset for branches sharing a row
  tree.forEach((b) => { off[b.id] = ((off[`n${b.node}`] = (off[`n${b.node}`] ?? -1) + 1) % 5 - 2) * 2.5; });
  const yy = (b) => Y(b.node) + off[b.id];
  tree.forEach((b) => {
    const ch = kids(b), end = b.end ? y1 : Math.max(...ch.map((c) => c.year));
    ctx.strokeStyle = metCol(b.met_share); ctx.lineWidth = lw(b);
    ctx.beginPath(); ctx.moveTo(X(b.year), yy(b)); ctx.lineTo(X(end), yy(b)); ctx.stroke();
    ch.forEach((c) => {
      ctx.fillStyle = 'rgba(100,116,139,.18)';           // timing of the move across the scenarios (earliest - latest)
      ctx.fillRect(X(c.year_range[0]), yy(c) - 7, Math.max(X(c.year_range[1]) - X(c.year_range[0]), 2), 14);
      ctx.strokeStyle = metCol(c.met_share); ctx.lineWidth = lw(c); ctx.beginPath(); ctx.moveTo(X(c.year), yy(b)); ctx.lineTo(X(c.year), yy(c)); ctx.stroke();
    });
    ctx.lineWidth = 1.5; ctx.fillStyle = '#fff'; ctx.strokeStyle = '#0f172a';
    ctx.beginPath(); ctx.arc(X(b.year), yy(b), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    if (b.end) {                       // pathways ending on this branch: met / not met (対応できない将来)
      const fail = b.end - b.end_met, same = tree.filter((x) => x.end && x.node === b.node);
      ctx.fillStyle = fail ? '#dc2626' : '#15803d';
      ctx.fillText(fail ? `✕ ${fail}/${b.end}` : `✓ ${b.end}`, X(y1) + 4 + 34 * Math.floor(same.indexOf(b) / 2), Y(b.node) + (same.indexOf(b) % 2) * 11 - 4);
    }
  });
}
function drawBranches() {
  const tree = E.trees[String(SEL)], tot = tree[0].n;
  const par = (b) => tree.find((x) => x.id === b.parent);
  $('r-branch').innerHTML = '<tr><th>年（中央値・範囲）</th><th>移行</th><th>シナリオ</th><th>観測した降雨倍率</th><th>観測した被害の比 ÷ 目標</th><th>トリガー</th><th>リードタイム</th><th>費用</th><th>目標の達成</th><th>移行先の不足</th></tr>'
    + tree.filter((b) => b.parent != null).sort((a, b) => a.year - b.year || b.n - a.n).map((b) => `<tr>
      <td>${b.year}<br><span class="hint">${b.year_range[0] === b.year_range[1] ? '' : `${b.year_range[0]}–${b.year_range[1]}`}</span></td><td>${esc(short(par(b).node))} → <b>${esc(short(b.node))}</b><br><span class="hint">${esc(changeText(par(b).node, b.node))}</span></td>
      <td>${b.n}（${pct(b.n / tot)}）</td><td>${b.obs_rain ? `${fmt(b.obs_rain[0], 3)}–${fmt(b.obs_rain[1], 3)}` : '—'}</td>
      <td>${b.obs_ratio ? `${fmt(b.obs_ratio[0], 2)}–${fmt(b.obs_ratio[1], 2)}` : '—'}</td>
      <td>${Object.entries(b.reasons).map(([k, v]) => `${REASON[k] || k} ${v}`).join('・')}</td><td>${fmt(b.lead, 0)} 年</td><td>${fmt(b.cost)} 億円</td>
      <td style="color:${metCol(b.met_share)}">${pct(b.met_share)}</td><td>${b.insufficient ? `${b.insufficient}（目標に届かない最善の移行）` : ''}</td></tr>`).join('');
}
function changeText(a, b) {
  const e = EDGES.find((x) => x.from === a && x.to === b);
  return e ? e.changes.map((c) => `${c.var} ${c.from}→${c.to}`).join('、') : '';
}

function drawATP() {
  const tree = E.trees[String(SEL)], used = [...new Set(tree.map((b) => b.node))].sort((a, b) => (D.network.cost_pv[a] - D.network.cost_pv[b]));
  const y1 = D.config.robust.model.years.end, sc = E.scenarios;
  const bins = [[2030, 2050, '〜2050', '#dc2626'], [2050, 2070, '2050–2070', '#f97316'], [2070, 2100, '2070–2100', '#facc15'], [null, null, '2100年まで到達しない', '#16a34a']];
  const share = (n, b) => { const ys = (ATP[String(n)]?.years || []).filter((_, s) => sc.includes(s)); if (!ys.length) return 0;
    return ys.filter((y) => (b[0] == null ? y == null : y != null && y >= b[0] && y < b[1])).length / ys.length; };
  chart('r-atp', { type: 'bar', data: { labels: used.map(short), datasets: bins.map((b) => ({ label: b[2], data: used.map((n) => share(n, b)), backgroundColor: b[3] })) },
    options: { indexAxis: 'y', animation: false, scales: { x: { stacked: true, max: 1, title: { display: true, text: 'シナリオの割合（ATP に達する時期）' } }, y: { stacked: true } } } });
  $('r-atp-note').textContent = `ATP＝その組合せが完成・成熟した状態で、目標（被害額 ≤ ${TGT.damage_ratio}×・浸水人口 ≤ ${TGT.pop_ratio}×）を初めて満たさなくなる期。経路の移行は、次の移行のリードタイム＋1期 前にこれを予測・観測して着手する（トリガー）。`;
}

function drawScen() {
  const recs = E.records.adaptive.filter((r) => r.init === SEL), po = E.records.perfect.filter((r) => r.init === SEL);
  const rain = (s) => D.worlds[s].rain_pts[D.worlds[s].rain_pts.length - 1][1];
  const order = recs.map((r) => r.s).sort((a, b) => rain(a) - rain(b));
  const prev = +$('r-scen').value;
  $('r-scen').innerHTML = order.map((s) => { const r = recs.find((x) => x.s === s); return `<option value="${s}">シナリオ ${s}（2100年 降雨 ×${fmt(rain(s), 2)}）${r.met ? '' : ' ✕'}</option>`; }).join('');
  if (order.includes(prev)) $('r-scen').value = prev; else $('r-scen').value = order[Math.floor(order.length / 2)];
  $('r-scen').onchange = drawScen;
  const s = +$('r-scen').value, a = recs.find((r) => r.s === s), p = po.find((r) => r.s === s);
  const xs = a.series.map((q) => q[0] + 2.5);
  const ds = [['被害額の比（経路）', a, 1, '#0e7490', []], ['浸水人口の比（経路）', a, 2, '#7c3aed', []], ['被害額の比（完全情報）', p, 1, '#0e7490', [5, 4]], ['浸水人口の比（完全情報）', p, 2, '#7c3aed', [5, 4]]]
    .map(([l, r, k, c, dash]) => ({ label: l, data: r.series.map((q) => q[k]), borderColor: c, borderDash: dash, pointRadius: 0, tension: 0.1 }));
  ds.push({ label: `目標 被害 ${TGT.damage_ratio}`, data: xs.map(() => TGT.damage_ratio), borderColor: '#0e7490', borderDash: [1, 3], pointRadius: 0 });
  ds.push({ label: `目標 人口 ${TGT.pop_ratio}`, data: xs.map(() => TGT.pop_ratio), borderColor: '#7c3aed', borderDash: [1, 3], pointRadius: 0 });
  chart('r-series', { type: 'line', data: { labels: xs, datasets: ds }, options: { animation: false, plugins: { legend: { labels: { boxWidth: 12, font: { size: 10 } } },
    subtitle: { display: true, text: `経路: ${a.events.map((e) => `${e.year} ${short(e.to)}`).join(' → ')}${a.met ? '' : `（${a.first_violation}年に未達）`}` } },
    scales: { y: { title: { display: true, text: '÷ 対策なし・現在気候' } } } } });
}

function drawEdges() {
  const tree = E.trees[String(SEL)];
  const pairs = new Set(tree.filter((b) => b.parent != null).map((b) => `${tree.find((x) => x.id === b.parent).node}>${b.node}`));
  const es = EDGES.filter((e) => pairs.has(`${e.from}>${e.to}`));
  $('r-edges').innerHTML = '<tr><th>移行</th><th>施策</th><th>変更</th><th>土地利用</th><th>可逆性</th><th>リードタイム（名目）</th><th>移行の初期費用</th></tr>'
    + es.map((e) => e.changes.map((c, k) => `<tr>${k === 0 ? `<td rowspan="${e.changes.length}">${esc(short(e.from))} → ${esc(short(e.to))}${e.conditional ? '<br><span class="hint">条件付き（再転換）</span>' : ''}</td>` : ''}
      <td>${esc(c.var)}</td><td>${esc(c.from)} → ${esc(c.to)}${c.up ? '' : '（縮小）'}</td><td>${esc(c.land || '—')}</td><td>${c.reversible ? '可逆' : '不可逆'}</td><td>${fmt(c.lead_years, 1)} 年</td>
      ${k === 0 ? `<td rowspan="${e.changes.length}">${fmt(e.cost)} 億円</td>` : ''}</tr>`).join('')).join('');
  const M = D.model.measures, kinds = ['levee', 'retention', 'paddy_dam', 'forest', 'raise', 'relocate', 'pump_urban', 'pump_agri', 'storage_public', 'storage_private'];
  const ph = (m) => ['consensus', 'build', 'diffuse', 'mature'].map((k) => { const v = m[k]; const x = typeof v === 'object' ? `${v.nominal}（${v.low}–${v.high}）` : v; return +v || typeof v === 'object' ? `${{ consensus: '合意', build: '施工', diffuse: '普及', mature: '成熟' }[k]} ${x}` : null; }).filter(Boolean).join('・');
  $('r-edge-note').innerHTML = `移行の規則（設定 <code>robust.model.measures</code>・<code>robust.landuse</code> で変更可）: 水準を上げる移行はいつでも可（予算の範囲で、超える分は次の期へ繰越）。下げる移行は可逆な施策（田んぼダム・各戸貯留）だけ。
    森林は植栽後 ${M.forest.reconvert?.max_age_years ?? '—'} 年以内なら植栽費の ${pct(M.forest.reconvert?.cost_fraction ?? 0)} で農地に再転換できる（条件付き）。遊水地・移転跡地は戻さない。嵩上げ→移転は可（変更費用 ${pct(M.raise.change_cost_fraction)}）、逆は不可。<br>`
    + kinds.filter((k) => M[k]).map((k) => `${M[k].label}: ${ph(M[k])} 年・変更費用 ${pct(M[k].change_cost_fraction)}・${M[k].reversible ? '可逆' : '不可逆'}`).join('／');
}

function drawLand() {
  const lu = node(SEL).landuse, zs = Object.keys(lu), keys = zs.length ? Object.keys(lu[zs[0]]) : [];
  const base = node(0).landuse;
  $('r-land').innerHTML = `<tr><th>地域</th>${keys.map((k) => `<th>${k}</th>`).join('')}</tr>` + zs.map((z) => `<tr><td>${z}</td>${keys.map((k) => {
    const d = lu[z][k] - (base?.[z]?.[k] ?? 0); return `<td>${fmt(lu[z][k], 1)}${Math.abs(d) > 1e-6 ? ` <span class="hint">(${d > 0 ? '+' : ''}${fmt(d, 1)})</span>` : ''}</td>`; }).join('')}</tr>`).join('');
}

function drawUnr() {
  const U = E.unreachable || {};
  const L = { adaptive: '観測に基づくどの経路でも対応できない', perfect: '完全情報でも対応できない', best_initial: '最も有望な初期施策から対応できない' };
  if (!$('r-unr').options.length || $('r-unr').dataset.v !== JSON.stringify(Object.keys(U))) {
    $('r-unr').innerHTML = Object.keys(U).map((k) => `<option value="${k}">${L[k] || k}</option>`).join(''); $('r-unr').dataset.v = JSON.stringify(Object.keys(U)); $('r-unr').onchange = drawUnr;
  }
  const k = $('r-unr').value, b = U[k];
  if (!b) { $('r-prim').innerHTML = '<p class="hint">—</p>'; return; }
  $('r-prim').innerHTML = `<p>${L[k]}シナリオ: <b>${b.n_cases}</b> / ${b.n}${k === 'best_initial' ? `（${esc(short(b.node))}）` : ''}</p>`
    + (b.box.length ? `<table class="tbl"><tr><th>条件（PRIM の範囲）</th><th>範囲</th></tr>${b.box.map((d) => `<tr><td>${esc(d.label)}</td><td>${fmt(d.low, 3)} – ${fmt(d.high, 3)}</td></tr>`).join('')}</table>
      <p class="hint">範囲内の該当割合（密度）${pct(b.density)}・該当シナリオの範囲内の割合（カバー）${pct(b.coverage)}・範囲内のシナリオ ${pct(b.mass)}</p>` : '<p class="hint">該当シナリオがない（または全シナリオ）ため範囲は求めない。</p>');
  const names = Object.keys(D.factors), j1 = names.indexOf('rain_2100'), dims = (b.box || []).map((d) => names.indexOf(d.factor)).filter((j) => j !== j1);
  const j2 = dims[0] ?? names.indexOf('asset_2100');
  const F = D.factors, v = (s, j) => F[names[j]].low + D.Z[s][j] * (F[names[j]].high - F[names[j]].low);
  const bad = new Set(k === 'best_initial' ? E.records.adaptive.filter((r) => r.init === b.node && !r.met).map((r) => r.s) : E.scenarios.filter((s, i) => E.uncovered[k][i]));
  chart('r-scat', { type: 'scatter', data: { datasets: [
    { label: '対応できない', data: E.scenarios.filter((s) => bad.has(s)).map((s) => ({ x: v(s, j1), y: v(s, j2) })), backgroundColor: '#dc2626' },
    { label: 'その他', data: E.scenarios.filter((s) => !bad.has(s)).map((s) => ({ x: v(s, j1), y: v(s, j2) })), backgroundColor: '#94a3b8' }] },
  options: { animation: false, scales: { x: { title: { display: true, text: F[names[j1]].label } }, y: { title: { display: true, text: F[names[j2]].label } } } } });
}

function drawVer() {
  const V = D.verification_rri;
  if (!V) { $('r-ver').innerHTML = ''; $('r-ver-note').innerHTML = '未実施。<code>python scripts/run_dapp.py --robust --verify 6</code> で、有望な初期施策の代表的な経路の状態（最初の移行後・最終期）を RRI で再計算し、代理モデルと比べる。'; return; }
  $('r-ver').innerHTML = '<tr><th>初期施策</th><th>シナリオ</th><th>期</th><th>状態</th><th>RRI 被害額</th><th>代理モデル</th><th>差</th></tr>'
    + V.states.map((r) => `<tr><td>${esc(short(r.init))}</td><td>${r.s}</td><td>${r.t}</td><td>${r.branch ? '分岐（移行後）' : '最終期'}</td><td>${fmt(r.rri)}</td><td>${fmt(r.surrogate)}</td><td>${pct((r.surrogate - r.rri) / Math.max(r.rri, 1e-9), 1)}</td></tr>`).join('');
  $('r-ver-note').textContent = V.r2 != null ? `R² ${fmt(V.r2, 3)}・相対 RMSE ${pct(V.rel_rmse, 1)}（被害額, 資産係数を掛ける前）。` : '';
}
