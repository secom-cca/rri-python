// Adaptive DAPP viewer (results.json of `scripts/run_dapp.py --adaptive`, docs/11 11.11): decision rules per initial action,
// Pareto comparison of the initial actions, adaptive pathways map, future options, comparison with the 11.9 pathways.
import { toCSV, save, pngFromSVG } from '/static/dapp_options.js';

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : `${(100 * v).toFixed(d)}%`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const metCol = (m) => `hsl(${Math.round(120 * m)}, 65%, ${m > 0.5 ? 38 : 48}%)`;
const MON = { observed: '観測した被害・浸水人口の比（直前の期）', projected: '現状のままの予測（観測した気候・資産・人口の傾向, 次の期＋標準のリードタイム後）', rain: '観測した降雨の傾向から予測した2100年の降雨倍率' };
const WHY = { observed: '観測', projected: '予測', rain: '降雨の傾向', scale_back: '縮小（余裕あり）' };
const FONT = 'font-family="Hiragino Sans, Noto Sans JP, sans-serif"';
let D = null, SEL = null;
const row = (n) => D.table.find((r) => r.node === n);
const short = (r) => (r.node === 0 ? '施策なし' : `#${r.node} ${fmt(r.initial_cost)} 億円`);

export function render(data) {
  D = data;
  const rec = D.table.find((r) => r.recommended);
  SEL = rec ? rec.node : D.table[0].node;
  $('meta').textContent = `作成 ${D.created}・計算 ${fmt(D.elapsed_s / 60, 1)} 分・11.9 の実行 ${D.robust_run}・評価シナリオ ${D.n_eval}（探索 ${D.n_search}）・行動の候補 ${D.catalog.length}`;
  $('a-map-png').onclick = () => { const a = document.createElement('a'); a.href = $('a-map').toDataURL('image/png'); a.download = `${D.name}_adaptive_map_${SEL}.png`; a.click(); };
  drawSummary(); select(SEL); drawInputs(); drawHistory();
}

// ---------------------------------------------------------------- sensitivity: targets and information quality -> Recalculate
const SENS = (d) => d.sensitivity || { targets: { damage_ratio: d.settings.targets.damage_ratio, pop_ratio: d.settings.targets.pop_ratio, from_year: d.settings.targets.from_year },
  noise_damage: 1, noise_climate: 1, trend_window: 3, rain_from_year: d.settings.policy.rain_from_year, forecast_noise: 0, mode: 'search' };
const INP = [['damage_ratio', 'Target: 被害額の比 ≤', 0.05, (x) => x.targets.damage_ratio], ['pop_ratio', 'Target: 浸水人口の比 ≤', 0.05, (x) => x.targets.pop_ratio],
  ['from_year', 'Target の適用開始年', 10, (x) => x.targets.from_year],
  ['noise_damage', '観測ノイズ（被害・浸水人口）× 基準', 0.5, (x) => x.noise_damage], ['noise_climate', '観測ノイズ（降雨・海面・資産・人口）× 基準', 0.5, (x) => x.noise_climate],
  ['trend_window', '予測: 傾向を推定する観測の数', 1, (x) => x.trend_window], ['rain_from_year', 'シグナル: 降雨の傾向を使い始める年', 10, (x) => x.rain_from_year],
  ['forecast_noise', '予測誤差 σ（予測した降雨増の相対誤差; 適応ルールのみ）', 0.1, (x) => x.forecast_noise]];
const base = () => (D.gui_inputs || {}).base || D.name;
function drawInputs() {
  const S = SENS(D), m = D.sensitivity?.monitoring;
  $('a-inputs').innerHTML = `<table class="tbl">${INP.map(([k, l, st, f]) => `<tr><td>${l}</td><td><input type="number" id="ai-${k}" step="${st}" value="${f(S)}" style="width:80px"></td></tr>`).join('')}</table>
    <div class="hint">${m ? `表示中の観測ノイズ: 被害 ${fmt(m.risk_noise, 3)}・降雨 ${fmt(m.rain_noise, 3)}・海面 ${fmt(m.slr_noise, 3)} m・資産と人口 ${fmt(m.exposure_noise, 3)}（相対・標準偏差）。` : ''}
      観測ノイズと Target・傾向の観測数は両方式に、降雨シグナルの開始年と予測誤差は適応ルールにだけ効く（11.9 方式は自身の予測器を使う）。</div>
    <div style="margin-top:6px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
      <select id="ai-mode"><option value="fixed">ルールを固定して再評価（約1分）</option><option value="search">ルールを再探索</option></select>
      <select id="ai-size"><option value="fast">速い（探索 12 シナリオ, 約3分）</option><option value="standard">標準（探索 24, 約13分）</option></select>
      <button id="ai-go" style="font-weight:700;padding:4px 14px">Recalculate</button><button id="ai-reset">表示中の条件に戻す</button></div>
    <div id="ai-note" class="hint" style="margin-top:4px"></div>
    <div id="ai-prog" style="display:none;margin-top:6px"><div style="height:8px;background:#e2e8f0;border-radius:4px"><div id="ai-bar" style="height:8px;width:0;background:#0e7490;border-radius:4px"></div></div><div id="ai-msg" class="hint"></div></div>`;
  $('a-inputs').querySelectorAll('input').forEach((el) => { el.oninput = mark; });
  $('ai-mode').onchange = () => { $('ai-size').disabled = $('ai-mode').value === 'fixed'; };
  $('ai-size').disabled = true;
  $('ai-reset').onclick = drawInputs; $('ai-go').onclick = recalc;
  mark();
}
function readInputs() {
  const v = (k) => { const x = $(`ai-${k}`).value; return x === '' ? null : +x; };
  return { targets: { damage_ratio: v('damage_ratio'), pop_ratio: v('pop_ratio'), from_year: v('from_year') }, noise_damage: v('noise_damage'), noise_climate: v('noise_climate'),
    trend_window: v('trend_window'), rain_from_year: v('rain_from_year'), forecast_noise: v('forecast_noise') };
}
function mark() {
  const S = SENS(D), I = readInputs();
  let n = 0;
  INP.forEach(([k, , , f]) => { const cur = ['damage_ratio', 'pop_ratio', 'from_year'].includes(k) ? I.targets[k] : I[k], ch = Math.abs((cur ?? 0) - (f(S) ?? 0)) > 1e-9;
    $(`ai-${k}`).style.background = ch ? '#fef08a' : ''; n += ch; });
  $('ai-note').innerHTML = n ? `<b style="color:#b45309">未計算の変更 ${n} 件</b>（黄色）→ Recalculate` : '表示中の結果の条件。値を変えて Recalculate。';
}
async function recalc() {
  const body = { name: D.name, mode: $('ai-mode').value, size: $('ai-size').value, ...readInputs() };
  $('ai-go').disabled = true; $('ai-prog').style.display = ''; $('ai-bar').style.width = '0'; $('ai-msg').textContent = '送信中';
  try {
    const r = await fetch('/api/dapp/adaptive/recalculate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    if (j.job) {
      for (;;) {
        await new Promise((ok) => setTimeout(ok, 1500));
        const st = await (await fetch(`/api/job/${j.job}`)).json();
        $('ai-bar').style.width = `${Math.round(100 * st.progress)}%`; $('ai-msg').textContent = `${Math.round(100 * st.progress)}%・${Math.round(st.elapsed_s)} 秒・${st.message}`;
        if (st.status === 'done') break;
        if (st.status === 'error') throw new Error(st.message.split('\n')[0]);
      }
    }
    await window.dappReload(j.name);
  } catch (e) { $('ai-msg').innerHTML = `<span style="color:#b91c1c">エラー: ${esc(e.message)}</span>`; $('ai-go').disabled = false; }
}
let HIST = [];
const SX = { noise_both: ['観測ノイズ（すべて）× 基準', (x) => (x.sensitivity?.noise_damage === x.sensitivity?.noise_climate ? x.sensitivity?.noise_damage : null)],
  damage_ratio: ['Target: 被害額の比', (x) => x.sensitivity?.targets?.damage_ratio], pop_ratio: ['Target: 浸水人口の比', (x) => x.sensitivity?.targets?.pop_ratio],
  noise_damage: ['観測ノイズ（被害）× 基準', (x) => x.sensitivity?.noise_damage], noise_climate: ['観測ノイズ（降雨・資産）× 基準', (x) => x.sensitivity?.noise_climate],
  trend_window: ['傾向の観測数', (x) => x.sensitivity?.trend_window], rain_from_year: ['降雨シグナルの開始年', (x) => x.sensitivity?.rain_from_year],
  forecast_noise: ['予測誤差 σ', (x) => x.sensitivity?.forecast_noise] };
async function drawHistory() {
  HIST = await (await fetch(`/api/dapp/adaptive/summaries/${encodeURIComponent(base())}`)).json();
  const cur = SENS(D), dif = (a, b) => (a != null && b != null && Math.abs(a - b) > 1e-9 ? ' style="background:#fef9c3"' : '');
  $('a-hist').innerHTML = `<tr><th>実行（作成）</th><th>モード</th><th>評価</th><th>Target 被害/人口/開始</th><th>観測ノイズ 被害×/気候×</th><th>傾向の観測数</th><th>降雨シグナル</th><th>予測誤差 σ</th>
    <th>平均 Coverage<br>適応 / 11.9</th><th>適応 ＞ 11.9</th><th>推奨初期行動<br>Coverage 適応 / 11.9</th><th>平均費用<br>適応 / 11.9</th></tr>`
    + HIST.map((r) => { const s = r.sensitivity || {}, t = s.targets || {}; return `<tr data-n="${esc(r.name)}" class="${r.name === D.name ? 'rec' : ''}">
      <td>${esc(r.name === r.base ? '基準（CLI）' : r.name.slice(r.base.length + 1))}<br><span class="hint">${esc(r.created)}</span></td>
      <td>${r.mode === 'fixed' ? 'ルール固定' : `再探索${r.size ? `（${r.size}）` : ''}`}</td><td>${r.n_eval}</td>
      <td${dif(t.damage_ratio, cur.targets.damage_ratio) || dif(t.pop_ratio, cur.targets.pop_ratio) || dif(t.from_year, cur.targets.from_year)}>${fmt(t.damage_ratio, 2)} / ${fmt(t.pop_ratio, 2)} / ${t.from_year ?? '—'}</td>
      <td${dif(s.noise_damage, cur.noise_damage) || dif(s.noise_climate, cur.noise_climate)}>${fmt(s.noise_damage, 1)} / ${fmt(s.noise_climate, 1)}</td><td${dif(s.trend_window, cur.trend_window)}>${s.trend_window ?? '—'}</td>
      <td${dif(s.rain_from_year, cur.rain_from_year)}>${s.rain_from_year ?? '—'}</td><td${dif(s.forecast_noise, cur.forecast_noise)}>${fmt(s.forecast_noise, 2)}</td>
      <td><b style="color:${metCol(r.mean_coverage.adaptive)}">${pct(r.mean_coverage.adaptive)}</b> / ${pct(r.mean_coverage.robust)}</td><td>${r.n_better} / ${r.n_inits}</td>
      <td>#${r.recommended.node}: ${pct(r.recommended.coverage)} / ${pct(r.recommended.robust_coverage)}</td><td>${fmt(r.mean_cost.adaptive)} / ${fmt(r.mean_cost.robust)}</td></tr>`; }).join('');
  $('a-hist').querySelectorAll('tr[data-n]').forEach((tr) => { tr.onclick = () => { if (tr.dataset.n !== D.name) window.dappReload(tr.dataset.n); }; });
  const flat = HIST.map((r) => ({ name: r.name, created: r.created, mode: r.mode, size: r.size, n_eval: r.n_eval, ...Object.fromEntries(Object.entries(SX).map(([k, [, f]]) => [k, f(r)])),
    from_year: r.sensitivity?.targets?.from_year, coverage_adaptive: r.mean_coverage.adaptive, coverage_robust: r.mean_coverage.robust, n_better: r.n_better,
    recommended: r.recommended.node, recommended_coverage: r.recommended.coverage, cost_adaptive: r.mean_cost.adaptive, cost_robust: r.mean_cost.robust }));
  buttons('a-hist-x', null, () => toCSV(flat), () => HIST, 'sensitivity_runs');
  if (!$('a-sens-x').options.length) { $('a-sens-x').innerHTML = Object.entries(SX).map(([k, [l]]) => `<option value="${k}">${l}</option>`).join(''); $('a-sens-x').onchange = drawSens; }
  drawSens();
}
function drawSens() {
  const k = $('a-sens-x').value, [lab, f] = SX[k];
  const me = HIST.find((r) => r.name === D.name) || HIST[0];
  const skip = (q) => q === k || q === 'noise_both' || (k === 'noise_both' && q.startsWith('noise'));
  const same = (r) => r.n_eval === me.n_eval && Object.entries(SX).every(([q, [, g]]) => skip(q) || Math.abs((g(r) ?? 0) - (g(me) ?? 0)) < 1e-9)
    && (r.sensitivity?.targets?.from_year ?? 0) === (me.sensitivity?.targets?.from_year ?? 0);
  const P = HIST.filter((r) => f(r) != null && same(r)).sort((a, b) => f(a) - f(b));          // one at a time: the other conditions as displayed
  const W = 560, H = 280, m = { l: 52, r: 16, t: 14, b: 40 };
  const xs = P.map(f), lo = Math.min(...xs, 0), hi = Math.max(...xs, 1);
  const pad = (hi - lo) * 0.08 || 1, X = lin(lo - pad, hi + pad, m.l, W - m.r), Y = lin(0, 1, H - m.b, m.t);
  let b = frame(W, H, m, X, Y, tk(lo, hi, 4), tk(0, 1), lab, 'Coverage', (v) => fmt(v, k.includes('year') || k === 'trend_window' ? 0 : 2), (v) => pct(v));
  const col = (r) => (r.mode === 'fixed' ? '#0e7490' : '#b45309');
  const line = (g, c, d) => (P.length > 1 ? `<polyline points="${P.map((r) => `${X(f(r))},${Y(g(r))}`).join(' ')}" fill="none" stroke="${c}" stroke-width="1.6" ${d ? 'stroke-dasharray="4 3"' : ''}/>` : '');
  b += line((r) => r.mean_coverage.adaptive, '#0e7490') + line((r) => r.mean_coverage.robust, '#64748b', true) + line((r) => r.recommended.coverage, '#f59e0b', true);
  P.forEach((r) => { const x = X(f(r)), cur = r.name === D.name;
    b += `<line x1="${x}" x2="${x}" y1="${Y(r.mean_coverage.robust)}" y2="${Y(r.mean_coverage.adaptive)}" stroke="#cbd5e1"/>`;
    b += `<circle cx="${x}" cy="${Y(r.mean_coverage.robust)}" r="4" fill="#fff" stroke="${col(r)}"/>`;
    b += `<circle cx="${x}" cy="${Y(r.mean_coverage.adaptive)}" r="${cur ? 7 : 5}" fill="${col(r)}" stroke="${cur ? '#0f172a' : 'none'}" stroke-width="2"><title>${esc(r.name)}</title></circle>`;
    b += `<path d="M${x},${Y(r.recommended.coverage) - 5}l5,5l-5,5l-5,-5z" fill="#f59e0b"/>`; });
  b += `<g font-size="11"><circle cx="${W - 210}" cy="${m.t + 6}" r="5" fill="#0e7490"/><text x="${W - 201}" y="${m.t + 10}">ルール固定</text><circle cx="${W - 130}" cy="${m.t + 6}" r="5" fill="#b45309"/><text x="${W - 121}" y="${m.t + 10}">再探索</text></g>`;
  if (P.length < 2) b += `<text x="${m.l + 10}" y="${m.t + 30}" font-size="12" fill="#b45309">この変数だけを変えた実行がまだない（他の条件を表示中の実行に合わせて Recalculate）</text>`;
  $('a-sens').innerHTML = svg(W, H, b);
  const rows = P.map((r) => ({ name: r.name, mode: r.mode, x: f(r), coverage_adaptive: r.mean_coverage.adaptive, coverage_robust: r.mean_coverage.robust, recommended_coverage: r.recommended.coverage }));
  buttons('a-sens-x-x', 'a-sens', () => toCSV(rows), () => rows, `sensitivity_${k}`);
}
function select(n) { SEL = n; drawTable(); drawScatter(); drawRule(); drawMap(); drawSeries(); drawFront(); drawLoss(); }

function ruleHTML(p) {
  const mon = p.monitor.map((m) => `<li>${MON[m]}</li>`).join('');
  const trig = [p.monitor.includes('observed') ? `観測した比 ÷ 目標 ＞ <b>${fmt(p.rho, 2)}</b>` : null, p.monitor.includes('projected') ? `予測した比 ÷ 目標 ＞ <b>${fmt(p.rho, 2)}</b>` : null,
    p.monitor.includes('rain') ? `予測した2100年の降雨 ≥ <b>×${fmt(p.rain, 2)}</b>（${D.settings.policy.rain_from_year}年以降）` : null].filter(Boolean).join(' または ');
  return `<div><b>監視指標</b><ul style="margin:2px 0 4px 18px;padding:0">${mon}</ul>
    <div><b>トリガー</b>: ${trig} のとき行動する（前の行動が予算待ちのときは待つ）</div>
    <div><b>次の行動の選び方</b>: いま実行可能な行動（土地・再転換期限・総予算の残り・2100年までに完成）を、完成後 <b>${fmt(p.horizon)}</b> 年の予測効果で採点し、上位 <b>${p.k}</b> 個を実施:
      <div class="hint" style="margin-left:12px">得点＝効果 ÷ (1 ＋ <b>${fmt(p.w_cost, 1)}</b> × 費用/${fmt(D.settings.policy.cost_ref_oku)}億円) − <b>${fmt(p.w_irr, 1)}</b> × ${D.settings.policy.irr_penalty} × 不可逆 − <b>${fmt(p.w_lead, 1)}</b> × ${D.settings.policy.lead_penalty} × リードタイム/${D.settings.policy.lead_ref_years}年（得点 ＞ 0、効果 ＞ ${D.settings.policy.min_benefit}）</div></div>
    <div><b>縮小</b>: ${p.scale_back ? `観測・予測とも ＜ ${fmt(p.scale_back, 2)} なら、可逆な施策（田んぼダム・各戸貯留）のうち維持費の最も大きいものを縮小（20年以内に変更したものは除く）` : 'しない'}</div></div>`;
}
function drawSummary() {
  const r = D.table.find((x) => x.recommended) || D.table[0], a = r.adaptive, b = r.robust;
  const kpi = (l, v, w) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="l">${w}</div></div>`;
  $('a-summary').innerHTML = `<h2>Adaptive DAPP: どの指標を監視し、どの条件で、次にどの施策を実施するか（状態依存の意思決定ルール）</h2>
    <div style="display:grid;grid-template-columns:minmax(380px,1fr) minmax(380px,1fr);gap:14px">
      <div><div class="rec-box"><b>推奨初期行動（今実施すべき施策）</b>: ${esc(short(r))}<div class="hint">${esc(r.label)}</div></div>
        <div class="kpis" style="margin-top:8px">${kpi('Coverage（適応ルール）', pct(a.coverage), `11.9 方式 ${pct(b.coverage)}`)}${kpi('平均累積費用 PV', `${fmt(a.mean_cost)} 億円`, `11.9 方式 ${fmt(b.mean_cost)}`)}
          ${kpi('最大 Regret', fmt(a.max_regret, 3), `11.9 方式 ${fmt(b.max_regret, 3)}`)}${kpi('残る選択肢（2050年）', pct(a.remaining_2050), `11.9 方式 ${pct(b.remaining_2050)}`)}</div></div>
      <div><b>推奨初期行動の後の適応ルール</b>${ruleHTML(r.policy)}</div></div>
    <p class="hint">経路を事前に与えず、各時点で実行可能な行動から次の施策を選ぶルールを、初期行動ごとに探索シナリオ（${D.n_search}）上の 2020–2100年の経路全体（Coverage・費用・効用・残る選択肢）で探索した。
      途中で一時的に非 Pareto な状態も捨てない。経路と分岐の数は固定しない（観測の違いから生じる）。</p>`;
}

function drawTable() {
  const H = `<tr><th rowspan="2">初期行動</th><th colspan="7">適応ルール（Adaptive DAPP）</th><th colspan="5">11.9 方式（同じ初期施策）</th><th rowspan="2">Lock-in<br><span class="hint">初期施策</span></th></tr>
    <tr><th>Coverage</th><th>費用</th><th>最大 Regret</th><th>残る選択肢<br><span class="hint">2050 / 平均</span></th><th>Option loss<br><span class="hint">2050</span></th><th>行動の数</th><th>監視指標</th>
    <th>Coverage</th><th>費用</th><th>最大 Regret</th><th>残る選択肢<br><span class="hint">2050 / 平均</span></th><th>移行の数</th></tr>`;
  $('a-table').innerHTML = H + D.table.map((r) => { const a = r.adaptive, b = r.robust; return `<tr data-n="${r.node}" class="${r.node === SEL ? 'rec' : ''}">
    <td>${r.recommended ? '◎ ' : ''}${r.pareto ? '★ ' : ''}${esc(short(r))}</td>
    <td style="color:${metCol(a.coverage)};font-weight:700">${pct(a.coverage)}</td><td>${fmt(a.mean_cost)}</td><td>${fmt(a.max_regret, 3)}</td>
    <td>${pct(a.remaining_2050)} / ${pct(a.remaining_mean)}</td><td>${pct(a.option_loss_2050)}</td><td>${fmt(a.n_acts, 1)}</td><td>${r.policy.monitor.map((m) => WHY[m]).join('・')}</td>
    <td style="color:${metCol(b.coverage)}">${pct(b.coverage)}</td><td>${fmt(b.mean_cost)}</td><td>${fmt(b.max_regret, 3)}</td><td>${pct(b.remaining_2050)} / ${pct(b.remaining_mean)}</td><td>${fmt(b.n_acts, 1)}</td>
    <td>${pct(r.lockin)}</td></tr>`; }).join('');
  $('a-table').querySelectorAll('tr[data-n]').forEach((tr) => { tr.onclick = () => select(+tr.dataset.n); });
  const flat = D.table.map((r) => ({ node: r.node, label: r.label, pareto: !!r.pareto, recommended: !!r.recommended, lockin: r.lockin,
    ...Object.fromEntries(['coverage', 'mean_cost', 'max_regret', 'mean_regret', 'remaining_2050', 'remaining_mean', 'option_loss_2050', 'n_acts'].flatMap((k) => [[`adaptive_${k}`, r.adaptive[k]], [`robust_${k}`, r.robust[k]]])),
    monitor: r.policy.monitor.join('+'), rho: r.policy.rho, rain: r.policy.rain, w_cost: r.policy.w_cost, w_irr: r.policy.w_irr, w_lead: r.policy.w_lead, horizon: r.policy.horizon, k: r.policy.k, scale_back: r.policy.scale_back }));
  buttons('a-table-x', null, () => toCSV(flat), () => flat, 'initial_actions');
}

// ---------------------------------------------------------------- SVG figures
const svg = (W, H, body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W * 1.25}px" ${FONT}><rect width="${W}" height="${H}" fill="#fff"/>${body}</svg>`;
const lin = (d0, d1, r0, r1) => (v) => r0 + (v - d0) * (r1 - r0) / ((d1 - d0) || 1);
function frame(W, H, m, X, Y, xt, yt, xl, yl, xf, yf) {
  let s = '';
  yt.forEach((v) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e2e8f0"/><text x="${m.l - 5}" y="${Y(v) + 4}" font-size="10" text-anchor="end" fill="#475569">${yf(v)}</text>`; });
  xt.forEach((v) => { s += `<text x="${X(v)}" y="${H - m.b + 14}" font-size="10" text-anchor="middle" fill="#475569">${xf(v)}</text>`; });
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="#334155"/><line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}" stroke="#334155"/>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" font-size="11" text-anchor="middle">${esc(xl)}</text><text transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" font-size="11" text-anchor="middle">${esc(yl)}</text>`;
  return s;
}
const tk = (a, b, n = 5) => Array.from({ length: n + 1 }, (_, k) => a + k * (b - a) / n);

function drawScatter() {
  const W = 560, H = 360, m = { l: 52, r: 16, t: 14, b: 40 }, X = lin(0, 1, m.l, W - m.r), Y = lin(0, 1, H - m.b, m.t);
  let b = frame(W, H, m, X, Y, tk(0, 1), tk(0, 1), '残る将来の選択肢（2020–2090年の平均）', 'Coverage', (v) => pct(v), (v) => pct(v));
  D.table.forEach((r) => {
    const a = r.adaptive, o = r.robust, sel = r.node === SEL;
    b += `<line x1="${X(o.remaining_mean)}" y1="${Y(o.coverage)}" x2="${X(a.remaining_mean)}" y2="${Y(a.coverage)}" stroke="#94a3b8" stroke-dasharray="3 3"/>`;
    b += `<circle cx="${X(o.remaining_mean)}" cy="${Y(o.coverage)}" r="4" fill="#fff" stroke="#64748b"/>`;
    b += `<circle cx="${X(a.remaining_mean)}" cy="${Y(a.coverage)}" r="${sel ? 8 : 5.5}" fill="${r.pareto ? '#0e7490' : '#94a3b8'}" stroke="${sel ? '#b45309' : '#0f172a'}" stroke-width="${sel ? 2.5 : 0.8}" data-n="${r.node}" style="cursor:pointer"><title>${esc(short(r))}</title></circle>`;
    b += `<text x="${X(a.remaining_mean) + 8}" y="${Y(a.coverage) + 4}" font-size="10">${r.node === 0 ? '施策なし' : `#${r.node}`}${r.recommended ? ' ◎' : ''}</text>`;
  });
  $('a-scatter').innerHTML = svg(W, H, b);
  $('a-scatter').querySelectorAll('circle[data-n]').forEach((c) => { c.onclick = () => select(+c.dataset.n); });
  const rows = D.table.map((r) => ({ node: r.node, adaptive_coverage: r.adaptive.coverage, adaptive_remaining: r.adaptive.remaining_mean, robust_coverage: r.robust.coverage, robust_remaining: r.robust.remaining_mean, pareto: !!r.pareto }));
  buttons('a-scatter-x', 'a-scatter', () => toCSV(rows), () => rows, 'coverage_vs_options');
}

function drawSeries() {
  const r = row(SEL), years = D.years, A = r.adaptive.options, B = r.robust.options;
  const W = 900, H = 240, w = 280, m = { l: 50, r: 10, t: 24, b: 36 };
  const panel = (k, title, ga, gb, lo, hi, yf, band) => {
    const ox = k * (w + 26), X = lin(2020, 2100, ox + m.l, ox + w - m.r), Y = lin(lo, hi, H - m.b, m.t), xs = years.map((y) => y + 5);
    let s = `<text x="${ox + m.l}" y="14" font-size="12" font-weight="700">${esc(title)}</text>`;
    tk(lo, hi, 4).forEach((v) => { s += `<line x1="${ox + m.l}" x2="${ox + w - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e2e8f0"/><text x="${ox + m.l - 4}" y="${Y(v) + 4}" font-size="10" text-anchor="end" fill="#475569">${yf(v)}</text>`; });
    [2020, 2060, 2100].forEach((y) => { s += `<text x="${X(y)}" y="${H - m.b + 14}" font-size="10" text-anchor="middle" fill="#475569">${y}</text>`; });
    if (band) s += `<path d="M${xs.map((x, n) => `${X(x)},${Y(band[0][n])}`).join('L')}L${xs.map((x, n) => `${X(x)},${Y(band[1][n])}`).reverse().join('L')}Z" fill="#0e7490" fill-opacity="0.15"/>`;
    const line = (v, c, d) => `<polyline points="${xs.map((x, n) => `${X(x)},${Y(v[n])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="2.2" ${d ? 'stroke-dasharray="5 4"' : ''}/>`;
    return s + line(ga, '#0e7490') + line(gb, '#64748b', true);
  };
  const cmax = Math.max(...A.cost_cum, ...B.cost_cum) * 1.1;
  let b = panel(0, 'Performance: 目標を満たし続けている割合', A.met_so_far, B.met_so_far, 0, 1, (v) => pct(v));
  b += panel(1, 'Cost: 累積費用 PV（億円）', A.cost_cum, B.cost_cum, 0, cmax, (v) => fmt(v));
  b += panel(2, 'Remaining Future Options', A.remaining.mean, B.remaining.mean, 0, 1, (v) => pct(v), [A.remaining.p10, A.remaining.p90]);
  $('a-series').innerHTML = svg(W, H, b);
  const rows = years.map((y, n) => ({ year: y, adaptive_met_so_far: A.met_so_far[n], robust_met_so_far: B.met_so_far[n], adaptive_cost_cum: A.cost_cum[n], robust_cost_cum: B.cost_cum[n],
    adaptive_remaining: A.remaining.mean[n], robust_remaining: B.remaining.mean[n], adaptive_used: A.used[n], robust_used: B.used[n] }));
  buttons('a-series-x', 'a-series', () => toCSV(rows), () => ({ node: SEL, rows }), `timeseries_${SEL}`);
}

function drawFront() {
  const F = row(SEL).front, W = 520, H = 300, m = { l: 52, r: 16, t: 14, b: 40 };
  if (!F.length) { $('a-front').innerHTML = '<p class="hint">ルールを固定した再評価の実行なので、ルールの探索はしていない（探索の非劣解は元の実行を参照）。</p>'; $('a-front-x').innerHTML = ''; return; }
  const c0 = Math.min(...F.map((f) => f.mean_cost)), c1 = Math.max(...F.map((f) => f.mean_cost));
  const X = lin(c0 * 0.95, c1 * 1.05, m.l, W - m.r), Y = lin(0, 1, H - m.b, m.t);
  let b = frame(W, H, m, X, Y, tk(c0 * 0.95, c1 * 1.05, 4), tk(0, 1), '平均累積費用 PV（億円）', 'Coverage（探索シナリオ）', (v) => fmt(v), (v) => pct(v));
  F.forEach((f) => { b += `<circle cx="${X(f.mean_cost)}" cy="${Y(f.coverage)}" r="${f.chosen ? 8 : 4.5}" fill="hsl(${Math.round(200 * f.remaining)},60%,45%)" stroke="${f.chosen ? '#b45309' : '#0f172a'}" stroke-width="${f.chosen ? 2.5 : 0.6}"><title>残る選択肢 ${pct(f.remaining)}・監視 ${f.policy.monitor.join('+')}・ρ ${f.policy.rho}</title></circle>`; });
  $('a-front').innerHTML = svg(W, H, b);
  const rows = F.map((f) => ({ coverage: f.coverage, mean_cost: f.mean_cost, remaining: f.remaining, chosen: f.chosen, ...f.policy, monitor: f.policy.monitor.join('+') }));
  buttons('a-front-x', 'a-front', () => toCSV(rows), () => F, `rules_${SEL}`);
}

function drawLoss() {
  const r = row(SEL), y = D.years, n50 = y.indexOf(2050), n90 = y.indexOf(2090), L = D.reason_labels;
  const v = (o, q, n) => o.options.loss[q][n];
  $('a-loss').innerHTML = '<tr><th>理由</th><th>適応 2050</th><th>適応 2090</th><th>11.9 2050</th><th>11.9 2090</th></tr>'
    + D.reasons.filter((q) => q !== 'transition_unavailable').map((q) => `<tr><td>${esc(L[q])}</td><td>${pct(v(r.adaptive, q, n50), 1)}</td><td>${pct(v(r.adaptive, q, n90), 1)}</td>
      <td>${pct(v(r.robust, q, n50), 1)}</td><td>${pct(v(r.robust, q, n90), 1)}</td></tr>`).join('')
    + `<tr><td>実施済み（used）</td><td>${pct(r.adaptive.options.used[n50])}</td><td>${pct(r.adaptive.options.used[n90])}</td><td>${pct(r.robust.options.used[n50])}</td><td>${pct(r.robust.options.used[n90])}</td></tr>`;
}

function drawRule() {
  const r = row(SEL);
  $('a-sel').textContent = short(r);
  $('a-rule').innerHTML = `<div class="hint">${esc(r.label)}</div>${ruleHTML(r.policy)}`;
}

// ---------------------------------------------------------------- adaptive pathways map (trie over the actions)
function drawMap() {
  const tree = D.trees[String(SEL)], tot = tree[0].n, cat = D.catalog;
  const minN = Math.max(1, Math.round(0.03 * tot)), T = tree.filter((b) => b.n >= minN), byId = Object.fromEntries(tree.map((b) => [b.id, b]));
  const key = (b) => (b.kind === 'act' ? `a${b.a}` : 'init');
  const first = {};
  T.filter((b) => b.kind === 'act').forEach((b) => { first[b.a] = Math.min(first[b.a] ?? 9999, b.year); });
  const rows = ['init', ...Object.keys(first).map(Number).sort((x, y) => first[x] - first[y] || x - y).map((x) => `a${x}`)];
  const cv = $('a-map'), ctx = cv.getContext('2d'), dpr = devicePixelRatio || 1, W = cv.clientWidth, H = Math.max(200, 26 * rows.length + 60);
  cv.style.height = `${H}px`; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
  const L = 250, R = 80, Tp = 16, X = (y) => L + (y - 2020) / 80 * (W - L - R), Y = (k) => Tp + rows.indexOf(k) * 26 + 10;
  ctx.font = '12px "Hiragino Sans", sans-serif'; ctx.textBaseline = 'middle';
  for (let yr = 2020; yr <= 2100; yr += 10) { ctx.strokeStyle = '#e2e8f0'; ctx.beginPath(); ctx.moveTo(X(yr), Tp - 8); ctx.lineTo(X(yr), H - 22); ctx.stroke(); ctx.fillStyle = '#64748b'; ctx.fillText(yr, X(yr) - 14, H - 9); }
  rows.forEach((k) => { const t = k === 'init' ? `初期: ${(tree[0].acts || []).length} 施策` : cat[+k.slice(1)]; ctx.fillStyle = '#334155'; ctx.fillText(t.length > 30 ? `${t.slice(0, 29)}…` : t, 6, Y(k)); });
  const off = {};
  T.forEach((b) => { const k = key(b); off[b.id] = (((off[k] = (off[k] ?? -1) + 1) % 5) - 2) * 2.2; });
  const yy = (b) => Y(key(b)) + (off[b.id] || 0), lw = (b) => 1.2 + 8 * b.n / tot, kids = (b) => T.filter((c) => c.parent === b.id);
  T.forEach((b) => {
    const ch = kids(b), end = b.end || !ch.length ? 2100 : Math.max(...ch.map((c) => c.year));
    ctx.strokeStyle = metCol(b.met_share); ctx.lineWidth = lw(b); ctx.beginPath(); ctx.moveTo(X(b.year), yy(b)); ctx.lineTo(X(end), yy(b)); ctx.stroke();
    ch.forEach((c) => {
      ctx.fillStyle = 'rgba(100,116,139,.15)'; ctx.fillRect(X(c.year_range[0]), yy(c) - 6, Math.max(X(c.year_range[1]) - X(c.year_range[0]), 2), 12);
      ctx.strokeStyle = metCol(c.met_share); ctx.lineWidth = lw(c); ctx.beginPath(); ctx.moveTo(X(c.year), yy(b)); ctx.lineTo(X(c.year), yy(c)); ctx.stroke();
    });
    ctx.lineWidth = 1.2; ctx.strokeStyle = '#0f172a'; ctx.fillStyle = Object.keys(b.reasons || {})[0] === 'scale_back' ? '#bfdbfe' : '#fff';
    ctx.beginPath(); ctx.arc(X(b.year), yy(b), 3.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  });
  ctx.fillStyle = '#64748b'; if (tree.length > T.length) ctx.fillText(`（${minN} シナリオ未満の枝 ${tree.length - T.length} は省略、全 ${tree.length - 1} 枝）`, L, H - 30);
  const path = (b) => { const o = []; for (let x = byId[b.parent]; x && x.parent != null; x = byId[x.parent]) o.unshift(x.label); return o.join(' → ') || '初期行動'; };
  $('a-branch').innerHTML = '<tr><th>年（中央値・範囲）</th><th>直前までの行動 → 次の行動</th><th>シナリオ</th><th>引いた指標</th><th>観測した比 ÷ 目標</th><th>予測した2100年の降雨</th><th>リードタイム</th><th>目標の達成</th></tr>'
    + tree.filter((b) => b.parent != null && b.n >= minN).sort((x, y) => x.year - y.year || y.n - x.n).map((b) => `<tr><td>${b.year}<br><span class="hint">${b.year_range[0] === b.year_range[1] ? '' : `${b.year_range[0]}–${b.year_range[1]}`}</span></td>
      <td><span class="hint">${esc(path(b))}</span> → <b>${esc(b.label)}</b></td><td>${b.n}（${pct(b.n / tot)}）</td>
      <td>${Object.entries(b.reasons).map(([k, v]) => `${WHY[k] || k} ${v}`).join('・')}</td><td>${b.obs_ratio ? `${fmt(b.obs_ratio[0], 2)}–${fmt(b.obs_ratio[1], 2)}` : '—'}</td>
      <td>${b.rain2100 ? `${fmt(b.rain2100[0], 3)}–${fmt(b.rain2100[1], 3)}` : '—'}</td><td>${b.lead != null ? `${fmt(b.lead)} 年` : '—'}</td><td style="color:${metCol(b.met_share)}">${pct(b.met_share)}</td></tr>`).join('');
}

function buttons(boxId, svgId, csv, json, base) {
  const box = $(boxId); if (!box) return;
  box.innerHTML = `${svgId ? '<button data-k="svg">SVG</button><button data-k="png">PNG</button>' : ''}<button data-k="csv">CSV</button><button data-k="json">JSON</button>`;
  box.querySelectorAll('button').forEach((bt) => { bt.onclick = () => {
    const k = bt.dataset.k, nm = `${D.name}_${base}`;
    if (k === 'svg') save(new Blob([new XMLSerializer().serializeToString($(svgId).querySelector('svg'))], { type: 'image/svg+xml' }), `${nm}.svg`);
    else if (k === 'png') pngFromSVG($(svgId), `${nm}.png`);
    else if (k === 'csv') save(new Blob(['﻿' + csv()], { type: 'text/csv' }), `${nm}.csv`);
    else save(new Blob([JSON.stringify(json(), null, 1)], { type: 'application/json' }), `${nm}.json`);
  }; });
}
