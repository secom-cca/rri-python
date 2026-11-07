// Future option space of Robust DAPP pathways (evaluation.option_space, docs/11 11.9.5): extended scorecard and presentation
// figures as SVG (each with SVG / PNG / CSV / JSON export).  Called by dapp_robust.js.
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (v, d = 0) => (v == null || !Number.isFinite(v) ? '—' : `${(100 * v).toFixed(d)}%`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const RCOL = { irreversible_land_use: '#7c3aed', land_use_conflict: '#b45309', irreversible_measure: '#be123c', transition_unavailable: '#475569',
  lead_time: '#0891b2', budget: '#ca8a04' };
const SHORT = { irreversible_land_use: '不可逆な土地利用', land_use_conflict: '土地利用の競合', irreversible_measure: '不可逆な施策',
  transition_unavailable: '移行できない', lead_time: 'リードタイム', budget: '予算' };
const FONT = 'font-family="Hiragino Sans, Noto Sans JP, sans-serif"';
let C = null;          // {D, E, SEL, short(node), select(node)}

export function renderOptions(ctx) {
  C = ctx;
  const O = C.E.option_space;
  document.querySelectorAll('.opt-card').forEach((el) => { el.style.display = O ? '' : 'none'; });
  if (!O) return;
  if (!$('o-y').options.length) {
    $('o-y').innerHTML = [['coverage', 'Coverage（目標を全期間満たす割合）'], ['mean_damage_ratio', 'Performance: 被害額の比（期間平均）'], ['max_regret', '最大 Regret']]
      .map(([k, l]) => `<option value="${k}">${l}</option>`).join('');
    $('o-x').innerHTML = [['remaining_options_2050', '2050年に残る選択肢'], ['remaining_options_mean', '残る選択肢（2020–2090年の平均）']]
      .map(([k, l]) => `<option value="${k}">${l}</option>`).join('');
    $('o-loss-year').innerHTML = O.years.filter((y) => y >= 2030).map((y) => `<option${y === 2050 ? ' selected' : ''}>${y}</option>`).join('');
    ['o-y', 'o-x'].forEach((id) => { $(id).onchange = drawScatter; }); $('o-loss-year').onchange = drawLoss;
  }
  drawScore(); drawScatter(); drawSeries(); drawLoss(); drawShrink();
}

// ---------------------------------------------------------------- extended scorecard
function scoreRows() {
  const O = C.E.option_space;
  return C.E.table.map((r) => ({ node: r.node, name: C.short(r.node), promising: !!r.promising, coverage: r.coverage, coverage_perfect: r.coverage_perfect,
    mean_damage_ratio: r.mean_damage_ratio, mean_cost: r.mean_cost, max_regret: r.max_regret, remaining_options_2050: r.remaining_options_2050,
    remaining_options_mean: r.remaining_options_mean, option_loss_2050: r.option_loss_2050, lockin_options: r.lockin.options,
    transition_feasibility: r.transition_feasibility,
    main_loss_2050: (() => { const L = O.inits[String(r.node)].checkpoints['2050']?.loss || {}; const k = Object.keys(L).sort((a, b) => L[b] - L[a])[0]; return k && L[k] > 0 ? SHORT[k] : '—'; })() }));
}
function drawScore() {
  const R = scoreRows(), O = C.E.option_space;
  $('o-score').innerHTML = `<tr><th>初期施策</th><th>Performance<br><span class="hint">被害額の比</span></th><th>Cost<br><span class="hint">PV 億円</span></th><th>Robustness<br><span class="hint">Coverage</span></th>
    <th>Regret<br><span class="hint">最大</span></th><th>Remaining future options<br><span class="hint">2050年 / 平均</span></th><th>Option loss<br><span class="hint">2050年（主な理由）</span></th>
    <th>Lock-in<br><span class="hint">選択肢</span></th><th>Transition feasibility</th></tr>`
    + R.map((r) => `<tr data-n="${r.node}" class="${r.node === C.SEL ? 'rec' : ''}"><td>${r.promising ? '★ ' : ''}${esc(r.name)}</td><td>${fmt(r.mean_damage_ratio, 2)}</td>
      <td>${fmt(r.mean_cost)}</td><td>${pct(r.coverage)}</td><td>${fmt(r.max_regret, 3)}</td>
      <td><b>${pct(r.remaining_options_2050)}</b> / ${pct(r.remaining_options_mean)}</td><td>${pct(r.option_loss_2050)}（${esc(r.main_loss_2050)}）</td>
      <td>${pct(r.lockin_options)}</td><td>${pct(r.transition_feasibility)}</td></tr>`).join('');
  $('o-score').querySelectorAll('tr[data-n]').forEach((tr) => { tr.onclick = () => C.select(+tr.dataset.n); });
  $('o-score-note').innerHTML = `選択肢の基準集合＝2020年（施策なし）に実行可能な施策・水準 ${O.n_base} 個（経路網に存在しない水準・最初から予算を超える水準は含めない）。
    Remaining future options＝まだ実行可能（若い森林の再転換が必要なものを含む）な割合。Option loss＝初期には選べたが経路上の選択や状態変化で実行できなくなった割合（実施済み＝used は含めない）。
    Transition feasibility＝不可逆性・土地利用・経路網の移行の制約で失われていない割合（リードタイム・予算を除く, 期間平均）。Lock-in＝既存の評価（施策なしと比べて到達できなくなる有用な組合せ）。`;
  exporter('o-score', null, () => toCSV(R), () => R, 'scorecard');
}

// ---------------------------------------------------------------- SVG helpers
function scale(d0, d1, r0, r1) { const k = (r1 - r0) / ((d1 - d0) || 1); return (v) => r0 + (v - d0) * k; }
function axes(W, H, m, X, Y, xt, yt, xl, yl, xf = (v) => v, yf = (v) => v) {
  let s = `<rect x="0" y="0" width="${W}" height="${H}" fill="#fff"/>`;
  yt.forEach((v) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e2e8f0"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="#475569">${yf(v)}</text>`; });
  xt.forEach((v) => { s += `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H - m.b}" stroke="#f1f5f9"/><text x="${X(v)}" y="${H - m.b + 15}" text-anchor="middle" font-size="11" fill="#475569">${xf(v)}</text>`; });
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="#334155"/><line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}" stroke="#334155"/>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" fill="#0f172a">${esc(xl)}</text>`;
  s += `<text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#0f172a">${esc(yl)}</text>`;
  return s;
}
const svg = (W, H, body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W * 1.25}px" ${FONT}>${body}</svg>`;
const ticks = (a, b, n = 5) => { const st = (b - a) / n; return Array.from({ length: n + 1 }, (_, k) => a + k * st); };

function drawScatter() {
  const R = scoreRows(), xk = $('o-x').value, yk = $('o-y').value;
  const W = 640, H = 380, m = { l: 62, r: 20, t: 16, b: 46 };
  const ys = R.map((r) => r[yk]).filter(Number.isFinite), ylo = yk === 'coverage' ? 0 : Math.min(...ys) * 0.95, yhi = yk === 'coverage' ? 1 : Math.max(...ys) * 1.05;
  const X = scale(0, 1, m.l, W - m.r), Y = scale(ylo, yhi, H - m.b, m.t);
  let b = axes(W, H, m, X, Y, ticks(0, 1), ticks(ylo, yhi), $('o-x').selectedOptions[0].text, $('o-y').selectedOptions[0].text, (v) => pct(v), (v) => (yk === 'coverage' ? pct(v) : fmt(v, yk === 'max_regret' ? 2 : 2)));
  R.forEach((r) => {
    if (!Number.isFinite(r[xk]) || !Number.isFinite(r[yk])) return;
    const sel = r.node === C.SEL, rad = 4 + 8 * (r.lockin_options || 0) + (sel ? 2 : 0);
    b += `<circle cx="${X(r[xk])}" cy="${Y(r[yk])}" r="${rad}" fill="${r.promising ? '#0e7490' : '#94a3b8'}" fill-opacity="0.85" stroke="${sel ? '#b45309' : '#0f172a'}" stroke-width="${sel ? 2.5 : 0.8}" data-n="${r.node}" style="cursor:pointer"><title>${esc(r.name)}</title></circle>`;
    b += `<text x="${X(r[xk]) + rad + 3}" y="${Y(r[yk]) + 4}" font-size="10" fill="#334155">${r.node === 0 ? '施策なし' : `#${r.node}`}</text>`;
  });
  b += `<g font-size="11"><circle cx="${W - 250}" cy="${m.t + 8}" r="5" fill="#0e7490"/><text x="${W - 241}" y="${m.t + 12}">有望（非劣）</text><circle cx="${W - 160}" cy="${m.t + 8}" r="5" fill="#94a3b8"/><text x="${W - 151}" y="${m.t + 12}">その他（大きさ＝Lock-in）</text></g>`;
  $('o-scatter').innerHTML = svg(W, H, b);
  $('o-scatter').querySelectorAll('circle[data-n]').forEach((c) => { c.onclick = () => C.select(+c.dataset.n); });
  exporter('o-scatter', 'o-scatter', () => toCSV(R.map((r) => ({ node: r.node, name: r.name, x: r[xk], y: r[yk], lockin: r.lockin_options, promising: r.promising }))), () => R, 'performance_vs_options');
}

function drawSeries() {
  const O = C.E.option_space, years = O.years, s = O.inits[String(C.SEL)].timeseries, z = O.inits['0']?.timeseries;
  const W = 900, H = 250, w = 280, m = { l: 52, r: 10, t: 24, b: 40 };
  const panel = (k, title, get, lo, hi, yf, band) => {
    const ox = k * (w + 26), X = scale(years[0], years[years.length - 1] + 10, ox + m.l, ox + w - m.r), Y = scale(lo, hi, H - m.b, m.t);
    let b = `<text x="${ox + m.l}" y="14" font-size="12" font-weight="700">${esc(title)}</text>`;
    ticks(lo, hi, 4).forEach((v) => { b += `<line x1="${ox + m.l}" x2="${ox + w - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e2e8f0"/><text x="${ox + m.l - 5}" y="${Y(v) + 4}" font-size="10" text-anchor="end" fill="#475569">${yf(v)}</text>`; });
    [2020, 2040, 2060, 2080, 2100].forEach((y) => { b += `<text x="${X(y)}" y="${H - m.b + 14}" font-size="10" text-anchor="middle" fill="#475569">${y}</text>`; });
    b += `<line x1="${ox + m.l}" x2="${ox + w - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="#334155"/>`;
    const xs = years.map((y) => y + 5);
    if (band) b += `<path d="M${xs.map((x, n) => `${X(x)},${Y(band[0][n])}`).join('L')}L${xs.map((x, n) => `${X(x)},${Y(band[1][n])}`).reverse().join('L')}Z" fill="#0e7490" fill-opacity="0.15"/>`;
    const line = (v, col, dash) => `<polyline points="${xs.map((x, n) => `${X(x)},${Y(v[n])}`).join(' ')}" fill="none" stroke="${col}" stroke-width="2.2" ${dash ? 'stroke-dasharray="5 4"' : ''}/>`;
    b += line(get(s), '#0e7490');
    if (z && C.SEL !== 0) b += line(get(z), '#64748b', true);
    return b;
  };
  const cmax = Math.max(...s.cost_cum, ...(z ? z.cost_cum : [0])) * 1.1;
  let b = `<rect width="${W}" height="${H}" fill="#fff"/>`;
  b += panel(0, 'Performance: 目標を満たし続けている割合', (t) => t.met_so_far, 0, 1, (v) => pct(v));
  b += panel(1, 'Cost: 累積費用 PV（億円）', (t) => t.cost_cum, 0, cmax, (v) => fmt(v));
  b += panel(2, 'Remaining Future Options', (t) => t.remaining.mean, 0, 1, (v) => pct(v), [s.remaining.p10, s.remaining.p90]);
  b += `<g font-size="11"><line x1="${W - 330}" x2="${W - 306}" y1="${H - 8}" y2="${H - 8}" stroke="#0e7490" stroke-width="2.2"/><text x="${W - 300}" y="${H - 4}">${esc(C.short(C.SEL))}</text>
    ${C.SEL !== 0 ? `<line x1="${W - 120}" x2="${W - 96}" y1="${H - 8}" y2="${H - 8}" stroke="#64748b" stroke-dasharray="5 4" stroke-width="2.2"/><text x="${W - 90}" y="${H - 4}">施策なし</text>` : ''}</g>`;
  $('o-series').innerHTML = svg(W, H, b);
  const rows = years.map((y, n) => ({ year: y, met_so_far: s.met_so_far[n], damage_ratio: s.damage_ratio[n], cost_cum_pv: s.cost_cum[n], remaining_mean: s.remaining.mean[n],
    remaining_p10: s.remaining.p10[n], remaining_p90: s.remaining.p90[n], used: s.used.mean[n], conditional: s.conditional.mean[n],
    ...Object.fromEntries(O.reasons.map((r) => [`loss_${r}`, s.loss[r][n]])) }));
  exporter('o-series', 'o-series', () => toCSV(rows), () => ({ node: C.SEL, rows }), `timeseries_${C.SEL}`);
}

function drawLoss() {
  const O = C.E.option_space, y = String($('o-loss-year').value), n = O.years.indexOf(+y);
  const R = scoreRows(), W = 900, rowH = 22, m = { l: 230, r: 20, t: 30, b: 30 }, H = m.t + m.b + rowH * R.length;
  const X = scale(0, 1, m.l, W - m.r);
  let b = `<rect width="${W}" height="${H}" fill="#fff"/>`;
  ticks(0, 1).forEach((v) => { b += `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t - 4}" y2="${H - m.b}" stroke="#e2e8f0"/><text x="${X(v)}" y="${H - m.b + 14}" font-size="10" text-anchor="middle" fill="#475569">${pct(v)}</text>`; });
  R.forEach((r, k) => {
    const ts = O.inits[String(r.node)].timeseries, yy = m.t + k * rowH;
    let x0 = 0;
    const seg = (v, col, lab) => { const s = `<rect x="${X(x0)}" y="${yy + 3}" width="${Math.max(X(x0 + v) - X(x0), 0)}" height="${rowH - 6}" fill="${col}"><title>${esc(lab)} ${pct(v, 1)}</title></rect>`; x0 += v; return s; };
    b += `<text x="${m.l - 6}" y="${yy + rowH / 2 + 4}" font-size="11" text-anchor="end" fill="${r.node === C.SEL ? '#b45309' : '#0f172a'}">${esc(r.name.length > 30 ? `${r.name.slice(0, 29)}…` : r.name)}</text>`;
    b += seg(ts.remaining.mean[n], '#bbf7d0', '残る選択肢') + seg(ts.used.mean[n], '#e2e8f0', '実施済み（used）');
    O.reasons.forEach((q) => { b += seg(ts.loss[q][n], RCOL[q], O.labels[q]); });
  });
  const leg = [['#bbf7d0', '残る選択肢'], ['#e2e8f0', '実施済み（used）'], ...O.reasons.map((q) => [RCOL[q], SHORT[q]])];
  let lx = 10;                       // legend items spaced by their label length
  b += `<g font-size="11">${leg.map(([c, l]) => { const s0 = `<rect x="${lx}" y="8" width="10" height="10" fill="${c}"/><text x="${lx + 14}" y="17">${esc(l)}</text>`; lx += 26 + 11 * l.length; return s0; }).join('')}</g>`;
  $('o-loss').innerHTML = svg(W, H, b);
  const rows = R.map((r) => { const ts = O.inits[String(r.node)].timeseries; return { node: r.node, name: r.name, year: +y, remaining: ts.remaining.mean[n], used: ts.used.mean[n],
    ...Object.fromEntries(O.reasons.map((q) => [q, ts.loss[q][n]])) }; });
  exporter('o-loss', 'o-loss', () => toCSV(rows), () => rows, `option_loss_${y}`);
}

// ---------------------------------------------------------------- where on the pathways map the option space shrinks
function drawShrink() {
  const O = C.E.option_space, d = O.inits[String(C.SEL)], tree = C.E.trees[String(C.SEL)], byId = Object.fromEntries(tree.map((b) => [b.id, b]));
  const rows = Object.entries(d.branches).map(([id, x]) => ({ id: +id, ...x, b: byId[id] })).filter((r) => r.b && Object.values(r.lost).some((v) => v > 0))
    .sort((a, b) => a.b.year - b.b.year);
  const timed = O.years.map((y, n) => ({ year: y, ...Object.fromEntries(O.reasons.map((q) => [q, d.time_driven[q][n]])) })).filter((r) => O.reasons.some((q) => r[q] > 0));
  $('o-shrink').innerHTML = `<tr><th>年</th><th>分岐（移行）</th><th>シナリオ</th><th>Option loss<br><span class="hint">失われた個数（平均）</span></th>
    <th>残る選択肢の変化<br><span class="hint">実施による使用を含む</span></th><th>失われた選択肢（平均の個数・理由）</th></tr>`
    + rows.map((r) => `<tr><td>${r.b.year}${r.b.year_range[0] !== r.b.year_range[1] ? `<br><span class="hint">${r.b.year_range[0]}–${r.b.year_range[1]}</span>` : ''}</td>
      <td>${r.id === 0 ? '<b>初期施策</b>' : '→ '}${esc(C.short(r.b.node))}</td><td>${r.n}</td>
      <td style="color:#b91c1c;font-weight:700">−${fmt(Object.values(r.lost).reduce((a, v) => a + v, 0), 1)}</td><td>${(100 * r.d_remaining).toFixed(1)} pt</td>
      <td>${r.options.map(([o, q, v]) => `<span style="color:${RCOL[q]}">■</span> ${esc(o)}（${esc(SHORT[q])}）${fmt(v, 2)}`).join('<br>')}</td></tr>`).join('')
    + (timed.length ? `<tr><td colspan="5" class="hint" style="text-align:left">時間の経過による縮小（移行なし: 森林の再転換期限・2100年までのリードタイム、1経路あたりの個数）: `
      + timed.map((r) => `${r.year}年 ${O.reasons.filter((q) => r[q] > 0).map((q) => `${SHORT[q]} ${fmt(r[q], 1)}`).join('・')}`).join('／') + '</td></tr>' : '');
  exporter('o-shrink', null, () => toCSV(rows.map((r) => ({ year: r.b.year, branch: r.id, node: r.b.node, n: r.n, d_remaining: r.d_remaining, ...r.lost,
    options: r.options.map(([o, q, v]) => `${o}/${q}/${v}`).join('; ') }))), () => ({ branches: rows.map(({ b, ...x }) => x), time_driven: d.time_driven }), `option_shrink_${C.SEL}`);
}

// markers on the branching pathways map (dapp_robust.js drawMap): ▼ where a move removes options
export function mapMarkers(ctx, b, x, y) {
  const O = C?.E?.option_space; if (!O) return;
  const d = O.inits[String(C.SEL)]?.branches[String(b.id)];
  if (!d || !Object.values(d.lost).some((v) => v > 0)) return;
  const main = Object.keys(d.lost).sort((p, q) => d.lost[q] - d.lost[p])[0];
  ctx.save();
  ctx.fillStyle = RCOL[main]; ctx.beginPath(); ctx.moveTo(x - 6, y - 16); ctx.lineTo(x + 6, y - 16); ctx.lineTo(x, y - 7); ctx.closePath(); ctx.fill();
  const lost = Object.values(d.lost).reduce((a, v) => a + v, 0);
  ctx.font = '10px "Hiragino Sans", sans-serif'; ctx.fillText(`−${fmt(lost, 1)}`, x + 7, y - 11);
  ctx.restore();
}

// ---------------------------------------------------------------- export (SVG / PNG / CSV / JSON)
export function toCSV(rows) {
  if (!rows.length) return '';
  const ks = Object.keys(rows[0]), q = (v) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return [ks.join(','), ...rows.map((r) => ks.map((k) => q(r[k])).join(','))].join('\n');
}
export function save(blob, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }
export function pngFromSVG(el, name) {
  const s = el.querySelector('svg'), vb = s.viewBox.baseVal, k = 2;
  const img = new Image();
  img.onload = () => { const cv = document.createElement('canvas'); cv.width = vb.width * k; cv.height = vb.height * k; const g = cv.getContext('2d');
    g.fillStyle = '#fff'; g.fillRect(0, 0, cv.width, cv.height); g.drawImage(img, 0, 0, cv.width, cv.height); cv.toBlob((b) => save(b, name)); };
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(s))}`;
}
function exporter(id, svgId, csv, json, base) {
  const box = $(`${id}-x`); if (!box) return;
  const run = C.D.name || 'robust';
  box.innerHTML = `${svgId ? '<button data-k="svg">SVG</button><button data-k="png">PNG</button>' : ''}<button data-k="csv">CSV</button><button data-k="json">JSON</button>`;
  box.querySelectorAll('button').forEach((bt) => { bt.onclick = () => {
    const k = bt.dataset.k, nm = `${run}_${base}`;
    if (k === 'svg') save(new Blob([new XMLSerializer().serializeToString($(svgId).querySelector('svg'))], { type: 'image/svg+xml' }), `${nm}.svg`);
    else if (k === 'png') pngFromSVG($(svgId), `${nm}.png`);
    else if (k === 'csv') save(new Blob(['﻿' + csv()], { type: 'text/csv' }), `${nm}.csv`);
    else save(new Blob([JSON.stringify(json(), null, 1)], { type: 'application/json' }), `${nm}.json`);
  }; });
}
