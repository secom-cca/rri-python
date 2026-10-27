// 流域治水チャレンジ — game UI for classes (budget, mission, score).
// Same engine and API as the expert UI (app.js): /api/prepare, /api/basin, /api/run, /api/job,
// plus /api/estimate (project cost of a plan without simulating).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ constants
const CLASS_JP = { forest: '森林', paddy: '水田', field: '畑', urban: '建物用地', water: '水域', paddy_dam: '田んぼダム',
  forest_new: '新規森林', green: '緑地（移転跡地）', wetland: '湿地（遊水地）' };
const CLASS_COLOR = { forest: '#4f8a57', paddy: '#d6e08a', field: '#dcbf85', urban: '#a3a3a3', water: '#6b9de0', paddy_dam: '#38c2c9',
  forest_new: '#1f5e33', green: '#93c97a', wetland: '#7fb7a6' };
const DEPTH_STOPS = [[0.1, '#bfe3ff'], [0.5, '#6cb6ff'], [1.0, '#2f80ed'], [2.0, '#1747a6'], [3.0, '#3b1a7a'], [5.0, '#5a0f5f']];
const DAMAGE_STOPS = [[0.01, '#fde68a'], [0.3, '#f97316'], [1, '#dc2626'], [3, '#7f1d1d']];   // 億円 per cell
const LEVEE_STOPS = [[0, '#e5e7eb'], [0.5, '#fde68a'], [2, '#f59e0b'], [4, '#c2410c'], [6, '#7f1d1d'], [10, '#3b0764']];
const BASIN_JP = {
  kusu_ogase: ['筑後川支川 玖珠川（小ヶ瀬）', '大分県の山あいの川。約530 km²。計算が速く、はじめての人向け'],
  abukuma_akutsu: ['阿武隈川上流（阿久津）', '福島県の阿武隈川の上流部。2019年 台風19号'],
  chikugo_whole: ['筑後川 全域', '九州最大の川。約2,900 km²。上流の山地から有明海の低平地まで'],
  abukuma_whole: ['阿武隈川 全域', '福島・宮城を流れる川。約5,400 km²。2019年 台風19号'],
};
const basinName = (k) => (BASIN_JP[k] || [k])[0];
const LEVELS = { // budget = factor x damage of this flood without measures; target = cut of 床上浸水世帯
  easy: { jp: 'やさしい', budget: 0.6, target: 0.3 },
  normal: { jp: 'ふつう', budget: 0.3, target: 0.5 },
  hard: { jp: 'むずかしい', budget: 0.15, target: 0.7 },
};
const TIPS = [
  '「流域治水」は、川の中だけでなく、田んぼや森、まちづくりも含めて流域全体で洪水を減らす考え方です。',
  '堤防を高くすると、あふれなかった水はそのまま下流へ流れます。下流のピーク流量も確かめよう。',
  '田んぼダムは、田んぼに雨を一時的にためて、ゆっくり流す取り組みです。',
  '家を高い場所に移すと、洪水そのものは変わらなくても被害は大きく減ります。でも、まちが分散すると暮らしの便利さは？',
  '遊水地は、洪水のときだけ川の水を引き込んでためる場所です。大きさと場所が大事です。',
  '気候変動で、大雨の雨量は2℃上昇で約1.1倍、4℃上昇で約1.3倍になると予測されています（国交省）。',
];

// ------------------------------------------------------------------ helpers
function dec(b64, T) {
  if (!b64) return null;
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new T(u8.buffer);
}
async function api(url, body) {
  const r = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
const rgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
function ramp(stops, v) {
  if (v <= stops[0][0]) return rgb(stops[0][1]);
  for (let k = 1; k < stops.length; k++) if (v <= stops[k][0]) {
    const t = (v - stops[k - 1][0]) / (stops[k][0] - stops[k - 1][0]), a = rgb(stops[k - 1][1]), b = rgb(stops[k][1]);
    return a.map((x, i) => x + (b[i] - x) * t);
  }
  return rgb(stops[stops.length - 1][1]);
}
const mix = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
const oku = (yen) => yen / 1e8;
const fmtOku = (x) => (Math.abs(x) >= 10 ? Math.round(x).toLocaleString() : x.toFixed(1));
const pct = (x) => `${x >= 0 ? '' : '−'}${Math.abs(x * 100).toFixed(0)}%`;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const niceRound = (x) => { const p = 10 ** Math.floor(Math.log10(Math.max(x, 1e-9))); return Math.max(1, Math.round(x / p * 2) / 2 * p); };
let toastT = null;
function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 2600); }

// ------------------------------------------------------------------ state
const G = {
  basins: [], basin: null, res: null, event: null, climate: 'current', level: 'normal',
  st: null, base: null, scen: null, plan: [], planCost: [], budget: 0, target: 0.3,
  view: 'stage', layer: 'landuse', frame: -1, measure: null, cfg: {}, paint: null, preview: [],
};

// ------------------------------------------------------------------ views
function show(view) {
  G.view = view;
  document.querySelectorAll('#panel > section').forEach((s) => s.classList.toggle('show', s.dataset.view === view));
  const order = ['stage', 'brief', 'plan', 'result'];
  const cur = view === 'busy' ? G.busyStep : view;
  document.querySelectorAll('#steps li').forEach((li) => {
    const k = order.indexOf(li.dataset.step), c = order.indexOf(cur);
    li.classList.toggle('on', k === c); li.classList.toggle('done', k < c);
  });
  const two = view === 'result';
  if ($('maps').classList.contains('two') !== two) {
    $('maps').classList.toggle('two', two);
    if (G.st) requestAnimationFrame(() => views.a.home());   // re-fit after the layout change
  }
  $('timebar').classList.toggle('show', view === 'result' || view === 'brief');
  setupLayers();
  syncPaintMode();
  drawMaps();
}
function busy(step, title, tip = true) {
  G.busyStep = step;
  $('busy-title').textContent = title; $('busy-bar').style.width = '2%'; $('busy-msg').textContent = '';
  $('busy-tip').textContent = tip ? '💡 ' + TIPS[Math.floor(Math.random() * TIPS.length)] : '';
  show('busy');
}
async function pollJob(job) {
  const t0 = Date.now();
  for (let n = 0; ; n++) {
    await new Promise((r) => setTimeout(r, 700));
    const j = await api(`/api/job/${job}`);
    $('busy-bar').style.width = `${Math.max(2, j.progress * 100)}%`;
    const s = Math.round((Date.now() - t0) / 1000);
    $('busy-msg').textContent = `${j.message}（${s >= 60 ? `${Math.floor(s / 60)}分` : ''}${s % 60}秒）`;
    if (n % 12 === 11) $('busy-tip').textContent = '💡 ' + TIPS[Math.floor(Math.random() * TIPS.length)];
    if (j.status === 'done') return j.result;
    if (j.status === 'error') throw new Error(j.message.split('\n')[0]);
  }
}
function climateParams() {
  if (G.climate === 'current' || !G.st) return { preset: 'current' };
  const c = G.st.climate_presets[G.climate];
  return { preset: G.climate, rain_factor: c.rain_factor, intensity_gamma: 1.0, sea_level_rise_m: c.sea_level_rise_m };
}
const gparams = () => ({ paddy_store_m: 0.15, paddy_release_mmh: 5, forest_n: 0.4, forest_soil_m: 1.0, climate: climateParams() });
async function runPlan(edits) {
  const { job } = await api('/api/run', { basin: G.basin, event: G.event, edits, params: gparams(), res: G.res });
  return prep(await pollJob(job));
}
function prep(r) {
  r._max = dec(r.max_depth_cm, Uint16Array); r._frames = dec(r.frames_depth_cm, Uint16Array);
  r._lc = dec(r.landclass, Uint8Array); r._damage = dec(r.damage, Float32Array); r._pop = dec(r.population, Float32Array);
  return r;
}

// ------------------------------------------------------------------ ① stage
async function initStage() {
  G.basins = await api('/api/basins');
  $('basin-cards').innerHTML = G.basins.map((b) => `<div class="card" data-k="${b.key}"><b>${basinName(b.key)}</b>
    <div class="hint">${(BASIN_JP[b.key] || [])[1] || ''}</div></div>`).join('');
  document.querySelectorAll('#basin-cards .card').forEach((c) => c.onclick = () => pickBasin(c.dataset.k));
  segment('seg-climate', (v) => { G.climate = v; });
  segment('seg-level', (v) => { G.level = v; });
  $('sel-res').onchange = () => { G.res = +$('sel-res').value; resHint(); };
  $('sel-event').onchange = () => { G.event = $('sel-event').value; };
  $('go-brief').onclick = startStage;
  pickBasin(G.basins[0].key);
}
function segment(id, fn) {
  document.querySelectorAll(`#${id} button`).forEach((b) => b.onclick = () => {
    document.querySelectorAll(`#${id} button`).forEach((x) => x.classList.toggle('on', x === b)); fn(b.dataset.v);
  });
}
const RES_JP = { 125: '詳細（とても遅い）', 250: '詳細（校正済み・遅い）', 500: 'ふつう', 1000: '速い（おすすめ）' };
function pickBasin(k) {
  G.basin = k;
  document.querySelectorAll('#basin-cards .card').forEach((c) => c.classList.toggle('on', c.dataset.k === k));
  const b = G.basins.find((x) => x.key === k);
  $('sel-event').innerHTML = b.events.map((e) => `<option value="${e}">${e.replace(/^.*_(\d{4})$/, '$1年の豪雨')}</option>`).join('');
  $('sel-event').value = b.events[b.events.length - 1]; G.event = $('sel-event').value;
  // 1000 m by default: every run takes seconds to tens of seconds (good for a class)
  $('sel-res').innerHTML = [1000, 500, 250].map((r) => `<option value="${r}">${r} m — ${RES_JP[r]}</option>`).join('');
  G.res = 1000; $('sel-res').value = G.res; resHint();
  $('go-brief').disabled = false;
}
function resHint() {
  $('res-hint').textContent = G.res >= 1000 ? '1マス＝1 km。計算が速い。川や町の形は粗いので、数字は目安として見よう。'
    : G.res >= 500 ? '1マス＝500 m。計算時間は 1000 m の数倍。' : '1マス＝250 m。モデルを合わせた（校正した）細かさ。大きな流域では1回の計算に数分かかります。';
}
async function startStage() {
  try {
    busy('stage', 'ステージを準備中…', true);
    const { job } = await api('/api/prepare', { basin: G.basin, res: G.res });
    if (job) await pollJob(job);
    $('busy-msg').textContent = '地図データを読み込み中…';
    const st = await api(`/api/basin/${G.basin}?res=${G.res}`);
    st.mask = dec(st.mask, Uint8Array); st.landclass = dec(st.landclass, Uint8Array); st.z = dec(st.z, Float32Array);
    st.hand = dec(st.hand, Float32Array); st.fc = dec(st.flood_cells, Int32Array); st.hh = dec(st.households, Float32Array);
    st.pop = dec(st.population, Float32Array); st.value = dec(st.asset_value, Float32Array);
    st.sub = Object.fromEntries(Object.entries(st.subbasins || {}).map(([g, b]) => [g, dec(b, Uint8Array)]));
    st.rivK = new Map(st.river.ri.map((i, k) => [i * st.nx + st.river.rj[k], k]));
    let lo = Infinity, hi = -Infinity;
    st.z.forEach((v) => { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    st.zlo = lo; st.zhi = hi;
    G.st = st; G.plan = []; G.planCost = []; G.scen = null; G.mainStem = null;
    buildViews();
    const ev = st.events.find((e) => e.key === G.event);
    G.eventLabel = ev ? `${+ev.eval_start.slice(0, 4)}年${+ev.eval_start.slice(5, 7)}月の豪雨` : G.event;
    busy('brief', '何も対策しないときの洪水を計算中…');
    G.base = await runPlan([]);
    const lv = LEVELS[G.level];
    G.budget = niceRound(G.base.damage_total_oku * lv.budget);
    G.target = lv.target;
    G.records = loadRecords();
    $('hud-stage').textContent = `${basinName(G.basin)}｜${G.eventLabel}｜${{ current: '現在の気候', '2C': '2℃上昇', '4C': '4℃上昇' }[G.climate]}｜${lv.jp}`;
    $('hud-mission').textContent = `🎯 床上浸水 ${pct(G.target)} 減`;
    G.layer = 'flood_base';
    renderBrief(); updateBudget(); show('brief');
  } catch (e) { toast('エラー: ' + e.message); show('stage'); }
}

// ------------------------------------------------------------------ ② brief
function kpi(l, v, d = '', cls = '') { return `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="d ${cls}">${d}</div></div>`; }
function renderBrief() {
  const b = G.base;
  $('brief-lead').textContent = `${G.eventLabel}がこの流域を襲うと、何も対策しない場合、次のような被害が出ると計算されました。`;
  $('brief-kpis').innerHTML = kpi('被害額', `${fmtOku(b.damage_total_oku)} 億円`) + kpi('床上浸水', `${Math.round(b.households_above_floor).toLocaleString()} 世帯`)
    + kpi('床下浸水', `${Math.round(b.households_below_floor).toLocaleString()} 世帯`) + kpi('浸水面積', `${b.inundated_km2.toFixed(1)} km²`);
  $('brief-mission').innerHTML = `<b>ミッション</b><br>予算 <b>${fmtOku(G.budget)} 億円</b> 以内で、床上浸水する世帯を <b>${pct(G.target)}</b> 以上減らそう。
    <div class="hint">予算＝施策の事業費（工事・用地・維持費の50年分）。暮らしやすさや自然への影響もスコアに入ります。</div>`;
}

// ------------------------------------------------------------------ ③ plan: measures
const sel = (name, label, opts) => ({ name, label, opts });
const MEASURES = {
  levee: { ic: '🧱', jp: '堤防', s: '川からあふれるのを防ぐ', color: '#7c4a1e',
    how: '川の両岸の堤防を、今の高さからさらに高くします（堤防のない川では新しく築きます）。',
    pro: 'あふれる水を確実に減らす', con: '下流に水が集中する・川と陸の生き物のつながりが切れる・長い区間は高い',
    opts: [sel('height', '高さ', { 1: '+1 m', 2: '+2 m', 3: '+3 m' }),
      sel('where', '場所', { main: '本流', trib: '支流', all: '本流＋支流', none: '堤防のない区間', paint: '地図で塗る' })] },
  retention: { ic: '🟦', jp: '遊水地', s: '洪水を一時的にためる', color: '#db2777',
    how: '川沿いの農地などを掘り下げて囲み、洪水のときだけ川の水を引き込んでためます。',
    pro: '下流のピーク流量を下げる', con: '広い用地が必要（農地が使えなくなる）・容量が小さいとすぐ満杯',
    opts: [sel('depth', 'ためる深さ', { 2: '2 m', 3: '3 m', 5: '5 m' }), sel('where', '場所', { paint: '地図で川沿いの土地を塗る' })] },
  paddy_dam: { ic: '🌾', jp: '田んぼダム', s: '田んぼに雨をためる', color: '#06b6d4',
    how: '田んぼの排水口に板をつけ、雨を15 cmまでためてゆっくり流します。',
    pro: '安い・広い範囲でできる', con: '長く続く大雨ではすぐ満杯になり効果が小さい',
    opts: [sel('where', '場所', { all: 'すべての田んぼ', up: '上流の田んぼ（川より20 m以上高い）', paint: '地図で塗る' })] },
  forest: { ic: '🌲', jp: '森林化', s: '森で雨をしみこませる', color: '#14532d',
    how: '畑や田んぼを森に戻し、雨が地面にしみこみやすく、ゆっくり流れるようにします。',
    pro: '自然が豊かになる・山から流れ出る水が遅くなる', con: '農地が減る（作物の収入がなくなる）・効果が出る場所は限られる',
    opts: [sel('where', '場所', { up: '高台の農地（川より20 m以上高い）', field: 'すべての畑', farm: 'すべての田畑', paint: '地図で塗る' })] },
  relocate: { ic: '🏠', jp: '移転', s: '危ない場所から引っ越す', color: '#8b5cf6',
    how: '浸水しやすい場所の家や事業所を、安全な場所へ移します。跡地は緑地にします。',
    pro: '被害が大きく減る・洪水の流れは変えない', con: '費用が高い・まちが分散すると暮らしの便利さが下がることも',
    opts: [sel('where', '対象', { flood1: '現況で1 m以上浸水する家', flood05: '現況で0.5 m以上浸水する家', low: '低地の家（川より3 m以下）', paint: '地図で塗る' }),
      sel('dest', '移転先', { new: '高台に新しく造成', existing: '安全な市街地に集約' })] },
  raise: { ic: '⬆️', jp: '嵩上げ', s: '家を高くする', color: '#f59e0b',
    how: '家の床の高さを上げ（高床・ピロティなど）、浸水しても床上まで水が来ないようにします。',
    pro: '移転より安い・住み慣れた場所に住み続けられる', con: '深い浸水には効かない・道路や田畑の浸水は減らない',
    opts: [sel('height', '高さ', { 1: '+1 m', 2: '+2 m' }),
      sel('where', '対象', { flood05: '現況で0.5 m以上浸水する家', low: '低地の家（川より3 m以下）', paint: '地図で塗る' })] },
};
function mainStem() {
  const r = G.st.river, n = r.ri.length, ups = Array.from({ length: n }, () => []);
  let out = -1;
  for (let k = 0; k < n; k++) { if (r.dn[k] >= 0) ups[r.dn[k]].push(k); else if (out < 0 || r.area[k] > r.area[out]) out = k; }
  const stem = new Set();
  for (let k = out; k >= 0;) { stem.add(k); const u = ups[k]; k = u.length ? u.reduce((a, b) => (r.area[b] > r.area[a] ? b : a)) : -1; }
  return stem;
}
function cellsWhere(type, v) {
  const st = G.st, n = st.nx * st.ny, cn = st.class_names, id = (x) => cn.indexOf(x), lc = st.landclass, out = [];
  if (v.where === 'paint') return G.paint ? [...G.paint] : [];
  if (type === 'levee') {
    const r = st.river, stem = G.mainStem || (G.mainStem = mainStem());
    for (let k = 0; k < r.ri.length; k++) {
      if (r.area[k] < 10) continue;
      const on = stem.has(k);
      if ((v.where === 'main' && !on) || (v.where === 'trib' && on) || (v.where === 'none' && r.embankment[k] >= 0.5)) continue;
      out.push(r.ri[k] * st.nx + r.rj[k]);
    }
    return out;
  }
  const depth = (c) => G.base._max[c] / 100;
  for (let c = 0; c < n; c++) {
    if (!st.mask[c]) continue;
    const h = st.hand[c], l = lc[c];
    let ok = false;
    if (type === 'paddy_dam') ok = l === id('paddy') && (v.where === 'all' || h >= 20);
    else if (type === 'forest') ok = v.where === 'up' ? [id('paddy'), id('field')].includes(l) && h >= 20
      : v.where === 'field' ? l === id('field') : [id('paddy'), id('field')].includes(l);
    else if (type === 'relocate' || type === 'raise') {
      const home = st.hh[c] + st.value[c] / 1e8 > 0.01 && st.value[c] > 0;
      ok = home && (v.where === 'low' ? h >= 0 && h <= 3 : depth(c) >= (v.where === 'flood1' ? 1.0 : 0.5));
    }
    if (ok) out.push(c);
  }
  return out;
}
function applicable(type, cells) {
  const st = G.st, cn = st.class_names, lc = st.landclass, id = (n) => cn.indexOf(n);
  const riv = (c) => st.rivK.has(c);
  if (type === 'levee') return cells.filter(riv);
  if (type === 'paddy_dam') return cells.filter((c) => lc[c] === id('paddy'));
  if (type === 'forest') return cells.filter((c) => ![id('forest'), id('water'), id('forest_new')].includes(lc[c]));
  if (type === 'relocate' || type === 'raise') return cells.filter((c) => st.value[c] > 0 && st.hh[c] + st.value[c] / 1e8 > 0.01);
  if (type === 'retention') return cells.filter((c) => !riv(c) && ![id('urban'), id('water')].includes(lc[c]));
  return cells;
}
function editFor(type, v) {
  const cells = applicable(type, cellsWhere(type, v));
  const M = MEASURES[type], label = M.opts.map((o) => o.opts[v[o.name]]).filter((x) => x && !/塗る/.test(x)).join('・') || '地図で指定';
  const e = { type, cells, label: `${M.jp}（${label}${v.where === 'paint' ? (label ? '・' : '') + '地図で指定' : ''}）` };
  if (type === 'levee') e.height = +v.height;
  if (type === 'raise') { e.height = +v.height; e.terrain = false; }
  if (type === 'retention') { e.depth = +v.depth; e.auto = true; e.q_ratio = 0.7; }
  if (type === 'relocate') { e.dest = v.dest; e.density = 40; }
  return e;
}
function renderMeasureCards() {
  $('measure-cards').innerHTML = Object.entries(MEASURES).map(([k, m]) => `<div class="mcard${G.measure === k ? ' on' : ''}" data-k="${k}">
    <div class="ic">${m.ic}</div><div class="n">${m.jp}</div><div class="s">${m.s}</div></div>`).join('');
  document.querySelectorAll('.mcard').forEach((c) => c.onclick = () => pickMeasure(c.dataset.k === G.measure ? null : c.dataset.k));
}
function pickMeasure(k) {
  G.measure = k; G.paint = null;
  renderMeasureCards();
  if (!k) { $('config').innerHTML = ''; G.preview = []; syncPaintMode(); drawMaps(); return; }
  const M = MEASURES[k];
  const v = G.cfg[k] || (G.cfg[k] = Object.fromEntries(M.opts.map((o) => [o.name, Object.keys(o.opts)[0]])));
  $('config').innerHTML = `<h3>${M.ic} ${M.jp}</h3><div class="hint">${M.how}</div>
    <div class="pc"><div><b class="good">◎ よいところ</b>${M.pro}</div><div><b class="bad">△ 気をつけること</b>${M.con}</div></div>
    ${M.opts.map((o) => `<div class="opt"><label>${o.label}</label><div class="seg" data-o="${o.name}">${Object.entries(o.opts).map(([x, l]) =>
      `<button data-v="${x}" class="${v[o.name] === x ? 'on' : ''}">${l}</button>`).join('')}</div></div>`).join('')}
    <div class="opt" id="brush-opt"><label>筆（地図を左ドラッグして塗る）</label><div class="seg" id="brush">
      <button data-v="0">小</button><button data-v="1" class="on">中</button><button data-v="3">大</button></div>
      <div class="seg" id="eraser"><button data-v="0" class="on">✏️ 塗る</button><button data-v="1">🧽 消す</button></div></div>
    ${Object.keys(G.st.sub).length ? `<div class="opt"><label>範囲をしぼる</label><select id="cfg-region"><option value="">流域全体</option>
      ${Object.keys(G.st.sub).map((g) => `<option value="${g}">${g} 観測所より上流</option>`).join('')}</select></div>` : ''}
    <div class="est" id="cfg-est"></div>
    <button id="cfg-add" class="primary big">＋ 計画に追加</button>`;
  $('config').querySelectorAll('.seg[data-o] button').forEach((b) => b.onclick = () => {
    const o = b.parentElement.dataset.o; v[o] = b.dataset.v;
    b.parentElement.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    syncConfig();
  });
  $('brush').querySelectorAll('button').forEach((b) => b.onclick = () => {
    G.brush = +b.dataset.v; $('brush').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
  });
  $('eraser').querySelectorAll('button').forEach((b) => b.onclick = () => {
    G.erase = b.dataset.v === '1'; $('eraser').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
  });
  G.erase = false;
  if ($('cfg-region')) $('cfg-region').onchange = syncConfig;
  $('cfg-add').onclick = addToPlan;
  if (k === 'levee') G.layer = 'levee';   // see where the levees already are
  setupLayers();
  syncConfig();
}
let estT = null;
function currentEdit() {
  const k = G.measure, v = G.cfg[k], e = editFor(k, v), g = $('cfg-region')?.value;
  if (g) { e.cells = e.cells.filter((c) => G.st.sub[g][c]); e.label = e.label.replace(/）$/, `・${g}より上流）`); }
  return e;
}
function syncConfig() {
  const k = G.measure; if (!k) return;
  const v = G.cfg[k], paint = v.where === 'paint';
  $('brush-opt').style.display = paint ? '' : 'none';
  syncPaintMode();
  const e = currentEdit();
  G.preview = e.cells;
  drawMaps();
  const n = e.cells.length;
  if (!n) { $('cfg-est').innerHTML = `<span class="hint">${paint ? '地図をドラッグして範囲を塗ろう' : '条件に合う場所がありません'}</span>`; $('cfg-add').disabled = true; return; }
  $('cfg-add').disabled = false;
  $('cfg-est').innerHTML = `<span class="hint">対象 ${n} セル … 費用を見積もり中</span>`;
  clearTimeout(estT);
  estT = setTimeout(async () => {
    try {
      const r = await api('/api/estimate', { basin: G.basin, event: G.event, edits: [e], params: gparams(), res: G.res });
      const c = oku(r.edit_cost_yen[0]), left = G.budget - G.planCost.reduce((a, x) => a + x, 0);
      const info = e.type === 'levee' ? `延長 約 ${(e.cells.reduce((a, cc) => a + G.st.river.length[G.st.rivK.get(cc)], 0) * 2 / 1000).toFixed(0)} km（両岸）`
        : `${(n * G.st.dx * G.st.dy / 1e6).toFixed(1)} km²${['relocate', 'raise'].includes(e.type) ? `・約 ${Math.round(e.cells.reduce((a, cc) => a + G.st.hh[cc], 0)).toLocaleString()} 世帯` : ''}`;
      $('cfg-est').innerHTML = `費用 約 <span class="${c > left ? 'bad' : ''}">${fmtOku(c)} 億円</span> <span class="hint">（${info}、残り予算 ${fmtOku(left)} 億円）</span>`;
    } catch (err) { $('cfg-est').textContent = '見積もりに失敗: ' + err.message; }
  }, 250);
}
async function addToPlan() {
  const e = currentEdit();
  if (!e.cells.length) return;
  G.plan.push(e); G.paint = null;
  toast(`${e.label} を計画に追加しました`);
  await refreshPlanCost();
  syncConfig();
}
async function refreshPlanCost() {
  renderPlan();
  if (!G.plan.length) { G.planCost = []; renderPlan(); updateBudget(); return; }
  try {
    const r = await api('/api/estimate', { basin: G.basin, event: G.event, edits: G.plan, params: gparams(), res: G.res });
    G.planCost = r.edit_cost_yen.map(oku);
  } catch (err) { toast('見積もりに失敗: ' + err.message); }
  renderPlan(); updateBudget();
}
function renderPlan() {
  const ul = $('plan-list');
  if (!G.plan.length) { ul.innerHTML = '<li class="empty">上のカードから施策を選んで「計画に追加」しよう。組み合わせてもOK。</li>'; }
  else ul.innerHTML = G.plan.map((e, i) => `<li><span class="sw" style="background:${MEASURES[e.type].color}"></span>
    <span class="t">${MEASURES[e.type].ic} ${e.label}</span>
    <span class="c">${G.planCost[i] != null ? fmtOku(G.planCost[i]) + ' 億円' : '…'}</span><button data-i="${i}" title="削除">✕</button></li>`).join('');
  ul.querySelectorAll('button').forEach((b) => b.onclick = () => { G.plan.splice(+b.dataset.i, 1); G.planCost.splice(+b.dataset.i, 1); refreshPlanCost(); drawMaps(); });
  const used = G.planCost.reduce((a, x) => a + x, 0);
  $('plan-sum').textContent = G.plan.length ? `合計 ${fmtOku(used)} 億円` : '';
  const over = used > G.budget;
  $('go-run').disabled = !G.plan.length || over;
  $('run-hint').innerHTML = over ? `<span class="bad">予算オーバーです（${fmtOku(used - G.budget)} 億円）。施策を減らすか、範囲をしぼろう。</span>`
    : G.plan.length ? '計算には少し時間がかかります。' : '';
  drawMaps();
}
function updateBudget() {
  const used = G.planCost.reduce((a, x) => a + x, 0);
  $('budget-bar').style.width = `${clamp(used / G.budget, 0, 1) * 100}%`;
  $('budget-bar').classList.toggle('over', used > G.budget);
  $('budget-txt').textContent = G.budget ? `${fmtOku(used)} / ${fmtOku(G.budget)} 億円` : '—';
}
async function runScenario() {
  try {
    busy('plan', 'あなたの計画で洪水を計算中…');
    G.scen = await runPlan(G.plan);
    G.layer = 'damage'; G.frame = -1;
    renderResult(); show('result');
  } catch (e) { toast('エラー: ' + e.message); show('plan'); }
}

// ------------------------------------------------------------------ ④ result: score
function scoreOf(b, s, cost) {
  const dRed = (b.damage_total_oku - s.damage_total_oku) / Math.max(b.damage_total_oku, 1e-9);
  const hRed = (b.households_above_floor - s.households_above_floor) / Math.max(b.households_above_floor, 1e-9);
  const ratio = cost > 0 ? (b.damage_total_oku - s.damage_total_oku) / cost : 0;   // 被害軽減額（この洪水1回）÷ 事業費
  const dL = s.livability.score - b.livability.score, dE = s.ecology.score - b.ecology.score;
  const ax = {
    '被害を減らす': 100 * clamp(dRed / 0.6, 0, 1), '住まいを守る': 100 * clamp(hRed / 0.6, 0, 1),
    '費用の効率': 100 * clamp(ratio / 2, 0, 1), '暮らしやすさ': clamp(50 + 10 * dL, 0, 100), '自然': clamp(50 + 10 * dE, 0, 100),
  };
  const w = { '被害を減らす': 0.25, '住まいを守る': 0.3, '費用の効率': 0.2, '暮らしやすさ': 0.125, '自然': 0.125 };
  const total = Math.round(Object.entries(ax).reduce((a, [k, x]) => a + w[k] * x, 0));
  const within = cost <= G.budget + 1e-6, hit = hRed >= G.target - 1e-9;
  const stars = !within || hRed <= 0 ? 0 : hit ? (dL >= -1 && dE >= -1 && ratio >= 1 ? 3 : 2) : 1;
  return { dRed, hRed, ratio, dL, dE, ax, total, stars, hit, within };
}
function outletGauge() {
  const st = G.st, r = st.river;
  return Object.entries(st.gauges).reduce((a, g) => (!a || r.area[g[1]] > r.area[a[1]] ? g : a), null)?.[0];
}
let radar = null, hydro = null;
function renderResult() {
  const b = G.base, s = G.scen, cost = oku(s.cost.pv_yen), sc = scoreOf(b, s, cost);
  G.lastScore = sc;
  $('stars').innerHTML = [0, 1, 2].map((i) => `<span class="${i < sc.stars ? 'on' : ''}">★</span>`).join('');
  $('score-n').innerHTML = `${sc.total}<small> 点</small>`;
  $('verdict').innerHTML = sc.hit ? '<span class="good">🎉 ミッション達成！</span>' : sc.hRed > 0 ? '<span class="warn">あと少し！ 目標には届かず</span>' : '<span class="bad">被害が減りませんでした</span>';
  const dcls = (x) => (x > 0.005 ? 'good' : x < -0.005 ? 'bad' : '');
  $('res-kpis').innerHTML = kpi('被害額', `${fmtOku(s.damage_total_oku)} 億円`, `${fmtOku(b.damage_total_oku)} → ${pct(-sc.dRed)}`, dcls(sc.dRed))
    + kpi('床上浸水', `${Math.round(s.households_above_floor).toLocaleString()} 世帯`, `${Math.round(b.households_above_floor).toLocaleString()} → ${pct(-sc.hRed)}（目標 ${pct(-G.target)}）`, dcls(sc.hRed))
    + kpi('事業費', `${fmtOku(cost)} 億円`, `予算 ${fmtOku(G.budget)} 億円`, cost <= G.budget ? '' : 'bad')
    + kpi('被害軽減 ÷ 事業費', sc.ratio.toFixed(2), 'この洪水1回あたり', sc.ratio >= 1 ? 'good' : '');
  // radar
  const labels = Object.keys(sc.ax);
  if (radar) radar.destroy();
  radar = new Chart($('radar'), { type: 'radar', data: { labels, datasets: [
    { label: 'あなたの計画', data: labels.map((k) => sc.ax[k]), backgroundColor: 'rgba(14,116,144,.25)', borderColor: '#0e7490', pointRadius: 3 },
    { label: '対策なし', data: [0, 0, 0, 50, 50], borderColor: '#94a3b8', borderDash: [4, 3], pointRadius: 0, backgroundColor: 'transparent' }] },
    options: { animation: { duration: 600 }, scales: { r: { min: 0, max: 100, ticks: { display: false, stepSize: 25 }, pointLabels: { font: { size: 11 } } } },
      plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } },
      onClick: (ev, els, chart) => {   // click an axis label or a point -> its details below
        const sc2 = chart.scales.r, x = ev.x, y = ev.y;
        let best = -1, bd = Infinity;
        labels.forEach((_, i) => {
          const p = sc2.getPointLabelPosition(i), cx = (p.left + p.right) / 2, cy = (p.top + p.bottom) / 2, d = (x - cx) ** 2 + (y - cy) ** 2;
          if (d < bd) { bd = d; best = i; }
        });
        if (els.length) best = els[0].index;
        if (best >= 0) openAxis(best, true);
      },
      onHover: (ev, els, chart) => { chart.canvas.style.cursor = 'pointer'; } } });
  renderAxes(sc, b, s, cost);
  // comments
  const g = outletGauge(), pk = (a) => Math.max(...a.filter((x) => x != null));
  const q0 = g ? pk(b.gauges[g].sim) : 0, q1 = g ? pk(s.gauges[g].sim) : 0, dq = q0 ? (q1 - q0) / q0 : 0;
  const c = [];
  c.push(sc.hit ? `床上浸水が ${pct(sc.hRed)} 減り、目標（${pct(G.target)}）を達成しました。` : `床上浸水は ${pct(Math.abs(sc.hRed))} ${sc.hRed >= 0 ? '減りました' : '増えました'}。目標は ${pct(G.target)} 減です。`);
  if (g && dq > 0.02) c.push(`⚠️ 下流（${g}）のピーク流量が ${pct(dq)} 増えました。川に水を集めると、下流の負担が大きくなります。`);
  if (g && dq < -0.02) c.push(`👍 下流（${g}）のピーク流量が ${pct(-dq)} 下がりました。流域で水をためた効果です。`);
  if (sc.dE <= -1) c.push(`🐟 自然の評価が ${sc.dE.toFixed(1)} 点下がりました（堤防で川と陸のつながりが切れる、造成で緑が減る など）。`);
  if (sc.dE >= 1) c.push(`🌿 自然の評価が +${sc.dE.toFixed(1)} 点上がりました。`);
  if (sc.dL <= -1) c.push(`🏘️ 暮らしやすさが ${sc.dL.toFixed(1)} 点下がりました（移転で病院・学校などから遠くなった など）。`);
  if (sc.ratio < 0.5 && cost > 0) c.push('💰 費用に比べて被害の減り方が小さめです。効果の大きい場所にしぼってみよう。');
  (s.retention || []).forEach((rt) => { if (rt.fill_ratio > 0.98) c.push(`🟦 遊水地がピーク前に満杯になりました。もっと広く・深くすると効果が上がるかも。`); });
  if (!sc.within) c.push('予算をオーバーしています。');
  $('comments').innerHTML = c.map((x) => `<li>${x}</li>`).join('');
  // hydrograph at the outlet gauge
  $('q-name').textContent = g || '—';
  if (hydro) hydro.destroy();
  if (g) {
    const labels2 = b.times.map((t) => t.slice(5, 13).replace('T', ' '));
    hydro = new Chart($('hydro'), { type: 'line', data: { labels: labels2, datasets: [
      { label: '対策なし', data: b.gauges[g].sim, borderColor: '#94a3b8', pointRadius: 0, borderWidth: 1.5 },
      { label: 'あなたの計画', data: s.gauges[g].sim, borderColor: '#0e7490', pointRadius: 0, borderWidth: 2 }] },
      options: { animation: false, interaction: { mode: 'index', intersect: false },
        scales: { x: { ticks: { maxTicksLimit: 5, font: { size: 9 } } }, y: { ticks: { font: { size: 9 } }, title: { display: true, text: 'm³/s', font: { size: 9 } } } },
        plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } } } });
  }
  $('time').max = s.nt - 1; $('time').value = -1; G.frame = -1; updateTime();
  renderRecords();
}

// ------------------------------------------------------------------ score axes: details
const tbl = (rows) => `<table><tr><th></th><th>対策なし</th><th>あなたの計画</th></tr>${rows.map(([n, a, z]) => `<tr><td>${n}</td><td>${a}</td><td>${z}</td></tr>`).join('')}</table>`;
const p100 = (x) => `${(x * 100).toFixed(0)}%`;
function renderAxes(sc, b, s, cost) {
  const CAT = { house: '家屋', household_goods: '家庭用品', car: '自動車', business_depreciable: '事業所の設備', business_inventory: '事業所の在庫',
    farm_depreciable: '農家の設備', farm_inventory: '農家の在庫', paddy: '水稲', field: '畑の作物' };
  const SVC = { medical: '医療（病院・診療所）', school: '教育（小・中学校）', commercial: '商業（スーパー等）', public: '公共（役所等）' };
  const cats = Object.keys(b.damage_by_category_oku).sort((x, y) => b.damage_by_category_oku[y] - b.damage_by_category_oku[x]).slice(0, 5);
  const L0 = b.livability, L1 = s.livability, E0 = b.ecology, E1 = s.ecology;
  const planRows = G.plan.map((e, i) => [`${MEASURES[e.type].ic} ${e.label}`, '', `${fmtOku(G.planCost[i] || 0)} 億円`]);
  const D = {
    '被害を減らす': {
      what: 'この洪水で出る被害額（家・家財・自動車・事業所・農作物など）がどれだけ減ったか。',
      body: tbl([['被害額', `${fmtOku(b.damage_total_oku)} 億円`, `${fmtOku(s.damage_total_oku)} 億円`],
        ...cats.map((k) => [`　${CAT[k] || k}`, `${fmtOku(b.damage_by_category_oku[k])}`, `${fmtOku(s.damage_by_category_oku[k] || 0)}`]),
        ['浸水面積', `${b.inundated_km2.toFixed(1)} km²`, `${s.inundated_km2.toFixed(1)} km²`]]),
      f: `点数 ＝ 被害額の減った割合 ${pct(sc.dRed)} ÷ 60% × 100（60%以上減れば満点）。被害額は国交省「治水経済調査マニュアル」の方法で、浸水深と資産から計算。` },
    '住まいを守る': {
      what: '床の上まで水が来る（床上浸水）世帯がどれだけ減ったか。ミッションの目標はこの数字。',
      body: tbl([['床上浸水', `${Math.round(b.households_above_floor).toLocaleString()} 世帯`, `${Math.round(s.households_above_floor).toLocaleString()} 世帯`],
        ['床下浸水', `${Math.round(b.households_below_floor).toLocaleString()} 世帯`, `${Math.round(s.households_below_floor).toLocaleString()} 世帯`],
        ['危険な場所（0.5 m以上浸水）に住む人の割合', p100(L0.pop_in_hazard_share), p100(L1.pop_in_hazard_share)]]),
      f: `点数 ＝ 床上浸水世帯の減った割合 ${pct(sc.hRed)} ÷ 60% × 100。床の高さは地面から 0.45 m（嵩上げでは＋高さ）。移転した世帯は移転先で数える。` },
    '費用の効率': {
      what: 'かけたお金に対して、どれだけ被害を減らせたか。',
      body: tbl([['被害の減少額（この洪水1回）', '', `${fmtOku(b.damage_total_oku - s.damage_total_oku)} 億円`], ['事業費（合計）', '', `${fmtOku(cost)} 億円`], ...planRows,
        ['被害の減少額 ÷ 事業費', '', sc.ratio.toFixed(2)]]),
      f: '点数 ＝ (被害の減少額 ÷ 事業費) ÷ 2 × 100（2以上で満点）。事業費は工事・用地・補償と、維持費50年分を割引率4%で今の価値にしたもの。'
        + '実際の事業評価（費用便益比 B/C）では、いろいろな大きさの洪水が起こる確率を考えた「年平均の被害軽減額」を使う。ここでは今回の洪水1回だけで比べている。' },
    '暮らしやすさ': {
      what: '病院・学校・お店・役所への行きやすさと、人がまとまって住んでいるか（まちのコンパクトさ）。移転や造成で変わる。',
      body: tbl([['暮らしやすさの点数（0〜100）', L0.score.toFixed(1), L1.score.toFixed(1)],
        ...Object.keys(L0.services).map((k) => [`　${SVC[k] || k}: 歩いて800 m以内の人`, p100(L0.services[k].walk800_coverage), p100(L1.services[k].walk800_coverage)]),
        ...Object.keys(L0.services).map((k) => [`　${SVC[k] || k}: 車での平均時間`, `${L0.services[k].mean_car_min.toFixed(1)} 分`, `${L1.services[k].mean_car_min.toFixed(1)} 分`]),
        ['　市街地に住む人の割合', p100(L0.zone_population_share), p100(L1.zone_population_share)],
        ['洪水のとき30分以内に病院へ行ける人', p100(L0.flood_medical_access_30min), p100(L1.flood_medical_access_30min)]]),
      f: `点数 ＝ 50 ＋ 10 ×（暮らしやすさの点数の変化 ${sc.dL >= 0 ? '+' : ''}${sc.dL.toFixed(1)}）。何もしなければ50点。国交省「都市構造の評価に関するハンドブック」の指標をもとにしている。` },
    '自然': {
      what: '生き物のすみかの質、森や川のつながり、雨を流域にためる力。堤防や造成で下がり、森林化や遊水地で上がることがある。',
      body: tbl([['自然の点数（0〜100）', E0.score.toFixed(1), E1.score.toFixed(1)],
        ['　生き物のすみかの質', p100(E0.habitat_quality), p100(E1.habitat_quality)],
        ['　森や緑のつながり', p100(E0.terrestrial_connectivity), p100(E1.terrestrial_connectivity)],
        ['　川と陸のつながり（堤防のない川の割合）', p100(E0.river_lateral_connectivity), p100(E1.river_lateral_connectivity)],
        ['　川の上流と下流のつながり（ダム）', p100(E0.river_longitudinal_dci), p100(E1.river_longitudinal_dci)],
        ['　雨を流域にためた割合', p100(E0.event_retention), p100(E1.event_retention)],
        ['　自然の氾濫原が水につかる割合', p100(E0.natural_flood_pulse), p100(E1.natural_flood_pulse)]]),
      f: `点数 ＝ 50 ＋ 10 ×（自然の点数の変化 ${sc.dE >= 0 ? '+' : ''}${sc.dE.toFixed(1)}）。何もしなければ50点。` },
  };
  $('axes').innerHTML = '<h2>スコアのくわしい内容（クリックで開く）</h2>' + Object.entries(sc.ax).map(([k, v], i) => `<div class="axis" data-i="${i}">
    <div class="h"><span>${k}</span><span class="bar"><i style="width:${v}%"></i></span><span class="n">${Math.round(v)}</span></div>
    <div class="body"><div>${D[k].what}</div>${D[k].body}<div class="f">${D[k].f}</div></div></div>`).join('');
  $('axes').querySelectorAll('.axis').forEach((el) => el.querySelector('.h').onclick = () => openAxis(+el.dataset.i));
}
function openAxis(i, scroll = false) {
  $('axes').querySelectorAll('.axis').forEach((el) => {
    const on = +el.dataset.i === i;
    el.classList.toggle('open', on ? !el.classList.contains('open') || scroll : false);
    if (on && scroll) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
}

// ------------------------------------------------------------------ records (per stage, this browser)
const recKey = () => `rri-play:${G.basin}:${G.res}:${G.event}:${G.climate}:${G.level}`;
function loadRecords() { try { return JSON.parse(localStorage.getItem(recKey()) || '[]'); } catch { return []; } }
function saveRecords() { try { localStorage.setItem(recKey(), JSON.stringify(G.records)); } catch { /* storage unavailable */ } }
function renderRecords() {
  const rs = G.records;
  if (!rs.length) { $('records').innerHTML = '<tr><td class="hint">まだ記録がありません。いろいろな計画を試して比べよう。</td></tr>'; return; }
  const best = rs.reduce((a, r, i) => (r.total > rs[a].total ? i : a), 0);
  $('records').innerHTML = '<tr><th>計画</th><th>★</th><th>点</th><th>床上</th><th>費用</th><th></th></tr>' + rs.map((r, i) => `<tr class="${i === best ? 'best' : ''}">
    <td>${r.name}</td><td>${'★'.repeat(r.stars) || '—'}</td><td>${r.total}</td><td>${pct(-r.hRed)}</td><td>${fmtOku(r.cost)}</td>
    <td><button class="link" data-i="${i}">読込</button></td></tr>`).join('');
  $('records').querySelectorAll('button').forEach((bt) => bt.onclick = () => {
    G.plan = JSON.parse(JSON.stringify(rs[+bt.dataset.i].plan)); G.scen = null; G.layer = 'landuse';
    refreshPlanCost(); renderMeasureCards(); show('plan'); toast(`「${rs[+bt.dataset.i].name}」を読み込みました`);
  });
}
function savePlan() {
  const sc = G.lastScore, name = prompt('計画の名前', `計画${G.records.length + 1}`);
  if (name == null) return;
  G.records.push({ name: name || `計画${G.records.length + 1}`, total: sc.total, stars: sc.stars, hRed: sc.hRed,
    cost: oku(G.scen.cost.pv_yen), plan: G.plan });
  saveRecords(); renderRecords(); toast('記録しました');
}

// ------------------------------------------------------------------ 3D map
// Each cell is its own quad (crisp colours, exact picking); corner heights are the
// mean of the surrounding cells.  Views a (対策なし) and b (あなたの計画) share the camera.
const LAYERS = {
  landuse: '土地利用', flood_base: '浸水（対策なし）', levee: '堤防の高さ', people: '人口',
  damage: '被害額', flood: '浸水深',
};
function setupLayers() {
  const v = G.view, opts = v === 'result' ? ['damage', 'flood', 'landuse', 'levee', 'people'] : v === 'stage' ? [] : ['landuse', 'flood_base', 'levee', 'people'];
  if (opts.length && !opts.includes(G.layer)) G.layer = opts[0];
  $('layers').innerHTML = opts.map((k) => `<button data-v="${k}" class="${k === G.layer ? 'on' : ''}">${LAYERS[k]}</button>`).join('');
  $('layers').querySelectorAll('button').forEach((b) => b.onclick = () => { G.layer = b.dataset.v; setupLayers(); drawMaps(); });
  $('map-bar').style.visibility = opts.length ? '' : 'hidden';
}
const EXAG = 3;   // vertical exaggeration
class View3D {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#dfe8ee');
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 5000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.48;
    this.scene.add(new THREE.HemisphereLight('#ffffff', '#6b7b6b', 1.15));
    const sun = new THREE.DirectionalLight('#ffffff', 1.5);
    sun.position.set(-30, 40, -20);
    this.scene.add(sun);
    this.objs = {};
    this.lineMat = new LineMaterial({ vertexColors: true, linewidth: 2.5 });
    new ResizeObserver(() => this.resize()).observe(el);
  }
  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.lineMat.resolution.set(w, h);
  }
  center(c) {   // [x, y, z] of a cell centre (km; y = exaggerated elevation)
    const st = G.st, i = Math.floor(c / st.nx), j = c % st.nx, z = st.z[c];
    return [(j + 0.5 - st.nx / 2) * st.dx / 1000, Number.isFinite(z) ? (z - st.zlo) / 1000 * EXAG : 0, (i + 0.5 - st.ny / 2) * st.dy / 1000];
  }
  set(key, obj) { if (this.objs[key]) this.scene.remove(this.objs[key]); this.objs[key] = obj; if (obj) this.scene.add(obj); }
  build() {
    const st = G.st, nx = st.nx, ny = st.ny;
    Object.keys(this.objs).forEach((k) => this.set(k, null));
    // corner heights
    const zc = new Float32Array((nx + 1) * (ny + 1));
    for (let i = 0; i <= ny; i++) for (let j = 0; j <= nx; j++) {
      let s = 0, n = 0;
      for (const [a, b] of [[i - 1, j - 1], [i - 1, j], [i, j - 1], [i, j]]) {
        if (a < 0 || b < 0 || a >= ny || b >= nx) continue;
        const c = a * nx + b;
        if (st.mask[c] && Number.isFinite(st.z[c])) { s += st.z[c]; n++; }
      }
      zc[i * (nx + 1) + j] = n ? (s / n - st.zlo) / 1000 * EXAG : 0;
    }
    this.cells = [];
    for (let c = 0; c < nx * ny; c++) if (st.mask[c]) this.cells.push(c);
    const n = this.cells.length, pos = new Float32Array(n * 12), idx = new Uint32Array(n * 6);
    this.cells.forEach((c, k) => {
      const i = Math.floor(c / nx), j = c % nx;
      [[i, j], [i, j + 1], [i + 1, j], [i + 1, j + 1]].forEach(([a, b], v) => {
        pos[12 * k + 3 * v] = (b - nx / 2) * st.dx / 1000;
        pos[12 * k + 3 * v + 1] = zc[a * (nx + 1) + b];
        pos[12 * k + 3 * v + 2] = (a - ny / 2) * st.dy / 1000;
      });
      idx.set([4 * k, 4 * k + 2, 4 * k + 1, 4 * k + 1, 4 * k + 2, 4 * k + 3], 6 * k);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 12), 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    this.terrain = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true }));
    this.set('terrain', this.terrain);
    this.lift = 0.02 + st.dx / 1000 * 0.03;
    // gauge labels
    const grp = new THREE.Group();
    for (const [name, k] of Object.entries(st.gauges)) {
      const c = st.river.ri[k] * nx + st.river.rj[k], [x, y, z] = this.center(c);
      const cv = document.createElement('canvas'), ctx = cv.getContext('2d');
      ctx.font = 'bold 26px "Hiragino Sans", sans-serif';
      cv.width = Math.ceil(ctx.measureText(name).width) + 44; cv.height = 40;
      ctx.font = 'bold 26px "Hiragino Sans", sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.beginPath(); ctx.roundRect(0, 2, cv.width, 36, 18); ctx.fill();
      ctx.fillStyle = '#0b2e3c'; ctx.beginPath(); ctx.arc(20, 20, 7, 0, 2 * Math.PI); ctx.fill(); ctx.fillText(name, 34, 29);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false, sizeAttenuation: false }));
      sp.scale.set(0.00042 * cv.width, 0.00042 * cv.height, 1); sp.center.set(0.08, 0); sp.position.set(x, y + this.lift, z); sp.renderOrder = 9;
      grp.add(sp);
    }
    this.set('labels', grp);
  }
  home() {   // whole basin in view (farther away on narrow views)
    this.resize();
    const st = G.st, ext = Math.max(st.nx * st.dx, st.ny * st.dy) / 1000 * Math.max(1, 1.3 / this.camera.aspect);
    this.camera.position.set(ext * 0.05, ext * 0.75, ext * 0.78);
    this.controls.target.set(0, 0, 0); this.controls.update();
  }
  setColors(col) {   // col: rgb 0-255 per grid cell
    const a = this.terrain.geometry.attributes.color;
    this.cells.forEach((c, k) => { for (let v = 0; v < 4; v++) for (let q = 0; q < 3; q++) a.array[12 * k + 3 * v + q] = col[3 * c + q] / 255; });
    a.needsUpdate = true;
  }
  setRivers(colorOf, minArea) {
    const st = G.st, r = st.river, nx = st.nx, pos = [], col = [];
    for (let k = 0; k < r.ri.length; k++) {
      const d = r.dn[k];
      if (d < 0 || r.area[k] < minArea) continue;
      const c = colorOf(k);
      for (const m of [k, d]) { const [x, y, z] = this.center(r.ri[m] * nx + r.rj[m]); pos.push(x, y + this.lift, z); col.push(c[0] / 255, c[1] / 255, c[2] / 255); }
    }
    const g = new LineSegmentsGeometry();
    g.setPositions(pos); g.setColors(col);
    this.set('rivers', new LineSegments2(g, this.lineMat));
  }
  setBars(cells, heights, colors) {
    if (!cells.length) { this.set('bars', null); return; }
    const st = G.st, w = 0.55 * st.dx / 1000;
    const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(w, 1, w), new THREE.MeshLambertMaterial(), cells.length);
    const m = new THREE.Matrix4(), color = new THREE.Color();
    cells.forEach((c, k) => {
      const [x, y, z] = this.center(c), h = Math.max(heights[k], 1e-4);
      m.makeScale(1, h, 1).setPosition(x, y + h / 2, z);
      mesh.setMatrixAt(k, m); mesh.setColorAt(k, color.setRGB(colors[k][0] / 255, colors[k][1] / 255, colors[k][2] / 255));
    });
    this.set('bars', mesh);
  }
  pick(ev) {
    if (!this.terrain) return -1;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster(); ray.setFromCamera(p, this.camera);
    const hit = ray.intersectObject(this.terrain)[0];
    return hit ? this.cells[Math.floor(hit.faceIndex / 2)] : -1;
  }
}
const views = { a: new View3D($('map-a')), b: new View3D($('map-b')) };
let syncing = false;
for (const [id, v] of Object.entries(views)) {   // both maps follow the same camera
  const o = views[id === 'a' ? 'b' : 'a'];
  v.controls.addEventListener('change', () => {
    if (syncing) return;
    syncing = true;
    o.camera.position.copy(v.camera.position); o.controls.target.copy(v.controls.target); o.controls.update();
    syncing = false;
  });
}
(function loop() {
  requestAnimationFrame(loop);
  for (const v of Object.values(views)) if (v.terrain && v.el.offsetParent) { v.controls.update(); v.renderer.render(v.scene, v.camera); }
})();
function buildViews() { views.a.build(); views.b.build(); views.a.home(); }
$('reset-view').onclick = () => views.a.home();

function embankmentOf(which) {
  const r = G.st.river, e = Float32Array.from(r.embankment);
  if (which === 'base') return e;
  const edits = which === 'scen' ? G.plan : [...G.plan, ...(G.measure === 'levee' ? [{ type: 'levee', cells: G.preview, height: +G.cfg.levee.height }] : [])];
  for (const ed of edits) if (ed.type === 'levee') for (const c of ed.cells) { const k = G.st.rivK.get(c); if (k !== undefined) e[k] += ed.height; }
  return e;
}
function depthOf(res) {
  const st = G.st, n = st.nx * st.ny, out = new Float32Array(n);
  if (!res) return out;
  if (G.frame < 0) { for (let c = 0; c < n; c++) out[c] = res._max[c] / 100; return out; }
  const fc = st.fc, nf = fc.length, off = G.frame * nf;
  for (let k = 0; k < nf; k++) { const v = res._frames[off + k] / 100; out[fc[k]] = v >= 0.1 ? v : 0; }
  return out;
}
function drawMaps() {
  if (!G.st || !views.a.terrain) return;
  const L = G.layer;
  if (G.view === 'result' && G.scen) {
    const tot = (r) => ({ damage: `被害 ${fmtOku(r.damage_total_oku)} 億円`, flood: `浸水 ${r.inundated_km2.toFixed(1)} km²`,
      levee: '', landuse: '', people: '' }[L]);
    drawMap('a', { kind: L, res: G.base, emb: 'base' }, `対策なし ${tot(G.base)}`);
    drawMap('b', { kind: L, res: G.scen, emb: 'scen', lc: G.scen._lc, pop: G.scen._pop }, `あなたの計画 ${tot(G.scen)}`);
  } else {
    drawMap('a', { kind: L === 'flood_base' ? 'flood' : L, res: G.base, emb: G.view === 'plan' ? 'plan' : 'base', overlay: G.view === 'plan' }, '');
  }
  legend();
}
function drawMap(id, o, caption) {
  const st = G.st, n = st.nx * st.ny, view = views[id];
  $(`cap-${id}`).textContent = caption;
  const col = new Uint8Array(n * 3), cn = st.class_names, lc = o.lc || st.landclass, pop = o.pop || st.pop;
  const depth = o.kind === 'flood' ? depthOf(o.res) : null;
  let popMax = 1;
  if (o.kind === 'people') for (let c = 0; c < n; c++) if (st.mask[c]) popMax = Math.max(popMax, st.pop[c]);
  const tint = new Map();
  if (o.overlay) {
    for (const e of G.plan) for (const c of e.cells) tint.set(c, MEASURES[e.type].color);
    if (G.measure) for (const c of G.preview) tint.set(c, '#ff00ff');
  }
  for (let c = 0; c < n; c++) {
    if (!st.mask[c]) continue;
    const zr = clamp((st.z[c] - st.zlo) / (st.zhi - st.zlo || 1), 0, 1);
    let v = o.kind === 'landuse' ? rgb(CLASS_COLOR[cn[lc[c]]] || '#999999') : mix([200, 208, 196], [244, 245, 240], zr);
    if (depth && depth[c] >= 0.1) v = ramp(DEPTH_STOPS, depth[c]);
    if (o.kind === 'people' && pop[c] > 0.5) v = ramp([[0, '#fff7bc'], [0.3, '#fe9929'], [1, '#993404']], Math.sqrt(pop[c] / popMax));
    if (o.kind === 'damage' && o.res._damage[c] > 1e6) v = mix(v, [254, 215, 170], 0.6);
    const t = tint.get(c);
    if (t) v = mix(v, rgb(t), t === '#ff00ff' ? 0.6 : 0.65);
    col[3 * c] = v[0]; col[3 * c + 1] = v[1]; col[3 * c + 2] = v[2];
  }
  view.setColors(col);
  // rivers: blue; the levee layer colours them by levee height (and shows walls)
  const emb = o.kind === 'levee' ? embankmentOf(o.emb) : null;
  view.setRivers(emb ? (k) => ramp(LEVEE_STOPS, emb[k]) : () => [37, 99, 235], emb ? 0 : 20);
  // bars: damage (height ∝ √damage, the same scale on both maps) or levee walls
  const ext = Math.max(st.nx * st.dx, st.ny * st.dy) / 1000, cells = [], hs = [], cs = [];
  if (o.kind === 'damage') {
    const dmax = Math.max(...G.base._damage);
    for (let c = 0; c < n; c++) {
      const d = o.res._damage[c];
      if (d > 1e6) { cells.push(c); hs.push(0.12 * ext * Math.sqrt(d / dmax)); cs.push(ramp(DAMAGE_STOPS, d / 1e8)); }
    }
  }
  if (emb) {
    const r = st.river;
    for (let k = 0; k < r.ri.length; k++) if (emb[k] > 0.05) { cells.push(r.ri[k] * st.nx + r.rj[k]); hs.push(emb[k] / 1000 * EXAG * 8); cs.push(ramp(LEVEE_STOPS, emb[k])); }
  }
  view.setBars(cells, hs, cs);
}
function legend() {
  const sw = (c, t) => `<div><span class="sw" style="background:${c}"></span>${t}</div>`, L = G.layer;
  let h = '';
  if (!G.st || G.view === 'stage' || G.view === 'busy') { $('legend').innerHTML = ''; return; }
  if (L === 'landuse') h = Object.entries(CLASS_JP).filter(([k]) => G.view === 'result' || k in { forest: 1, paddy: 1, field: 1, urban: 1, water: 1 }).map(([k, v]) => sw(CLASS_COLOR[k], v)).join('');
  if (L === 'flood' || L === 'flood_base') h = '<b>浸水深</b>' + DEPTH_STOPS.map(([v, c]) => sw(c, `${v} m〜`)).join('');
  if (L === 'levee') h = '<b>堤防の高さ（地面から）</b>' + LEVEE_STOPS.map(([v, c]) => sw(c, v ? `${v} m〜` : 'なし')).join('') + '<div class="hint">柱の高さ∝堤防の高さ</div>';
  if (L === 'people') h = '<b>人口</b>' + sw('#fff7bc', '少') + sw('#993404', '多');
  if (L === 'damage') h = '<b>被害額（1マスあたり）</b>' + DAMAGE_STOPS.map(([v, c]) => sw(c, `${v} 億円〜`)).join('') + '<div class="hint">柱の高さ∝√被害額（左右で同じ尺度）。家・家財・事業所・農作物</div>';
  if (G.view === 'plan') h += '<hr style="border:0;border-top:1px solid #ddd">' + sw('#ff00ff', '選択中の範囲') + '<div class="hint">計画の施策はカードの色</div>';
  $('legend').innerHTML = h;
}
function brushAt(c) {
  const st = G.st, R = G.brush ?? 1, i0 = Math.floor(c / st.nx), j0 = c % st.nx, out = [];
  for (let i = i0 - R; i <= i0 + R; i++) for (let j = j0 - R; j <= j0 + R; j++)
    if (i >= 0 && j >= 0 && i < st.ny && j < st.nx && (i - i0) ** 2 + (j - j0) ** 2 <= R * R + 0.5 && st.mask[i * st.nx + j]) out.push(i * st.nx + j);
  return out;
}
const painting = () => G.view === 'plan' && G.measure && G.cfg[G.measure].where === 'paint';
function syncPaintMode() {   // painting: left drag paints, right drag pans; otherwise left drag rotates
  views.a.controls.enableRotate = !painting();
  $('view-hint').textContent = painting() ? '左ドラッグ＝塗る（「消す」で消しゴム）、右ドラッグ＝移動、ホイール＝拡大'
    : 'ドラッグ＝回転、右ドラッグ＝移動、ホイール＝拡大';
}
let dragging = false;
function paintAt(c) {
  if (c < 0) return;
  G.paint = G.paint || new Set();
  for (const x of brushAt(c)) (G.erase ? G.paint.delete(x) : G.paint.add(x));
  syncConfig();
}
for (const [id, v] of Object.entries(views)) {
  const el = v.renderer.domElement;
  el.addEventListener('pointerdown', (ev) => {
    if (!painting() || ev.button !== 0 || id !== 'a') return;
    dragging = true; paintAt(v.pick(ev));
  });
  el.addEventListener('pointermove', (ev) => {
    const c = v.pick(ev);
    if (dragging) paintAt(c);
    tip(id, c, ev);
  });
  el.addEventListener('pointerleave', () => { $('tip').style.display = 'none'; });
}
window.addEventListener('pointerup', () => { dragging = false; });
function tip(id, c, ev) {
  const el = $('tip');
  if (c < 0 || !G.st) { el.style.display = 'none'; return; }
  const st = G.st, res = id === 'b' ? G.scen : G.base, lc = id === 'b' && G.scen ? G.scen._lc : st.landclass;
  let t = `${CLASS_JP[st.class_names[lc[c]]]}　標高 ${st.z[c].toFixed(0)} m`;
  if (st.hh[c] >= 0.5) t += `\n世帯 約 ${Math.round(st.hh[c])}`;
  if (res) {
    const d = depthOf(res)[c]; if (d >= 0.1) t += `\n浸水深 ${d.toFixed(2)} m`;
    if (res._damage[c] > 1e6) t += `\n被害 ${(res._damage[c] / 1e8).toFixed(2)} 億円`;
  }
  const k = st.rivK.get(c);
  if (k !== undefined) t += `\n川（流域 ${st.river.area[k].toFixed(0)} km²）堤防 ${embankmentOf(id === 'b' ? 'scen' : 'base')[k].toFixed(1)} m`;
  const rect = $('stage-area').getBoundingClientRect();
  el.textContent = t; el.style.display = 'block';
  el.style.left = `${ev.clientX - rect.left + 14}px`; el.style.top = `${ev.clientY - rect.top + 10}px`;
}
// time slider (brief: no-measure flood; result: both maps)
let playT = null;
function updateTime() {
  const res = G.scen || G.base;
  $('time-l').textContent = !res || G.frame < 0 ? '期間最大' : res.times[G.frame].replace('T', ' ').slice(5, 16);
  drawMaps();
}
$('time').oninput = (e) => { G.frame = +e.target.value; if (G.view === 'brief') G.layer = 'flood_base'; updateTime(); };
$('play').onclick = () => {
  if (playT) { clearInterval(playT); playT = null; $('play').textContent = '▶'; return; }
  const res = G.scen && G.view === 'result' ? G.scen : G.base;
  if (!res) return;
  $('time').max = res.nt - 1;
  if (G.view === 'brief') { G.layer = 'flood_base'; setupLayers(); }
  if (G.view === 'result' && G.layer !== 'flood') { G.layer = 'flood'; setupLayers(); }
  $('play').textContent = '❚❚';
  playT = setInterval(() => {
    G.frame = G.frame + 1 >= res.nt ? -1 : G.frame + 1; $('time').value = G.frame; updateTime();
    if (G.frame < 0) { clearInterval(playT); playT = null; $('play').textContent = '▶'; }
  }, 160);
};

// ------------------------------------------------------------------ wiring
$('go-plan').onclick = () => { G.layer = 'landuse'; renderMeasureCards(); renderPlan(); show('plan'); };
$('go-run').onclick = runScenario;
$('go-back').onclick = () => { G.layer = 'landuse'; G.frame = -1; renderMeasureCards(); renderPlan(); show('plan'); };
$('save-plan').onclick = savePlan;
$('go-stage').onclick = () => { G.st = null; G.base = G.scen = null; G.plan = []; G.planCost = []; G.budget = 0; updateBudget(); $('hud-stage').textContent = ''; $('hud-mission').textContent = ''; show('stage'); };
show('stage');
initStage().catch((e) => toast('エラー: ' + e.message));
