import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ constants
const CLASS_JP = { forest: '森林', paddy: '水田', field: '畑', urban: '建物用地', water: '水域', paddy_dam: '田んぼダム', forest_new: '新規森林' };
const CLASS_COLOR = { forest: '#3d7a46', paddy: '#cfdc78', field: '#d9b46e', urban: '#9b9b9b', water: '#5b8fd6', paddy_dam: '#38c2c9', forest_new: '#14532d' };
const TOOL = {
  forest:    { jp: '森林化',     color: '#14532d' },
  paddy_dam: { jp: '田んぼダム', color: '#06b6d4' },
  relocate:  { jp: '移転',       color: '#8b5cf6' },
  raise:     { jp: '嵩上げ',     color: '#f59e0b' },
  levee:     { jp: '堤防',       color: '#7c4a1e' },
  retention: { jp: '遊水地',     color: '#db2777' },
};
const CAT_JP = { house: '家屋', household_goods: '家庭用品', car: '自動車', business_depreciable: '事業所償却',
  business_inventory: '事業所在庫', farm_depreciable: '農漁家償却', farm_inventory: '農漁家在庫', paddy: '水稲', field: '畑作物' };
const DEPTH_STOPS = [[0.1, '#bfe3ff'], [0.5, '#6cb6ff'], [1.0, '#2f80ed'], [2.0, '#1747a6'], [3.0, '#3b1a7a'], [5.0, '#5a0f5f']];

// --------------------------------------------------------------------- state
const S = {
  st: null, basin: null, event: null, base: null, scen: null, edits: [],
  tool: 'view', mode: 'landuse', which: 'scen', frame: -1, exag: 2.5, playing: null,
  painting: null, hoverCell: -1,
};

// ------------------------------------------------------------------- helpers
function dec(b64, T) {
  if (!b64) return null;
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new T(u8.buffer);
}
const hex = (h) => new THREE.Color(h);
function lerpColor(stops, v) {
  if (v <= stops[0][0]) return hex(stops[0][1]);
  for (let k = 1; k < stops.length; k++) {
    if (v <= stops[k][0]) {
      const t = (v - stops[k - 1][0]) / (stops[k][0] - stops[k - 1][0]);
      return hex(stops[k - 1][1]).lerp(hex(stops[k][1]), t);
    }
  }
  return hex(stops[stops.length - 1][1]);
}
const fmtOku = (v) => (v == null ? '—' : `${v.toFixed(1)} 億円`);
async function getJSON(url, opt) {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
function setProgress(f, msg) { $('bar').style.width = `${Math.round(f * 100)}%`; $('pmsg').textContent = msg || ''; }
function log(msg) { $('log').textContent = `${msg}\n` + $('log').textContent; }

// --------------------------------------------------------------------- scene
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#dfe6ea');
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight('#ffffff', '#6b7b6b', 1.1));
const sun = new THREE.DirectionalLight('#ffffff', 1.6);
sun.position.set(-30, 40, -20);
scene.add(sun);

let terrain = null, riverLines = null, assetBars = null, damageBars = null, leveeWalls = null;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

// cell geometry -------------------------------------------------------------
function cellXZ(i, j) {
  const st = S.st;
  return [(j - st.nx / 2) * st.dx / 1000, (i - st.ny / 2) * st.dy / 1000];
}
function zAt(c) {
  const z = S.zcur[c];
  return Number.isFinite(z) ? (z - S.zmin) / 1000 * S.exag : 0;
}

function buildTerrain() {
  const st = S.st, nx = st.nx, ny = st.ny;
  [terrain, riverLines, assetBars, damageBars, leveeWalls].forEach((o) => o && scene.remove(o));
  const pos = new Float32Array(nx * ny * 3);
  const col = new Float32Array(nx * ny * 3);
  const idx = [];
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const c = i * nx + j;
    const [x, zz] = cellXZ(i, j);
    pos[3 * c] = x; pos[3 * c + 1] = zAt(c); pos[3 * c + 2] = zz;
    if (i < ny - 1 && j < nx - 1) {
      const a = c, b = c + 1, d = c + nx, e = c + nx + 1;
      if (st.mask[a] && st.mask[b] && st.mask[d] && st.mask[e]) idx.push(a, d, b, b, d, e);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  terrain = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true }));
  scene.add(terrain);
  buildRiver();
  buildAssetBars();
  const ext = Math.max(nx * st.dx, ny * st.dy) / 1000;
  camera.position.set(ext * 0.15, ext * 0.75, ext * 0.85);
  controls.target.set(0, 0, 0);
  controls.update();
}

function updateTerrainHeights() {
  const pos = terrain.geometry.attributes.position;
  for (let c = 0; c < S.st.nx * S.st.ny; c++) pos.array[3 * c + 1] = zAt(c);
  pos.needsUpdate = true;
  terrain.geometry.computeVertexNormals();
  buildRiver(); buildAssetBars(); refreshBars();
}

function buildRiver() {
  if (riverLines) scene.remove(riverLines);
  const r = S.st.river, nx = S.st.nx, n = r.ri.length;
  const pos = [], col = [];
  for (let k = 0; k < n; k++) {
    const d = r.dn[k];
    if (d < 0) continue;
    for (const m of [k, d]) {
      const [x, zz] = cellXZ(r.ri[m], r.rj[m]);
      pos.push(x, zAt(r.ri[m] * nx + r.rj[m]) + 0.012, zz);
      col.push(0.2, 0.45, 0.9);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  riverLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true }));
  riverLines.visible = $('show-river').checked;
  scene.add(riverLines);
}

function makeBars(cells, heights, colors, widthFrac) {
  const st = S.st, w = widthFrac * st.dx / 1000;
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(w, 1, w), new THREE.MeshLambertMaterial(), Math.max(cells.length, 1));
  const m = new THREE.Matrix4();
  cells.forEach((c, k) => {
    const i = Math.floor(c / st.nx), j = c % st.nx;
    const [x, zz] = cellXZ(i, j);
    const h = Math.max(heights[k], 1e-4);
    m.makeScale(1, h, 1).setPosition(x, zAt(c) + h / 2, zz);
    mesh.setMatrixAt(k, m);
    mesh.setColorAt(k, colors[k]);
  });
  mesh.count = cells.length;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return mesh;
}

function buildAssetBars() {
  if (assetBars) scene.remove(assetBars);
  const hh = S.hh, cells = [], hs = [], cs = [];
  for (let c = 0; c < hh.length; c++) if (S.st.mask[c] && hh[c] >= 1) {
    cells.push(c); hs.push(Math.sqrt(hh[c]) * 0.018); cs.push(hex('#f5f5f0'));
  }
  assetBars = makeBars(cells, hs, cs, 0.45);
  scene.add(assetBars);
}

function refreshBars() {
  if (damageBars) { scene.remove(damageBars); damageBars = null; }
  if (leveeWalls) { scene.remove(leveeWalls); leveeWalls = null; }
  const res = currentResult();
  assetBars.visible = $('show-assets').checked && S.mode !== 'damage';
  if (S.mode === 'damage' && res) {
    const dmg = res._damage, val = S.value, cells = [], hs = [], cs = [];
    for (let c = 0; c < dmg.length; c++) if (dmg[c] > 1e6) {
      cells.push(c);
      hs.push(Math.sqrt(dmg[c] / 1e8) * 0.25);
      const ratio = val[c] > 0 ? dmg[c] / val[c] : 0;
      cs.push(lerpColor([[0, '#fde68a'], [0.2, '#f97316'], [0.5, '#b91c1c'], [1, '#450a0a']], ratio));
    }
    damageBars = makeBars(cells, hs, cs, 0.7);
    scene.add(damageBars);
  }
  // levee walls: result embankment (or base) + pending levee edits preview
  const r = S.st.river, nx = S.st.nx;
  const emb = Float32Array.from(res ? res.embankment : r.embankment);
  for (const e of S.edits) if (e.type === 'levee') {
    const set = new Set(e.cells);
    for (let k = 0; k < r.ri.length; k++) if (set.has(r.ri[k] * nx + r.rj[k])) emb[k] = Math.max(emb[k], e.height);
  }
  const cells = [], hs = [], cs = [];
  for (let k = 0; k < r.ri.length; k++) if (emb[k] > 0.05) {
    cells.push(r.ri[k] * nx + r.rj[k]); hs.push(Math.max(emb[k] / 1000 * S.exag * 6, 0.02)); cs.push(hex('#5b3a1a'));
  }
  if (cells.length) {
    leveeWalls = makeBars(cells, hs, cs, 0.6);
    leveeWalls.visible = $('show-river').checked;
    scene.add(leveeWalls);
  }
}

// ------------------------------------------------------------------- results
function currentResult() { return S.which === 'base' ? S.base : (S.scen || S.base); }

function prepResult(res) {
  res._max = dec(res.max_depth_cm, Uint16Array);
  res._damage = dec(res.damage, Float32Array);
  res._frames = dec(res.frames_depth_cm, Uint16Array);
  res._rframes = dec(res.frames_river_cm, Uint16Array);
  res._lc = dec(res.landclass, Uint8Array);
  res._z = res.z_changed ? dec(res.z_changed, Float32Array) : null;
  const nr = S.st.river.ri.length;
  res._rmax = new Float32Array(nr);
  for (let t = 0; t < res.nt; t++) for (let k = 0; k < nr; k++)
    res._rmax[k] = Math.max(res._rmax[k], res._rframes[t * nr + k] / 100);
  return res;
}

function depthField(res) {
  // full-grid depth [m] for the current frame (or max)
  const n = S.st.nx * S.st.ny, out = new Float32Array(n);
  if (!res) return out;
  if (S.frame < 0) { for (let c = 0; c < n; c++) out[c] = res._max[c] / 100; return out; }
  const fc = S.fc, nf = fc.length, off = S.frame * nf;
  for (let k = 0; k < nf; k++) {
    const v = res._frames[off + k] / 100;
    out[fc[k]] = v >= 0.1 ? v : 0;
  }
  return out;
}

// ------------------------------------------------------------------ colouring
function pendingOverlay() {
  const m = new Map();
  for (const e of S.edits) for (const c of e.cells) m.set(c, e.type);
  if (S.painting) for (const c of S.painting.cells) m.set(c, S.painting.type === 'erase' ? null : S.painting.type);
  return m;
}

function colorize() {
  if (!terrain) return;
  const st = S.st, n = st.nx * st.ny, col = terrain.geometry.attributes.color.array;
  const res = currentResult();
  const lc = (res && S.which === 'scen' && S.scen) ? S.scen._lc : st.landclass;
  const over = pendingOverlay();
  const neutral = hex('#c9cfc4'), neutralHigh = hex('#e8ebe4');
  let depth = null, dDiff = null;
  if (S.mode === 'flood') depth = depthField(res);
  if (S.mode === 'diff' && S.base && S.scen) {
    const a = depthField(S.base), b = depthField(S.scen);
    dDiff = b.map((v, k) => v - a[k]);
  }
  for (let c = 0; c < n; c++) {
    if (!st.mask[c]) continue;
    const zr = Math.min(1, Math.max(0, (S.zcur[c] - S.zmin) / (S.zmax - S.zmin)));
    let color;
    if (S.mode === 'landuse') color = hex(CLASS_COLOR[st.class_names[lc[c]]] || '#888');
    else color = neutral.clone().lerp(neutralHigh, zr);
    if (depth && depth[c] >= 0.1) color = lerpColor(DEPTH_STOPS, depth[c]);
    if (dDiff) {
      const v = dDiff[c];
      if (v > 0.05) color = lerpColor([[0.05, '#fecaca'], [0.5, '#ef4444'], [2, '#7f1d1d']], v);
      else if (v < -0.05) color = lerpColor([[0.05, '#bbf7d0'], [0.5, '#22c55e'], [2, '#14532d']], -v);
    }
    const t = over.get(c);
    if (t !== undefined) {
      if (t === null) color = color.clone().lerp(hex('#ffffff'), 0.6);
      else color = color.clone().lerp(hex(TOOL[t].color), S.mode === 'landuse' ? 0.75 : 0.45);
    }
    col[3 * c] = color.r; col[3 * c + 1] = color.g; col[3 * c + 2] = color.b;
  }
  terrain.geometry.attributes.color.needsUpdate = true;
  colorRiver(res);
  refreshLegend();
}

function colorRiver(res) {
  if (!riverLines) return;
  const r = S.st.river, col = riverLines.geometry.attributes.color.array;
  const nr = r.ri.length;
  const emb = res ? res.embankment : r.embankment;
  let p = 0;
  for (let k = 0; k < nr; k++) {
    if (r.dn[k] < 0) continue;
    for (const m of [k, r.dn[k]]) {
      let c = hex('#3b82f6');
      if (res) {
        const h = S.frame < 0 ? res._rmax[m] : res._rframes[S.frame * nr + m] / 100;
        const ratio = h / Math.max(r.depth[m] + emb[m], 0.1);
        c = lerpColor([[0.3, '#93c5fd'], [0.7, '#2563eb'], [0.95, '#facc15'], [1.05, '#dc2626']], ratio);
      }
      col[p++] = c.r; col[p++] = c.g; col[p++] = c.b;
    }
  }
  riverLines.geometry.attributes.color.needsUpdate = true;
}

function refreshLegend() {
  const L = $('legend');
  const sw = (c, t) => `<div class="item"><span class="sw" style="background:${c}"></span>${t}</div>`;
  let h = '';
  if (S.mode === 'landuse') h = Object.entries(CLASS_JP).map(([k, v]) => sw(CLASS_COLOR[k], v)).join('');
  if (S.mode === 'flood') h = '<b>浸水深</b>' + DEPTH_STOPS.map(([v, c]) => sw(c, `${v} m〜`)).join('');
  if (S.mode === 'damage') h = '<b>被害額（柱の高さ∝√被害額）</b>' + sw('#fde68a', '被害率 小') + sw('#b91c1c', '被害率 大');
  if (S.mode === 'diff') h = '<b>最大浸水深の差（施策−現況）</b>' + sw('#22c55e', '浅くなった') + sw('#ef4444', '深くなった');
  h += '<hr style="border:0;border-top:1px solid #ddd">' + sw('#2563eb', '河道（水位/堤防高）') + sw('#dc2626', '越水') +
    sw('#8a5a2b', '堤防') + Object.entries(TOOL).map(([k, v]) => sw(v.color, `施策: ${v.jp}`)).join('');
  L.innerHTML = h;
}

// ------------------------------------------------------------------- picking
function pickCell(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(terrain)[0];
  if (!hit) return -1;
  const st = S.st;
  const j = Math.round(hit.point.x * 1000 / st.dx + st.nx / 2);
  const i = Math.round(hit.point.z * 1000 / st.dy + st.ny / 2);
  if (i < 0 || j < 0 || i >= st.ny || j >= st.nx) return -1;
  const c = i * st.nx + j;
  return st.mask[c] ? c : -1;
}
function brushCells(c) {
  const st = S.st, R = +$('brush').value, i0 = Math.floor(c / st.nx), j0 = c % st.nx, out = [];
  for (let di = -R; di <= R; di++) for (let dj = -R; dj <= R; dj++) {
    if (di * di + dj * dj > R * R + 0.5) continue;
    const i = i0 + di, j = j0 + dj;
    if (i < 0 || j < 0 || i >= st.ny || j >= st.nx) continue;
    const k = i * st.nx + j;
    if (st.mask[k]) out.push(k);
  }
  return out;
}

renderer.domElement.addEventListener('pointerdown', (ev) => {
  if (S.tool === 'view' || ev.button !== 0 || !terrain) return;
  const c = pickCell(ev);
  if (c < 0) return;
  S.painting = { type: S.tool, cells: new Set(brushCells(c)) };
  colorize();
});
renderer.domElement.addEventListener('pointermove', (ev) => {
  if (!terrain) return;
  const c = pickCell(ev);
  if (S.painting && c >= 0) {
    brushCells(c).forEach((k) => S.painting.cells.add(k));
    colorize();
  }
  if (c !== S.hoverCell) { S.hoverCell = c; showHover(c); }
});
window.addEventListener('pointerup', () => {
  if (!S.painting) return;
  const p = S.painting;
  S.painting = null;
  if (p.type === 'erase') {
    S.edits.forEach((e) => { e.cells = e.cells.filter((c) => !p.cells.has(c)); });
    S.edits = S.edits.filter((e) => e.cells.length);
  } else if (p.cells.size) {
    const cells = applicable(p.type, [...p.cells]);
    if (cells.length) S.edits.push({ type: p.type, cells, ...toolParams(p.type) });
    setProgress(0, `${TOOL[p.type].jp}: 塗った ${p.cells.size} セルのうち適用 ${cells.length} セル`);
  }
  renderEdits(); refreshBars(); colorize();
});

// cells a measure can actually be applied to (mirrors the engine's rules)
function applicable(type, cells) {
  const st = S.st, cn = st.class_names, lc = st.landclass;
  const id = (n) => cn.indexOf(n);
  if (type === 'paddy_dam') return cells.filter((c) => lc[c] === id('paddy'));
  if (type === 'forest') return cells.filter((c) => ![id('forest'), id('water'), id('forest_new')].includes(lc[c]));
  if (type === 'levee') { const rs = S.riverSet; return cells.filter((c) => rs.has(c)); }
  if (type === 'relocate' || type === 'raise') return cells.filter((c) => S.value[c] > 0 && S.hh[c] + S.value[c] / 1e8 > 0.01);
  return cells;
}

function toolParams(t) {
  if (t === 'raise') return { height: +$('raise-h').value, terrain: $('raise-terrain').checked };
  if (t === 'levee') return { height: +$('levee-h').value };
  if (t === 'retention') return { depth: +$('ret-d').value, inlet: +$('ret-inlet').value };
  return {};
}
function describeEdit(e) {
  const km2 = (e.cells.length * S.st.dx * S.st.dy / 1e6).toFixed(2);
  const extra = e.type === 'raise' ? ` +${e.height}m${e.terrain ? '盛土' : ''}` : e.type === 'levee' ? ` ${e.height}m`
    : e.type === 'retention' ? ` 掘削${e.depth}m 越流堤${e.inlet}m` : '';
  return `${TOOL[e.type].jp}${extra}（${e.cells.length}セル, ${km2} km²）`;
}
function renderEdits() {
  const ul = $('edits');
  ul.innerHTML = '';
  S.edits.forEach((e, k) => {
    const li = document.createElement('li');
    li.innerHTML = `<span><span class="sw" style="background:${TOOL[e.type].color}"></span>${describeEdit(e)}</span>`;
    const b = document.createElement('button');
    b.textContent = '✕';
    b.onclick = () => { S.edits.splice(k, 1); renderEdits(); refreshBars(); colorize(); };
    li.appendChild(b);
    ul.appendChild(li);
  });
  if (!S.edits.length) ul.innerHTML = '<li class="hint">施策を地図上で塗ってください</li>';
}

function showHover(c) {
  const el = $('hover');
  if (c < 0) { el.textContent = ''; return; }
  const st = S.st, res = currentResult();
  const lc = (S.scen && S.which === 'scen') ? S.scen._lc[c] : st.landclass[c];
  let t = `標高 ${S.zcur[c].toFixed(1)} m\n土地利用 ${CLASS_JP[st.class_names[lc]]}\n世帯数 ${S.hh[c].toFixed(0)}\n`;
  t += `HAND ${st.hand[c] >= 0 ? st.hand[c].toFixed(1) + ' m' : '—'}\n資産額 ${(S.value[c] / 1e8).toFixed(2)} 億円`;
  if (res) {
    const d = depthField(res)[c];
    t += `\n浸水深 ${d.toFixed(2)} m\n被害額 ${(res._damage[c] / 1e8).toFixed(2)} 億円`;
  }
  el.textContent = t;
}

// ------------------------------------------------------------------- charts
let qChart = null, dChart = null;
function drawCharts() {
  const g = $('gauge').value, base = S.base, scen = S.scen;
  if (!base) return;
  const labels = base.times.map((t) => t.slice(5, 13).replace('T', ' '));
  const ds = [];
  if (base.gauges[g].obs.some((v) => v != null))
    ds.push({ label: '観測', data: base.gauges[g].obs, borderColor: '#111', pointRadius: 1.5, showLine: false, yAxisID: 'q' });
  ds.push({ label: '現況', data: base.gauges[g].sim, borderColor: '#2563eb', pointRadius: 0, borderWidth: 1.6, yAxisID: 'q' });
  if (scen) ds.push({ label: '施策', data: scen.gauges[g].sim, borderColor: '#dc2626', pointRadius: 0, borderWidth: 1.6, yAxisID: 'q' });
  ds.push({ type: 'bar', label: '流域平均雨量', data: base.basin_rain.slice(0, labels.length), backgroundColor: 'rgba(100,140,200,.45)', yAxisID: 'r' });
  if (qChart) qChart.destroy();
  qChart = new Chart($('q-chart'), {
    type: 'line', data: { labels, datasets: ds },
    options: { animation: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: { x: { ticks: { maxTicksLimit: 8, font: { size: 10 } } },
        q: { position: 'left', title: { display: true, text: 'm³/s' } },
        r: { position: 'right', reverse: true, max: Math.max(...base.basin_rain) * 3, grid: { display: false }, title: { display: true, text: 'mm/h' } } } },
  });
  const cats = Object.keys(base.damage_by_category_oku);
  if (dChart) dChart.destroy();
  dChart = new Chart($('dmg-chart'), {
    type: 'bar',
    data: { labels: cats.map((c) => CAT_JP[c] || c),
      datasets: [{ label: '現況', data: cats.map((c) => base.damage_by_category_oku[c]), backgroundColor: '#93c5fd' },
        ...(scen ? [{ label: '施策', data: cats.map((c) => scen.damage_by_category_oku[c] || 0), backgroundColor: '#f87171' }] : [])] },
    options: { animation: false, indexAxis: 'y', plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: { x: { title: { display: true, text: '億円' } } } },
  });
  // KPIs
  const k = (h, b, s, unit, better) => {
    const d = s == null ? null : s - b;
    const cls = d == null || Math.abs(d) < 1e-9 ? '' : ((better === 'lower' ? d < 0 : d > 0) ? 'good' : 'bad');
    return `<div><div class="h">${h}</div><div class="v">${b.toFixed(1)}${unit}</div></div>
      <div><div class="h">施策</div><div class="v">${s == null ? '—' : s.toFixed(1) + unit}</div></div>
      <div><div class="h">差</div><div class="v ${cls}">${d == null ? '—' : (d > 0 ? '+' : '') + d.toFixed(1) + unit}</div></div>`;
  };
  const pk = (r) => Math.max(...r.gauges[g].sim);
  $('kpis').innerHTML =
    k('被害額（現況）', base.damage_total_oku, scen?.damage_total_oku, ' 億円', 'lower') +
    k('浸水面積', base.inundated_km2, scen?.inundated_km2, ' km²', 'lower') +
    k('床上浸水世帯', base.households_above_floor, scen?.households_above_floor, '', 'lower') +
    k(`ピーク流量 ${g}`, pk(base), scen ? pk(scen) : null, ' m³/s', 'lower') +
    (scen && scen.retention_max_storage_mcm > 0
      ? `<div style="grid-column:1/-1" class="h">遊水地 最大貯留量（概算） ${scen.retention_max_storage_mcm.toFixed(2)} 百万m³（遊水地内は被害・浸水面積の集計対象外）</div>` : '');
}

// ------------------------------------------------------------------- jobs
async function runJob(edits) {
  const params = { paddy_store_m: +$('pd-store').value / 100, paddy_release_mmh: +$('pd-rel').value,
    forest_n: +$('fo-n').value, forest_soil_m: +$('fo-soil').value };
  const { job } = await getJSON('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ basin: S.basin, event: S.event, edits, params }) });
  $('run').disabled = true;
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 700));
      const j = await getJSON(`/api/job/${job}`);
      setProgress(j.progress, j.message);
      if (j.status === 'done') return prepResult(j.result);
      if (j.status === 'error') { log(j.message); throw new Error(j.message.split('\n')[0]); }
    }
  } finally { $('run').disabled = false; }
}

function setResultUI() {
  const res = currentResult();
  S.zcur = (S.which === 'scen' && S.scen && S.scen._z) ? S.scen._z : S.st._z;
  updateTerrainHeights();
  $('time').max = res ? res.nt - 1 : 0;
  colorize(); refreshBars(); drawCharts(); updateTimeLabel();
}

async function runBaseline() {
  S.base = null; S.scen = null;
  setProgress(0, '現況（施策なし）を計算中… 初回はスピンアップ計算のため数分かかります');
  try {
    S.base = await runJob([]);
    log(`現況 計算完了: ${S.base.wall_s.toFixed(1)} s, 水収支誤差 ${S.base.mass_rel_err.toExponential(1)}`);
    S.which = 'base'; selectButton('which', 'base');
    if (S.mode === 'landuse') { S.mode = 'flood'; selectButton('modes', 'flood'); }
    setResultUI();
    setProgress(1, '現況の計算完了。施策を塗って「洪水計算を実行」');
  } catch (e) { setProgress(0, `エラー: ${e.message}`); }
}

async function runScenario() {
  if (!S.base) { await runBaseline(); }
  if (!S.edits.length) { setProgress(0, '施策がありません（現況のみ）'); return; }
  try {
    S.scen = await runJob(S.edits);
    log(`施策 計算完了: ${S.scen.wall_s.toFixed(1)} s, 水収支誤差 ${S.scen.mass_rel_err.toExponential(1)}\n` + S.scen.edits_log.join('\n'));
    S.which = 'scen'; selectButton('which', 'scen');
    setResultUI();
    const dd = S.scen.damage_total_oku - S.base.damage_total_oku;
    setProgress(1, `完了: 被害額 ${dd <= 0 ? '' : '+'}${dd.toFixed(1)} 億円（現況比）`);
  } catch (e) { setProgress(0, `エラー: ${e.message}`); }
}

// -------------------------------------------------------------------- time
function updateTimeLabel() {
  const res = currentResult();
  $('time-label').textContent = (!res || S.frame < 0) ? '期間最大' : res.times[S.frame].replace('T', ' ').slice(0, 16);
}
$('time').addEventListener('input', (e) => { S.frame = +e.target.value; updateTimeLabel(); colorize(); });
$('tmax').onclick = () => { S.frame = -1; updateTimeLabel(); colorize(); };
$('play').onclick = () => {
  if (S.playing) { clearInterval(S.playing); S.playing = null; $('play').textContent = '▶'; return; }
  const res = currentResult();
  if (!res) return;
  if (S.mode !== 'flood') { S.mode = 'flood'; selectButton('modes', 'flood'); }
  $('play').textContent = '❚❚';
  S.playing = setInterval(() => {
    S.frame = (S.frame + 1) % res.nt;
    $('time').value = S.frame; updateTimeLabel(); colorize();
  }, 160);
};

// -------------------------------------------------------------------- UI wiring
function selectButton(group, val) {
  document.querySelectorAll(`#${group} button`).forEach((b) => b.classList.toggle('active', b.dataset[group === 'modes' ? 'mode' : group === 'which' ? 'which' : 'tool'] === val));
}
document.querySelectorAll('#tools button').forEach((b) => b.onclick = () => {
  S.tool = b.dataset.tool; selectButton('tools', S.tool);
  controls.enableRotate = S.tool === 'view';
  document.querySelectorAll('#tool-params > div').forEach((d) => d.classList.toggle('show', d.dataset.for === S.tool));
  $('tool-hint').textContent = S.tool === 'view' ? '左ドラッグ＝回転、右ドラッグ＝移動、ホイール＝拡大'
    : S.tool === 'levee' ? '河道に沿って塗ると、範囲内の河道セルに堤防を設置（樋門付き）'
    : S.tool === 'retention' ? '河道沿いの低地を塗る：河道外は掘削、隣接河道は越流堤（この高さを超えると流入）'
    : S.tool === 'paddy_dam' ? '水田セルのみ有効（他の土地利用は無視）'
    : S.tool === 'erase' ? '塗った範囲の施策を取り消し' : '左ドラッグで塗る（右ドラッグ＝移動）';
});
document.querySelectorAll('#modes button').forEach((b) => b.onclick = () => { S.mode = b.dataset.mode; selectButton('modes', S.mode); colorize(); refreshBars(); });
document.querySelectorAll('#which button').forEach((b) => b.onclick = () => { S.which = b.dataset.which; selectButton('which', S.which); setResultUI(); });
$('brush').oninput = (e) => { $('brush-v').textContent = e.target.value; };
$('exag').oninput = (e) => { S.exag = +e.target.value; $('exag-v').textContent = S.exag; updateTerrainHeights(); };
$('show-assets').onchange = refreshBars;
$('show-river').onchange = () => { riverLines.visible = $('show-river').checked; refreshBars(); };
$('run').onclick = runScenario;
$('clear').onclick = () => { S.edits = []; S.scen = null; renderEdits(); setResultUI(); };
$('gauge').onchange = drawCharts;
$('event').onchange = (e) => { S.event = e.target.value; runBaseline(); };
$('basin').onchange = (e) => loadBasin(e.target.value);

async function loadBasin(key) {
  setProgress(0.02, '流域データ読み込み中（初回は DEM・土地利用・資産データを処理）…');
  const st = await getJSON(`/api/basin/${key}`);
  st.mask = dec(st.mask, Uint8Array);
  st.landclass = dec(st.landclass, Uint8Array);
  st._z = dec(st.z, Float32Array);
  st.hand = dec(st.hand, Float32Array);
  S.fc = dec(st.flood_cells, Int32Array);
  S.hh = dec(st.households, Float32Array);
  S.value = dec(st.asset_value, Float32Array);
  S.st = st; S.basin = key; S.edits = []; S.base = null; S.scen = null;
  S.riverSet = new Set(st.river.ri.map((i, k) => i * st.nx + st.river.rj[k]));
  S.zcur = st._z;
  let zmin = Infinity, zmax = -Infinity;
  st._z.forEach((v) => { if (Number.isFinite(v)) { zmin = Math.min(zmin, v); zmax = Math.max(zmax, v); } });
  S.zmin = zmin; S.zmax = zmax;
  $('basin-info').textContent = `${st.nx}×${st.ny} セル（${st.dx} m）、河道 ${st.river.ri.length} セル、世帯 ${Math.round(st.households_total)}、資産 ${st.asset_total_oku.toFixed(0)} 億円`;
  $('event').innerHTML = st.events.map((e) => `<option value="${e.key}">${e.label}</option>`).join('');
  $('gauge').innerHTML = Object.keys(st.gauges).map((g) => `<option>${g}</option>`).join('');
  const pref = st.events.find((e) => e.role === 'validation' && e.key.endsWith('2020')) || st.events[st.events.length - 1];
  $('event').value = pref.key; S.event = pref.key;
  buildTerrain(); renderEdits(); colorize(); refreshBars();
  runBaseline();
}

async function init() {
  resize();
  const basins = await getJSON('/api/basins');
  $('basin').innerHTML = basins.map((b) => `<option value="${b.key}">${b.key}</option>`).join('');
  await loadBasin(basins[0].key);
}
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();
init().catch((e) => setProgress(0, `エラー: ${e.message}`));
