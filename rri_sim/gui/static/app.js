import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ constants
const CLASS_JP = { forest: '森林', paddy: '水田', field: '畑', urban: '建物用地', water: '水域', paddy_dam: '田んぼダム',
  forest_new: '新規森林', green: '緑地（移転跡地）', wetland: '湿地（遊水地）' };
const CLASS_COLOR = { forest: '#3d7a46', paddy: '#cfdc78', field: '#d9b46e', urban: '#9b9b9b', water: '#5b8fd6',
  paddy_dam: '#38c2c9', forest_new: '#14532d', green: '#86c06c', wetland: '#7fb7a6' };
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
const SVC_JP = { medical: '医療', school: '教育（小・中学校）', commercial: '商業', public: '公共サービス' };
const FAC_COLOR = { medical: '#e11d48', school: '#2563eb', commercial: '#f59e0b', public: '#7c3aed' };
const DEPTH_STOPS = [[0.1, '#bfe3ff'], [0.5, '#6cb6ff'], [1.0, '#2f80ed'], [2.0, '#1747a6'], [3.0, '#3b1a7a'], [5.0, '#5a0f5f']];
const RIVER_STOPS = [[0.3, '#93c5fd'], [0.7, '#2563eb'], [0.95, '#facc15'], [1.05, '#dc2626']];

// --------------------------------------------------------------------- state
const S = {
  st: null, basin: null, event: null, base: null, scen: null, edits: [],
  tool: 'view', mode: 'landuse', which: 'scen', frame: -1, exag: 2.5, playing: null,
  painting: null, hoverCell: -1, unitCosts: {}, colors: null,
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
const oku = (yen) => yen / 1e8;
const fmt = (v, d = 1) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(d));
async function getJSON(url, opt) {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
function setProgress(f, msg) { $('bar').style.width = `${Math.round(f * 100)}%`; $('pmsg').textContent = msg || ''; }
function log(msg) { $('log').textContent = `${msg}\n` + $('log').textContent; }
function diffCell(b, s, unit, better, d = 1) {
  if (s == null) return `<td>${fmt(b, d)}${unit}</td><td>—</td><td>—</td>`;
  const dd = s - b;
  const cls = Math.abs(dd) < 1e-9 ? '' : ((better === 'lower' ? dd < 0 : dd > 0) ? 'good' : 'bad');
  return `<td>${fmt(b, d)}${unit}</td><td>${fmt(s, d)}${unit}</td><td class="${cls}">${dd > 0 ? '+' : ''}${fmt(dd, d)}</td>`;
}

// --------------------------------------------------------------------- scene
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#dfe6ea');
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.addEventListener('change', () => drawMinimapOverlay());
scene.add(new THREE.HemisphereLight('#ffffff', '#6b7b6b', 1.1));
const sun = new THREE.DirectionalLight('#ffffff', 1.6);
sun.position.set(-30, 40, -20);
scene.add(sun);

let terrain = null, riverLines = null, assetBars = null, damageBars = null, leveeWalls = null, damMarks = null;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

function cellXZ(i, j) { const st = S.st; return [(j - st.nx / 2) * st.dx / 1000, (i - st.ny / 2) * st.dy / 1000]; }
function zAt(c) { const z = S.zcur[c]; return Number.isFinite(z) ? (z - S.zmin) / 1000 * S.exag : 0; }

function buildTerrain() {
  const st = S.st, nx = st.nx, ny = st.ny;
  [terrain, riverLines, assetBars, damageBars, leveeWalls, damMarks].forEach((o) => o && scene.remove(o));
  const pos = new Float32Array(nx * ny * 3), col = new Float32Array(nx * ny * 3), idx = [];
  for (let i = 0; i < ny; i++) for (let j = 0; j < nx; j++) {
    const c = i * nx + j, [x, zz] = cellXZ(i, j);
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
  buildRiver(); buildAssetBars(); buildDams();
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
  buildRiver(); buildDams(); refreshBars();
}
function buildRiver() {
  if (riverLines) scene.remove(riverLines);
  const r = S.st.river, nx = S.st.nx, pos = [], col = [];
  for (let k = 0; k < r.ri.length; k++) {
    const d = r.dn[k];
    if (d < 0) continue;
    for (const m of [k, d]) {
      const [x, zz] = cellXZ(r.ri[m], r.rj[m]);
      pos.push(x, zAt(r.ri[m] * nx + r.rj[m]) + 0.012, zz); col.push(0.2, 0.45, 0.9);
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
    const i = Math.floor(c / st.nx), j = c % st.nx, [x, zz] = cellXZ(i, j), h = Math.max(heights[k], 1e-4);
    m.makeScale(1, h, 1).setPosition(x, zAt(c) + h / 2, zz);
    mesh.setMatrixAt(k, m); mesh.setColorAt(k, colors[k]);
  });
  mesh.count = cells.length;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return mesh;
}
function buildAssetBars() {
  if (assetBars) scene.remove(assetBars);
  const hh = currentHouseholds(), cells = [], hs = [], cs = [];
  for (let c = 0; c < hh.length; c++) if (S.st.mask[c] && hh[c] >= 1) { cells.push(c); hs.push(Math.sqrt(hh[c]) * 0.018); cs.push(hex('#f5f5f0')); }
  assetBars = makeBars(cells, hs, cs, 0.45);
  scene.add(assetBars);
}
function buildDams() {
  if (damMarks) scene.remove(damMarks);
  damMarks = new THREE.Group();
  for (const [name, dm] of Object.entries(S.st.dams || {})) {
    const i = Math.floor(dm.cell / S.st.nx), j = dm.cell % S.st.nx, [x, zz] = cellXZ(i, j);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.35, 0.8, 4), new THREE.MeshLambertMaterial({ color: '#334155' }));
    cone.position.set(x, zAt(dm.cell) + 0.5, zz);
    cone.rotation.x = Math.PI;
    cone.userData.name = name;
    damMarks.add(cone);
  }
  scene.add(damMarks);
}
function refreshBars() {
  if (!S.st) return;
  if (damageBars) { scene.remove(damageBars); damageBars = null; }
  if (leveeWalls) { scene.remove(leveeWalls); leveeWalls = null; }
  const res = currentResult();
  buildAssetBars();
  assetBars.visible = $('show-assets').checked && S.mode !== 'damage';
  if (S.mode === 'damage' && res) {
    const dmg = res._damage, val = S.value, cells = [], hs = [], cs = [];
    for (let c = 0; c < dmg.length; c++) if (dmg[c] > 1e6) {
      cells.push(c); hs.push(Math.sqrt(dmg[c] / 1e8) * 0.25);
      const ratio = val[c] > 0 ? dmg[c] / val[c] : 0;
      cs.push(lerpColor([[0, '#fde68a'], [0.2, '#f97316'], [0.5, '#b91c1c'], [1, '#450a0a']], ratio));
    }
    damageBars = makeBars(cells, hs, cs, 0.7);
    scene.add(damageBars);
  }
  const emb = currentEmbankment(), r = S.st.river, nx = S.st.nx, cells = [], hs = [], cs = [];
  for (let k = 0; k < r.ri.length; k++) if (emb[k] > 0.05) {
    cells.push(r.ri[k] * nx + r.rj[k]); hs.push(Math.max(emb[k] / 1000 * S.exag * 6, 0.02)); cs.push(hex('#5b3a1a'));
  }
  if (cells.length) { leveeWalls = makeBars(cells, hs, cs, 0.6); leveeWalls.visible = $('show-river').checked; scene.add(leveeWalls); }
}

// ------------------------------------------------------------------- results
function currentResult() { return S.which === 'base' ? S.base : (S.scen || S.base); }
function currentHouseholds() { const r = currentResult(); return (r && r._hh) ? r._hh : S.hh; }
function currentEmbankment() {
  const res = currentResult(), r = S.st.river, nx = S.st.nx;
  const emb = Float32Array.from(res ? res.embankment : r.embankment);
  for (const e of S.edits) if (e.type === 'levee') {
    const set = new Set(e.cells);
    for (let k = 0; k < r.ri.length; k++) if (set.has(r.ri[k] * nx + r.rj[k])) emb[k] = Math.max(emb[k], e.height);
  }
  return emb;
}
function prepResult(res) {
  res._max = dec(res.max_depth_cm, Uint16Array);
  res._damage = dec(res.damage, Float32Array);
  res._frames = dec(res.frames_depth_cm, Uint16Array);
  res._rframes = dec(res.frames_river_cm, Uint16Array);
  res._lc = dec(res.landclass, Uint8Array);
  res._z = res.z_changed ? dec(res.z_changed, Float32Array) : null;
  res._hh = dec(res.households, Float32Array);
  res._pop = dec(res.population, Float32Array);
  const nr = S.st.river.ri.length;
  res._rmax = new Float32Array(nr);
  for (let t = 0; t < res.nt; t++) for (let k = 0; k < nr; k++) res._rmax[k] = Math.max(res._rmax[k], res._rframes[t * nr + k] / 100);
  return res;
}
function depthField(res) {
  const n = S.st.nx * S.st.ny, out = new Float32Array(n);
  if (!res) return out;
  if (S.frame < 0) { for (let c = 0; c < n; c++) out[c] = res._max[c] / 100; return out; }
  const fc = S.fc, nf = fc.length, off = S.frame * nf;
  for (let k = 0; k < nf; k++) { const v = res._frames[off + k] / 100; out[fc[k]] = v >= 0.1 ? v : 0; }
  return out;
}

// ------------------------------------------------------------------ colouring
function pendingOverlay() {
  const m = new Map();
  for (const e of S.edits) for (const c of e.cells) m.set(c, e.type);
  if (S.painting) for (const c of S.painting.cells) m.set(c, S.painting.type === 'erase' ? null : S.painting.type);
  return m;
}
function computeColors() {
  const st = S.st, n = st.nx * st.ny, col = new Float32Array(n * 3).fill(1);
  const res = currentResult();
  const lc = (S.scen && S.which === 'scen') ? S.scen._lc : st.landclass;
  const over = pendingOverlay();
  const neutral = hex('#c9cfc4'), neutralHigh = hex('#e8ebe4');
  let depth = null, dDiff = null, pop = null, popMax = 1;
  if (S.mode === 'flood') depth = depthField(res);
  if (S.mode === 'diff' && S.base && S.scen) { const a = depthField(S.base), b = depthField(S.scen); dDiff = b.map((v, k) => v - a[k]); }
  if (S.mode === 'population') { pop = (res && res._pop) || S.pop; for (let c = 0; c < n; c++) if (st.mask[c]) popMax = Math.max(popMax, pop[c]); }
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
    if (pop && pop[c] > 0.5) color = lerpColor([[0, '#fff7bc'], [0.3, '#fe9929'], [1, '#993404']], Math.sqrt(pop[c] / popMax));
    const t = over.get(c);
    if (t !== undefined) {
      if (t === null) color = color.clone().lerp(hex('#ffffff'), 0.6);
      else color = color.clone().lerp(hex(TOOL[t].color), S.mode === 'landuse' ? 0.75 : 0.45);
    }
    col[3 * c] = color.r; col[3 * c + 1] = color.g; col[3 * c + 2] = color.b;
  }
  return col;
}
function colorize() {
  if (!terrain) return;
  S.colors = computeColors();
  terrain.geometry.attributes.color.array.set(S.colors);
  terrain.geometry.attributes.color.needsUpdate = true;
  colorRiver(currentResult());
  refreshLegend();
  drawMinimap();
}
function riverRatio(res, m) {
  const r = S.st.river;
  if (!res) return 0.3;
  const h = S.frame < 0 ? res._rmax[m] : res._rframes[S.frame * r.ri.length + m] / 100;
  return h / Math.max(r.depth[m] + res.embankment[m], 0.1);
}
function colorRiver(res) {
  if (!riverLines) return;
  const r = S.st.river, col = riverLines.geometry.attributes.color.array;
  let p = 0;
  for (let k = 0; k < r.ri.length; k++) {
    if (r.dn[k] < 0) continue;
    for (const m of [k, r.dn[k]]) { const c = lerpColor(RIVER_STOPS, riverRatio(res, m)); col[p++] = c.r; col[p++] = c.g; col[p++] = c.b; }
  }
  riverLines.geometry.attributes.color.needsUpdate = true;
}
function refreshLegend() {
  const sw = (c, t) => `<div class="item"><span class="sw" style="background:${c}"></span>${t}</div>`;
  let h = '';
  if (S.mode === 'landuse') h = Object.entries(CLASS_JP).map(([k, v]) => sw(CLASS_COLOR[k], v)).join('');
  if (S.mode === 'flood') h = '<b>浸水深</b>' + DEPTH_STOPS.map(([v, c]) => sw(c, `${v} m〜`)).join('');
  if (S.mode === 'damage') h = '<b>被害額（柱の高さ∝√被害額）</b>' + sw('#fde68a', '被害率 小') + sw('#b91c1c', '被害率 大');
  if (S.mode === 'diff') h = '<b>最大浸水深の差（施策−現況）</b>' + sw('#22c55e', '浅くなった') + sw('#ef4444', '深くなった');
  if (S.mode === 'population') h = '<b>人口（2020国勢調査, 移転後）</b>' + sw('#fff7bc', '少') + sw('#993404', '多');
  h += '<hr style="border:0;border-top:1px solid #ddd">' + sw('#2563eb', '河道（水位/堤防高）') + sw('#dc2626', '越水') +
    sw('#5b3a1a', '堤防') + sw('#334155', 'ダム（▼）');
  if ($('show-fac').checked) h += Object.entries(FAC_COLOR).map(([k, c]) => sw(c, `施設: ${SVC_JP[k]}`)).join('');
  $('legend').innerHTML = h;
}

// ------------------------------------------------------------------- 2D map
const mm = $('minimap'), mctx = mm.getContext('2d');
const mmBase = document.createElement('canvas');
let mmScale = 1;
function setupMinimap() {
  const st = S.st, maxW = 270, maxH = 240;
  mmScale = Math.min(maxW / st.nx, maxH / st.ny);
  mm.width = Math.round(st.nx * mmScale); mm.height = Math.round(st.ny * mmScale);
  mmBase.width = st.nx; mmBase.height = st.ny;
}
function drawMinimap() {
  if (!S.colors) return;
  const st = S.st, bctx = mmBase.getContext('2d'), img = bctx.createImageData(st.nx, st.ny);
  for (let c = 0; c < st.nx * st.ny; c++) {
    const o = 4 * c;
    if (!st.mask[c]) { img.data[o + 3] = 0; continue; }
    img.data[o] = S.colors[3 * c] * 255; img.data[o + 1] = S.colors[3 * c + 1] * 255; img.data[o + 2] = S.colors[3 * c + 2] * 255; img.data[o + 3] = 255;
  }
  bctx.putImageData(img, 0, 0);
  drawMinimapOverlay();
}
function drawMinimapOverlay() {
  if (!S.st) return;
  const st = S.st, s = mmScale, res = currentResult(), r = st.river;
  mctx.clearRect(0, 0, mm.width, mm.height);
  mctx.imageSmoothingEnabled = false;
  mctx.drawImage(mmBase, 0, 0, mm.width, mm.height);
  if ($('show-river').checked) {
    mctx.lineWidth = 1;
    for (let k = 0; k < r.ri.length; k++) {
      const d = r.dn[k];
      if (d < 0 || r.area[k] < 20) continue;
      mctx.strokeStyle = '#' + lerpColor(RIVER_STOPS, riverRatio(res, k)).getHexString();
      mctx.beginPath(); mctx.moveTo((r.rj[k] + 0.5) * s, (r.ri[k] + 0.5) * s); mctx.lineTo((r.rj[d] + 0.5) * s, (r.ri[d] + 0.5) * s); mctx.stroke();
    }
  }
  if ($('show-fac').checked) for (const [k, cells] of Object.entries(st.facilities || {})) {
    mctx.fillStyle = FAC_COLOR[k];
    for (const c of cells) { const i = Math.floor(c / st.nx), j = c % st.nx; mctx.fillRect(j * s, i * s, Math.max(2, s), Math.max(2, s)); }
  }
  mctx.font = '10px "Hiragino Sans", sans-serif';
  for (const [name, dm] of Object.entries(st.dams || {})) {
    const i = Math.floor(dm.cell / st.nx), j = dm.cell % st.nx, x = (j + 0.5) * s, y = (i + 0.5) * s;
    mctx.fillStyle = '#334155'; mctx.beginPath(); mctx.moveTo(x - 4, y - 3); mctx.lineTo(x + 4, y - 3); mctx.lineTo(x, y + 4); mctx.fill();
    mctx.fillText(name, x + 5, y + 3);
  }
  for (const [g, k] of Object.entries(st.gauges)) {
    const x = (r.rj[k] + 0.5) * s, y = (r.ri[k] + 0.5) * s;
    mctx.fillStyle = '#fff'; mctx.strokeStyle = '#111'; mctx.lineWidth = 1.2;
    mctx.beginPath(); mctx.arc(x, y, 3.2, 0, 2 * Math.PI); mctx.fill(); mctx.stroke();
    mctx.fillStyle = '#111'; mctx.fillText(g, x + 5, y - 3);
  }
  const t = controls.target, j = t.x * 1000 / st.dx + st.nx / 2, i = t.z * 1000 / st.dy + st.ny / 2;
  const dir = new THREE.Vector3().subVectors(camera.position, t);
  mctx.strokeStyle = '#dc2626'; mctx.lineWidth = 1.5;
  mctx.beginPath(); mctx.arc(j * s, i * s, 5, 0, 2 * Math.PI); mctx.stroke();
  const L = 18, n = Math.hypot(dir.x, dir.z) || 1;
  mctx.beginPath(); mctx.moveTo(j * s, i * s); mctx.lineTo(j * s + dir.x / n * L, i * s + dir.z / n * L); mctx.stroke();
}
function minimapCell(ev) {
  const rect = mm.getBoundingClientRect(), st = S.st;
  const j = Math.floor((ev.clientX - rect.left) / mmScale), i = Math.floor((ev.clientY - rect.top) / mmScale);
  if (i < 0 || j < 0 || i >= st.ny || j >= st.nx) return -1;
  const c = i * st.nx + j;
  return st.mask[c] ? c : -1;
}
mm.addEventListener('pointerdown', (ev) => {
  const c = minimapCell(ev);
  if (c < 0) return;
  if (S.tool === 'view') {
    const i = Math.floor(c / S.st.nx), j = c % S.st.nx, [x, zz] = cellXZ(i, j);
    const off = new THREE.Vector3().subVectors(camera.position, controls.target);
    controls.target.set(x, zAt(c), zz); camera.position.copy(controls.target).add(off); controls.update();
    return;
  }
  S.painting = { type: S.tool, cells: new Set(brushCells(c)) };
  colorize();
});
mm.addEventListener('pointermove', (ev) => {
  const c = minimapCell(ev);
  if (S.painting && c >= 0) { brushCells(c).forEach((k) => S.painting.cells.add(k)); colorize(); }
  if (c !== S.hoverCell) { S.hoverCell = c; showHover(c); }
});

// ------------------------------------------------------------------- picking / painting
function pickCell(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(terrain)[0];
  if (!hit) return -1;
  const st = S.st, j = Math.round(hit.point.x * 1000 / st.dx + st.nx / 2), i = Math.round(hit.point.z * 1000 / st.dy + st.ny / 2);
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
  if (S.painting && c >= 0) { brushCells(c).forEach((k) => S.painting.cells.add(k)); colorize(); }
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
function applicable(type, cells) {
  const st = S.st, cn = st.class_names, lc = st.landclass, id = (n) => cn.indexOf(n);
  if (type === 'paddy_dam') return cells.filter((c) => lc[c] === id('paddy'));
  if (type === 'forest') return cells.filter((c) => ![id('forest'), id('water'), id('forest_new')].includes(lc[c]));
  if (type === 'levee') return cells.filter((c) => S.riverSet.has(c));
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
    li.appendChild(b); ul.appendChild(li);
  });
  if (!S.edits.length) ul.innerHTML = '<li class="hint">施策を地図上で塗ってください</li>';
}
function showHover(c) {
  const el = $('hover');
  if (c < 0) { el.textContent = ''; return; }
  const st = S.st, res = currentResult();
  const lc = (S.scen && S.which === 'scen') ? S.scen._lc[c] : st.landclass[c];
  const pop = (res && res._pop) ? res._pop[c] : S.pop[c];
  let t = `標高 ${S.zcur[c].toFixed(1)} m\n土地利用 ${CLASS_JP[st.class_names[lc]]}\n人口 ${pop.toFixed(0)}  世帯 ${currentHouseholds()[c].toFixed(0)}\n`;
  t += `HAND ${st.hand[c] >= 0 ? st.hand[c].toFixed(1) + ' m' : '—'}\n資産額 ${(S.value[c] / 1e8).toFixed(2)} 億円`;
  if (res) t += `\n浸水深 ${depthField(res)[c].toFixed(2)} m\n被害額 ${(res._damage[c] / 1e8).toFixed(2)} 億円`;
  el.textContent = t;
}

// ------------------------------------------------------------------- panels
let dChart = null;
const qCharts = {};
function drawFlowCharts() {
  const base = S.base, scen = S.scen, box = $('q-charts');
  if (!base) return;
  const labels = base.times.map((t) => t.slice(5, 13).replace('T', ' '));
  const names = Object.keys(base.gauges);
  if (box.dataset.basin !== S.basin + S.event) {
    box.innerHTML = names.map((g, k) => `<div class="qchart"><div class="t"><span>${g}</span><span id="qk-${k}"></span></div><canvas id="qc-${k}" height="120"></canvas></div>`).join('');
    box.dataset.basin = S.basin + S.event;
    for (const k of Object.keys(qCharts)) { qCharts[k].destroy(); delete qCharts[k]; }
  }
  names.forEach((g, k) => {
    const ds = [];
    const hasObs = base.gauges[g].obs.some((v) => v != null);
    if (hasObs) ds.push({ label: '観測', data: base.gauges[g].obs, borderColor: '#111', backgroundColor: '#111', pointRadius: 1.2, showLine: false });
    ds.push({ label: '現況', data: base.gauges[g].sim, borderColor: '#2563eb', pointRadius: 0, borderWidth: 1.4 });
    if (scen) ds.push({ label: '施策', data: scen.gauges[g].sim, borderColor: '#dc2626', pointRadius: 0, borderWidth: 1.4 });
    if (qCharts[k]) qCharts[k].destroy();
    qCharts[k] = new Chart($(`qc-${k}`), { type: 'line', data: { labels, datasets: ds },
      options: { animation: false, interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: k === 0, labels: { boxWidth: 8, font: { size: 9 } } } },
        scales: { x: { ticks: { maxTicksLimit: 5, font: { size: 9 } } }, y: { ticks: { font: { size: 9 } }, title: { display: true, text: 'm³/s', font: { size: 9 } } } } } });
    const pk = (a) => Math.max(...a.filter((v) => v != null));
    $(`qk-${k}`).textContent = `ピーク 観測 ${hasObs ? fmt(pk(base.gauges[g].obs), 0) : '—'} / 現況 ${fmt(pk(base.gauges[g].sim), 0)}${scen ? ' / 施策 ' + fmt(pk(scen.gauges[g].sim), 0) : ''} m³/s`;
  });
}
function drawDamagePanel() {
  const base = S.base, scen = S.scen;
  if (!base) return;
  const rows = [
    ['被害額 [億円]', base.damage_total_oku, scen?.damage_total_oku, 1],
    ['浸水面積 [km²]', base.inundated_km2, scen?.inundated_km2, 1],
    ['床上浸水世帯', base.households_above_floor, scen?.households_above_floor, 0],
    ['床下浸水世帯', base.households_below_floor, scen?.households_below_floor, 0],
  ];
  $('kpis').innerHTML = `<table class="ind"><tr><th></th><th>現況</th><th>施策</th><th>差</th></tr>` +
    rows.map(([n, b, s, d]) => `<tr><td>${n}</td>${diffCell(b, s, '', 'lower', d)}</tr>`).join('') + '</table>' +
    (scen && scen.retention_max_storage_mcm > 0 ? `<div class="hint">遊水地 最大貯留量（概算）${scen.retention_max_storage_mcm.toFixed(2)} 百万m³（遊水地内は集計対象外）</div>` : '') +
    (Object.keys(base.dams || {}).length ? `<div class="hint">ダム最大貯留: ${Object.entries((scen || base).dams).map(([n, v]) => `${n} ${v.max_storage_mcm.toFixed(1)}/${v.capacity_mcm.toFixed(1)} 百万m³`).join('、')}</div>` : '');
  const cats = Object.keys(base.damage_by_category_oku);
  if (dChart) dChart.destroy();
  dChart = new Chart($('dmg-chart'), { type: 'bar',
    data: { labels: cats.map((c) => CAT_JP[c] || c), datasets: [{ label: '現況', data: cats.map((c) => base.damage_by_category_oku[c]), backgroundColor: '#93c5fd' },
      ...(scen ? [{ label: '施策', data: cats.map((c) => scen.damage_by_category_oku[c] || 0), backgroundColor: '#f87171' }] : [])] },
    options: { animation: false, indexAxis: 'y', plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 } } } }, scales: { x: { title: { display: true, text: '億円' } } } } });
  drawCosts();
}
function costTable(cost) {
  const q = (it) => {
    const x = it.q || {};
    if (it.type === 'levee') return `延長 ${fmt((x.length_m || 0) / 1000)} km・平均嵩上 ${fmt(x.height_m)} m`;
    if (it.type === 'retention') return `${fmt((x.area_m2 || 0) / 1e6, 2)} km²・掘削 ${fmt(x.depth_m)} m`;
    if (it.type === 'paddy_dam') return `水田 ${fmt((x.area_m2 || 0) / 1e6, 2)} km²`;
    if (it.type === 'forest') return `${fmt((x.area_m2 || 0) / 1e6, 2)} km²`;
    if (it.type === 'relocate') return `${fmt(x.households, 0)} 世帯・${fmt(x.employees, 0)} 人`;
    if (it.type === 'raise') return `${fmt(x.households, 0)} 世帯・+${fmt(x.height_m)} m${x.terrain ? ' 盛土' : ''}`;
    return '';
  };
  return `<table class="ind"><tr><th>施策</th><th>数量</th><th>初期 [億円]</th><th>年 [百万円]</th><th>現在価値 [億円]</th></tr>` +
    cost.items.map((it) => `<tr><td>${TOOL[it.type].jp}</td><td>${q(it)}</td><td>${fmt(oku(it.initial_yen), 2)}</td><td>${fmt(it.annual_yen / 1e6, 1)}</td><td>${fmt(oku(it.pv_yen), 2)}</td></tr>`).join('') +
    `<tr class="total"><td>合計</td><td></td><td>${fmt(oku(cost.initial_yen), 2)}</td><td>${fmt(cost.annual_yen / 1e6, 1)}</td><td>${fmt(oku(cost.pv_yen), 2)}</td></tr></table>`;
}
function drawCosts() {
  const scen = S.scen, base = S.base, el = $('costs');
  if (!scen || !scen.cost.items.length) { el.innerHTML = '<div class="hint">施策を計算すると事業費（初期費用＋維持費の現在価値: 50年・割引率4%）を表示します。</div>'; return; }
  const red = base.damage_total_oku - scen.damage_total_oku, pv = oku(scen.cost.pv_yen);
  el.innerHTML = costTable(scen.cost) +
    `<div class="score"><span>本イベントの被害軽減額 <b>${fmt(red)}</b> 億円</span><span>軽減額／事業費(PV) <b>${pv > 0 ? fmt(red / pv, 2) : '—'}</b></span></div>` +
    '<div class="hint">※軽減額は単一イベントの値。正式な費用便益比（年平均被害軽減期待額による）ではない。単価は仮定値を含む（単価設定で変更可）。</div>';
}
async function recomputeCosts() {
  if (!S.scen) return;
  S.scen.cost = await getJSON('/api/costs', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items: S.scen.cost.items.map((i) => ({ type: i.type, q: i.q })), unit_costs: S.unitCosts }) });
  drawCosts();
}
function buildUnitCostEditor() {
  const uc = S.st.unit_costs, el = $('unit-costs');
  el.innerHTML = Object.entries(uc).map(([k, v]) => `<label title="${v.basis}">${v.label} [${v.unit}]<input type="number" data-k="${k}" value="${S.unitCosts[k] ?? v.value}" step="any"></label>`).join('') +
    '<div class="hint">カーソルを合わせると単価の根拠を表示。変更すると事業費を再計算（洪水計算は不要）。</div>';
  el.querySelectorAll('input').forEach((inp) => inp.onchange = () => { S.unitCosts[inp.dataset.k] = +inp.value; recomputeCosts(); });
}
function drawIndicators() {
  const b = S.base, s = S.scen;
  if (!b) return;
  const L0 = b.livability, L1 = s?.livability, E0 = b.ecology, E1 = s?.ecology;
  const sc = (x0, x1) => `<div class="score"><span class="big">${fmt(x0)}</span><span>現況</span>${x1 != null ? `<span class="big ${x1 - x0 >= 0 ? 'good' : 'bad'}">${fmt(x1)}</span><span>施策（${x1 - x0 >= 0 ? '+' : ''}${fmt(x1 - x0)}）</span>` : ''}</div>`;
  let h = sc(L0.score, L1?.score) + '<table class="ind"><tr><th>サービス</th><th>徒歩800m圏<br>人口カバー率</th><th>車10–30分<br>アクセス</th><th>平均所要<br>（車, 分）</th></tr>';
  for (const k of Object.keys(L0.services)) {
    const a = L0.services[k], z = L1?.services[k];
    h += `<tr><td>${SVC_JP[k]}</td><td>${fmt(a.walk800_coverage * 100, 0)}%${z ? ' → ' + fmt(z.walk800_coverage * 100, 0) + '%' : ''}</td><td>${fmt(a.car_access * 100, 0)}%${z ? ' → ' + fmt(z.car_access * 100, 0) + '%' : ''}</td><td>${fmt(a.mean_car_min)}${z ? ' → ' + fmt(z.mean_car_min) : ''}</td></tr>`;
  }
  h += `<tr><td>居住集約度（人口加重密度）</td><td colspan="3">${fmt(L0.population_weighted_density_per_ha)} 人/ha${L1 ? ' → ' + fmt(L1.population_weighted_density_per_ha) : ''}（基準 40 人/ha = DID）</td></tr></table>`;
  $('liv').innerHTML = h;
  const row = (n, k) => `<tr><td>${n}</td>${diffCell(E0[k] * 100, E1 ? E1[k] * 100 : null, '', 'higher', 1)}</tr>`;
  $('eco').innerHTML = sc(E0.score, E1?.score) + '<table class="ind"><tr><th></th><th>現況</th><th>施策</th><th>差</th></tr>' +
    row('E1 生息地の質', 'habitat_quality') + row('E2 生態的ネットワーク連結性', 'connectivity') +
    row('　陸域連結性（PC近似）', 'terrestrial_connectivity') + row('　河川横断連結性（堤防なし延長率）', 'river_lateral_connectivity') +
    row('　河川縦断連結性 DCI（ダム）', 'river_longitudinal_dci') + row('E3 水文・生態条件', 'hydro_ecological') +
    row('　流域内保水率（本イベント）', 'event_retention') + row('　自然的氾濫原の冠水率', 'natural_flood_pulse') + '</table>';
  $('safety').innerHTML = '<table class="ind"><tr><th></th><th>現況</th><th>施策</th><th>差</th></tr>' +
    `<tr><td>浸水深0.5m以上の区域の居住人口割合</td>${diffCell(L0.pop_in_hazard_share * 100, L1 ? L1.pop_in_hazard_share * 100 : null, '%', 'lower')}</tr>` +
    `<tr><td>洪水時に医療機関へ30分以内で到達可能な人口割合</td>${diffCell(L0.flood_medical_access_30min * 100, L1 ? L1.flood_medical_access_30min * 100 : null, '%', 'higher')}</tr></table>`;
}
function drawPanels() { drawDamagePanel(); drawIndicators(); drawFlowCharts(); }

// ------------------------------------------------------------------- jobs
async function runJob(edits) {
  const params = { paddy_store_m: +$('pd-store').value / 100, paddy_release_mmh: +$('pd-rel').value,
    forest_n: +$('fo-n').value, forest_soil_m: +$('fo-soil').value, unit_costs: S.unitCosts };
  const { job } = await getJSON('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ basin: S.basin, event: S.event, edits, params }) });
  $('run').disabled = true;
  try {
    for (;;) {
      await new Promise((r) => setTimeout(r, 800));
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
  colorize(); refreshBars(); drawPanels(); updateTimeLabel();
}
async function runBaseline() {
  S.base = null; S.scen = null;
  setProgress(0, '現況（施策なし）を計算中… 初回はデータ取得とスピンアップ計算のため時間がかかります');
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
  if (!S.base) await runBaseline();
  if (!S.edits.length) { setProgress(0, '施策がありません（現況のみ）'); return; }
  try {
    S.scen = await runJob(S.edits);
    log(`施策 計算完了: ${S.scen.wall_s.toFixed(1)} s, 水収支誤差 ${S.scen.mass_rel_err.toExponential(1)}\n` + S.scen.edits_log.join('\n'));
    S.which = 'scen'; selectButton('which', 'scen');
    setResultUI();
    const dd = S.scen.damage_total_oku - S.base.damage_total_oku;
    setProgress(1, `完了: 被害額 ${dd <= 0 ? '' : '+'}${dd.toFixed(1)} 億円、事業費(PV) ${oku(S.scen.cost.pv_yen).toFixed(1)} 億円`);
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
  S.playing = setInterval(() => { S.frame = (S.frame + 1) % res.nt; $('time').value = S.frame; updateTimeLabel(); colorize(); }, 180);
};

// -------------------------------------------------------------------- UI wiring
function selectButton(group, val) {
  const key = group === 'modes' ? 'mode' : group === 'which' ? 'which' : group === 'tabs' ? 'tab' : 'tool';
  const sel = group === 'tools' ? '#tools button, #ops button' : `#${group} button`;
  document.querySelectorAll(sel).forEach((b) => b.classList.toggle('active', b.dataset[key] === val));
}
document.querySelectorAll('#tools button, #ops button').forEach((b) => b.onclick = () => {
  S.tool = b.dataset.tool; selectButton('tools', S.tool);
  controls.enableRotate = S.tool === 'view';
  document.querySelectorAll('#tool-params > div').forEach((d) => d.classList.toggle('show', d.dataset.for === S.tool));
  $('tool-hint').textContent = S.tool === 'view' ? '左ドラッグ＝回転、右ドラッグ＝移動、ホイール＝拡大。2D地図クリックで移動。'
    : S.tool === 'levee' ? '河道に沿って塗ると、範囲内の河道セルに堤防を設置（樋門付き）'
    : S.tool === 'retention' ? '河道沿いの低地を塗る：河道外は掘削、隣接河道は越流堤（この高さを超えると流入）'
    : S.tool === 'paddy_dam' ? '水田セルのみ有効（他の土地利用は無視）'
    : S.tool === 'erase' ? '塗った範囲の施策を取り消し' : '左ドラッグで塗る（3D・2D地図）。右ドラッグ＝移動';
});
document.querySelectorAll('#modes button').forEach((b) => b.onclick = () => { S.mode = b.dataset.mode; selectButton('modes', S.mode); colorize(); refreshBars(); });
document.querySelectorAll('#which button').forEach((b) => b.onclick = () => { S.which = b.dataset.which; selectButton('which', S.which); setResultUI(); });
document.querySelectorAll('#tabs button').forEach((b) => b.onclick = () => {
  selectButton('tabs', b.dataset.tab);
  document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('show', p.dataset.panel === b.dataset.tab));
  if (b.dataset.tab === 'flow') drawFlowCharts();
});
$('edit-costs').onclick = () => $('unit-costs').classList.toggle('show');
$('brush').oninput = (e) => { $('brush-v').textContent = e.target.value; };
$('exag').oninput = (e) => { S.exag = +e.target.value; $('exag-v').textContent = S.exag; updateTerrainHeights(); };
$('show-assets').onchange = refreshBars;
$('show-river').onchange = () => { riverLines.visible = $('show-river').checked; refreshBars(); drawMinimapOverlay(); };
$('show-fac').onchange = () => { refreshLegend(); drawMinimapOverlay(); };
$('run').onclick = runScenario;
$('clear').onclick = () => { S.edits = []; S.scen = null; renderEdits(); setResultUI(); };
$('event').onchange = (e) => { S.event = e.target.value; runBaseline(); };
$('basin').onchange = (e) => loadBasin(e.target.value);

async function loadBasin(key) {
  setProgress(0.02, '流域データ読み込み中（初回は DEM・土地利用・資産・施設データを処理）…');
  const st = await getJSON(`/api/basin/${key}`);
  st.mask = dec(st.mask, Uint8Array); st.landclass = dec(st.landclass, Uint8Array);
  st._z = dec(st.z, Float32Array); st.hand = dec(st.hand, Float32Array);
  S.fc = dec(st.flood_cells, Int32Array); S.hh = dec(st.households, Float32Array);
  S.pop = dec(st.population, Float32Array); S.value = dec(st.asset_value, Float32Array);
  S.st = st; S.basin = key; S.edits = []; S.base = null; S.scen = null; S.zcur = st._z;
  S.riverSet = new Set(st.river.ri.map((i, k) => i * st.nx + st.river.rj[k]));
  let zmin = Infinity, zmax = -Infinity;
  st._z.forEach((v) => { if (Number.isFinite(v)) { zmin = Math.min(zmin, v); zmax = Math.max(zmax, v); } });
  S.zmin = zmin; S.zmax = zmax;
  $('basin-info').textContent = `流域面積 ${st.area_km2.toFixed(0)} km²、${st.nx}×${st.ny} セル（${st.dx} m）、河道 ${st.river.ri.length} セル、人口 ${Math.round(st.population_total).toLocaleString()}、世帯 ${Math.round(st.households_total).toLocaleString()}、資産 ${st.asset_total_oku.toFixed(0)} 億円、ダム ${Object.keys(st.dams).length}`;
  $('event').innerHTML = st.events.map((e) => `<option value="${e.key}">${e.label}</option>`).join('');
  const pref = st.events.find((e) => e.role === 'validation' && /20(19|20)$/.test(e.key)) || st.events[st.events.length - 1];
  $('event').value = pref.key; S.event = pref.key;
  setupMinimap(); buildTerrain(); renderEdits(); buildUnitCostEditor(); colorize(); refreshBars();
  runBaseline();
}
async function init() {
  resize();
  const basins = await getJSON('/api/basins');
  const JP = { kusu_ogase: '筑後川支川 玖珠川（小ヶ瀬）', abukuma_akutsu: '阿武隈川上流（阿久津）', chikugo_whole: '筑後川 全域', abukuma_whole: '阿武隈川 全域' };
  $('basin').innerHTML = basins.map((b) => `<option value="${b.key}">${JP[b.key] || b.key}</option>`).join('');
  await loadBasin(basins[0].key);
}
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();
init().catch((e) => setProgress(0, `エラー: ${e.message}`));
