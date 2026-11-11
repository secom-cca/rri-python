"""1K-RRI: macro-scale (1 km) rainfall-runoff-inundation model (Sugawara, Sayama, Yamada and Yamamoto, 2024:
「広域かつ高速な降雨流出氾濫計算のためのマクロスケール RRI モデルの開発」, 土木学会論文集 80(16), 23-16125).

Every 1 km cell holds a mountain part (山地, area A_s, mean slope i_s), a plain part (平野, area A_f = cell area - A_s,
mean slope i_f, elevation z_f) and a river channel (every cell has one).  Equations of the paper:

    mountain  (1)  A_s dh_s/dt = -q_s l_s + r A_s
                   q_s: RRI saturated-unsaturated discharge-stage relation (立川ら 2004) with the slope i_s;
                   the mountain drains into the channel from one side only, over l_s
              (2)  l_s = A_s / (A_s + A_f) l_r,   l_r = l = sqrt(dx dy),   l_f = l_r - l_s
    plain     (3)  A_f dh_f/dt = Q_xy - 2 q_f l_f + 2 q_rf l_f - v_i A_f + r A_f
                   Q_xy: diffusive wave to / from the 8 neighbouring plains, the cell taken as a grid cell of mean length l
                         with water depth h_f over the plain elevation z_f
                   q_f : kinematic wave with the mean plain slope i_f, into the channel whatever the river state
                   q_rf: overflow formula (本間) from the channel when the river level exceeds h_f and the levee crest
                   v_i : Green-Ampt infiltration
    river     diffusive wave as in RRI (here by default the local inertial form, workflow.K1_RIVER_SCHEME: same steady state,
              explicit steps not limited by the diffusive stability condition) along the network of the upscaled flow directions (最大集水河道追跡法 with relay
              zones, 舛谷ら 2006); width w, depth d, levee height h_b and z_f from the representative fine cell; bed z_f - d;
              kinematic wave from a channel cell with catchment area <= 16 km² into a larger one (no backwater, paper §2(2)c)
    land use  the most frequent class of the mountain and of the plain fine cells

Built by upscaling the basin's fine model (here the 250 m model, 4 x 4 cells; the paper upscales the 150 m JRRI 6 x 6).

ASSUMPTIONS (not in the paper; listed in ASSUMPTIONS below and in docs/14_1krri.md):
    mountain / plain split: fine cells of the forest class are mountain, all others plain
    relay zones: the four corner fine cells of each block
    a flow path that reaches another block's representative cell that is not a neighbour: the downstream cell is the first
        neighbouring block the path enters
    diagonal faces of Q_xy: width dx dy / distance (orthogonal faces: the cell side)
    plain kinematic wave: q_f = (1/n) (h_f - d_a - s_d)^{5/3} i_f^{1/2} (+ the RRI subsurface term when the class has one)
    mean slopes of the parts: mean magnitude of the 250 m raw-DEM gradient (floor 1e-4)
    cells on the basin divide: A_s, A_f within the basin only (rain volume = basin area), channel length l_r in proportion
    representative cells that are not channels of the fine model: w, d from the RRI geometry relation of the fine model,
        no levee, the fine model's default channel roughness
    measures on the plain: lowland drainage pumps lift into the first channel downstream larger than 16 km² (the main river);
        slow release of retained water (paddy dams, storage) goes to the cell's own channel
    damage (outside the paper): plain water depth h_f over z_f, sub-grid asset elevations relative to z_f
"""
from __future__ import annotations

import copy
import math
import time as _time
from dataclasses import dataclass

import numpy as np
from numba import njit

from . import kernels as K
from .domain import Domain, RiverNetwork, enforce_monotone_bed
from .hydro import D8_DI, D8_DJ
from .solver import MassBalance, Results

KIN_AREA_KM2 = 16.0           # paper §2(2)c: kinematic wave from channels below ~16 km² into larger ones
MIN_SLOPE = 1e-4

ASSUMPTIONS = {
    "mountain_plain_split": "250 m セルの土地利用が森林なら山地、それ以外は平野（論文は『山地（主として森林）』）",
    "relay_zone": "中継領域は 4×4 区画の四隅の 1 セル（論文に大きさの記載なし）",
    "non_adjacent_target": "流下経路が隣接しない区画の代表セルに達した場合、経路が最初に入った隣接区画を下流とする",
    "diagonal_width": "Q_xy の斜め方向の流下幅 = ΔxΔy / 距離（直交方向はセル幅）",
    "plain_kinematic": "q_f = (1/n)(h_f − d_a − s_d)^{5/3} i_f^{1/2}（Manning 型。地中流のある土地利用はその項を加える）",
    "slopes": "山地・平野の平均勾配 = 250 m DEM（未補正）の勾配の大きさの平均（下限 1e-4）",
    "divide_cells": "流域界のセルは流域内の面積だけを A_s, A_f とし、河道長 l_r も流域内の面積に比例させる（l_r = l (A_s+A_f)/ΔxΔy）",
    "non_channel_representative": "代表セルが 250 m モデルの河道でない場合、川幅・深さは同じ河道断面式、堤防なし、既定の粗度",
    "plain_measures": "内水の排水ポンプは下流で最初に集水面積 16 km² を超える河道（本川側）へ。田んぼダム・貯留施設の放流はそのセルの河道へ",
    "damage": "被害計算の浸水深は平野の水深 h_f（標高 z_f 基準）。建物・農地の標高分布は z_f からの差",
    "river_scheme": "河道は局所慣性式（論文は diffusive wave。定常解は同じ。陽解法の安定条件で時間刻みが数秒になるのを避ける）",
}


# ------------------------------------------------------------------ domain
@dataclass
class Domain1K(Domain):
    """Domain with the 1K-RRI sub-grid structure.  z = plain elevation z_f; landclass = dominant class of the cell
    (indicators, assets); lc_f / lc_s = plain / mountain classes (hydrology)."""
    As: np.ndarray = None          # mountain area in the basin [m2]
    Af: np.ndarray = None          # plain area in the basin [m2]
    i_s: np.ndarray = None
    i_f: np.ndarray = None
    lc_s: np.ndarray = None
    lc_f: np.ndarray = None
    kinematic: np.ndarray = None   # per river cell: link to the downstream cell by kinematic wave
    kcell: np.ndarray = None       # river index of each cell (-1 outside)

    def copy(self) -> "Domain1K":
        out = copy.copy(self)
        for k in ("z", "mask", "landclass", "As", "Af", "i_s", "i_f", "lc_s", "lc_f", "kinematic", "kcell", "open_boundary"):
            v = getattr(self, k)
            setattr(out, k, None if v is None else v.copy())
        out.params = copy.deepcopy(self.params)
        out.river = self.river.copy()
        return out

    @property
    def l(self) -> float:
        return math.sqrt(self.dx * self.dy)

    def lengths(self):
        """(l_s, l_f) per cell [m] (paper eq. 2: l_s = A_s / (A_s + A_f) l_r, l_f = l_r - l_s, l_r = l).  Cells on the basin
        divide (A_s + A_f < dx dy): l_r = l (A_s + A_f) / (dx dy), the same channel length per area as a full cell (assumption)."""
        tot = np.maximum(self.As + self.Af, 1e-9)
        lr = self.l * np.minimum(tot / (self.dx * self.dy), 1.0)
        ls = self.As / tot * lr
        return np.where(self.mask, ls, 0.0), np.where(self.mask, lr - ls, 0.0)


def _block_reduce(a, f, fn):
    ny, nx = a.shape
    by, bx = -(-ny // f), -(-nx // f)
    pad = np.full((by * f, bx * f), np.nan if a.dtype.kind == "f" else 0, dtype=a.dtype)
    pad[:ny, :nx] = a
    return fn(pad.reshape(by, f, bx, f).swapaxes(1, 2).reshape(by, bx, f * f))


def upscale(d, meta, factor=4, kin_area_km2=KIN_AREA_KM2, log=lambda m: None):
    """Fine Domain (+ build meta) -> (Domain1K, meta) by factor x factor blocks."""
    from .basin import height_above_nearest_drainage  # noqa: F401  (fine HAND is reused)
    f = int(factor)
    ny, nx = d.shape
    by, bx = -(-ny // f), -(-nx // f)
    fa = d.dx * d.dy
    mask = d.mask
    names = [c.name for c in d.params.land_classes]
    forest = names.index("forest")
    lc = d.landclass
    mtn = mask & (lc == forest)
    pln = mask & ~mtn
    zr = np.where(mask, meta["z_raw"], np.nan)
    gy, gx = np.gradient(np.where(np.isfinite(zr), zr, np.nanmean(zr)), d.dy, d.dx)
    slope = np.where(mask, np.hypot(gx, gy), np.nan)
    acc = np.where(mask, meta["acc_km2"], 0.0)
    fdir = meta["fdir"]
    # block index of every fine cell
    bi = np.arange(ny)[:, None] // f + 0 * np.arange(nx)[None]
    bj = 0 * np.arange(ny)[:, None] + np.arange(nx)[None] // f
    cmask = np.zeros((by, bx), bool)
    np.logical_or.at(cmask, (bi[mask], bj[mask]), True)
    nin = np.zeros((by, bx)); np.add.at(nin, (bi[mask], bj[mask]), 1)
    nmt = np.zeros((by, bx)); np.add.at(nmt, (bi[mtn], bj[mtn]), 1)
    As, Af = nmt * fa, (nin - nmt) * fa

    def mean_over(sel, val, floor):
        s = np.zeros((by, bx)); c = np.zeros((by, bx))
        np.add.at(s, (bi[sel], bj[sel]), val[sel]); np.add.at(c, (bi[sel], bj[sel]), 1)
        return np.where(c > 0, np.maximum(s / np.maximum(c, 1), floor), floor)
    i_s = mean_over(mtn & np.isfinite(slope), slope, MIN_SLOPE)
    i_f = mean_over(pln & np.isfinite(slope), slope, MIN_SLOPE)
    # land use: the most frequent class of each part (fine land classes are the basin model's base classes)
    ncl = len(names)
    cnt = np.zeros((ncl, by, bx))
    for c in range(ncl):
        sel = pln & (lc == c)
        np.add.at(cnt[c], (bi[sel], bj[sel]), 1)
    lc_f = np.argmax(cnt, 0)
    lc_f[cnt.sum(0) == 0] = names.index("field")
    lc_s = np.full((by, bx), forest)
    # dominant class of the whole cell (indicators / assets)
    frac = meta["landuse_fraction"]
    cfrac = np.stack([_block_reduce(np.where(mask, frac[c], np.nan), f, lambda x: np.nanmean(x, 2)) for c in range(frac.shape[0])])
    cfrac = np.nan_to_num(cfrac)
    landclass = np.argmax(cfrac, 0).astype(np.int64)
    # representative cells: largest catchment area outside the relay zones (block corners)
    rep = -np.ones((by, bx), np.int64)
    corner = np.zeros((ny, nx), bool)
    for a in (0, f - 1):
        for b in (0, f - 1):
            corner[(np.arange(ny) % f == a)[:, None] & (np.arange(nx) % f == b)[None]] = True
    flat_acc = acc.ravel()
    order = np.argsort(-flat_acc)
    for p in order:
        i, j = divmod(int(p), nx)
        if not mask[i, j] or corner[i, j]:
            continue
        if rep[bi[i, j], bj[i, j]] < 0:
            rep[bi[i, j], bj[i, j]] = p
    for p in order:                                      # blocks with only corner cells in the basin
        i, j = divmod(int(p), nx)
        if mask[i, j] and rep[bi[i, j], bj[i, j]] < 0:
            rep[bi[i, j], bj[i, j]] = p
    is_rep = np.zeros(ny * nx, bool)
    is_rep[rep[cmask]] = True
    # downstream block: follow the fine D8 path from the representative cell to the next representative cell
    cdn = -np.full((by, bx), 1, np.int64) * 0 - 1
    n_far = 0
    for a, b in zip(*np.nonzero(cmask)):
        p = int(rep[a, b])
        i, j = divmod(p, nx)
        first = None
        target = None
        for _ in range(ny * nx):
            k = fdir[i, j]
            if k < 0:
                break
            i, j = i + D8_DI[k], j + D8_DJ[k]
            if not (0 <= i < ny and 0 <= j < nx) or not mask[i, j]:
                break
            blk = (bi[i, j], bj[i, j])
            if blk != (a, b) and first is None:
                first = blk
            if is_rep[i * nx + j] and blk != (a, b):
                target = blk
                break
        if target is None:
            continue                                     # leaves the domain: outlet
        if max(abs(target[0] - a), abs(target[1] - b)) > 1:
            n_far += 1
            target = first
        cdn[a, b] = target[0] * bx + target[1]
    # river network: every basin cell
    ri, rj = np.nonzero(cmask)
    nr = len(ri)
    kcell = -np.ones((by, bx), np.int64)
    kcell[ri, rj] = np.arange(nr)
    dn = np.array([kcell.ravel()[cdn[a, b]] if cdn[a, b] >= 0 else -1 for a, b in zip(ri, rj)], np.int64)
    # break cycles (should not occur): a cell whose path returns to itself drains to its largest-area neighbour downhill
    dn = _break_cycles(dn, np.array([flat_acc[rep[a, b]] for a, b in zip(ri, rj)]), log)
    area = np.array([flat_acc[rep[a, b]] for a, b in zip(ri, rj)])
    di_, dj_ = np.zeros(nr, int), np.zeros(nr, int)
    has = dn >= 0
    di_[has], dj_[has] = ri[dn[has]] - ri[has], rj[dn[has]] - rj[has]
    D = d.dx * f
    length = np.where(has, np.hypot(di_ * D, dj_ * D), D)
    # channel geometry, levee, roughness and plain elevation of the representative cell
    fr = d.river
    fk = -np.ones(ny * nx, np.int64)
    fk[fr.ri * nx + fr.rj] = np.arange(fr.n)
    rp = d.params.river
    width, depth, emb, nriv = np.empty(nr), np.empty(nr), np.zeros(nr), np.full(nr, rp.n_river)
    zf_cell = np.empty(nr)
    zmodel = np.where(mask, d.z, np.nan)
    n_nonch = 0
    for k, (a, b) in enumerate(zip(ri, rj)):
        p = int(rep[a, b])
        i, j = divmod(p, nx)
        zf_cell[k] = zmodel[i, j]
        q = fk[p]
        if q >= 0:
            width[k], depth[k], emb[k], nriv[k] = fr.width[q], fr.depth[q], fr.embankment[q], fr.n_river[q]
        else:
            n_nonch += 1
            width[k], depth[k] = float(rp.width(area[k])), float(rp.depth(area[k]))
    width = np.minimum(width, 0.9 * D)
    zb = zf_cell - depth
    river = RiverNetwork(ri=ri, rj=rj, dn=dn, length=length, width=width, depth=depth.copy(), zb=zb, embankment=emb,
                         n_river=nriv, area_km2=area, outlet_slope=np.full(nr, 1e-3))
    enforce_monotone_bed(river, min_slope=1e-5)
    river.depth[:] = zf_cell - river.zb
    for k in np.nonzero(river.dn < 0)[0]:              # outlet slope: mean bed slope over ~5 km upstream (as build_river)
        path, dist = [k], 0.0
        while dist < 5000.0:
            ups = np.nonzero(river.dn == path[-1])[0]
            if not len(ups):
                break
            u = ups[np.argmax(river.area_km2[ups])]
            dist += river.length[u]
            path.append(u)
        if dist > 0:
            river.outlet_slope[k] = max((river.zb[path[-1]] - river.zb[k]) / dist, 1e-4)
    kin = np.zeros(nr, bool)
    kin[has] = (area[has] <= kin_area_km2) & (area[dn[has]] > kin_area_km2)
    zf = np.full((by, bx), np.nan)
    zf[ri, rj] = zf_cell
    # coast (open boundary of the plain), HAND (damage masks: lowest plain cell of the block)
    ob = None
    if d.open_boundary is not None:
        ob = _block_reduce(np.where(mask, d.open_boundary, 0.0), f, lambda x: np.nanmax(x, 2))
        ob = np.where(cmask, np.nan_to_num(ob), 0.0)
    hand_f = np.where(mask, meta["hand"], np.nan)
    hp = _block_reduce(np.where(pln, hand_f, np.nan), f, lambda x: np.nanmin(np.where(np.isnan(x), np.inf, x), 2))
    ha = _block_reduce(hand_f, f, lambda x: np.nanmin(np.where(np.isnan(x), np.inf, x), 2))
    hand = np.where(np.isfinite(hp), hp, ha)
    hand = np.where(cmask & np.isfinite(hand), hand, np.nan)
    from rasterio.transform import from_origin
    tf = d.transform
    tfc = from_origin(tf.c, tf.f, D, D)
    dk = Domain1K(z=zf, mask=cmask, dx=D, dy=D, landclass=landclass, params=copy.deepcopy(d.params), river=river,
                  transform=tfc, crs=d.crs, open_boundary=ob, As=np.where(cmask, As, 0.0), Af=np.where(cmask, Af, 0.0),
                  i_s=i_s, i_f=i_f, lc_s=lc_s.astype(np.int64), lc_f=lc_f.astype(np.int64), kinematic=kin, kcell=kcell)
    # coarse D8 code of the links (drain targets, displays)
    fd = -np.ones((by, bx), np.int64)
    for k in np.nonzero(has)[0]:
        for c in range(8):
            if D8_DI[c] == di_[k] and D8_DJ[c] == dj_[k]:
                fd[ri[k], rj[k]] = c
    acc_c = np.zeros((by, bx)); acc_c[ri, rj] = area
    out_k = int(np.argmax(np.where(dn < 0, area, -1)))
    zraw_c = _block_reduce(zr, f, lambda x: np.nanmean(x, 2))
    m2 = {"outlet_rc": (int(ri[out_k]), int(rj[out_k])), "area_km2": float(nin.sum() * fa / 1e6), "z_raw": np.where(cmask, zraw_c, np.nan),
          "coast_cells": int((ob > 0).sum()) if ob is not None else 0, "hand": hand, "landuse_fraction": np.where(cmask, cfrac, 0.0),
          "acc_km2": acc_c, "fdir": fd, "res": D, "model": "1K-RRI", "fine_res": d.dx,
          "upscale": {"factor": f, "relay_far_paths": n_far, "non_channel_representative": n_nonch, "kinematic_links": int(kin.sum()),
                      "mountain_share": float(As.sum() / max(As.sum() + Af.sum(), 1)), "assumptions": ASSUMPTIONS}}
    # gauges: the coarse channel next to the fine gauge whose catchment area is closest to the fine one
    g = {}
    for name, kf in meta.get("gauge_index", {}).items():
        i, j = int(fr.ri[kf]), int(fr.rj[kf])
        a0, b0 = i // f, j // f
        best, bk = np.inf, -1
        for a in range(a0 - 1, a0 + 2):
            for b in range(b0 - 1, b0 + 2):
                if 0 <= a < by and 0 <= b < bx and kcell[a, b] >= 0:
                    e = abs(math.log(max(area[kcell[a, b]], 1e-6) / max(fr.area_km2[kf], 1e-6)))
                    if e < best:
                        best, bk = e, int(kcell[a, b])
        if bk >= 0:
            g[name] = bk
    m2["gauge_index"] = g
    log(f"1K-RRI: {cmask.sum()} cells, mountain {m2['upscale']['mountain_share']:.0%}, kinematic links {kin.sum()}, "
        f"non-adjacent relay paths {n_far}, non-channel representative cells {n_nonch}")
    return dk, m2


def _break_cycles(dn, area, log):
    n = len(dn)
    state = np.zeros(n, np.int8)
    fixed = 0
    for s in range(n):
        path = []
        k = s
        while k >= 0 and state[k] == 0:
            state[k] = 1; path.append(k); k = dn[k]
        if k >= 0 and state[k] == 1:                    # cycle: cut at its smallest-area... largest-area member drains out
            cyc = path[path.index(k):]
            m = max(cyc, key=lambda q: area[q])
            dn[m] = -1
            fixed += 1
        for q in path:
            state[q] = 2
    if fixed:
        log(f"1K-RRI: {fixed} flow-direction cycles cut")
    return dn


# ------------------------------------------------------------------ kernels
@njit(cache=True)
def _mountain(hs, mask, As, ls, i_s, n, ka, da, dm, beta, kcell, hr, width, length, dt):
    """Eq. (1): outflow q_s l_s into the cell's channel.  Returns (volume to rivers, max c / l)."""
    ny, nx = hs.shape
    vol = 0.0
    rate = 0.0
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or As[i, j] <= 0.0 or hs[i, j] <= 0.0:
                continue
            q = K.slope_discharge(hs[i, j], i_s[i, j], n[i, j], ka[i, j], da[i, j], dm[i, j], beta[i, j])
            V = q * ls[i, j] * dt
            avail = hs[i, j] * As[i, j]
            if V > avail:
                V = avail
            if V <= 0.0:
                continue
            hs[i, j] -= V / As[i, j]
            k = kcell[i, j]
            hr[k] += V / (width[k] * length[k])
            vol += V
            # stability of the lumped reservoir: d(q l_s/A_s)/dh ~ (5/3) q / h l_s / A_s
            r = 5.0 / 3.0 * q / max(hs[i, j] + V / As[i, j], 1e-9) * ls[i, j] / As[i, j]
            if r > rate:
                rate = r
    return vol, rate


@njit(cache=True)
def _plain_to_river(h, mask, Af, lf, i_f, n, ka, da, dm, beta, sd, kcell, hr, width, length, dt):
    """Eq. (3) term -2 q_f l_f: kinematic wave of the plain into the channel (any river state)."""
    ny, nx = h.shape
    vol = 0.0
    rate = 0.0
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or Af[i, j] <= 0.0:
                continue
            hsurf = h[i, j] - da[i, j] - sd[i, j]
            qsub = K.subsurface_discharge(h[i, j], i_f[i, j], ka[i, j], da[i, j], dm[i, j], beta[i, j])
            q = qsub
            if hsurf > 0.0:
                q += hsurf ** (5.0 / 3.0) * math.sqrt(i_f[i, j]) / n[i, j]
            if q <= 0.0:
                continue
            V = 2.0 * q * lf[i, j] * dt
            free = max(hsurf, 0.0) + (min(h[i, j], da[i, j]) if qsub > 0.0 else 0.0)
            avail = free * Af[i, j]
            if V > avail:
                V = avail
            if V <= 0.0:
                continue
            h[i, j] -= V / Af[i, j]
            k = kcell[i, j]
            hr[k] += V / (width[k] * length[k])
            vol += V
            if hsurf > 0.0:
                r = 5.0 / 3.0 * q / max(hsurf, 1e-9) * 2.0 * lf[i, j] / Af[i, j]
                if r > rate:
                    rate = r
    return vol, rate


@njit(cache=True)
def _plain_xy(h, z, mask, Af, n, da, sd, dx, dy, dt, lim_frac, dV):
    """Eq. (3) term Q_xy: diffusive wave between plains in 8 directions (cell = grid cell of depth h_f at z_f).
    Fluxes limited so that a face never more than lim_frac x equalises the two levels; positivity per donor."""
    ny, nx = h.shape
    DI = (0, 1, 1, 1)
    DJ = (1, 1, 0, -1)
    for i in range(ny):
        for j in range(nx):
            dV[i, j] = 0.0
    out = np.zeros((ny, nx))
    F = np.zeros((4, ny, nx))
    diag = math.sqrt(dx * dx + dy * dy)
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or Af[i, j] <= 0.0:
                continue
            for d in range(4):
                a, b = i + DI[d], j + DJ[d]
                if not (0 <= a < ny and 0 <= b < nx) or not mask[a, b] or Af[a, b] <= 0.0:
                    continue
                ha = max(h[i, j] - da[i, j], 0.0)
                hb = max(h[a, b] - da[a, b], 0.0)
                Ha, Hb = z[i, j] + ha, z[a, b] + hb
                dH = Ha - Hb
                if dH == 0.0:
                    continue
                if DI[d] != 0 and DJ[d] != 0:
                    dist, w = diag, dx * dy / diag
                elif DI[d] == 0:
                    dist, w = dx, dy
                else:
                    dist, w = dy, dx
                if dH > 0.0:
                    hd, nd, sdd = ha, n[i, j], sd[i, j]
                else:
                    hd, nd, sdd = hb, n[a, b], sd[a, b]
                hf = max(Ha, Hb) - max(z[i, j], z[a, b])
                if hf > hd - sdd:
                    hf = hd - sdd
                if hf <= 1e-6:
                    continue
                q = hf ** (5.0 / 3.0) * math.sqrt(abs(dH) / dist) / nd * w
                veq = abs(dH) * Af[i, j] * Af[a, b] / (Af[i, j] + Af[a, b])
                qmax = lim_frac * veq / dt
                if q > qmax:
                    q = qmax
                if dH < 0.0:
                    q = -q
                F[d, i, j] = q
                if q > 0.0:
                    out[i, j] += q * dt
                else:
                    out[a, b] -= q * dt
    r = np.ones((ny, nx))
    for i in range(ny):
        for j in range(nx):
            if out[i, j] > 0.0:
                avail = max(h[i, j] - da[i, j] - sd[i, j], 0.0) * Af[i, j]
                if out[i, j] > avail:
                    r[i, j] = avail / out[i, j]
    for i in range(ny):
        for j in range(nx):
            for d in range(4):
                q = F[d, i, j]
                if q == 0.0:
                    continue
                a, b = i + DI[d], j + DJ[d]
                q *= r[i, j] if q > 0.0 else r[a, b]
                dV[i, j] -= q * dt
                dV[a, b] += q * dt
    for i in range(ny):
        for j in range(nx):
            if dV[i, j] != 0.0:
                h[i, j] = max(h[i, j] + dV[i, j] / Af[i, j], 0.0)


@njit(cache=True)
def _river_to_plain(h, z, mask, Af, lf, da, ri, rj, hr, zb, width, length, embank, dt, mu1, mu2):
    """Eq. (3) term +2 q_rf l_f: overflow from the channel onto the plain when the river level exceeds the plain water level
    and the levee crest (本間).  Capped at the level that equalises channel and plain."""
    vol = 0.0
    for k in range(hr.shape[0]):
        i, j = ri[k], rj[k]
        if Af[i, j] <= 0.0 or lf[i, j] <= 0.0:
            continue
        Hr = zb[k] + hr[k]
        Hs = z[i, j] + max(h[i, j] - da[i, j], 0.0)
        zc = z[i, j] + embank[k]
        if Hr <= Hs or Hr <= zc:
            continue
        h1 = Hr - zc
        h2 = max(Hs - zc, 0.0)
        V = K.honda_unit(h1, h2, mu1, mu2) * 2.0 * lf[i, j] * dt
        Ar = width[k] * length[k]
        Vmax = min(h1, hr[k]) * Ar
        Vmax = min(Vmax, (Hr - Hs) * Af[i, j] * Ar / (Af[i, j] + Ar))
        if V > Vmax:
            V = Vmax
        if V <= 0.0:
            continue
        hr[k] -= V / Ar
        h[i, j] += V / Af[i, j]
        vol += V
    return vol


@njit(cache=True)
def _kinematic_links(Q, hr, zb, width, length, nr, dn, kin, h_dry):
    """Paper §2(2)c: kinematic wave (normal depth with the bed slope, no backwater) from a small tributary channel into a
    larger one."""
    cmax = 0.0
    for k in range(hr.shape[0]):
        if not kin[k]:
            continue
        d = dn[k]
        h = hr[k]
        if h <= h_dry:
            Q[k] = 0.0
            continue
        s0 = max((zb[k] - zb[d]) / (0.5 * (length[k] + length[d])), 1e-5)
        w = width[k]
        A = w * h
        R = A / (w + 2.0 * h)
        q = A * R ** (2.0 / 3.0) * math.sqrt(s0) / nr[k]
        Q[k] = q
        u = q / A
        cmax = max(cmax, 5.0 / 3.0 * u)
    return cmax


@njit(cache=True)
def _drain_own(h, mask, Af, da, sd, rate, kcell, hr, width, length, dt):
    """Lowland drainage (pumps) of the plain into the cell's own channel at most `rate` [m/s]."""
    ny, nx = h.shape
    tot = 0.0
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or rate[i, j] <= 0.0 or Af[i, j] <= 0.0:
                continue
            hs = h[i, j] - da[i, j] - sd[i, j]
            if hs <= 0.0:
                continue
            dh = min(hs, rate[i, j] * dt)
            h[i, j] -= dh
            k = kcell[i, j]
            hr[k] += dh * Af[i, j] / (width[k] * length[k])
            tot += dh * Af[i, j]
    return tot


@njit(cache=True)
def _release_own(h, mask, Af, da, sd, leak, kcell, hr, width, length, dt):
    """Slow release of the water held below the store depth (田んぼダム, storage) into the cell's own channel
    (linear reservoir, rate `leak`)."""
    ny, nx = h.shape
    tot = 0.0
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or leak[i, j] <= 0.0 or sd[i, j] <= 0.0 or Af[i, j] <= 0.0:
                continue
            hs = h[i, j] - da[i, j]
            if hs <= 0.0:
                continue
            ret = min(hs, sd[i, j])
            dh = ret * (1.0 - math.exp(-leak[i, j] * dt))
            if dh <= 0.0:
                continue
            h[i, j] -= dh
            k = kcell[i, j]
            hr[k] += dh * Af[i, j] / (width[k] * length[k])
            tot += dh * Af[i, j]
    return tot


@njit(cache=True)
def _coast(h, mask, Af, bnd, n, ka, da, dm, beta, l, dt):
    """Plain outflow to the sea through the open boundary (normal depth with a gentle slope; the earlier RRI setting)."""
    ny, nx = h.shape
    tot = 0.0
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or bnd[i, j] <= 0.0 or Af[i, j] <= 0.0 or h[i, j] <= 0.0:
                continue
            q = K.slope_discharge(h[i, j], bnd[i, j], n[i, j], ka[i, j], da[i, j], dm[i, j], beta[i, j])
            V = min(q * l * dt, max(h[i, j] - da[i, j], 0.0) * Af[i, j])
            if V > 0.0:
                h[i, j] -= V / Af[i, j]
                tot += V
    return tot


# ------------------------------------------------------------------ simulation
class Simulation1K:
    """1K-RRI with the interface of solver.Simulation used by the GUI engine: h (plain water depth h_f), da / sd (plain soil
    and store depths), hr, Q, F, dam_S, get_state / set_state, run() -> solver.Results (max_depth = plain surface depth)."""
    STATE_FIELDS = ("h", "hs", "hr", "F", "Q", "dam_S")

    def __init__(self, domain: Domain1K, rainfall=None, inflows=None, h0=None, hr0=None, dams=None, stage_bc=None,
                 drainage=None):
        from .solver import Simulation
        self.d = d = domain
        self.rain = rainfall
        self.inflows = inflows or {}
        self.stage_bc = stage_bc or {}
        self.drain_rate = None if drainage is None else drainage[0]
        nr = d.river.n
        self.bc_stage = np.full(nr, np.nan)
        self.dams = list(dams or [])
        self.dam_S = np.array([dm.storage0 for dm in self.dams], float)
        self.dam_qin = np.zeros(len(self.dams))
        self.dam_qout = np.zeros(len(self.dams))
        self._build_param_arrays()
        self.h = np.zeros(d.shape) if h0 is None else np.array(h0, float)
        self.hs = np.zeros(d.shape)
        self.hr = np.zeros(nr) if hr0 is None else np.array(hr0, float)
        self.F = np.zeros(d.shape)
        self.Q = np.zeros(nr)
        self.t = 0.0
        self.dV = np.zeros(d.shape)
        self.bnd = np.zeros(d.shape) if d.open_boundary is None else np.asarray(d.open_boundary, float)
        self.ls, self.lf = d.lengths()
        # lowland drainage pumps lift the plain water into the main river: the first channel downstream of the cell whose
        # catchment exceeds KIN_AREA_KM2 (assumption; the cell's own channel when it is that large)
        r = d.river
        tgt = np.arange(nr)
        for k in range(nr):
            q = k
            while r.area_km2[q] <= KIN_AREA_KM2 and r.dn[q] >= 0:
                q = r.dn[q]
            tgt[k] = q
        self.kdrain = np.where(d.kcell >= 0, tgt[np.maximum(d.kcell, 0)], -1)
        # reuse the dam / retention operation of the 2D solver (same river arrays)
        self._ops = Simulation.__new__(Simulation)
        self._ops.d, self._ops.dams = d, self.dams

    def _build_param_arrays(self):
        d = self.d
        cl = d.params.land_classes

        def arr(lc, attr):
            vals = np.array([getattr(c, attr) for c in cl], dtype=float)
            return vals[np.clip(lc, 0, len(cl) - 1)]
        f, s = d.lc_f, d.lc_s
        self.n, self.ka, self.da, self.dm, self.beta = (arr(f, a) for a in ("n", "ka", "d_a", "d_m", "beta"))
        self.kv, self.phi_theta, self.sf, self.inf_limit = (arr(f, a) for a in ("kv", "phi_minus_theta", "sf", "infilt_limit"))
        self.sd, self.leak = arr(f, "store_depth"), arr(f, "leak_rate")
        self.n_s, self.ka_s, self.da_s, self.dm_s, self.beta_s = (arr(s, a) for a in ("n", "ka", "d_a", "d_m", "beta"))
        self.z = np.where(d.mask, d.z, 0.0)
        self.mask = d.mask.astype(np.bool_)
        self.has_retention = bool((self.leak > 0).any() and (self.sd > 0).any())

    def get_state(self):
        return {"t": self.t, **{k: getattr(self, k).copy() for k in self.STATE_FIELDS}}

    def set_state(self, st, reset_time=True):
        for k in self.STATE_FIELDS:
            if k in st and getattr(self, k).shape == st[k].shape:
                setattr(self, k, st[k].copy())
        self.t = 0.0 if reset_time else st["t"]

    def storage(self):
        d, r = self.d, self.d.river
        v = float((self.h * d.Af)[d.mask].sum() + (self.hs * d.As)[d.mask].sum())
        v += float((self.hr * r.width * r.length).sum())
        return v + float(self.dam_S.sum())

    def surface_depth(self):
        return np.where(self.mask, np.maximum(self.h - self.da, 0.0), np.nan)

    def step(self, dt, mb):
        d, num, r = self.d, self.d.params.numerics, self.d.river
        if self.rain is not None:
            depth = np.where(self.mask, self.rain.depth(self.t, self.t + dt), 0.0)
            self.h += np.where(d.Af > 0, depth, 0.0)
            self.hs += np.where(d.As > 0, depth, 0.0)
            mb.rain += float((depth * (d.Af + d.As)).sum())
        h0 = self.h.copy()
        K.green_ampt(self.h, self.F, self.mask & (d.Af > 0), self.da, self.kv, self.phi_theta, self.sf, self.inf_limit, dt)
        mb.infiltration += float(((h0 - self.h) * d.Af).sum())
        # mountain and plain into the channels, flood spreading between plains
        _, rate_s = _mountain(self.hs, self.mask, d.As, self.ls, d.i_s, self.n_s, self.ka_s, self.da_s, self.dm_s, self.beta_s,
                              d.kcell, self.hr, r.width, r.length, dt)
        _, rate_f = _plain_to_river(self.h, self.mask, d.Af, self.lf, d.i_f, self.n, self.ka, self.da, self.dm, self.beta, self.sd,
                                    d.kcell, self.hr, r.width, r.length, dt)
        _plain_xy(self.h, self.z, self.mask, d.Af, self.n, self.da, self.sd, d.dx, d.dy, dt, num.limiter_frac, self.dV)
        if self.bnd.any():
            mb.boundary_outflow += _coast(self.h, self.mask, d.Af, self.bnd, self.n, self.ka, self.da, self.dm, self.beta, d.l, dt)
        if self.drain_rate is not None:
            _drain_own(self.h, self.mask, d.Af, self.da, self.sd, self.drain_rate, self.kdrain, self.hr, r.width, r.length, dt)
        if self.has_retention:
            _release_own(self.h, self.mask, d.Af, self.da, self.sd, self.leak, d.kcell, self.hr, r.width, r.length, dt)
        # river
        for k, hyd in self.inflows.items():
            v = hyd.volume(self.t, self.t + dt)
            self.hr[k] += v / (r.width[k] * r.length[k])
            mb.inflow += v
        for k, (tt, hh) in self.stage_bc.items():
            self.bc_stage[k] = np.interp(self.t + 0.5 * dt, tt, hh)
        inertial = num.scheme == "inertial"
        c_r, dts_r = K.river_fluxes(self.hr, r.zb, r.width, r.length, r.n_river, r.dn, r.outlet_slope, dt, inertial,
                                    num.limiter_frac, num.h_dry, self.Q, self.bc_stage)
        c_k = _kinematic_links(self.Q, self.hr, r.zb, r.width, r.length, r.n_river, r.dn, d.kinematic, num.h_dry)
        ops = self._ops
        ops.hr, ops.Q, ops.dam_S, ops.dam_qin, ops.dam_qout = self.hr, self.Q, self.dam_S, self.dam_qin, self.dam_qout
        for n_, dm in enumerate(self.dams):
            if dm.release_q is None:
                self.Q[dm.k] = 0.0
            else:
                ops._retention_divert(n_, dm, dt)
        mb.river_outflow += K.apply_river_fluxes(self.hr, r.width, r.length, r.dn, self.Q, dt)
        if self.dams:
            mb.river_outflow += ops._operate_dams(dt)
        _river_to_plain(self.h, self.z, self.mask, d.Af, self.lf, self.da, r.ri, r.rj, self.hr, r.zb, r.width, r.length,
                        r.embankment, dt, num.mu_complete, num.mu_submerged)
        self.t += dt
        dt_new = num.dt_max
        c = max(c_r, c_k)
        if c > 0:
            dt_new = min(dt_new, num.cfl * float(r.length.min()) / c)
        rate = max(rate_s, rate_f)
        self._limits = (c_r, c_k, rate_s, rate_f)
        if rate > 0:
            dt_new = min(dt_new, 0.5 / rate)
        if not inertial:
            dt_new = min(dt_new, num.diff_safety * dts_r)
        return max(dt_new, num.dt_min)

    def run(self, t_end, gauges=None, output_interval=600.0, snapshot_interval=None, dt0=1.0, verbose=False, progress=None,
            snapshot_fn=None):
        gauges = gauges or {}
        d = self.d
        mb = MassBalance(storage0=self.storage())
        times = [self.t]
        gq = {g: [self.Q[k] if k >= 0 else 0.0] for g, k in gauges.items()}
        gh = {g: [self.hr[k]] for g, k in gauges.items()}
        maxd = np.zeros(d.shape); tmax = np.zeros(d.shape)
        nr = d.river.n
        maxhr, maxq = np.zeros(nr), np.zeros(nr)
        outlets = np.nonzero(d.river.dn < 0)[0]
        outq, snaps = [0.0], []
        next_out = self.t + output_interval
        next_snap = self.t + snapshot_interval if snapshot_interval else np.inf
        dt, nsteps, dtmin_used = dt0, 0, np.inf
        wall0 = _time.perf_counter()
        while self.t < t_end - 1e-9:
            dt_step = max(min(dt, t_end - self.t, next_out - self.t, next_snap - self.t), 1e-6)
            dt = self.step(dt_step, mb)
            dtmin_used = min(dtmin_used, dt_step)
            nsteps += 1
            sd_ = np.maximum(self.h - self.da, 0.0)
            upd = sd_ > maxd
            maxd[upd] = sd_[upd]; tmax[upd] = self.t
            np.maximum(maxhr, self.hr, out=maxhr)
            np.maximum(maxq, np.abs(self.Q), out=maxq)
            if self.t >= next_out - 1e-9:
                times.append(self.t)
                for g, k in gauges.items():
                    gq[g].append(self.Q[k]); gh[g].append(self.hr[k])
                outq.append(float(self.Q[outlets].sum()))
                next_out += output_interval
                if progress is not None:
                    progress(min(self.t / t_end, 1.0))
                if verbose:
                    print(f"t={self.t / 3600:8.2f} h  dt={dt_step:7.3f}s  steps={nsteps}  Qout={outq[-1]:.2f}")
            if self.t >= next_snap - 1e-9:
                snaps.append((self.t, self.surface_depth(), self.hr.copy()) if snapshot_fn is None else (self.t, snapshot_fn(self)))
                next_snap += snapshot_interval
        mb.storage = self.storage()
        return Results(times=np.array(times), gauge_q={g: np.array(v) for g, v in gq.items()},
                       gauge_h={g: np.array(v) for g, v in gh.items()}, max_depth=np.where(d.mask, maxd, np.nan),
                       time_of_max=tmax, max_river_h=maxhr, max_river_q=maxq, snapshots=snaps, mass=mb, n_steps=nsteps,
                       wall_time=_time.perf_counter() - wall0, outlet_q=np.array(outq), dt_min_used=dtmin_used)


def is_1k(domain) -> bool:
    return isinstance(domain, Domain1K)


def make_simulation(domain, *a, **kw):
    """solver.Simulation or Simulation1K for the domain."""
    if is_1k(domain):
        return Simulation1K(domain, *a, **kw)
    from .solver import Simulation
    return Simulation(domain, *a, **kw)
