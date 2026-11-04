"""Candidate measures from site-suitability conditions -> integer decision variables.

Each Variable is one gene: an index into `options`; option 0 is always "do
nothing".  `edits(x, scen)` turns a decision vector into GUI-engine edits
(the same measures as the GUI: levee / retention / paddy_dam / forest /
relocate / raise), so the RRI model is used unchanged.

    levee      per zone: existing levee + 0/1/2 m on the river cells with upstream area >= min_area_km2;
               select "overflow": only those whose peak water level in the baseline reaches the crest
               (ground + existing levee) within margin_m (where the river actually overtops)
    retention  n sites: farmland/forest next to a large river, low HAND, not built-up
    paddy_dam  per zone: all paddies
    forest     per zone: farmland with HAND >= hand_min_m (upland)
    relocate   per zone: homes flooded >= depth_min_m in the baseline: none / raise / relocate
    pump_urban      per zone: 排水ポンプ（下水道）on low (HAND <= hand_max_m) built-up cells flooded >= depth_min_m
    pump_agri       per zone: 排水機場（農業用排水）on low farmland cells flooded >= depth_min_m
    storage_public  per zone: 雨水貯留浸透（公共: 調整池・貯留管）on built-up cells flooded >= depth_min_m
    storage_private per zone: 雨水貯留浸透（各戸: 民間・個人）on the same cells (implementation rate in the exploration)

A variable with `continuous=True` (paddy_dam / forest in the exploration, explore.py)
takes an implementation rate in [0, 1] instead of an option index: the first
round(rate x n) of its cells (in the order of `cells`) get option 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..basin import upstream_mask


@dataclass
class Variable:
    name: str
    type: str
    zone: str
    options: list            # labels; option 0 = none
    cells: list              # flat cell indices of the candidate site / area
    params: list = field(default_factory=list)   # per option: dict of edit parameters (None = no edit)
    continuous: bool = False                      # gene = implementation rate in [0, 1] (option 1 on a share of cells)

    def to_dict(self):
        d = asdict(self)
        if not self.continuous:                   # unchanged dict (and cache keys) for integer variables
            d.pop("continuous")
        return d

    def active(self, xi):
        """Does gene value xi apply an edit?"""
        if self.continuous:
            return self.n_cells(xi) > 0
        return self.params[int(xi)] is not None

    def n_cells(self, rate):
        return int(round(min(max(float(rate), 0.0), 1.0) * len(self.cells)))


def zone_masks(d, meta, zones):
    """{zone name: bool mask}; zones = [[name, gauge or None], ...] upstream first (nested catchments)."""
    out, taken = {}, np.zeros(d.shape, bool)
    r = d.river
    for name, g in zones:
        if g is None:
            m = d.mask & ~taken
        else:
            k = meta["gauge_index"][g]
            m = upstream_mask(meta["fdir"], int(r.ri[k]), int(r.rj[k])) & d.mask & ~taken
        out[name] = m
        taken |= m
    return out


def _grow_site(seed, ok, nx, n_cells):
    site, front = [seed], [seed]
    seen = {seed}
    while front and len(site) < n_cells:
        c = front.pop(0)
        for nb in (c - 1, c + 1, c - nx, c + nx):
            if 0 <= nb < ok.size and nb not in seen and ok[nb]:
                seen.add(nb); site.append(nb); front.append(nb)
                if len(site) >= n_cells:
                    break
    return site


def build(d, meta, hand, assets, class_names, baseline_depth, cfg, baseline_river_hmax=None) -> list[Variable]:
    c = cfg["candidates"]
    ny, nx = d.shape
    zm = zone_masks(d, meta, cfg["zones"])
    zone_of = np.full(ny * nx, "", object)
    for z, m in zm.items():
        zone_of[m.ravel()] = z
    lc = d.landclass.ravel()
    ci = {n: i for i, n in enumerate(class_names)}
    h = np.nan_to_num(hand, nan=1e9).ravel()
    r = d.river
    riv_flat = r.ri * nx + r.rj
    is_riv = np.zeros(ny * nx, bool); is_riv[riv_flat] = True
    out = []
    # levee: main rivers per zone
    lv = c.get("levee")
    if lv is not None:
        ok = r.area_km2 >= lv["min_area_km2"]
        if lv.get("select") == "overflow" and baseline_river_hmax is not None:
            crest = d.z[r.ri, r.rj] + r.embankment
            ok &= r.zb + np.asarray(baseline_river_hmax) >= crest - lv.get("margin_m", 0.2)
        for z in zm:
            ks = np.flatnonzero(ok & (zone_of[riv_flat] == z))
            if len(ks):
                hs = lv["heights_m"]
                out.append(Variable(f"堤防・{z}", "levee", z, [f"+{x} m" if x else "なし" for x in hs], riv_flat[ks].tolist(),
                                    [None if not x else {"height": float(x)} for x in hs]))
    # retention sites
    rt = c.get("retention")
    if rt is not None:
        farm = np.isin(lc, [ci["paddy"], ci["field"], ci["forest"]])
        ok = d.mask.ravel() & farm & ~is_riv & (h <= rt["hand_max_m"])
        big = r.area_km2 >= rt["min_river_km2"]
        near = np.zeros(ny * nx)                      # upstream area of an adjacent large river
        for k in np.flatnonzero(big):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    i, j = r.ri[k] + di, r.rj[k] + dj
                    if 0 <= i < ny and 0 <= j < nx:
                        near[i * nx + j] = max(near[i * nx + j], r.area_km2[k])
        ok &= near > 0
        n_cells = max(1, int(round(rt["site_km2"] * 1e6 / d.cell_area)))
        used = np.zeros(ny * nx, bool)
        for s in range(rt["n_sites"]):
            cand = np.flatnonzero(ok & ~used)
            if not len(cand):
                break
            seed = int(cand[np.argmax(near[cand])])  # most downstream first (largest river)
            site = _grow_site(seed, ok & ~used, nx, n_cells)
            i0, j0 = divmod(seed, nx)
            ii, jj = np.divmod(np.arange(ny * nx), nx)
            used |= (np.abs(ii - i0) <= 4) & (np.abs(jj - j0) <= 4)     # keep sites apart
            out.append(Variable(f"遊水地{s + 1}・{zone_of[seed]}", "retention", zone_of[seed], ["なし", f"貯留深 {rt['depth_m']} m"],
                                site, [None, {"depth": float(rt["depth_m"]), "auto": True, "q_ratio": 0.7}]))
    for z, m in zm.items():
        mz = m.ravel()
        if c.get("paddy_dam") is not None:
            cells = np.flatnonzero(mz & (lc == ci["paddy"]))
            if len(cells):
                out.append(Variable(f"田んぼダム・{z}", "paddy_dam", z, ["なし", "全水田"], cells.tolist(), [None, {}]))
        fo = c.get("forest")
        if fo is not None:
            cells = np.flatnonzero(mz & np.isin(lc, [ci["paddy"], ci["field"]]) & (h >= fo["hand_min_m"]) & (h < 1e8))
            if len(cells):
                out.append(Variable(f"森林化・{z}", "forest", z, ["なし", f"HAND {fo['hand_min_m']:.0f} m以上の農地"], cells.tolist(), [None, {}]))
        rl = c.get("relocate")
        if rl is not None and baseline_depth is not None:
            home = (assets.households.ravel() >= 0.5)
            cells = np.flatnonzero(mz & home & (np.nan_to_num(baseline_depth).ravel() >= rl["depth_min_m"]))
            if len(cells):
                out.append(Variable(f"住宅対策・{z}", "relocate", z, ["なし", f"嵩上げ +{rl['raise_m']} m", "移転"], cells.tolist(),
                                    [None, {"type": "raise", "height": float(rl["raise_m"]), "terrain": False},
                                     {"type": "relocate", "dest": rl["dest"], "density": 40}]))
    # 内水対策 (inland flooding): where the baseline floods (depth >= depth_min_m)
    inl = {k: c[k] for k in ("pump_urban", "pump_agri", "storage_public", "storage_private") if c.get(k) is not None}
    if inl and baseline_depth is not None:
        bd = np.nan_to_num(baseline_depth).ravel()
        urban = lc == ci["urban"]
        farm = np.isin(lc, [ci[n] for n in ("paddy", "field") if n in ci])
        for z, m in zm.items():
            mz = m.ravel()
            for k, spec in inl.items():
                wet = bd >= spec.get("depth_min_m", 0.1)
                low = h <= spec.get("hand_max_m", 5.0)
                if k == "pump_urban":
                    cells, lab, par = mz & low & urban & wet, f"+{spec['mmh']:.0f} mm/h", {"type": "pump", "target": "urban", "mmh": float(spec["mmh"])}
                    name = f"排水ポンプ（下水道）・{z}"
                elif k == "pump_agri":
                    cells, lab, par = mz & low & farm & wet, f"+{spec['mmh']:.0f} mm/h", {"type": "pump", "target": "agri", "mmh": float(spec["mmh"])}
                    name = f"排水機場（農業）・{z}"
                else:
                    kind = "public" if k == "storage_public" else "private"
                    cells, lab = mz & urban & wet, f"{spec['depth_mm']:.0f} mm"
                    par = {"type": "storage", "depth_mm": float(spec["depth_mm"]), "release_mmh": float(spec.get("release_mmh", 2.0)), "kind": kind}
                    name = f"雨水貯留（{'公共' if kind == 'public' else '各戸'}）・{z}"
                idx = np.flatnonzero(cells)
                if len(idx) >= spec.get("min_cells", 1):
                    out.append(Variable(name, k, z, ["なし", lab], idx.tolist(), [None, par]))
    return out


def edits(variables, x, scen=None):
    """Decision vector -> GUI edits.  scen may scale measure parameters:
    levee_effect x levee raise, retention_effect x retention depth, raise_effect x house raise."""
    scen = scen or {}
    out = []
    for v, xi in zip(variables, x):
        if not v.active(xi):
            continue
        p = v.params[1] if v.continuous else v.params[int(xi)]
        cells = v.cells[:v.n_cells(xi)] if v.continuous else v.cells
        e = {"type": v.type, "cells": cells, **p}
        if v.type == "retention":
            e["depth"] = p["depth"] * scen.get("retention_effect", 1.0)
        if v.type == "levee":
            e["height"] = p["height"] * scen.get("levee_effect", 1.0)
        if e["type"] == "raise":
            e["height"] = p["height"] * scen.get("raise_effect", 1.0)
        out.append(e)
    return out


def describe(variables, x):
    return [f"{v.name}: {float(xi):.0%}" if v.continuous else f"{v.name}: {v.options[int(xi)]}"
            for v, xi in zip(variables, x) if v.active(xi)]
