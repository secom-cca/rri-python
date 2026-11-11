"""Candidate measures from site-suitability conditions -> integer decision variables.

Each Variable is one gene: an index into `options`; option 0 is always "do
nothing".  `edits(x, scen)` turns a decision vector into GUI-engine edits
(the same measures as the GUI: levee / retention / paddy_dam / forest /
forest_mgmt / relocate / raise / pump / storage), so the RRI model is used unchanged.

    levee      per zone: existing levee + 0/1/2 m on the river cells with upstream area >= min_area_km2;
               select "overflow": only those whose peak water level in the baseline reaches the crest
               (ground + existing levee) within margin_m (where the river actually overtops)
    retention  n sites: farmland/forest next to a large river, low HAND, not built-up; or `sites` (a planned
               project, e.g. 阿武隈川上流遊水地群): one variable per site, the n = round(area / cell) farmland/forest
               cells next to a river >= min_river_km2 within radius_km of lonlat with the lowest HAND (HAND limit
               not applied); the site's stated area and volume set the capacity and cost (engine area_m2 / volume_m3)
    paddy_dam  per zone: all paddies
    forest     per zone: farmland with HAND >= hand_min_m (upland) -> new forest (森林化)
    forest_mgmt_national / forest_mgmt
               per zone: existing forest -> 森林整備 (more soil storage), split into 国有林 (KSJ A45,
               actor 国) and the other forest (民有林等) when `national_forest` is given
    relocate   per zone: homes flooded >= depth_min_m in the baseline: none / relocate
    raise      per zone: homes flooded >= raise.depth_min_m: none / + heights_m (床高の嵩上げ);
               listed after relocate, so homes moved by an active relocation are not raised (nor paid)
               candidates.housing "combined" (configs without the key, e.g. stored explorations used by
               the roadmap / DAPP): one variable `relocate` with options none / raise / relocate
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


def _planned_sites(d, r, rt, lc, ci, is_riv, h, zone_of, basin=None):
    """Retention basins of a planned project (`retention.sites` of this basin), one variable each; [] when none can be
    placed (other basin, a domain without coordinates or without suitable cells: the automatic sites are used)."""
    from pyproj import Transformer
    sites = [st for st in rt["sites"] if st.get("basin") in (None, basin)]
    if not sites or not getattr(d, "crs", None) or getattr(d, "transform", None) is None:
        return []
    ny, nx = d.shape
    cell = float(d.dx) ** 2
    farm = np.isin(lc, [ci["paddy"], ci["field"], ci["forest"]]) & d.mask.ravel() & ~is_riv
    near = np.zeros(ny * nx, bool)                    # next to (8 neighbours) a river >= min_river_km2
    for k in np.flatnonzero(r.area_km2 >= rt.get("min_river_km2", 300.0)):
        i0, j0 = r.ri[k], r.rj[k]
        near.reshape(ny, nx)[max(i0 - 1, 0):i0 + 2, max(j0 - 1, 0):j0 + 2] = True
    ii, jj = np.divmod(np.arange(ny * nx), nx)
    try:
        tr = Transformer.from_crs("EPSG:4326", d.crs, always_xy=True)
    except Exception:                                  # synthetic domains (tests)
        return []
    out = []
    for st in sites:
        x, y = tr.transform(*st["lonlat"])
        col, row = (x - d.transform.c) / d.transform.a - 0.5, (y - d.transform.f) / d.transform.e - 0.5
        dist = np.hypot(ii - row, jj - col) * d.dx / 1000.0
        cand = np.flatnonzero(farm & near & (dist <= st.get("radius_km", 2.5)))
        n = max(1, int(round(st["area_ha"] * 1e4 / cell)))
        cells = cand[np.lexsort((dist[cand], h[cand]))][:n].tolist()      # lowest HAND, then nearest
        if not cells:
            continue
        z = str(zone_of[cells[0]])
        area = float(st["area_ha"]) * 1e4
        p = {"depth": float(st["volume_m3"]) / area, "area_m2": area, "volume_m3": float(st["volume_m3"]),
             "auto": True, "q_ratio": 0.7}
        out.append(Variable(f"遊水地・{z}", "retention", z,
                            ["なし", f"{st.get('label', '遊水地')}（{st['area_ha']:.0f} ha, {st['volume_m3'] / 1e4:.0f} 万m³）"],
                            cells, [None, p]))
    return out


def build(d, meta, hand, assets, class_names, baseline_depth, cfg, baseline_river_hmax=None,
          national_forest=None) -> list[Variable]:
    c = cfg["candidates"]
    ny, nx = d.shape
    zm = zone_masks(d, meta, cfg["zones"])
    zone_of = np.full(ny * nx, "", object)
    for z, m in zm.items():
        zone_of[m.ravel()] = z
    from ..k1rri import is_1k
    k1 = is_1k(d)
    # 1K-RRI: measures on the plain select by the plain's land use; every cell holds a channel (not excluded from sites)
    lc = (d.lc_f if k1 else d.landclass).ravel()
    ci = {n: i for i, n in enumerate(class_names)}
    h = np.nan_to_num(hand, nan=1e9).ravel()
    r = d.river
    riv_flat = r.ri * nx + r.rj
    is_riv = np.zeros(ny * nx, bool)
    if not k1:
        is_riv[riv_flat] = True
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
    planned = _planned_sites(d, r, rt, lc, ci, is_riv, h, zone_of, cfg.get("basin")) if rt and rt.get("sites") else []
    out += planned
    if rt is not None and not planned and rt.get("n_sites"):
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
        fm = c.get("forest_mgmt")
        if fm is not None:
            fcells = mz & (((d.As.ravel() > 0) & (d.lc_s.ravel() == ci["forest"])) if k1 else (lc == ci["forest"])) \
                & (h >= fm.get("hand_min_m", 0.0))
            nat = np.zeros_like(fcells) if national_forest is None or not fm.get("split_owner", True) \
                else np.asarray(national_forest, bool).ravel()
            for typ, lab, sel, owner in (("forest_mgmt_national", "森林整備（国有林）", fcells & nat, "national"),
                                         ("forest_mgmt", "森林整備（民有林等）" if nat.any() else "森林整備", fcells & ~nat, "other")):
                cells = np.flatnonzero(sel)
                if len(cells) >= fm.get("min_cells", 1):
                    out.append(Variable(f"{lab}・{z}", typ, z, ["なし", "全域"], cells.tolist(),
                                        [None, {"type": "forest_mgmt", "owner": owner}]))
        rl = c.get("relocate")
        if baseline_depth is not None:
            home = (assets.households.ravel() >= 0.5)
            bd = np.nan_to_num(baseline_depth).ravel()
            if rl is not None and c.get("housing", "combined") == "combined":
                cells = np.flatnonzero(mz & home & (bd >= rl["depth_min_m"]))
                if len(cells):
                    out.append(Variable(f"住宅対策・{z}", "relocate", z, ["なし", f"嵩上げ +{rl['raise_m']} m", "移転"], cells.tolist(),
                                        [None, {"type": "raise", "height": float(rl["raise_m"]), "terrain": False},
                                         {"type": "relocate", "dest": rl["dest"], "density": 40}]))
            else:
                if rl is not None:
                    cells = np.flatnonzero(mz & home & (bd >= rl["depth_min_m"]))
                    if len(cells):
                        out.append(Variable(f"移転・{z}", "relocate", z, ["なし", "移転"], cells.tolist(),
                                            [None, {"type": "relocate", "dest": rl["dest"], "density": 40}]))
                ra = c.get("raise")
                if ra is not None:
                    cells = np.flatnonzero(mz & home & (bd >= ra["depth_min_m"]))
                    hs = [float(x) for x in ra["heights_m"]]
                    if len(cells):
                        out.append(Variable(f"嵩上げ・{z}", "raise", z, ["なし"] + [f"+{x:g} m" for x in hs], cells.tolist(),
                                            [None] + [{"type": "raise", "height": x, "terrain": False} for x in hs]))
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


def superseded_raise(variables):
    """[(i, j)]: relocation variable i and raise variable j of the same zone whose homes are all
    relocated when i is active (raise cells within the relocation cells: same or higher depth
    threshold, i.e. no more cells).  With i active, j has no effect and no cost (the engine applies
    the relocation first); additive cost tables / surrogate features use this to stay consistent.
    Accepts Variables or the variable dicts stored in results.json."""
    g = lambda v, k: v[k] if isinstance(v, dict) else getattr(v, k)  # noqa: E731
    n = lambda v: v["n_cells"] if isinstance(v, dict) else len(v.cells)  # noqa: E731
    rel = {g(v, "zone"): i for i, v in enumerate(variables) if g(v, "type") == "relocate" and len(g(v, "options")) == 2}
    return [(rel[g(v, "zone")], j) for j, v in enumerate(variables)
            if g(v, "type") == "raise" and g(v, "zone") in rel and n(v) <= n(variables[rel[g(v, "zone")]])]


def edits(variables, x, scen=None):
    """Decision vector -> GUI edits.  scen may scale measure parameters:
    levee_effect x levee raise, retention_effect x retention depth (and volume of a planned site), raise_effect x house raise."""
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
            if p.get("volume_m3"):              # a planned site: the stated volume x the effect coefficient
                e["volume_m3"] = p["volume_m3"] * scen.get("retention_effect", 1.0)
        if v.type == "levee":
            e["height"] = p["height"] * scen.get("levee_effect", 1.0)
        if e["type"] == "raise":
            e["height"] = p["height"] * scen.get("raise_effect", 1.0)
        out.append(e)
    return out


def describe(variables, x):
    return [f"{v.name}: {float(xi):.0%}" if v.continuous else f"{v.name}: {v.options[int(xi)]}"
            for v, xi in zip(variables, x) if v.active(xi)]
