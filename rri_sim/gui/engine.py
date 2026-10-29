"""Simulation engine behind the 3D GUI.

A BasinSession holds one basin (domain with calibrated preset, assets,
population, facilities, HAND, dams, tide).  For each event the spin-up
(observed rain from `event.start` to `event.eval_start`) is computed once and
cached; scenario runs restart from that state and only simulate the flood
window.  Scenario = list of edits painted in the GUI:

  {"type": "forest",     "cells": [...]}                       森林化（土地利用→新規森林）
  {"type": "paddy_dam",  "cells": [...]}                       田んぼダム（水田セルのみ有効）
  {"type": "relocate",   "cells": [...]}                       住宅・事業所の移転（安全な市街地へ）
  {"type": "raise",      "cells": [...], "height": 1.0,
                         "terrain": false}                     嵩上げ（床高 / 盛土）
  {"type": "levee",      "cells": [...], "height": 1.0}        堤防（範囲内の河道セル, 既存堤防 + height）
  {"type": "retention",  "cells": [...], "depth": 2.0,
                         "inlet": 1.0}                         遊水地（掘削＋越流堤）
Global scenario parameters (gparams): paddy_store_m, paddy_release_mmh,
forest_n, forest_soil_m, unit_costs{...}.

Outputs per run: inundation frames, damage, hydrographs at all gauges, project
cost, Livability and Ecology scores (rri_sim.indicators).

Approximation (documented in the UI): land-use edits do not alter the
spin-up state, i.e. antecedent soil moisture is that of the baseline.
"""
from __future__ import annotations

import base64
import copy
import logging
import pickle
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .. import climate as CL
from .. import costs as C
from .. import damage as D
from .. import indicators as I
from .. import subgrid as SG
from ..assets import build_assets
from ..basin import floodplain_inundation, lonlat_to_rc, upstream_mask
from ..cases import BASINS, EVENTS
from ..data import estat, facilities, landuse, mlit_water
from ..events import collect_gauges, gridded_rainfall
from ..solver import Simulation
from ..workflow import (CACHE, FOREST_NEW, PADDY_DAM, PRESETS, boundary_conditions, build_preset, cache_key,
                        default_res, domain_cached,
                        drainage_arrays)

log = logging.getLogger("rri_sim.gui")
HAND_DISPLAY_MAX = 15.0       # m; cells that can show inundation in the GUI
HAND_DAMAGE_MAX = 10.0        # m; floodplain definition for inundation / damage statistics (docs/04)
DEPTH_MIN = 0.1               # m; inundation threshold for damage / extent
FLOOR_HEIGHT = 0.45           # m; manual default
SAFE_HAND_M = 10.0
# grid sizes offered in the GUI [m] and their standing (docs/08_resolution.md)
NEW_SITE_DENSITY = 40.0       # 人/ha; default density of new residential land for relocation (DID criterion)
NEW_SITE_MAX_SLOPE = 0.08     # ground slope limit for new residential land
RETENTION_Q_RATIO = 0.7       # default 越流開始流量 = 0.7 x baseline peak discharge at the inlet
RETENTION_RELEASE_RATIO = 0.5 # the basin drains back (排水門) only while the river is below 0.5 x 越流開始流量
RESOLUTIONS = {125.0: "詳細・低速", 250.0: "標準・校正済み", 500.0: "高速・比較向け", 1000.0: "概略・精度低"}
# bump when the asset / indicator-data derivation changes (v2: e-Stat meshes without
# a cell centre are no longer dropped, +8% households at 250 m)
ASSET_CACHE_VERSION = 2            # relocation destinations: built-up cells this high above drainage
ASSET_FILE_VERSION = 3             # assets only (v3: farm households 2020年農林業センサス)
SUBGRID_VERSION = 1                # sub-grid building / farmland elevations (rri_sim.subgrid)


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def event_label(ev):
    return f"{ev.name}（{pd.Timestamp(ev.eval_start):%Y/%m/%d}〜, {ev.role}）"


class BasinSession:
    def __init__(self, key: str, res: float | None = None, progress=None):
        say = progress or (lambda f, msg: None)
        self.key = key
        self.res = float(res) if res else default_res(key)
        self.ckey = cache_key(key, self.res)          # cache-file stem (basin + resolution)
        cached = domain_cached(key, self.res)
        say(0.05, "流域モデルを読み込み中" if cached else
            "流域モデルを構築中（初回: 標高タイル・土地利用・河川データの取得、流域界・河道網・既設堤防の推定。数分〜数十分）")
        self.d, self.meta, self.b = build_preset(key, use_cache=True, res=self.res)
        d, b = self.d, self.b
        pref = b.get("prefecture", "福岡")
        apath = CACHE / f"assets_{self.ckey}_v{ASSET_FILE_VERSION}.pkl"
        if apath.exists():
            self.assets = pickle.load(open(apath, "rb"))
        else:
            say(0.6, "資産データを作成中（初回: e-Stat 国勢調査・経済センサス・農林業センサスのメッシュ統計）")
            self.assets = build_assets(d, self.meta["landuse_fraction"], b["cfg"].bbox, pref)
            CACHE.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.assets, open(apath, "wb"))
        self._load_indicator_data(say)
        say(0.95, "表示データを準備中")
        self.gs = D.ground_slope_from_dem(d.z, d.dx, d.dy)
        self.hand = self.meta["hand"]
        spath = CACHE / f"subgrid_{self.ckey}_v{SUBGRID_VERSION}.pkl"
        if spath.exists():
            self.subgrid = pickle.load(open(spath, "rb"))
        else:
            say(0.7, "建物・農地の標高分布を作成中（初回: 5 m標高・100 m土地利用。数分）")
            self.subgrid = SG.elevation_offsets(d, b["cfg"].bbox, self.hand, HAND_DAMAGE_MAX,
                                                say=lambda f, m: say(0.7 + 0.2 * f, m))
            pickle.dump(self.subgrid, open(spath, "wb"))
        self.flood_cells = np.flatnonzero(d.mask & (np.nan_to_num(self.hand, nan=1e9) <= HAND_DISPLAY_MAX))
        self.class_names = [c.name for c in d.params.land_classes]
        # 居住誘導区域 proxy for 居住集約度: built-up land of the current land use (fixed across scenarios)
        self.zone = d.mask & (d.landclass == self.class_names.index("urban"))
        self.events = [e for e in EVENTS.values() if e.basin == key]
        self._prep, self._baseline = {}, {}

    def _load_indicator_data(self, say=lambda f, msg: None):
        d, b = self.d, self.b
        path = CACHE / f"indicator_data_{self.ckey}_v{ASSET_CACHE_VERSION}.pkl"
        if path.exists():
            self.road, self.pop, self.fac = pickle.load(open(path, "rb"))
            return
        say(0.75, "指標データを作成中（初回: 道路・人口メッシュ、医療・学校・公共施設の位置）")
        bbox = b["cfg"].bbox
        frac, extra = landuse.class_fractions(d.transform, d.shape, d.crs, *bbox,
                                              extra_codes={"road": ["0901"]})
        meshes = landuse.first_meshes(*bbox)
        w = frac[landuse.CLASSES.index("urban")] + 1e-3
        pop = estat.population(meshes, d.transform, d.shape, d.crs, w)
        prefs = b.get("prefs", ())
        fac = {k: facilities.to_cells(facilities.points(k, prefs), d.transform, d.shape, d.crs)
               for k in ("medical", "school", "public")}
        emp_i = self.assets.employees.get("I")
        fac["commercial"] = np.flatnonzero(np.nan_to_num(emp_i).ravel()
                                           >= I.AccessParams().commercial_min_employees)
        self.road, self.pop, self.fac = extra["road"], pop, fac
        pickle.dump((self.road, self.pop, self.fac), open(path, "wb"))

    # ----------------------------------------------------------- static data
    def static_payload(self) -> dict:
        d, r = self.d, self.d.river
        ny, nx = d.shape
        value = D.asset_value(self.assets)
        dams = {}
        for n, v in self.b.get("dams", {}).items():
            rr, cc, _, _ = lonlat_to_rc(*v["lonlat"], d.transform, d.crs)
            dams[n] = {"capacity": v["capacity"], "q_cut": v["q_cut"], "source": v["source"],
                       "cell": int(rr * nx + cc)}
        return {
            "key": self.key, "nx": nx, "ny": ny, "dx": d.dx, "dy": d.dy,
            "res": self.res, "default_res": default_res(self.key), "resolutions": [{"res": r, "label": lb} for r, lb in RESOLUTIONS.items()],
            "z": b64(np.where(d.mask, d.z, np.nan), "<f4"),
            "mask": b64(d.mask, "u1"), "landclass": b64(d.landclass, "u1"),
            "class_names": self.class_names, "hand": b64(np.nan_to_num(self.hand, nan=-1), "<f4"),
            "flood_cells": b64(self.flood_cells, "<i4"),
            "river": {"ri": r.ri.tolist(), "rj": r.rj.tolist(), "dn": r.dn.tolist(),
                      "width": r.width.round(1).tolist(), "depth": r.depth.round(2).tolist(),
                      "embankment": r.embankment.round(2).tolist(),
                      "area": r.area_km2.round(1).tolist(), "length": r.length.round(0).tolist()},
            "households": b64(self.assets.households, "<f4"),
            "population": b64(self.pop, "<f4"),
            "asset_value": b64(value, "<f4"),
            "asset_total_oku": float(value.sum() / 1e8),
            "households_total": float(self.assets.households.sum()),
            "population_total": float(self.pop[d.mask].sum()),
            "gauges": {g: int(k) for g, k in self.meta["gauge_index"].items()},
            # catchment of each gauge (for the GUI's bulk selection "upstream of gauge X")
            "subbasins": {g: b64(upstream_mask(self.meta["fdir"], int(r.ri[k]), int(r.rj[k])) & d.mask, "u1")
                          for g, k in self.meta["gauge_index"].items()},
            "gauges_observed": list(self.b["discharge_gauges"]),
            "facilities": {k: v.tolist() for k, v in self.fac.items()},
            "dams": dams,
            "area_km2": float(self.meta["area_km2"]),
            "events": [{"key": e.name, "label": event_label(e), "role": e.role,
                        "eval_start": e.eval_start, "eval_end": e.eval_end} for e in self.events],
            "preset": PRESETS.get(self.key, {}),
            "unit_costs": C.UNIT_COSTS,
            "note": self.b.get("gui_note", ""),
            "res_note": (f"注意: パラメータは {default_res(self.key):.0f} m メッシュで設定・校正した値。"
                         f"{self.res:.0f} m では流量・浸水面積・被害額の絶対値が変わる（docs/08_resolution.md）。"
                         + ("施策の効果の大小関係は概ね保たれるが、最終評価は 250 m で確認すること。" if self.res <= 500
                            else "流域界・観測所・低地の市街地の表現が崩れるため、施策評価には用いないこと。"))
            if self.res != default_res(self.key) else "",
            "climate_presets": {lv: CL.preset(lv, *self.b.get("climate_region", ("その他", "I"))).to_dict()
                                for lv in ("2C", "4C")},
            "climate_region": list(self.b.get("climate_region", ("その他", "I"))),
        }

    # -------------------------------------------------------------- spin-up
    def prepare(self, event_key, progress=None):
        if event_key in self._prep:
            return self._prep[event_key]
        ev = EVENTS[event_key]
        b = self.b
        tic = time.time()
        T0, Te = pd.Timestamp(ev.start), pd.Timestamp(ev.eval_start)
        start = str(T0 + pd.Timedelta(hours=1))
        if progress:
            progress(0.02, "雨量データを準備中（初回: 国交省・気象庁の観測所データを取得。数分〜）")
        data, stations = collect_gauges(
            b["cfg"].bbox, start, ev.end, mlit_suikei=b["mlit_suikei"],
            jma_prec_nos=b["jma_prec_nos"],
            cache_tag=f"{ev.basin}_{T0:%Y%m%d}_{pd.Timestamp(ev.end):%Y%m%d}", verbose=False)
        log.info("prepare %s: rain ready (%.1fs)", event_key, time.time() - tic)
        spath = CACHE / f"spinup_{self.ckey}_{event_key}.pkl"
        if spath.exists():
            if progress:
                progress(0.1, "洪水前の状態（スピンアップ）を読み込み中")
            state = pickle.load(open(spath, "rb"))
        else:
            rain0, _ = gridded_rainfall(self.d, data, stations, T0)
            dams0, sbc0, _ = boundary_conditions(self.key, self.d, self.meta, T0, ev.end)
            sim = Simulation(self.d, rain0, dams=dams0, stage_bc=sbc0,
                             drainage=drainage_arrays(self.key, self.d, self.meta))
            sim.run((Te - T0).total_seconds(), output_interval=6 * 3600,
                    progress=(lambda f: progress(0.05 + 0.4 * f, "洪水前の土壌水分・河川流量を計算中（スピンアップ, 初回のみ）"))
                    if progress else None)
            state = sim.get_state()
            pickle.dump(state, open(spath, "wb"))
        log.info("prepare %s: spin-up ready (%.1fs)", event_key, time.time() - tic)
        if progress:
            progress(0.45, "観測流量・潮位データを準備中")
        rain_ev, fields = gridded_rainfall(self.d, data, stations, Te)
        dams, sbc, bc_info = boundary_conditions(self.key, self.d, self.meta, Te, ev.end)
        t_end = (pd.Timestamp(ev.end) - Te).total_seconds()
        times = Te + pd.to_timedelta(np.arange(0, t_end + 1, 3600), unit="s")
        obs = {}
        for g, sid in b["discharge_gauges"].items():
            try:
                obs[g] = mlit_water.hourly(sid, mlit_water.KIND_DISCHARGE, str(times[0]),
                                           str(times[-1])).reindex(times).values
            except Exception:  # observation missing -> still allow simulation
                obs[g] = np.full(len(times), np.nan)
        log.info("prepare %s: observations ready (%.1fs)", event_key, time.time() - tic)
        hours = (data.index - Te) / pd.Timedelta(hours=1)
        w = (hours > 0) & (hours <= t_end / 3600)
        basin_rain = fields[np.asarray(w)][:, self.d.mask].mean(1)
        prep = {"state": state, "rain": rain_ev, "t_end": t_end, "times": times, "obs": obs,
                "basin_rain": basin_rain, "dams": dams, "stage_bc": sbc, "bc_info": bc_info}
        self._prep[event_key] = prep
        return prep

    # ------------------------------------------------------------ scenarios
    def _relocate(self, src, assets, pop, dest_ok, mode="existing", landclass=None, painted=None,
                  density_per_ha=NEW_SITE_DENSITY):
        """Move households, population, floor area and businesses of `src` cells.

        mode "existing": to the nearest safe built-up cells (`dest_ok`), proportionally
                         to their current population (集約: residential density rises)
             "new"     : to new residential land developed on the nearest safe, gentle
                         non-built-up cells at `density_per_ha` (高台の新規造成: these
                         cells become built-up; usually lowers the population-weighted density)
             "painted" : to the user-painted destination cells (`painted`), equally
        `landclass` is modified in place for new built-up cells.  Returns moved quantities."""
        arrays = [assets.households, assets.floor_area_m2, pop, *assets.employees.values()]
        moved = {"households": float(assets.households[src].sum()),
                 "employees": float(sum(v[src].sum() for v in assets.employees.values())),
                 "population": float(pop[src].sum()), "dest_cells": 0, "dest_mean_distance_km": 0.0,
                 "dest_mode": mode, "new_urban_cells": 0, "new_urban_area_m2": 0.0, "dest_hazard_cells": 0}
        d = self.d
        cell_ha = d.cell_area / 1e4
        ci = {n: k for k, n in enumerate(self.class_names)}
        if moved["households"] + moved["employees"] + moved["population"] > 0 and src.any():
            si, sj = np.nonzero(src)
            if mode == "painted":
                cand = (painted if painted is not None else np.zeros_like(src)) & ~src
            elif mode == "new":
                buildable = ~np.isin(landclass, [ci["urban"], ci["water"], ci["wetland"]])
                cand = dest_ok["new"] & buildable & ~src
            else:
                cand = dest_ok["existing"] & ~src
            if cand.any():
                ci_, cj_ = np.nonzero(cand)
                dist = np.hypot(ci_ - si.mean(), cj_ - sj.mean())
                order = np.argsort(dist)
                if mode == "existing":
                    n_dest = int(min(len(dist), max(3, np.ceil(src.sum() / 2))))
                elif mode == "new":
                    n_dest = int(min(len(dist), max(1, np.ceil(moved["population"] / (density_per_ha * cell_ha)))))
                else:
                    n_dest = len(dist)
                pick = order[:n_dest]
                dest = np.zeros_like(src)
                dest[ci_[pick], cj_[pick]] = True
                w = pop[dest] + 1.0 if mode == "existing" else np.ones(int(dest.sum()))
                w = w / w.sum()
                for arr in arrays:
                    arr[dest] += float(arr[src].sum()) * w
                if mode in ("new", "painted"):
                    newu = dest & (landclass != ci["urban"])
                    landclass[newu] = ci["urban"]
                    moved["new_urban_cells"] = int(newu.sum())
                    moved["new_urban_area_m2"] = float(newu.sum() * d.cell_area)
                    moved["dest_hazard_cells"] = int((dest & ~dest_ok["existing_hazard_free"]).sum())
                moved["dest_cells"] = int(dest.sum())
                moved["dest_mean_distance_km"] = float(dist[pick].mean() * d.dx / 1000)
                moved["dest_density_per_ha"] = float(moved["population"] / (dest.sum() * cell_ha))
        for arr in arrays:
            arr[src] = 0.0
        return moved

    def apply_edits(self, edits, gparams, baseline_depth=None, baseline_qmax=None, baseline_qseries=None):
        """Apply GUI edits to a copy of the basin.

        Returns (domain, assets, population, floor height, notes, storage mask,
        applied measures with quantities, retention list).  `retention` holds one
        dict per 遊水地 {"dam": solver.Dam, "site": mask, ...}: the basin is an
        off-channel storage fed over an overflow weir (越流堤) when the river
        discharge at its inlet exceeds q_cut, drained back after the peak.
        `baseline_qmax` (peak river discharge per river cell of the baseline run)
        sets the default q_cut; `baseline_qseries` (hourly discharge, (nt, nriv))
        gives the flood volume above q_cut and the automatic q_cut ("auto": the
        weir discharge at which this flood just fills the basin)."""
        from ..solver import Dam
        d = self.d.copy()
        assets = copy.deepcopy(self.assets)
        pop = self.pop.copy()
        floor = np.full(d.shape, FLOOR_HEIGHT)
        storage = np.zeros(d.shape, bool)       # planned flood storage (遊水地), not damage
        ny, nx = d.shape
        r = d.river
        riv_flat = r.ri * nx + r.rj
        is_riv = np.zeros(ny * nx, bool)
        is_riv[riv_flat] = True
        is_riv = is_riv.reshape(d.shape)
        ci = {n: k for k, n in enumerate(self.class_names)}
        pd_cls = d.params.land_classes[ci[PADDY_DAM]]
        store = float(gparams.get("paddy_store_m", 0.15))
        rel = float(gparams.get("paddy_release_mmh", 5.0))
        pd_cls.store_depth = store
        pd_cls.leak_rate = (rel / 1000 / 3600) / max(store, 1e-3)
        fn_cls = d.params.land_classes[ci[FOREST_NEW]]
        fn_cls.n = float(gparams.get("forest_n", 0.4))
        fn_cls.soil_depth = float(gparams.get("forest_soil_m", 1.0))
        cell_area = d.cell_area
        # relocation destinations: high above drainage and dry in the baseline;
        # existing built-up land (集約) or gentle land for new residential sites
        hazard_free = d.mask & (np.nan_to_num(self.hand, nan=0) > SAFE_HAND_M)
        if baseline_depth is not None:
            hazard_free &= np.nan_to_num(baseline_depth) < DEPTH_MIN
        gs = getattr(self, "gs", None)
        gentle = np.ones(d.shape, bool) if gs is None else np.nan_to_num(gs, nan=1.0) <= NEW_SITE_MAX_SLOPE
        dest_ok = {"existing": hazard_free & (d.landclass == ci["urban"]),
                   "new": hazard_free & gentle & ~is_riv,
                   "existing_hazard_free": hazard_free}
        painted_dest = np.zeros(d.shape, bool)
        for e in edits:
            if e["type"] == "relocate_dest":
                c = np.asarray(e.get("cells", []), dtype=np.int64)
                c = c[(c >= 0) & (c < ny * nx)]
                painted_dest.ravel()[c] = True
        painted_dest &= d.mask & ~is_riv
        notes, applied, retention = [], [], []
        for ei, e in enumerate(edits):
            if e["type"] == "relocate_dest":
                continue
            cells = np.asarray(e.get("cells", []), dtype=np.int64)
            cells = cells[(cells >= 0) & (cells < ny * nx)]
            m = np.zeros(ny * nx, bool)
            m[cells] = True
            m = m.reshape(d.shape) & d.mask
            t = e["type"]
            q = {}
            if t == "forest":
                sel = m & ~np.isin(d.landclass, [ci["water"], ci["forest"], ci[FOREST_NEW]])
                farm = float((assets.paddy_area_m2[sel] + assets.field_area_m2[sel]).sum())
                d.landclass[sel] = ci[FOREST_NEW]
                assets.paddy_area_m2[sel] = 0.0
                assets.field_area_m2[sel] = 0.0
                q = {"area_m2": float(sel.sum() * cell_area), "farm_area_m2": farm}
                notes.append(f"森林化 {sel.sum()} セル ({sel.sum() * cell_area / 1e6:.2f} km²; "
                             f"n={fn_cls.n:.2f}, 土層厚 {fn_cls.soil_depth:.1f} m)")
            elif t == "paddy_dam":
                sel = m & (d.landclass == ci["paddy"])
                d.landclass[sel] = ci[PADDY_DAM]
                q = {"area_m2": float(assets.paddy_area_m2[sel].sum())}
                notes.append(f"田んぼダム {sel.sum()} セル（水田 {q['area_m2'] / 1e6:.2f} km²）, "
                             f"貯留 {store * 100:.0f} cm, 放流 {rel:.1f} mm/h")
            elif t == "relocate":
                mode = e.get("dest", "new")
                if mode == "painted" and not painted_dest.any():
                    mode = "new"
                    notes.append("移転先が塗られていないため、高台の新規造成地（自動）に移転")
                vac = m & (d.landclass == ci["urban"])
                mv = self._relocate(m, assets, pop, dest_ok, mode, d.landclass, painted_dest,
                                    float(e.get("density", NEW_SITE_DENSITY)))
                d.landclass[vac] = ci["green"]                 # 移転跡地 -> 緑地
                q = dict(mv)
                where = {"existing": "既存の安全な市街地", "new": "高台の新規造成地", "painted": "指定した移転先"}[mode]
                msg = (f"移転 {m.sum()} セル（約 {mv['households']:.0f} 世帯・{mv['employees']:.0f} 従業者"
                       f" → {where} {mv['dest_cells']} セル, 平均 {mv['dest_mean_distance_km']:.1f} km")
                if mv["dest_cells"]:
                    msg += f", 移転先の人口密度 約 {mv.get('dest_density_per_ha', 0):.0f} 人/ha"
                if mv["new_urban_cells"]:
                    msg += f", 新たな建物用地 {mv['new_urban_area_m2'] / 1e4:.0f} ha"
                if mv["dest_hazard_cells"]:
                    msg += f", うち {mv['dest_hazard_cells']} セルは低地・現況浸水域"
                if not mv["dest_cells"] and mv["households"] > 0:
                    msg += " — 移転先の候補なし（世帯は流域外へ）"
                notes.append(msg + "）")
            elif t == "raise":
                hgt = float(e.get("height", 1.0))
                floor[m] += hgt
                if e.get("terrain"):
                    d.z[m] += hgt
                q = {"height_m": hgt, "terrain": bool(e.get("terrain")),
                     "households": float(assets.households[m].sum()), "area_m2": float(m.sum() * cell_area)}
                notes.append(f"嵩上げ +{hgt:.1f} m {m.sum()} セル（約 {q['households']:.0f} 世帯）"
                             + ("（盛土）" if e.get("terrain") else "（床高）"))
            elif t == "levee":
                # 築堤・嵩上げ: + height on top of the existing levee (0 m where there is none)
                hgt = float(e.get("height", 1.0))
                sel = m.ravel()[riv_flat]
                h0 = r.embankment[sel].copy()
                r.embankment[sel] = h0 + hgt
                L = float(r.length[sel].sum() * 2)   # both banks
                q = {"height_m": hgt, "base_height_m": float(h0.mean()) if len(h0) else 0.0, "length_m": L}
                notes.append(f"堤防 +{hgt:.1f} m 河道 {sel.sum()} セル（延長 {L / 1000:.1f} km, 両岸; "
                             f"既存 平均 {q['base_height_m']:.1f} m → {q['base_height_m'] + hgt:.1f} m）")
            elif t == "retention":
                dep = float(e.get("depth", 3.0))
                ratio = float(e.get("q_ratio", RETENTION_Q_RATIO))
                # site: painted land except built-up land (建物用地), channels and water
                site = m & ~is_riv & ~np.isin(d.landclass, [ci["urban"], ci["water"]])
                n_urban = int((m & (d.landclass == ci["urban"])).sum())
                if not site.any():
                    notes.append(f"遊水地: 用地にできるセルがありません（建物用地 {n_urban} セルは対象外）")
                    continue
                # inlet: the largest river next to the site (searching up to ~1 km away)
                ring, k_in = site.copy(), -1
                for _ in range(max(1, int(round(1000.0 / d.dx)))):
                    grow = ring.copy()
                    grow[1:] |= ring[:-1]
                    grow[:-1] |= ring[1:]
                    grow[:, 1:] |= ring[:, :-1]
                    grow[:, :-1] |= ring[:, 1:]
                    ring = grow
                    cand = np.flatnonzero(ring.ravel()[riv_flat])
                    if len(cand):
                        k_in = int(cand[np.argmax(r.area_km2[cand])])
                        break
                if k_in < 0:
                    notes.append("遊水地: 約1 km 以内に河道がないため設置できません")
                    continue
                capacity = float(site.sum() * cell_area * dep)
                q_peak = float(baseline_qmax[k_in]) if baseline_qmax is not None else np.nan
                series = (np.asarray(baseline_qseries)[:, k_in] if baseline_qseries is not None else None)
                excess = (lambda qc: float(np.maximum(series - qc, 0.0).sum() * 3600.0)) if series is not None else None
                auto = bool(e.get("auto")) and excess is not None
                if e.get("q_cut") is not None:
                    q_cut, how = float(e["q_cut"]), "指定"
                elif auto:                               # this flood just fills the basin (perfect foresight)
                    lo, hi = 0.0, float(series.max())
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if excess(mid) > capacity else (lo, mid)
                    q_cut, how = hi, "自動（この洪水で容量がちょうど満杯）"
                elif np.isfinite(q_peak) and q_peak > 0:
                    q_cut, how = ratio * q_peak, f"現況ピーク {q_peak:.0f} の {ratio * 100:.0f}%"
                else:
                    q_cut, how = 100.0, "既定値"
                vol_over = excess(q_cut) if excess is not None else np.nan
                mv = self._relocate(site, assets, pop, dest_ok, "existing", d.landclass)   # 用地内の住宅
                assets.paddy_area_m2[site] = 0.0
                assets.field_area_m2[site] = 0.0
                d.landclass[site] = ci["wetland"]
                storage |= site
                retention.append({"dam": Dam(k=k_in, capacity=capacity, q_cut=q_cut, name=f"遊水地{len(retention) + 1}",
                                             release_q=RETENTION_RELEASE_RATIO * q_cut),
                                  "site": site, "area_m2": float(site.sum() * cell_area)})
                weir = float(np.sqrt(site.sum() * cell_area))           # 越流堤長 ~ one side of the site
                q = {"area_m2": float(site.sum() * cell_area), "depth_m": dep, "weir_length_m": weir,
                     "households": mv["households"], "capacity_m3": capacity, "q_cut": q_cut,
                     "flood_volume_over_qcut_m3": vol_over,
                     "inlet_area_km2": float(r.area_km2[k_in]), "excluded_urban_cells": n_urban}
                msg = (f"遊水地 {site.sum()} セル（{q['area_m2'] / 1e6:.2f} km², 貯留深 {dep:.1f} m, 容量 {capacity / 1e6:.2f} 百万m³）"
                       f": 集水面積 {r.area_km2[k_in]:.0f} km² の河道から、流量 {q_cut:.0f} m³/s（{how}）を超える分を"
                       f"越流堤から取り込み、河道の流量がその半分を下回ってから排水")
                if np.isfinite(vol_over):
                    msg += f"。現況でこの流量を超える水量 {vol_over / 1e6:.2f} 百万m³"
                    if vol_over > 1.05 * capacity:
                        msg += (f"（容量の {vol_over / capacity:.1f} 倍: ピーク前に満杯になり、ピークはほとんど下がらない。"
                                "面積・貯留深を増やすか、越流開始流量を上げる）")
                if n_urban:
                    msg += f"。建物用地 {n_urban} セルは対象外"
                if mv["households"] >= 0.5:
                    msg += f"。用地内の {mv['households']:.0f} 世帯は移転"
                notes.append(msg)
            applied.append({"type": t, "q": q, "edit": ei})
        return d, assets, pop, floor, notes, storage, applied, retention

    def estimate(self, event_key, edits, gparams=None):
        """Project cost of `edits` without simulating (game UI budget).  Uses the
        baseline of this event/climate when it has been run (relocation targets,
        retention weirs); returns the cost with one item per applied edit and
        `edit_cost_yen` (present value per edit index, 0 where nothing applied)."""
        gparams = gparams or {}
        base = self._baseline.get((event_key, self.climate(gparams).key()))
        design = self._baseline.get((event_key, CL.ClimateScenario().key()), base)
        *_, notes, storage, applied, retention = self.apply_edits(
            edits, gparams, base["_depth"] if base else None, design["_river_qmax"] if design else None,
            design["_river_q_series"] if design else None)
        cost = C.estimate(applied, gparams.get("unit_costs"))
        per, ini, ann = [0.0] * len(edits), [0.0] * len(edits), [0.0] * len(edits)
        for a, it in zip(applied, cost["items"]):
            per[a["edit"]] += it["pv_yen"]
            ini[a["edit"]] += it["initial_yen"]
            ann[a["edit"]] += it["annual_yen"]
        return {"cost": {k: v for k, v in cost.items() if k != "unit_costs"}, "edit_cost_yen": per,
                "edit_initial_yen": ini, "edit_annual_yen": ann, "edits_log": notes}

    def climate(self, gparams) -> "CL.ClimateScenario":
        rr, sr = self.b.get("climate_region", ("その他", "I"))
        return CL.from_params((gparams or {}).get("climate"), rr, sr)

    @staticmethod
    def _summary(res):
        return {"damage_total_oku": res["damage_total_oku"], "inundated_km2": res["inundated_km2"],
                "households_above_floor": res["households_above_floor"],
                "households_below_floor": res["households_below_floor"],
                "peaks": {g: float(np.nanmax(v["sim"])) for g, v in res["gauges"].items()},
                "gauges": {g: v["sim"] for g, v in res["gauges"].items()},
                "basin_rain_total_mm": float(np.sum(res["basin_rain"])),
                "livability": res["livability"]["score"], "ecology": res["ecology"]["score"]}

    def run(self, event_key, edits=None, gparams=None, progress=None):
        edits = edits or []
        gparams = gparams or {}
        prep = self.prepare(event_key, progress)
        sc = self.climate(gparams)
        bkey = (event_key, sc.key())
        is_base = not edits
        if is_base and bkey in self._baseline:
            return self._baseline[bkey]
        # gparams["baseline"] == "current": measures are set against the current-climate baseline
        # (no extra no-measures run per scenario climate; used by the RDM exploration)
        cur_key = (event_key, CL.ClimateScenario().key())
        use_cur = gparams.get("baseline") == "current"
        if not is_base and use_cur and cur_key not in self._baseline:
            self.run(event_key, [], {k: v for k, v in gparams.items() if k not in ("climate", "baseline")})
        elif not is_base and not use_cur and bkey not in self._baseline:
            # measures are evaluated against the baseline (relocation targets dry land,
            # retention weirs are set from the baseline peak discharge)
            self.run(event_key, [], gparams, (lambda f, msg: progress(0.5 * f, "現況: " + msg)) if progress else None)
        base = self._baseline.get(bkey) or (self._baseline.get(cur_key) if use_cur else None)
        # 遊水地 are designed for the current climate (the facility does not change with the scenario)
        design = self._baseline.get((event_key, CL.ClimateScenario().key()), base)
        if progress and not is_base:
            progress(0.5, "施策を反映中")
        d, assets, pop, floor, edit_log, storage, applied, retention = self.apply_edits(
            edits, gparams, base["_depth"] if base else None, design["_river_qmax"] if design else None,
            design["_river_q_series"] if design else None)
        dams = [copy.copy(dm) for dm in prep["dams"]]
        n_dam0 = len(dams)
        for rt in retention:
            dup = [dm for dm in dams if dm.k == rt["dam"].k]
            if dup:                       # two basins on the same reach: one combined storage
                dup[0].capacity += rt["dam"].capacity
                rt["dam"] = dup[0]
            else:
                dams.append(rt["dam"])
        rain = CL.scale_rainfall(prep["rain"], sc, t_start=0.0)   # event window (after spin-up)
        sim = Simulation(d, rain, dams=dams, stage_bc=CL.shift_stage(prep["stage_bc"], sc),
                         drainage=drainage_arrays(self.key, d, self.meta, d.landclass))
        sim.set_state(prep["state"])
        fc = self.flood_cells

        # 遊水地: stored volume shown as a water depth over the site
        ret_view = []
        fc_pos = -np.ones(d.shape[0] * d.shape[1], np.int64)
        fc_pos[fc] = np.arange(len(fc))
        for rt in retention:
            n = dams.index(rt["dam"])
            same = [x for x in retention if x["dam"] is rt["dam"]]
            area = sum(x["area_m2"] for x in same)
            ret_view.append((n, fc_pos[np.flatnonzero(rt["site"].ravel())], area))

        def snap(s):
            # water intentionally held in 田んぼダム plots is not inundation
            depth = np.maximum(s.h - s.da - s.sd, 0.0).ravel()[fc].astype(np.float32)
            for n, pos, area in ret_view:
                depth[pos[pos >= 0]] = s.dam_S[n] / area
            return depth, s.hr.astype(np.float32), s.dam_S.copy(), s.Q.astype(np.float32)

        t0 = time.time()
        log.info("run %s: %d edits, simulating %.0f h", event_key, len(edits), prep["t_end"] / 3600)
        res = sim.run(prep["t_end"], gauges=self.meta["gauge_index"], output_interval=3600.0,
                      snapshot_interval=3600.0, snapshot_fn=snap,
                      progress=(lambda f: progress(0.5 + 0.4 * f, f"洪水計算中（{'現況' if is_base else '施策あり'}, {f * 100:.0f}%）"))
                      if progress else None)
        wall = time.time() - t0
        log.info("run %s: done in %.1fs (%d steps)", event_key, wall, res.n_steps)
        if progress:
            progress(0.92, "被害額・指標を計算中")
        surf = np.maximum(np.nan_to_num(res.max_depth) - sim.sd, 0.0)
        dam_S = np.stack([s[1][2] for s in res.snapshots]) if dams else np.zeros((0, 0))
        ret_info = []
        for k, (n, pos, area) in enumerate(ret_view):
            smax = float(dam_S[:, n].max()) if dam_S.size else 0.0
            surf[retention[k]["site"]] = smax / area
            dm = dams[n]
            ret_info.append({"name": dm.name, "q_cut": dm.q_cut, "capacity_mcm": dm.capacity / 1e6,
                             "max_storage_mcm": smax / 1e6, "fill_ratio": smax / dm.capacity if dm.capacity else 0.0,
                             # hourly values (as the gauges): the per-step maximum has a brief spike when the basin fills
                             "inlet_peak_before": float(design["_river_q_series"][:, dm.k].max()) if design else None,
                             "inlet_peak_after": float(max(x[1][3][dm.k] for x in res.snapshots))})
        retention_mcm = sum(x["max_storage_mcm"] for x in {x["name"]: x for x in ret_info}.values())
        depth = floodplain_inundation(surf, self.hand, DEPTH_MIN, HAND_DAMAGE_MAX)
        disp = floodplain_inundation(surf, self.hand, DEPTH_MIN, HAND_DISPLAY_MAX)
        disp[storage] = surf[storage]   # shown in the map (遊水地 filling) ...
        depth[storage] = 0.0            # ... but not counted as inundation / damage
        # inundation duration [days] (hourly snapshots of the floodplain cells) for crops
        frames_d = np.stack([s[1][0] for s in res.snapshots])     # (nt, nflood)
        dur = np.zeros(d.shape)
        dur.ravel()[fc] = (frames_d >= DEPTH_MIN).sum(0) * 3600.0 / 86400.0
        # sub-grid: houses / farmland flooded by their own ground elevation within the cell
        use_sg = gparams.get("subgrid", True) and getattr(self, "subgrid", None) is not None
        level = SG.water_level(depth, self.subgrid["all"]) if use_sg else None
        dmg = D.estimate_damage(depth, assets, ground_slope=self.gs, floor_height=floor, duration_days=dur,
                                subgrid=(self.subgrid, level) if use_sg else None)
        hh = assets.households
        if use_sg:
            du = SG.depths(depth, self.subgrid["urban"], level)
            k = du.shape[0]
            hh_above = float((hh[None] / k * (du >= floor[None])).sum())
            hh_below = float((hh[None] / k * ((du > 0) & (du < floor[None]))).sum())
            area_frac = (SG.depths(depth, self.subgrid["all"], level) >= DEPTH_MIN).mean(0)
        else:
            hh_above = float(hh[(depth >= floor)].sum())
            hh_below = float(hh[(depth > 0) & (depth < floor)].sum())
            area_frac = (depth > 0).astype(float)
        names = self.class_names
        liv = I.livability(pop, d.landclass, names, self.road, self.fac, d.dx, d.dy, d.mask,
                           flood_depth=depth, zone=getattr(self, "zone", None))
        eco = I.ecology(d.landclass, names, self.road, d.mask, d.dx, d.dy, d.river, d.river.embankment,
                        [dm.k for dm in dams[:n_dam0]], self.hand, surf, res.mass.rain,   # 遊水地 are off-channel
                        res.mass.river_outflow + res.mass.boundary_outflow)
        eco.pop("_quality_map", None)
        cost = C.estimate(applied, gparams.get("unit_costs"))
        frames_r = np.stack([s[1][1] for s in res.snapshots])     # (nt, nriv)
        out = {
            "event": event_key, "edits_log": edit_log, "wall_s": wall, "steps": res.n_steps,
            "mass_rel_err": res.mass.relative_error,
            "times": [str(t) for t in prep["times"][1:len(res.snapshots) + 1]],
            "basin_rain": self._basin_rain(rain, prep["t_end"]).round(2).tolist(),
            "frames_depth_cm": b64(np.clip(frames_d * 100, 0, 65535), "<u2"),
            "frames_river_cm": b64(np.clip(frames_r * 100, 0, 65535), "<u2"),
            "nt": len(res.snapshots),
            "max_depth_cm": b64(np.clip(np.nan_to_num(disp) * 100, 0, 65535), "<u2"),
            "damage": b64(dmg.total_map, "<f4"),
            "damage_total_oku": dmg.total / 1e8,
            "damage_by_category_oku": {k: v / 1e8 for k, v in dmg.totals.items()},
            "inundated_km2": float(area_frac.sum() * d.cell_area / 1e6),
            "subgrid": bool(use_sg),
            "retention_max_storage_mcm": retention_mcm,
            "retention": ret_info,
            "storage_cells": b64(np.flatnonzero(storage), "<i4"),
            "households_above_floor": hh_above,
            "households_below_floor": hh_below,
            "embankment": d.river.embankment.round(2).tolist(),
            "z_changed": b64(np.where(d.mask, d.z, np.nan), "<f4") if any(
                e["type"] in ("retention", "raise") for e in edits) else None,
            "landclass": b64(d.landclass, "u1"),
            "households": b64(hh, "<f4"),
            "population": b64(pop, "<f4"),
            "gauges": {g: {"sim": np.asarray(res.gauge_q[g][1:], float).round(2).tolist(),
                           "obs": [None if not np.isfinite(v) else round(float(v), 2)
                                   for v in prep["obs"].get(g, [])][1:]}
                       for g in self.meta["gauge_index"]},
            "dams": {dm.name: {"max_storage_mcm": float(dam_S[:, n].max() / 1e6) if dam_S.size else 0.0,
                               "capacity_mcm": dm.capacity / 1e6} for n, dm in enumerate(dams[:n_dam0])},
            "cost": cost,
            "livability": liv,
            "ecology": eco,
            "climate": sc.to_dict(),
        }
        cur = self._baseline.get((event_key, CL.ClimateScenario().key()))
        if not sc.is_current and cur is not None:
            out["climate_reference"] = self._summary(cur)      # current-climate baseline for comparison
        if is_base:
            out["_depth"] = depth
            out["_river_qmax"] = np.asarray(res.max_river_q, float)
            out["_river_q_series"] = np.stack([x[1][3] for x in res.snapshots])
            self._baseline[bkey] = out
        return out

    def _basin_rain(self, rain, t_end):
        mid = 0.5 * (rain.edges[:-1] + rain.edges[1:])
        w = (mid > 0) & (mid <= t_end)
        return rain.rates[w][:, self.d.mask].mean(1) * 3.6e6


def _raise_thread_qos():
    """macOS: run the simulation thread with USER_INITIATED QoS.

    A server started from a background/non-interactive shell can inherit
    background QoS, which confines it to efficiency cores (observed 10-50x
    slow-downs).  Best effort; no-op elsewhere."""
    import sys
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        QOS_CLASS_USER_INITIATED = 0x19
        lib.pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0)
    except Exception:  # pragma: no cover
        pass


class Engine:
    def __init__(self):
        self.sessions: dict[tuple, BasinSession] = {}
        self.jobs: dict[str, dict] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, initializer=_raise_thread_qos)  # one at a time
        self.lock = threading.Lock()

    def session(self, key, res=None, progress=None) -> BasinSession:
        res = float(res) if res else default_res(key)
        with self.lock:
            if (key, res) not in self.sessions:
                # keep at most one other resolution of the same basin in memory
                for k in [k for k in self.sessions if k[0] == key and k[1] != default_res(key)]:
                    del self.sessions[k]
                self.sessions[(key, res)] = BasinSession(key, res, progress)
            return self.sessions[(key, res)]

    def is_loaded(self, key, res=None) -> bool:
        return (key, float(res) if res else default_res(key)) in self.sessions

    def basins_keys(self):
        return set(BASINS)

    def basins(self):
        return [{"key": k, "name": v["cfg"].name,
                 "events": [e.name for e in EVENTS.values() if e.basin == k]}
                for k, v in BASINS.items()]

    def _start(self, fn):
        jid = uuid.uuid4().hex[:10]
        job = {"status": "queued", "progress": 0.0, "message": "待機中（他の計算の終了待ち）", "result": None,
               "started": None, "phase_started": None}
        self.jobs[jid] = job

        def progress(f, msg):
            if msg.split("（")[0] != job["message"].split("（")[0]:      # new phase (ignore % updates)
                job["phase_started"] = time.time()
            job["progress"], job["message"] = float(f), msg

        def work():
            job["status"] = "running"
            job["started"] = job["phase_started"] = time.time()
            try:
                job["result"] = fn(progress)
                job["status"], job["progress"], job["message"] = "done", 1.0, "完了"
            except Exception as e:  # report to UI
                job["status"], job["message"] = "error", f"{e}\n{traceback.format_exc()}"

        self.pool.submit(work)
        return jid

    def prepare_basin(self, basin, grid_res=None):
        """Job: load / build the basin model, assets and indicator data."""
        def fn(progress):
            self.session(basin, grid_res, progress)
            return {"ready": True}
        return self._start(fn)

    def estimate(self, basin, event, edits, gparams, grid_res=None):
        with self.lock:   # sessions are built by prepare jobs; estimating needs a loaded one
            s = self.sessions.get((basin, float(grid_res) if grid_res else default_res(basin)))
        if s is None:
            raise KeyError(basin)
        return s.estimate(event, edits, gparams)

    def submit(self, basin, event, edits, gparams, grid_res=None):
        def fn(progress):
            s = self.session(basin, grid_res, progress)
            res = s.run(event, edits, gparams, progress)
            return {k: v for k, v in res.items() if not k.startswith("_")}
        return self._start(fn)
