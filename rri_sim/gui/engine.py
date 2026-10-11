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
  {"type": "levee",      "cells": [...], "height": 3.0}        堤防（範囲内の河道セル）
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

from .. import costs as C
from .. import damage as D
from .. import indicators as I
from ..assets import build_assets
from ..basin import floodplain_inundation, lonlat_to_rc
from ..cases import BASINS, EVENTS
from ..data import estat, facilities, landuse, mlit_water
from ..events import collect_gauges, gridded_rainfall
from ..solver import Simulation
from ..workflow import CACHE, FOREST_NEW, PADDY_DAM, PRESETS, boundary_conditions, build_preset

log = logging.getLogger("rri_sim.gui")
HAND_DISPLAY_MAX = 15.0       # m; cells that can show inundation in the GUI
HAND_DAMAGE_MAX = 10.0        # m; floodplain definition for inundation / damage statistics (docs/04)
DEPTH_MIN = 0.1               # m; inundation threshold for damage / extent
FLOOR_HEIGHT = 0.45           # m; manual default
SAFE_HAND_M = 10.0            # relocation destinations: built-up cells this high above drainage


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def event_label(ev):
    return f"{ev.name}（{pd.Timestamp(ev.eval_start):%Y/%m/%d}〜, {ev.role}）"


class BasinSession:
    def __init__(self, key: str):
        self.key = key
        self.d, self.meta, self.b = build_preset(key, use_cache=True)
        d, b = self.d, self.b
        pref = b.get("prefecture", "福岡")
        apath = CACHE / f"assets_{key}.pkl"
        if apath.exists():
            self.assets = pickle.load(open(apath, "rb"))
        else:
            self.assets = build_assets(d, self.meta["landuse_fraction"], b["cfg"].bbox, pref)
            CACHE.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.assets, open(apath, "wb"))
        self._load_indicator_data()
        self.gs = D.ground_slope_from_dem(d.z, d.dx, d.dy)
        self.hand = self.meta["hand"]
        self.flood_cells = np.flatnonzero(d.mask & (np.nan_to_num(self.hand, nan=1e9) <= HAND_DISPLAY_MAX))
        self.class_names = [c.name for c in d.params.land_classes]
        self.events = [e for e in EVENTS.values() if e.basin == key]
        self._prep, self._baseline = {}, {}

    def _load_indicator_data(self):
        d, b = self.d, self.b
        path = CACHE / f"indicator_data_{self.key}.pkl"
        if path.exists():
            self.road, self.pop, self.fac = pickle.load(open(path, "rb"))
            return
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
            "z": b64(np.where(d.mask, d.z, np.nan), "<f4"),
            "mask": b64(d.mask, "u1"), "landclass": b64(d.landclass, "u1"),
            "class_names": self.class_names, "hand": b64(np.nan_to_num(self.hand, nan=-1), "<f4"),
            "flood_cells": b64(self.flood_cells, "<i4"),
            "river": {"ri": r.ri.tolist(), "rj": r.rj.tolist(), "dn": r.dn.tolist(),
                      "width": r.width.round(1).tolist(), "depth": r.depth.round(2).tolist(),
                      "embankment": r.embankment.round(2).tolist(),
                      "area": r.area_km2.round(1).tolist()},
            "households": b64(self.assets.households, "<f4"),
            "population": b64(self.pop, "<f4"),
            "asset_value": b64(value, "<f4"),
            "asset_total_oku": float(value.sum() / 1e8),
            "households_total": float(self.assets.households.sum()),
            "population_total": float(self.pop[d.mask].sum()),
            "gauges": {g: int(k) for g, k in self.meta["gauge_index"].items()},
            "gauges_observed": list(self.b["discharge_gauges"]),
            "facilities": {k: v.tolist() for k, v in self.fac.items()},
            "dams": dams,
            "area_km2": float(self.meta["area_km2"]),
            "events": [{"key": e.name, "label": event_label(e), "role": e.role,
                        "eval_start": e.eval_start, "eval_end": e.eval_end} for e in self.events],
            "preset": PRESETS.get(self.key, {}),
            "unit_costs": C.UNIT_COSTS,
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
            progress(0.02, "雨量データ取得中（初回は数分〜）")
        data, stations = collect_gauges(
            b["cfg"].bbox, start, ev.end, mlit_suikei=b["mlit_suikei"],
            jma_prec_nos=b["jma_prec_nos"],
            cache_tag=f"{ev.basin}_{T0:%Y%m%d}_{pd.Timestamp(ev.end):%Y%m%d}", verbose=False)
        log.info("prepare %s: rain ready (%.1fs)", event_key, time.time() - tic)
        spath = CACHE / f"spinup_{self.key}_{event_key}.pkl"
        if spath.exists():
            state = pickle.load(open(spath, "rb"))
        else:
            rain0, _ = gridded_rainfall(self.d, data, stations, T0)
            dams0, sbc0, _ = boundary_conditions(self.key, self.d, self.meta, T0, ev.end)
            sim = Simulation(self.d, rain0, dams=dams0, stage_bc=sbc0)
            sim.run((Te - T0).total_seconds(), output_interval=6 * 3600,
                    progress=(lambda f: progress(0.05 + 0.45 * f, "スピンアップ計算中（初回のみ）"))
                    if progress else None)
            state = sim.get_state()
            pickle.dump(state, open(spath, "wb"))
        log.info("prepare %s: spin-up ready (%.1fs)", event_key, time.time() - tic)
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
    def _relocate(self, src, assets, pop, dest_ok):
        """Move households, population, floor area and businesses of `src`
        cells to the nearest safe built-up cells (`dest_ok`), proportionally to
        their current population (densification).  Returns moved quantities."""
        arrays = [assets.households, assets.floor_area_m2, pop, *assets.employees.values()]
        moved = {"households": float(assets.households[src].sum()),
                 "employees": float(sum(v[src].sum() for v in assets.employees.values())),
                 "population": float(pop[src].sum()), "dest_cells": 0, "dest_mean_distance_km": 0.0}
        cand = dest_ok & ~src
        if moved["households"] + moved["employees"] + moved["population"] > 0 and cand.any() and src.any():
            si, sj = np.nonzero(src)
            ci_, cj_ = np.nonzero(cand)
            dist = np.hypot(ci_ - si.mean(), cj_ - sj.mean())
            n_dest = int(min(len(dist), max(3, np.ceil(src.sum() / 2))))
            pick = np.argsort(dist)[:n_dest]
            dest = np.zeros_like(src)
            dest[ci_[pick], cj_[pick]] = True
            w = pop[dest] + 1.0
            w = w / w.sum()
            for arr in arrays:
                arr[dest] += float(arr[src].sum()) * w
            moved["dest_cells"] = int(dest.sum())
            moved["dest_mean_distance_km"] = float(dist[pick].mean() * self.d.dx / 1000)
        for arr in arrays:
            arr[src] = 0.0
        return moved

    def apply_edits(self, edits, gparams, baseline_depth=None):
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
        # relocation destinations: built-up, high above drainage, dry in the baseline
        safe = d.mask & (d.landclass == ci["urban"]) & (np.nan_to_num(self.hand, nan=0) > SAFE_HAND_M)
        if baseline_depth is not None:
            safe &= np.nan_to_num(baseline_depth) < DEPTH_MIN
        notes, applied = [], []
        for e in edits:
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
                mv = self._relocate(m, assets, pop, safe)
                vac = m & (d.landclass == ci["urban"])
                d.landclass[vac] = ci["green"]                 # 移転跡地 -> 緑地
                q = dict(mv)
                notes.append(f"移転 {m.sum()} セル（約 {mv['households']:.0f} 世帯・{mv['employees']:.0f} 従業者"
                             f" → 安全な市街地 {mv['dest_cells']} セル, 平均 {mv['dest_mean_distance_km']:.1f} km）")
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
                hgt = float(e.get("height", 3.0))
                sel = m.ravel()[riv_flat]
                add = np.maximum(hgt - r.embankment[sel], 0.0)
                r.embankment[sel] = np.maximum(r.embankment[sel], hgt)
                L = float((r.length[sel] * (add > 0)).sum() * 2)   # both banks
                q = {"height_m": float(add[add > 0].mean()) if (add > 0).any() else 0.0, "length_m": L}
                notes.append(f"堤防 {hgt:.1f} m 河道 {sel.sum()} セル（延長 {L / 1000:.1f} km, 両岸）")
            elif t == "retention":
                dep = float(e.get("depth", 2.0))
                inlet = float(e.get("inlet", 1.0))
                exc = m & ~is_riv
                d.z[exc] -= dep
                storage |= m
                mv = self._relocate(m, assets, pop, safe)       # 用地内の住宅は移転
                assets.paddy_area_m2[m] = 0.0
                assets.field_area_m2[m] = 0.0
                d.landclass[exc] = ci["wetland"]
                near = m.copy()
                near[1:] |= m[:-1]
                near[:-1] |= m[1:]
                near[:, 1:] |= m[:, :-1]
                near[:, :-1] |= m[:, 1:]
                sel = near.ravel()[riv_flat]
                r.embankment[sel] = inlet
                q = {"area_m2": float(exc.sum() * cell_area), "depth_m": dep,
                     "weir_length_m": float(r.length[sel].sum()), "households": mv["households"]}
                notes.append(f"遊水地 {exc.sum()} セル（{q['area_m2'] / 1e6:.2f} km², 掘削 {dep:.1f} m,"
                             f" 越流堤 {inlet:.1f} m, 河道 {sel.sum()} セル, 移転 {mv['households']:.0f} 世帯）")
            applied.append({"type": t, "q": q})
        return d, assets, pop, floor, notes, storage, applied

    def run(self, event_key, edits=None, gparams=None, progress=None):
        edits = edits or []
        gparams = gparams or {}
        prep = self.prepare(event_key, progress)
        is_base = not edits
        if is_base and event_key in self._baseline:
            return self._baseline[event_key]
        base_depth = self._baseline[event_key]["_depth"] if (not is_base and event_key in self._baseline) else None
        d, assets, pop, floor, edit_log, storage, applied = self.apply_edits(edits, gparams, base_depth)
        dams = [copy.copy(dm) for dm in prep["dams"]]
        sim = Simulation(d, prep["rain"], dams=dams, stage_bc=prep["stage_bc"])
        sim.set_state(prep["state"])
        fc = self.flood_cells

        def snap(s):
            # water intentionally held in 田んぼダム plots is not inundation
            return (np.maximum(s.h - s.da - s.sd, 0.0).ravel()[fc].astype(np.float32),
                    s.hr.astype(np.float32), s.dam_S.copy())

        t0 = time.time()
        log.info("run %s: %d edits, simulating %.0f h", event_key, len(edits), prep["t_end"] / 3600)
        res = sim.run(prep["t_end"], gauges=self.meta["gauge_index"], output_interval=3600.0,
                      snapshot_interval=3600.0, snapshot_fn=snap,
                      progress=(lambda f: progress(0.5 + 0.4 * f, "洪水計算中")) if progress else None)
        wall = time.time() - t0
        log.info("run %s: done in %.1fs (%d steps)", event_key, wall, res.n_steps)
        if progress:
            progress(0.92, "被害額・指標を計算中")
        surf = np.maximum(np.nan_to_num(res.max_depth) - sim.sd, 0.0)
        retention_mcm = float(surf[storage].sum() * d.cell_area / 1e6)
        depth = floodplain_inundation(surf, self.hand, DEPTH_MIN, HAND_DAMAGE_MAX)
        disp = floodplain_inundation(surf, self.hand, DEPTH_MIN, HAND_DISPLAY_MAX)
        disp[storage] = surf[storage]   # shown in the map (遊水地 filling) ...
        depth[storage] = 0.0            # ... but not counted as inundation / damage
        dmg = D.estimate_damage(depth, assets, ground_slope=self.gs, floor_height=floor)
        names = self.class_names
        liv = I.livability(pop, d.landclass, names, self.road, self.fac, d.dx, d.dy, d.mask,
                           flood_depth=depth)
        eco = I.ecology(d.landclass, names, self.road, d.mask, d.dx, d.dy, d.river, d.river.embankment,
                        [dm.k for dm in dams], self.hand, surf, res.mass.rain,
                        res.mass.river_outflow + res.mass.boundary_outflow)
        eco.pop("_quality_map", None)
        cost = C.estimate(applied, gparams.get("unit_costs"))
        frames_d = np.stack([s[1][0] for s in res.snapshots])     # (nt, nflood)
        frames_r = np.stack([s[1][1] for s in res.snapshots])     # (nt, nriv)
        dam_S = np.stack([s[1][2] for s in res.snapshots]) if dams else np.zeros((0, 0))
        hh = assets.households
        out = {
            "event": event_key, "edits_log": edit_log, "wall_s": wall, "steps": res.n_steps,
            "mass_rel_err": res.mass.relative_error,
            "times": [str(t) for t in prep["times"][1:len(res.snapshots) + 1]],
            "basin_rain": prep["basin_rain"].round(2).tolist(),
            "frames_depth_cm": b64(np.clip(frames_d * 100, 0, 65535), "<u2"),
            "frames_river_cm": b64(np.clip(frames_r * 100, 0, 65535), "<u2"),
            "nt": len(res.snapshots),
            "max_depth_cm": b64(np.clip(np.nan_to_num(disp) * 100, 0, 65535), "<u2"),
            "damage": b64(dmg.total_map, "<f4"),
            "damage_total_oku": dmg.total / 1e8,
            "damage_by_category_oku": {k: v / 1e8 for k, v in dmg.totals.items()},
            "inundated_km2": float((depth > 0).sum() * d.cell_area / 1e6),
            "retention_max_storage_mcm": retention_mcm,
            "storage_cells": b64(np.flatnonzero(storage), "<i4"),
            "households_above_floor": float(hh[(depth >= floor)].sum()),
            "households_below_floor": float(hh[(depth > 0) & (depth < floor)].sum()),
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
                               "capacity_mcm": dm.capacity / 1e6} for n, dm in enumerate(dams)},
            "cost": cost,
            "livability": liv,
            "ecology": eco,
        }
        if is_base:
            out["_depth"] = depth
            self._baseline[event_key] = out
        return out


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
        self.sessions: dict[str, BasinSession] = {}
        self.jobs: dict[str, dict] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, initializer=_raise_thread_qos)  # one at a time
        self.lock = threading.Lock()

    def session(self, key) -> BasinSession:
        with self.lock:
            if key not in self.sessions:
                self.sessions[key] = BasinSession(key)
            return self.sessions[key]

    def basins(self):
        return [{"key": k, "name": v["cfg"].name,
                 "events": [e.name for e in EVENTS.values() if e.basin == k]}
                for k, v in BASINS.items()]

    def submit(self, basin, event, edits, gparams):
        jid = uuid.uuid4().hex[:10]
        job = {"status": "queued", "progress": 0.0, "message": "待機中", "result": None}
        self.jobs[jid] = job

        def progress(f, msg):
            job["progress"], job["message"] = float(f), msg

        def work():
            job["status"] = "running"
            try:
                progress(0.0, "流域データ準備中")
                s = self.session(basin)
                res = s.run(event, edits, gparams, progress)
                job["result"] = {k: v for k, v in res.items() if not k.startswith("_")}
                job["status"], job["progress"], job["message"] = "done", 1.0, "完了"
            except Exception as e:  # report to UI
                job["status"], job["message"] = "error", f"{e}\n{traceback.format_exc()}"

        self.pool.submit(work)
        return jid
