"""Simulation engine behind the 3D GUI.

A BasinSession holds one basin (domain with calibrated preset, assets, HAND).
For each event the spin-up (observed rain from `event.start` to
`event.eval_start`) is computed once and cached; scenario runs restart from
that state and only simulate the flood window, so a run takes seconds to a
minute.  Scenario = list of edits painted in the GUI:

  {"type": "forest",     "cells": [...]}                       森林化（土地利用→森林）
  {"type": "paddy_dam",  "cells": [...]}                       田んぼダム（水田セルのみ有効）
  {"type": "relocate",   "cells": [...]}                       住宅・事業所の移転（資産を除去）
  {"type": "raise",      "cells": [...], "height": 1.0,
                         "terrain": false}                     嵩上げ（床高 / 盛土）
  {"type": "levee",      "cells": [...], "height": 3.0}        堤防（範囲内の河道セル）
  {"type": "retention",  "cells": [...], "depth": 2.0,
                         "inlet": 1.0}                         遊水地（掘削＋越流堤）
Global scenario parameters: {"paddy_store_m": 0.15, "paddy_release_mmh": 5.0}

Approximation (documented in the UI): land-use edits do not alter the
spin-up state, i.e. antecedent soil moisture is that of the baseline.
"""
from __future__ import annotations

import base64
import logging
import copy
import pickle
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .. import damage as D
from ..assets import build_assets
from ..basin import floodplain_inundation
from ..cases import BASINS, EVENTS
from ..data import mlit_water
from ..data.landuse import CLASSES
from ..events import collect_gauges, gridded_rainfall
from ..solver import Simulation
from ..workflow import CACHE, FOREST_NEW, PADDY_DAM, PRESETS, build_preset

log = logging.getLogger("rri_sim.gui")
PREFECTURE = {"kusu_ogase": "大分", "abukuma_akutsu": "福島"}
HAND_DISPLAY_MAX = 15.0       # m; cells that can show inundation in the GUI
HAND_DAMAGE_MAX = 10.0        # m; floodplain definition for inundation / damage statistics (docs/04)
DEPTH_MIN = 0.1               # m; inundation threshold for damage / extent
FLOOR_HEIGHT = 0.45           # m; manual default


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def event_label(ev):
    return f"{ev.name}（{pd.Timestamp(ev.eval_start):%Y/%m/%d}〜, {ev.role}）"


class BasinSession:
    def __init__(self, key: str):
        self.key = key
        self.d, self.meta, self.b = build_preset(key, use_cache=True)
        apath = CACHE / f"assets_{key}.pkl"
        if apath.exists():
            self.assets = pickle.load(open(apath, "rb"))
        else:
            self.assets = build_assets(self.d, self.meta["landuse_fraction"], self.b["cfg"].bbox,
                                       PREFECTURE.get(key, "福岡"))
            CACHE.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.assets, open(apath, "wb"))
        self.gs = D.ground_slope_from_dem(self.d.z, self.d.dx, self.d.dy)
        self.hand = self.meta["hand"]
        self.flood_cells = np.flatnonzero(self.d.mask & (np.nan_to_num(self.hand, nan=1e9)
                                                          <= HAND_DISPLAY_MAX))
        self.class_names = [c.name for c in self.d.params.land_classes]
        self.events = [e for e in EVENTS.values() if e.basin == key]
        self._prep = {}
        self._baseline = {}
        self.lock = threading.Lock()

    # ----------------------------------------------------------- static data
    def static_payload(self) -> dict:
        d, r = self.d, self.d.river
        ny, nx = d.shape
        value = D.asset_value(self.assets)
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
            "asset_value": b64(value, "<f4"),
            "asset_total_oku": float(value.sum() / 1e8),
            "households_total": float(self.assets.households.sum()),
            "gauges": {g: int(k) for g, k in self.meta["gauge_index"].items()},
            "events": [{"key": e.name, "label": event_label(e), "role": e.role,
                        "eval_start": e.eval_start, "eval_end": e.eval_end} for e in self.events],
            "preset": PRESETS.get(self.key, {}),
        }

    # -------------------------------------------------------------- spin-up
    def prepare(self, event_key, progress=None):
        if event_key in self._prep:
            return self._prep[event_key]
        ev = EVENTS[event_key]
        b = self.b
        tic = time.time()
        log.info("prepare %s: collecting rain", event_key)
        T0 = pd.Timestamp(ev.start)
        Te = pd.Timestamp(ev.eval_start)
        start = str(T0 + pd.Timedelta(hours=1))
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
            sim = Simulation(self.d, rain0)
            sim.run((Te - T0).total_seconds(), output_interval=6 * 3600,
                    progress=(lambda f: progress(0.5 * f, "スピンアップ計算中")) if progress else None)
            state = sim.get_state()
            pickle.dump(state, open(spath, "wb"))
        log.info("prepare %s: spin-up ready (%.1fs)", event_key, time.time() - tic)
        rain_ev, fields = gridded_rainfall(self.d, data, stations, Te)
        t_end = (pd.Timestamp(ev.eval_end) - Te).total_seconds()
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
                "basin_rain": basin_rain}
        self._prep[event_key] = prep
        return prep

    # ------------------------------------------------------------ scenarios
    def apply_edits(self, edits, gparams):
        d = self.d.copy()
        assets = copy.deepcopy(self.assets)
        floor = np.full(d.shape, FLOOR_HEIGHT)
        storage = np.zeros(d.shape, bool)       # planned flood storage (遊水地), not damage
        ny, nx = d.shape
        r = d.river
        riv_flat = r.ri * nx + r.rj
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
        notes = []
        for e in edits:
            cells = np.asarray(e.get("cells", []), dtype=np.int64)
            cells = cells[(cells >= 0) & (cells < ny * nx)]
            m = np.zeros(ny * nx, bool)
            m[cells] = True
            m = m.reshape(d.shape) & d.mask
            t = e["type"]
            if t == "forest":
                sel = m & ~np.isin(d.landclass, [ci["water"], ci["forest"]])
                d.landclass[sel] = ci[FOREST_NEW]
                assets.paddy_area_m2[sel] = 0.0
                assets.field_area_m2[sel] = 0.0
                notes.append(f"森林化 {sel.sum()} セル ({sel.sum() * cell_area / 1e6:.2f} km²; "
                             f"n={fn_cls.n:.2f}, 土層厚 {fn_cls.soil_depth:.1f} m)")
            elif t == "paddy_dam":
                sel = m & (d.landclass == ci["paddy"])
                d.landclass[sel] = ci[PADDY_DAM]
                notes.append(f"田んぼダム {sel.sum()} セル（水田のみ）, 貯留 {store * 100:.0f} cm, "
                           f"放流 {rel:.1f} mm/h")
            elif t == "relocate":
                hh = float(assets.households[m].sum())
                assets.households[m] = 0.0
                assets.floor_area_m2[m] = 0.0
                for k in assets.employees:
                    assets.employees[k][m] = 0.0
                notes.append(f"移転 {m.sum()} セル（約 {hh:.0f} 世帯）")
            elif t == "raise":
                hgt = float(e.get("height", 1.0))
                floor[m] += hgt
                if e.get("terrain"):
                    d.z[m] += hgt
                notes.append(f"嵩上げ +{hgt:.1f} m {m.sum()} セル" + ("（盛土）" if e.get("terrain") else "（床高）"))
            elif t == "levee":
                hgt = float(e.get("height", 3.0))
                sel = m.ravel()[riv_flat]
                r.embankment[sel] = np.maximum(r.embankment[sel], hgt)
                notes.append(f"堤防 {hgt:.1f} m 河道 {sel.sum()} セル")
            elif t == "retention":
                dep = float(e.get("depth", 2.0))
                inlet = float(e.get("inlet", 1.0))
                is_riv = np.zeros(ny * nx, bool)
                is_riv[riv_flat] = True
                is_riv = is_riv.reshape(d.shape)
                exc = m & ~is_riv
                d.z[exc] -= dep
                # 遊水地用地: land acquired -> no exposed assets; flooding there is by design
                storage |= m
                assets.households[m] = 0.0
                assets.floor_area_m2[m] = 0.0
                assets.paddy_area_m2[m] = 0.0
                assets.field_area_m2[m] = 0.0
                for k in assets.employees:
                    assets.employees[k][m] = 0.0
                # river cells inside or adjacent to the basin act as overflow weir (越流堤)
                near = m.copy()
                near[1:] |= m[:-1]; near[:-1] |= m[1:]; near[:, 1:] |= m[:, :-1]; near[:, :-1] |= m[:, 1:]
                sel = near.ravel()[riv_flat]
                r.embankment[sel] = inlet
                notes.append(f"遊水地 {exc.sum()} セル（{exc.sum() * cell_area / 1e6:.2f} km², 掘削 {dep:.1f} m,"
                           f" 越流堤 {inlet:.1f} m, 河道 {sel.sum()} セル）")
        return d, assets, floor, notes, storage

    def run(self, event_key, edits=None, gparams=None, progress=None):
        edits = edits or []
        gparams = gparams or {}
        prep = self.prepare(event_key, progress)
        is_base = not edits
        if is_base and event_key in self._baseline:
            return self._baseline[event_key]
        d, assets, floor, edit_log, storage = self.apply_edits(edits, gparams)
        sim = Simulation(d, prep["rain"])
        sim.set_state(prep["state"])
        fc = self.flood_cells

        def snap(s):
            # water intentionally held in 田んぼダム plots is not inundation
            return (np.maximum(s.h - s.da - s.sd, 0.0).ravel()[fc].astype(np.float32),
                    s.hr.astype(np.float32))

        t0 = time.time()
        log.info("run %s: %d edits, simulating %.0f h", event_key, len(edits), prep["t_end"] / 3600)
        res = sim.run(prep["t_end"], gauges=self.meta["gauge_index"], output_interval=3600.0,
                      snapshot_interval=3600.0, snapshot_fn=snap,
                      progress=(lambda f: progress(0.5 + 0.5 * f, "洪水計算中")) if progress else None)
        wall = time.time() - t0
        log.info("run %s: done in %.1fs (%d steps)", event_key, wall, res.n_steps)
        surf = np.maximum(np.nan_to_num(res.max_depth) - sim.sd, 0.0)
        retention_mcm = float(surf[storage].sum() * d.cell_area / 1e6)
        depth = floodplain_inundation(surf, self.hand, DEPTH_MIN, HAND_DAMAGE_MAX)
        disp = floodplain_inundation(surf, self.hand, DEPTH_MIN, HAND_DISPLAY_MAX)
        disp[storage] = surf[storage]   # shown in the map (遊水地 filling) ...
        depth[storage] = 0.0            # ... but not counted as inundation / damage
        dmg = D.estimate_damage(depth, assets, ground_slope=self.gs, floor_height=floor)
        frames_d = np.stack([s[1][0] for s in res.snapshots])     # (nt, nflood)
        frames_r = np.stack([s[1][1] for s in res.snapshots])     # (nt, nriv)
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
            "gauges": {g: {"sim": np.asarray(res.gauge_q[g][1:], float).round(2).tolist(),
                           "obs": [None if not np.isfinite(v) else round(float(v), 2)
                                   for v in prep["obs"].get(g, [])][1:]}
                       for g in self.meta["gauge_index"]},
        }
        if is_base:
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
                job["result"] = s.run(event, edits, gparams, progress)
                job["status"], job["progress"], job["message"] = "done", 1.0, "完了"
            except Exception as e:  # report to UI
                job["status"], job["message"] = "error", f"{e}\n{traceback.format_exc()}"

        self.pool.submit(work)
        return jid
