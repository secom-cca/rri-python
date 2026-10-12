"""Run one registered flood event and score it against observed discharge.

python scripts/run_event.py kusu_2018 --tag base --set forest.n=0.2 river.cw=8
"""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from rri_sim import metrics, viz
from rri_sim.cases import BASINS, EVENTS
from rri_sim.data import ROOT, mlit_water
from rri_sim.events import basin_mean_series, collect_gauges, gridded_rainfall
from rri_sim.workflow import apply_overrides, boundary_conditions, build, drainage_arrays, PRESETS  # noqa: F401
from rri_sim.solver import Simulation
from rri_sim import climate as CL


def run_event(event_key, tag="base", sets=None, threshold=None, save=True, verbose=True,
              climate=None):
    """climate: rri_sim.climate.ClimateScenario applied from eval_start (spin-up unchanged
    unless climate.apply_to_spinup)."""
    ev = EVENTS[event_key]
    d, meta, b = build(ev.basin, sets, threshold)
    T0 = pd.Timestamp(ev.start)
    start = str(T0 + pd.Timedelta(hours=1))
    data, stations = collect_gauges(b["cfg"].bbox, start, ev.end, mlit_suikei=b["mlit_suikei"],
                                    jma_prec_nos=b["jma_prec_nos"],
                                    cache_tag=f"{ev.basin}_{T0:%Y%m%d}_{pd.Timestamp(ev.end):%Y%m%d}",
                                    verbose=verbose)
    rain, fields = gridded_rainfall(d, data, stations, T0)
    bm = basin_mean_series(fields, d.mask)
    dams, stage_bc, bc_info = boundary_conditions(ev.basin, d, meta, T0, ev.end)
    if climate is not None and isinstance(climate.factor_field, str):
        climate.factor_field = CL.load_factor_raster(climate.factor_field, d)
    if climate is not None and not climate.is_current:
        t_from = -np.inf if climate.apply_to_spinup else (pd.Timestamp(ev.eval_start) - T0).total_seconds()
        rain = CL.scale_rainfall(rain, climate, t_start=t_from)
        stage_bc = CL.shift_stage(stage_bc, climate)
        bm = basin_mean_series(rain.rates * 3.6e6, d.mask)
    sim = Simulation(d, rain, dams=dams, stage_bc=stage_bc, drainage=drainage_arrays(ev.basin, d, meta))
    t_end = (pd.Timestamp(ev.end) - T0).total_seconds()
    t = time.time()
    res = sim.run(t_end, gauges=meta["gauge_index"], output_interval=3600.0)
    wall = time.time() - t
    times = T0 + pd.to_timedelta(res.times, unit="s")
    w = (times >= ev.eval_start) & (times <= ev.eval_end)
    scores = {}
    obs_all = {}
    for g, sid in b["discharge_gauges"].items():
        obs = mlit_water.hourly(sid, mlit_water.KIND_DISCHARGE, start, ev.end).reindex(times)
        obs_all[g] = obs.values
        scores[g] = metrics.hydrograph_scores(times.values[w], obs.values[w], res.gauge_q[g][w])
    rain_t = T0 + pd.to_timedelta(rain.edges[:-1], unit="s")
    rain_on_t = pd.Series(bm, index=rain_t).reindex(times).values
    summary = {"event": event_key, "role": ev.role, "tag": tag, "overrides": sets or [],
               "threshold": float(b["cfg"].river_threshold_km2),
               "cells": int(d.mask.sum()), "river_cells": d.river.n, "steps": res.n_steps,
               "wall_s": wall, "mass_rel_err": res.mass.relative_error,
               "rain_eval_mm": float(np.nansum(rain_on_t[w])), "scores": scores,
               "area_km2": meta["area_km2"], "boundary": bc_info,
               "dam_max_storage_mcm": {dm.name: None for dm in dams},
               "climate": climate.to_dict() if climate is not None else None}
    if verbose:
        print(event_key, tag, {g: {k: round(v, 3) for k, v in s.items()} for g, s in scores.items()},
              f"wall={wall:.0f}s")
    if save:
        out = ROOT / "outputs" / "events" / event_key / tag
        out.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
        for g in b["discharge_gauges"]:
            sc = scores[g]
            viz.hydrograph(times[w], {"RRI-python": res.gauge_q[g][w]}, obs_all[g][w],
                           rain=rain_on_t[w],
                           title=f"{event_key} {g} ({ev.role}) NSE={sc['NSE']:.2f} "
                                 f"KGE={sc['KGE']:.2f} [{tag}]",
                           path=out / f"hydrograph_{g}.png")
        viz.depth_map(res.max_depth, d, title=f"max depth {event_key} [{tag}]",
                      path=out / "max_depth.png")
        with open(out / "result.pkl", "wb") as f:
            pickle.dump({"times": times, "gauge_q": res.gauge_q, "gauge_h": res.gauge_h,
                         "obs": obs_all, "max_depth": res.max_depth, "rain_bm": bm,
                         "eval_mask": w}, f)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--preset", action="store_true", help="use workflow.PRESETS for the basin")
    ap.add_argument("--climate", choices=["current", "2C", "4C", "custom"], default="current",
                    help="MLIT/JMA preset (2C, 4C) or custom with --rain-factor/--gamma/--slr")
    ap.add_argument("--rain-factor", type=float)
    ap.add_argument("--gamma", type=float, help="intensity exponent (>1 concentrates the increase)")
    ap.add_argument("--slr", type=float, help="sea-level rise [m] added to the tide boundary")
    ap.add_argument("--climate-spinup", action="store_true", help="scale spin-up rain too")
    ap.add_argument("--factor-raster", help="GeoTIFF of spatial rainfall change factors (e.g. from d4PDF)")
    a = ap.parse_args()
    _rr, _sr = BASINS[EVENTS[a.event].basin].get("climate_region", ("その他", "I"))
    clim = CL.from_params({"preset": a.climate, "rain_factor": a.rain_factor, "intensity_gamma": a.gamma,
                           "sea_level_rise_m": a.slr, "apply_to_spinup": a.climate_spinup}, _rr, _sr)
    sets, thr = a.set, a.threshold
    if a.preset:
        p = PRESETS.get(EVENTS[a.event].basin, {})
        sets, thr = (sets or []) + p.get("sets", []), thr or p.get("threshold")
    if a.factor_raster:
        clim.name, clim.source = "custom", f"factor raster {a.factor_raster}"
        clim.factor_field = a.factor_raster        # resolved to the grid inside run_event
    run_event(a.event, a.tag, sets, thr, climate=clim)
