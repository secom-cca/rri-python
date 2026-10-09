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
from rri_sim.basin import build_basin, default_land_params
from rri_sim.cases import BASINS, EVENTS
from rri_sim.data import ROOT, mlit_water
from rri_sim.events import basin_mean_series, collect_gauges, gridded_rainfall
from rri_sim.params import ModelParams
from rri_sim.solver import Simulation


def apply_overrides(params, sets):
    names = {c.name: c for c in params.land_classes}
    for s in sets or []:
        key, val = s.split("=")
        obj, attr = key.split(".")
        if obj == "all":                       # apply to every land class having attr>0
            for c in params.land_classes:
                setattr(c, attr, float(val))
            continue
        target = {"river": params.river, "num": params.numerics}.get(obj) or names[obj]
        cur = getattr(target, attr)
        setattr(target, attr, type(cur)(val) if not isinstance(cur, bool) else val == "True")


def build(basin_key, sets=None, threshold=None):
    b = BASINS[basin_key]
    cfg = b["cfg"]
    if threshold is not None:
        cfg.river_threshold_km2 = threshold
    params = ModelParams(land_classes=default_land_params())
    apply_overrides(params, sets)
    d, meta = build_basin(cfg, params, verbose=False)
    return d, meta, b


def run_event(event_key, tag="base", sets=None, threshold=None, save=True, verbose=True):
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
    sim = Simulation(d, rain)
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
               "rain_eval_mm": float(np.nansum(rain_on_t[w])), "scores": scores}
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
    a = ap.parse_args()
    run_event(a.event, a.tag, a.set, a.threshold)
