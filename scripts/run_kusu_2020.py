"""Stage 3a: 玖珠川 (above 小ヶ瀬) — July 2020 Kyushu heavy rain.

Usage: python scripts/run_kusu_2020.py [--res 250] [--tag name] [--set forest.ka=0.05 ...]
"""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from rri_sim import metrics, viz
from rri_sim.basin import BasinConfig, build_basin
from rri_sim.data import PROCESSED, ROOT, mlit_water
from rri_sim.events import basin_mean_series, collect_gauges, gridded_rainfall
from rri_sim.solver import Simulation

OUT = ROOT / "outputs" / "kusu_2020"
START, END = "2020-07-01 01:00", "2020-07-10 00:00"
T0 = pd.Timestamp("2020-07-01 00:00")
GAUGES = {"ogase": ("309061289901330", (130.96444, 33.29806))}
STAGE_ONLY = {"takenaka": ("309061289920090", (131.11806, 33.27667))}


def apply_overrides(params, sets):
    names = {c.name: c for c in params.land_classes}
    for s in sets or []:
        key, val = s.split("=")
        obj, attr = key.split(".")
        target = {"river": params.river, "num": params.numerics}.get(obj) or names[obj]
        setattr(target, attr, type(getattr(target, attr))(val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=250.0)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--threshold", type=float, default=5.0)
    a = ap.parse_args()
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)

    cfg = BasinConfig(name="kusu_ogase", outlet_lonlat=GAUGES["ogase"][1],
                      bbox=(130.90, 33.05, 131.40, 33.40), crs="EPSG:6690", res=a.res,
                      river_threshold_km2=a.threshold,
                      gauges={k: v[1] for k, v in {**GAUGES, **STAGE_ONLY}.items()})
    t = time.time()
    d, meta = build_basin(cfg)
    apply_overrides(d.params, a.set)
    print(f"domain {d.shape}, {d.mask.sum()} cells, {d.river.n} river cells, "
          f"build {time.time()-t:.1f}s")

    data, stations = collect_gauges(cfg.bbox, START, END, mlit_suikei="89094000",
                                    jma_prec_nos=(82, 83, 86), cache_tag="kusu_2020")
    rain, fields = gridded_rainfall(d, data, stations, T0)
    bm = basin_mean_series(fields, d.mask)
    print(f"basin-mean rain total {bm.sum():.0f} mm, max {bm.max():.1f} mm/h")

    sim = Simulation(d, rain)
    t_end = (pd.Timestamp(END) - T0).total_seconds()
    gidx = meta["gauge_index"]
    t = time.time()
    res = sim.run(t_end, gauges=gidx, output_interval=3600.0, snapshot_interval=3600.0)
    wall = time.time() - t
    mb = res.mass
    print(f"run: {res.n_steps} steps, wall {wall:.1f}s, mass rel.err {mb.relative_error:.2e}")

    times = T0 + pd.to_timedelta(res.times, unit="s")
    obs = mlit_water.hourly(GAUGES["ogase"][0], mlit_water.KIND_DISCHARGE, START, END)
    obs = obs.reindex(times)
    q = res.gauge_q["ogase"]
    sc = metrics.hydrograph_scores(times.values, obs.values, q)
    # event window scores
    w = (times >= "2020-07-06") & (times <= "2020-07-10")
    scw = metrics.hydrograph_scores(times.values[w], obs.values[w], q[w])
    print("scores (full):", {k: round(v, 3) for k, v in sc.items()})
    print("scores (7/6-7/10):", {k: round(v, 3) for k, v in scw.items()})

    rain_t = T0 + pd.to_timedelta(rain.edges[:-1], unit="s")
    rain_on_t = pd.Series(bm, index=rain_t).reindex(times).values
    viz.hydrograph(times, {"RRI-python": q}, obs.values, rain=rain_on_t,
                   title=f"玖珠川 小ヶ瀬 2020/7  NSE={sc['NSE']:.2f} KGE={sc['KGE']:.2f}",
                   path=out / "hydrograph_ogase.png")
    viz.depth_map(res.max_depth, d, title="max inundation depth (2020/7, 玖珠川)",
                  path=out / "max_depth.png")
    summary = {"tag": a.tag, "res": a.res, "cells": int(d.mask.sum()), "river_cells": d.river.n,
               "area_km2": meta["area_km2"], "steps": res.n_steps, "wall_s": wall,
               "mass_rel_err": mb.relative_error, "rain_total_mm": float(bm.sum()),
               "scores_full": sc, "scores_event": scw, "overrides": a.set or [],
               "rain_gauges": int(len(stations))}
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    with open(out / "result.pkl", "wb") as f:
        pickle.dump({"times": times, "q": q, "obs": obs.values, "max_depth": res.max_depth,
                     "hr_ogase": res.gauge_h["ogase"], "hr_takenaka": res.gauge_h["takenaka"],
                     "q_takenaka": res.gauge_q["takenaka"], "rain_bm": bm}, f)


if __name__ == "__main__":
    main()
