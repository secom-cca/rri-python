"""Adaptive DAPP with forecasts and learning (docs/11 11.12; rri_sim.dapp.forecast_dapp).

    python scripts/run_forecast_dapp.py                         # settings: rri_sim/dapp/config.py DEFAULT["forecast"]
    python scripts/run_forecast_dapp.py --quick                 # small scenario sets and grids (check)
    python scripts/run_forecast_dapp.py --config my.json --name abukuma_dapp_forecast_v5b

Needs the exploration surrogate (forecast.explore_run, default abukuma_explore_v5) and its cost table
(`python scripts/run_roadmap.py --prepare` with physics.explore_run = that run).
Results: data/processed/dapp/<forecast.name>/{config.json, results.json}.
"""
import argparse
import json

from rri_sim.dapp import config as CF
from rri_sim.dapp import forecast_dapp as FD

QUICK = {"forecast": {"name": "abukuma_dapp_forecast_quick", "scenarios": {"n_eval": 48, "n_search": 16, "seeds": [11, 12]},
                      "initial": {"pop": 32, "gens": 10, "n_max": 6},
                      "policy": {"rho_grid": [0.7, 0.9, 1.1], "lookahead_grid": [10, 30], "k_grid": [1], "wait_grid": [0, 1]}}}

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--name")
a = ap.parse_args()
over = json.loads(open(a.config, encoding="utf-8").read()) if a.config else {}
if a.quick:
    over = CF._merge(QUICK, over)
cfg = CF.load(over)
if a.name:
    cfg["forecast"]["name"] = a.name
FD.run(cfg, log=lambda m: print(m, flush=True))
