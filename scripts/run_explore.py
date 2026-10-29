"""RDM exploration without stakeholder weights (docs/12_explore.md).

    python scripts/run_explore.py --quick                   # small design (about 100 RRI runs, 阿武隈川全域 1000 m)
    python scripts/run_explore.py                           # full design (about 450 RRI runs)
    python scripts/run_explore.py --config my.json          # settings merged over rri_sim/rdm/config.py DEFAULT
    python scripts/run_explore.py --print-default > my.json

Results: data/processed/explore/<explore.name>/{config_explore.json, evaluations.jsonl, results.json};
GUI: http://127.0.0.1:8765/explore .  Re-runs re-use the simulation cache.
"""
import argparse
import json
import sys

from rri_sim.rdm import config as CF
from rri_sim.rdm import explore as EX

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--name")
ap.add_argument("--print-default", action="store_true")
a = ap.parse_args()
if a.print_default:
    json.dump(CF.DEFAULT, sys.stdout, ensure_ascii=False, indent=2)
    sys.exit()
cfg = CF.load(a.config) if a.config else CF.load()
if a.quick:
    cfg = CF._merge(cfg, CF.QUICK_EX)
if a.name:
    cfg["explore"]["name"] = a.name
EX.run(cfg, log=lambda m: print(m, flush=True))
