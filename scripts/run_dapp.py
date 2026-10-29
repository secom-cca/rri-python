"""Dynamic Adaptive Policy Pathways analysis (docs/11_dapp.md).

    python scripts/run_dapp.py --quick                      # 4 strategies x 3 climate paths (阿武隈川全域 1000 m)
    python scripts/run_dapp.py --config my_dapp.json        # settings merged over rri_sim/dapp/config.py DEFAULT
    python scripts/run_dapp.py --print-default > my_dapp.json

Results: data/processed/dapp/<name>/{config.json, evaluations.jsonl, results.json};
GUI: http://127.0.0.1:8765/dapp .  Re-runs re-use the simulation cache.
"""
import argparse
import json
import sys

from rri_sim.dapp import config as CF
from rri_sim.dapp import run as RUN

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--quick", action="store_true", help="only cfg['quick_strategies']")
ap.add_argument("--name")
ap.add_argument("--print-default", action="store_true")
a = ap.parse_args()
if a.print_default:
    json.dump(CF.DEFAULT, sys.stdout, ensure_ascii=False, indent=2)
    sys.exit()
cfg = CF.load(a.config) if a.config else CF.load()
if a.quick:
    cfg = CF._merge(cfg, CF.QUICK)
if a.name:
    cfg["name"] = a.name
RUN.run(cfg, quick=a.quick, log=lambda m: print(m, flush=True))
