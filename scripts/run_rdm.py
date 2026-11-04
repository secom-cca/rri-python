"""Robust Decision Making analysis on the RRI model (docs/09_rdm.md).

    python scripts/run_rdm.py --quick                     # trial: 阿武隈川全域 1000 m, few policies/scenarios
    python scripts/run_rdm.py --config my_rdm.json        # settings merged over rri_sim/rdm/config.py DEFAULT
    python scripts/run_rdm.py --print-default > my_rdm.json
    python scripts/run_rdm.py --surrogate                 # on the exploration surrogate: no RRI, no budget (docs/09 9.8)

Results: data/processed/rdm/<name>/{config.json, candidates.json, evaluations.jsonl, results.json};
view them in the GUI at http://127.0.0.1:8765/rdm .  Re-running with the same
name re-uses the simulation cache (evaluations.jsonl).
"""
import argparse
import json
import sys

from rri_sim.rdm import config as CF
from rri_sim.rdm import run as RUN

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--quick", action="store_true", help="small trial (pop 8, 2 generations, 3 climate scenarios)")
ap.add_argument("--name")
ap.add_argument("--print-default", action="store_true")
ap.add_argument("--surrogate", action="store_true", help="RDM on the exploration surrogate (settings: surrogate_rdm)")
a = ap.parse_args()
if a.print_default:
    json.dump(CF.DEFAULT, sys.stdout, ensure_ascii=False, indent=2)
    sys.exit()
over = CF.load(a.config) if a.config else CF.load()
if a.surrogate:
    from rri_sim.rdm import rdm_surrogate
    if a.quick:
        over = CF._merge(over, CF.QUICK_SUR)
    if a.name:
        over["surrogate_rdm"]["name"] = a.name
    rdm_surrogate.run(over, log=lambda m: print(m, flush=True))
    sys.exit()
if a.quick:
    over = CF._merge(over, CF.QUICK)
if a.name:
    over["name"] = a.name
RUN.run(over, log=lambda m: print(m, flush=True))
