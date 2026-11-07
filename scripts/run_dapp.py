"""Dynamic Adaptive Policy Pathways analysis (docs/11_dapp.md).

    python scripts/run_dapp.py --quick                      # 4 strategies x 3 climate paths (阿武隈川全域 1000 m)
    python scripts/run_dapp.py --config my_dapp.json        # settings merged over rri_sim/dapp/config.py DEFAULT
    python scripts/run_dapp.py --print-default > my_dapp.json
    python scripts/run_dapp.py --robust [--quick]           # Robust DAPP on the exploration surrogate (docs/11 11.9)
    python scripts/run_dapp.py --robust --verify 6          # RRI check of representative pathway states of that run
    python scripts/run_dapp.py --multi [--quick]            # Multi-Actor Robust DAPP: Individual / Joint / Single (11.10)
    python scripts/run_dapp.py --multi --verify 4           # RRI check of joint states of that run
    python scripts/run_dapp.py --adaptive [--quick]         # Adaptive DAPP: state-dependent decision rules (11.11; needs --robust)

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
ap.add_argument("--robust", action="store_true", help="Robust DAPP: pathways network on the surrogate (cfg['robust'])")
ap.add_argument("--adaptive", action="store_true", help="Adaptive DAPP (cfg['adaptive']): decision rules per initial action of the --robust run")
ap.add_argument("--multi", action="store_true", help="Multi-Actor Robust DAPP (cfg['multi']): 国・上流自治体・下流自治体")
ap.add_argument("--verify", type=int, default=0, help="with --robust / --multi: re-run RRI for N states of the representative pathways")
a = ap.parse_args()
if a.print_default:
    json.dump(CF.DEFAULT, sys.stdout, ensure_ascii=False, indent=2)
    sys.exit()
cfg = CF.load(a.config) if a.config else CF.load()
log = lambda m: print(m, flush=True)  # noqa: E731
if a.adaptive:
    from rri_sim.dapp import adaptive as AD
    if a.quick:
        cfg = CF._merge(cfg, CF.QUICK_ADAPTIVE)
    if a.name:
        cfg["adaptive"]["name"] = a.name
    AD.run(cfg, log=log)
    sys.exit()
if a.multi:
    from rri_sim.dapp import multi as MU
    if a.quick:
        cfg = CF._merge(cfg, CF.QUICK_MULTI)
    if a.name:
        cfg["multi"]["name"] = a.name
    if a.verify:
        MU.verify_rri(MU.DAPP_DIR / cfg["multi"]["name"], a.verify, log)
    else:
        MU.run(cfg, log=log)
    sys.exit()
if a.robust:
    from rri_sim.dapp import robust as RB
    if a.quick:
        cfg = CF._merge(cfg, CF.QUICK_ROBUST)
    if a.name:
        cfg["robust"]["name"] = a.name
    if a.verify:
        RB.verify_rri(RB.DAPP_DIR / cfg["robust"]["name"], a.verify, log)
    else:
        RB.run(cfg, log=log)
    sys.exit()
if a.quick:
    cfg = CF._merge(cfg, CF.QUICK)
if a.name:
    cfg["name"] = a.name
RUN.run(cfg, quick=a.quick, log=log)
