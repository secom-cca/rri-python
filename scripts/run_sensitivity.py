"""Sensitivity analysis of an RDM run (docs/10_sensitivity.md).

    python scripts/run_sensitivity.py abukuma_quick --quick     # Morris r=2, GP 10+5 points
    python scripts/run_sensitivity.py abukuma_trial              # settings of the run's config["sensitivity"]
    python scripts/run_sensitivity.py abukuma_trial --config sa.json   # overrides, e.g. parameter ranges

Reads data/processed/rdm/<run>/results.json (scripts/run_rdm.py first), re-uses
the run's simulation cache and writes config_sensitivity.json + sensitivity.json
next to it (shown in the GUI page /rdm).
"""
import argparse
import json

from rri_sim.rdm import config as CF
from rri_sim.rdm import sensitivity as SA
from rri_sim.rdm.run import RDM_DIR

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--config", help="JSON overrides merged over the run's config (e.g. {'sensitivity': {...}})")
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
over = json.load(open(a.config, encoding="utf-8")) if a.config else {}
if a.quick:
    over = CF._merge(CF.QUICK_SA, over)
SA.run(RDM_DIR / a.run, over, log=lambda m: print(m, flush=True))
