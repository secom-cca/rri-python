"""Exploration v5 (docs/12, model revision 11.12): the v4 settings on the 1K-RRI model (rri_sim.k1rri; 1 km = the 250 m model
upscaled 4 x 4), without effect coefficients (measures act with their physical parameters), with forest management of the
existing mountain forest (森林整備, one variable per zone, an implementation rate) instead of afforestation.  Runs before
(v2-v4) are kept; their surrogates still use their stored configs.

    .venv/bin/python data/processed/explore/run_v5.py
"""
import copy
import json

from rri_sim.data import PROCESSED
from rri_sim.rdm import config as CF, explore as EX

cfg = json.loads((PROCESSED / "explore" / "abukuma_explore_v4" / "config_explore.json").read_text(encoding="utf-8"))
cfg["explore"]["name"] = "abukuma_explore_v5"
P = cfg["sensitivity"]["parameters"]
for k in [k for k in P if k.endswith("_effect")]:          # no effect coefficients
    P.pop(k)
cfg["candidates"]["forest"] = None                          # afforestation: not represented by 1K-RRI (changes A_s / A_f)
cfg["candidates"]["forest_mgmt"] = copy.deepcopy(CF.DEFAULT["candidates"]["forest_mgmt"])    # split_owner False
cfg["explore"]["continuous"] = ["paddy_dam", "forest_mgmt", "storage_private"]
EX.CF.load = lambda c=None: copy.deepcopy(c)               # no merge over the current defaults
EX.run(cfg, log=lambda m: print(m, flush=True))
