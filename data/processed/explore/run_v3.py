"""Exploration v3 (docs/12): the v2 settings with the 遊水地 option replaced by the planned 阿武隈川上流遊水地群
(one variable, rdm config candidates.retention.sites).  v2's stored config is used (the defaults have changed since)."""
import copy, json
from rri_sim.rdm import config as CF, explore as EX
from rri_sim.data import PROCESSED
cfg = json.loads((PROCESSED / "explore" / "abukuma_explore_v2" / "config_explore.json").read_text(encoding="utf-8"))
cfg["candidates"]["retention"] = copy.deepcopy(CF.DEFAULT["candidates"]["retention"])
cfg["explore"]["name"] = "abukuma_explore_v3"
EX.CF.load = lambda c=None: copy.deepcopy(c)      # no merge over the current defaults (exactly v2 + the new 遊水地)
EX.run(cfg, log=lambda m: print(m, flush=True))
