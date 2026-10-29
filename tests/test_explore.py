"""Exploration without weights (rri_sim/rdm/explore.py) on an analytic stand-in for RRI."""
import json

import numpy as np

from rri_sim.rdm import candidates as CA
from rri_sim.rdm import config as CF
from rri_sim.rdm import explore as EX


class FakeEvaluator:
    """Damage falls with levees / paddy-dam rate / retention, rises with rain; zones 上流 / 下流."""

    def __init__(self):
        self.variables = [
            CA.Variable("堤防・下流", "levee", "下流", ["なし", "+1 m", "+2 m"], [1], [None, {"height": 1.0}, {"height": 2.0}]),
            CA.Variable("遊水地1・下流", "retention", "下流", ["なし", "3 m"], [2], [None, {"depth": 3.0}]),
            CA.Variable("田んぼダム・上流", "paddy_dam", "上流", ["なし", "全水田"], list(range(10, 30)), [None, {}]),
            CA.Variable("森林化・上流", "forest", "上流", ["なし", "全域"], list(range(30, 40)), [None, {}]),
            CA.Variable("住宅対策・下流", "relocate", "下流", ["なし", "嵩上げ", "移転"], [3],
                        [None, {"type": "raise", "height": 1.0}, {"type": "relocate", "dest": "new"}]),
        ]
        self.n_sim = 0

    def cost(self, x):
        c = {"上流": 40 * float(x[2]) + 300 * float(x[3]), "下流": [0, 300, 700][int(x[0])] + 500 * int(x[1]) + [0, 200, 800][int(x[4])]}
        return {"total": sum(c.values()), "by_zone": c, "by_type": {}}

    def simulate(self, x, sim, strict=True):
        self.n_sim += 1
        k = sim.get("rain_factor", 1.0) ** 3 * sim.get("rain_gamma", 1.0)
        up = 200 * k * (1 - 0.4 * float(x[2]) * sim.get("paddy_effect", 1.0)) * (1 - 0.1 * float(x[3]) * sim.get("forest_effect", 1.0))
        lv = [0, 1, 2][int(x[0])] * sim.get("levee_effect", 1.0)
        down = 500 * k * (1 - 0.2 * lv) * (1 - 0.3 * int(x[1]) * sim.get("retention_effect", 1.0)) * (1 - 0.3 * (int(x[4]) > 0))
        return {"damage": up + down, "damage_zone": {"上流": up, "下流": down}, "households": 3 * (up + down),
                "livability": 60.0, "ecology": 60 + 3 * float(x[3]) - 0.5 * lv, "pop_hazard": 5 * (up + down),
                "inundated_km2": 1.0, "peak_q": 1.0}


def test_continuous_variable_takes_a_share_of_cells():
    v = FakeEvaluator().variables[2]
    v.continuous = True
    e = CA.edits([v], [0.25])
    assert len(e) == 1 and len(e[0]["cells"]) == 5
    assert CA.edits([v], [0.0]) == [] and CA.describe([v], [0.5]) == ["田んぼダム・上流: 50%"]
    assert "continuous" not in FakeEvaluator().variables[0].to_dict()      # integer variables: unchanged dict / cache keys


def test_explore_runs_and_finds_policies_meeting_the_target(tmp_path):
    cfg = CF.load({"explore": {"design": {"n_train": 40, "n_none": 4, "n_val": 12, "adaptive_rounds": 1, "adaptive_batch": 4, "pool": 300},
                               "ensemble": {"n": 128}, "optimisation": {"pop": 16, "gens": 6},
                               "verify": {"n_policies": 3, "n_scenarios": 4}}})
    res = EX.run(cfg, tmp_path, evaluator=FakeEvaluator(), log=lambda *a: None)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["reference"]["damage"] == 700.0
    P = saved["policies"]
    assert P[0]["desc"] == ["施策なし"]
    rel = [p["reliability"] for p in P]
    assert max(rel) > P[0]["reliability"]                # measures raise the share of scenarios meeting the target
    assert all(0 <= p["reliability"] <= 1 for p in P)
    assert any(0 < x < 1 for p in P for x in (p["x"][2], p["x"][3]))   # rates between 0 and 1 are used
    assert saved["surrogate"]["validation"]["damage"]["r2"] > 0.8
    assert len(saved["verification"]["policies"]) >= 2
    assert len(P[0]["ratio_samples"]) == 128 and len(P[0]["reliability_curve"]) == len(cfg["explore"]["target_curve"])
    res2 = EX.run(cfg, tmp_path / "again", evaluator=FakeEvaluator(), log=lambda *a: None)
    assert res2["pareto"] == res["pareto"]                # reproducible with the seed
