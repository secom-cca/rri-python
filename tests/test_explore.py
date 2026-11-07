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


def test_trade_spaces_by_actor(tmp_path):
    from rri_sim.rdm import explore_scope as ES
    cfg = CF.load({"explore": {"design": {"n_train": 40, "n_none": 4, "n_val": 8, "adaptive_rounds": 0, "pool": 100},
                               "ensemble": {"n": 64}, "optimisation": {"pop": 12, "gens": 3},
                               "verify": {"n_policies": 2, "n_scenarios": 2}}})
    EX.run(cfg, tmp_path, evaluator=FakeEvaluator(), log=lambda *a: None)
    doc = ES.run("fake", {"explore": {"scopes": {"optimisation": {"pop": 10, "gens": 3}, "max_points": 60}}},
                 log=lambda *a: None, out_dir=tmp_path)
    assert doc["actors"] == ["国", "下流自治体", "地域主体"]        # 住宅対策 is municipal: 下流 zone -> 下流自治体
    assert len(doc["scopes"]) == 7
    P = doc["points"]
    rel = np.frombuffer(__import__("base64").b64decode(P["rel"]), np.uint8).reshape(P["n"], *P["rel_shape"])
    assert rel.shape[1:] == (len(cfg["explore"]["target_curve"]), cfg["explore"]["rain_bins"] + 1)
    for x, need, cost, ca in zip(P["x"], P["need"], P["cost"], P["cost_actor"]):
        used = {doc["actor_of"][i] for i, a in enumerate(x) if a > 0}
        assert need == sum(1 << doc["actors"].index(a) for a in used)          # actors a policy needs
        assert abs(sum(ca) - cost) < 0.5
        assert abs(cost - FakeEvaluator().cost(x)["total"]) < 0.5             # same costs as the exploration
    kuni = [i for i, n in enumerate(P["need"]) if n & ~1 == 0]
    assert any(P["x"][i][0] > 0 for i in kuni) and all(P["x"][i][2] == 0 for i in kuni)   # 国: levee / retention only


def test_rdm_on_the_surrogate(tmp_path):
    from rri_sim.rdm import rdm_surrogate as RS
    cfg = CF.load({"explore": {"design": {"n_train": 40, "n_none": 4, "n_val": 8, "adaptive_rounds": 0, "pool": 100},
                               "ensemble": {"n": 64}, "optimisation": {"pop": 12, "gens": 3},
                               "verify": {"n_policies": 2, "n_scenarios": 2}}})
    EX.run(cfg, tmp_path / "ex", evaluator=FakeEvaluator(), log=lambda *a: None)
    rc = CF.load({"zones": [["上流", None], ["下流", None]],
                  "stakeholders": {"国": CF.DEFAULT["stakeholders"]["国"],
                                   "下流自治体": CF.DEFAULT["stakeholders"]["下流自治体"]},
                  "surrogate_rdm": {"explore_dir": str(tmp_path / "ex"), "n_scenarios": 64, "n_display": 6,
                                    "optimisation": {"pop": 8, "gens": 2, "n_scenarios": 16},
                                    "sobol": {"n_base": 32, "pool_per_stakeholder": 20}}})
    res = RS.run(rc, log=lambda *a: None, out_dir=tmp_path / "rdm", evaluator=FakeEvaluator())
    saved = json.loads((tmp_path / "rdm" / "results.json").read_text())
    assert saved["mode"] == "surrogate" and saved["config"]["budget_oku"] is None
    P, rob = saved["policies"], saved["robustness"]
    assert P[0]["label"] == "施策なし" and len(P) == 6
    for n, s in rob["stakeholders"].items():
        R = np.array(s["regret"])
        assert R.shape == (6, 64) and (R >= -1e-6).all()                     # regret against the best of the whole pool
        assert s["summary"]["max"][s["minimax"]] <= min(s["summary"]["max"]) + 1e-9
    assert rob["overall"]["minimax"] == 1
    assert all(len(v["cells"]) > 0 for v in saved["variables"])               # map cells from the evaluator
    a = saved["archive"][P[0]["archive"]]
    assert a["cost"] == 0 and abs(a["damage"] - 700) < 70                      # no measures, nominal scenario ~ RRI reference
    assert set(saved["sobol"]) == {"被害額（総合推奨案）", "Regret・国（国の推奨案）", "Regret・下流自治体（下流自治体の推奨案）"}


class SplitEvaluator(FakeEvaluator):
    """Housing as separate 移転 / 嵩上げ variables (same cells), 森林整備 as a rate, a jointly implemented
    農業用排水機場; the RRI-side cost of raising is zero while the same homes are relocated."""

    def __init__(self):
        self.variables = [
            CA.Variable("堤防・下流", "levee", "下流", ["なし", "+1 m", "+2 m"], [1], [None, {"height": 1.0}, {"height": 2.0}]),
            CA.Variable("移転・下流", "relocate", "下流", ["なし", "移転"], [3, 4], [None, {"type": "relocate", "dest": "new"}]),
            CA.Variable("嵩上げ・下流", "raise", "下流", ["なし", "+1 m"], [3, 4], [None, {"type": "raise", "height": 1.0}]),
            CA.Variable("森林整備（民有林等）・上流", "forest_mgmt", "上流", ["なし", "全域"], list(range(30, 40)),
                        [None, {"type": "forest_mgmt", "owner": "other"}]),
            CA.Variable("排水機場（農業）・上流", "pump_agri", "上流", ["なし", "+3 mm/h"], [5], [None, {"type": "pump"}]),
        ]
        self.n_sim = 0

    def cost(self, x):
        reloc = int(x[1]) > 0
        c = {"上流": 100 * float(x[3]) + 200 * int(x[4]),
             "下流": [0, 300, 700][int(x[0])] + 800 * reloc + (0 if reloc else 150 * int(x[2]))}
        return {"total": sum(c.values()), "by_zone": c, "by_type": {}}

    def simulate(self, x, sim, strict=True):
        self.n_sim += 1
        k = sim.get("rain_factor", 1.0) ** 3
        reloc, raised = int(x[1]) > 0, int(x[2]) > 0 and int(x[1]) == 0
        up = 200 * k * (1 - 0.2 * float(x[3]) * sim.get("forest_mgmt_effect", 1.0)) * (1 - 0.2 * int(x[4]))
        down = 500 * k * (1 - 0.2 * int(x[0]) * sim.get("levee_effect", 1.0)) * (1 - 0.5 * reloc) * (1 - 0.3 * raised * sim.get("raise_effect", 1.0))
        return {"damage": up + down, "damage_zone": {"上流": up, "下流": down}, "households": 3 * (up + down),
                "livability": 60.0, "ecology": 60.0, "pop_hazard": 5 * (up + down), "inundated_km2": 1.0, "peak_q": 1.0,
                "relocated_households": 40.0 * reloc, "relocated_population": 100.0 * reloc, "service_access": 0.7 - 0.01 * reloc,
                "natural_habitat_km2": 50.0, "habitat_connectivity": 0.4 + 0.01 * float(x[3])}


def test_split_housing_forest_management_and_shared_actors(tmp_path):
    from rri_sim.rdm import explore_scope as ES
    V = SplitEvaluator().variables
    assert CA.superseded_raise(V) == [(1, 2)]
    V[3].continuous = True
    e = CA.edits(V, [0, 1, 1, 0.5, 0], {"forest_mgmt_effect": 0.5})
    assert [x["type"] for x in e] == ["relocate", "raise", "forest_mgmt"] and len(e[2]["cells"]) == 5   # relocation first
    cfg = CF.load({"explore": {"design": {"n_train": 40, "n_none": 4, "n_val": 8, "adaptive_rounds": 0, "pool": 100},
                               "ensemble": {"n": 64}, "optimisation": {"pop": 12, "gens": 4},
                               "verify": {"n_policies": 2, "n_scenarios": 2}}})
    assert cfg["explore"]["optimisation"]["objectives"] == ["cost", "damage_mean", "reliability"]
    res = EX.run(cfg, tmp_path, evaluator=SplitEvaluator(), log=lambda *a: None)
    assert [v["continuous"] for v in res["variables"]] == [False, False, False, True, False]
    for p in res["policies"]:
        assert abs(p["cost"] - SplitEvaluator().cost(p["x"])["total"]) < 1e-6      # raise not paid when relocated
        assert set(p["sub"]) == set(EX.SUB_KEYS)
    assert res["reference"]["sub"]["relocated_households"] == 0.0
    doc = ES.run("fake", {"explore": {"scopes": {"optimisation": {"pop": 10, "gens": 3}, "max_points": 60}}},
                 log=lambda *a: None, out_dir=tmp_path)
    A = doc["actors"]
    assert A == ["国", "上流自治体", "下流自治体", "地域主体"]
    assert doc["actor_shares"][4] == {"国": 0.6, "上流自治体": 0.25, "地域主体": 0.15}   # 排水機場: joint
    assert doc["actor_shares"][3] == {"国": 0.3, "上流自治体": 0.4, "地域主体": 0.3}     # 民有林の森林整備: joint
    P = doc["points"]
    for x, need, cost, ca in zip(P["x"], P["need"], P["cost"], P["cost_actor"]):
        bits = 0
        for i, a in enumerate(x):
            if a > 0:
                bits |= sum(1 << A.index(n) for n in doc["actor_shares"][i])
        assert need == bits                                                  # a joint measure needs all its actors
        assert abs(sum(ca) - cost) < 0.5 and abs(cost - SplitEvaluator().cost(x)["total"]) < 0.5
    assert "relocated_households" in P and "habitat_connectivity" in P
