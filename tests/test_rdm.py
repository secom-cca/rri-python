"""RDM analysis (rri_sim.rdm) without RRI runs: NSGA-II, Sobol, regret,
candidate generation and the whole pipeline with an analytic evaluator."""
import json

import numpy as np
import pytest

from rri_sim.rdm import candidates as CA
from rri_sim.rdm import config as CF
from rri_sim.rdm import nsga2
from rri_sim.rdm import robustness as RB
from rri_sim.rdm import run as RUN
from rri_sim.rdm import uncertainty as UQ


def test_nsga2_finds_the_pareto_front_of_a_knapsack_like_problem():
    # 6 binary measures, effect e_i and cost c_i; minimise (damage, cost) with cost <= 10
    e = np.array([5, 4, 3, 3, 2, 1.0]); c = np.array([6, 4, 3, 2, 2, 1.0])

    def f(x):
        return [100 - e @ x, c @ x], max(0.0, c @ x - 10)
    arch, _ = nsga2.run([2] * 6, f, pop=16, gens=12, rng=np.random.default_rng(3), log=lambda *a: None)
    F = np.array([a["f"] for a in arch]); V = np.array([a["v"] for a in arch])
    par = F[nsga2.pareto_mask(F, V)]
    # exhaustive front
    allx = np.array(np.meshgrid(*[[0, 1]] * 6)).reshape(6, -1).T
    G = np.array([[100 - e @ x, c @ x] for x in allx if c @ x <= 10])
    true = G[nsga2.pareto_mask(G)]
    found = {tuple(p) for p in par}
    assert sum(tuple(t) in found for t in true) >= len(true) - 1      # (nearly) the whole front
    assert (par[:, 1] <= 10).all()


def test_sobol_indices_of_the_ishigami_function():
    a, b = 7.0, 0.1

    def ish(U):
        x = -np.pi + 2 * np.pi * U
        return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])
    S1, ST = UQ.saltelli(ish, 3, 20000, np.random.default_rng(0))
    assert np.allclose(S1, [0.314, 0.442, 0.0], atol=0.04)
    assert np.allclose(ST, [0.558, 0.442, 0.244], atol=0.04)


def test_regret_uses_the_best_feasible_policy_of_each_scenario():
    U = np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 3.0]])
    F = np.array([[1, 1], [1, 1], [0, 1]], bool)       # policy 2 over budget in scenario 0
    R = RB.regret(U, F)
    assert np.allclose(R[:, 0], [1.0, 0.0, -1.0])     # best feasible in s0 = policy 1
    assert np.allclose(R[:, 1], [3.0, 2.8, 0.0])
    s = RB.summary(R, F, 0.5)
    assert s["satisficing"] == [0.0, 0.5, 0.5]
    assert RB.minimax(R) == 2 and RB.minimax(R, eligible=np.array([1, 1, 0], bool)) == 1


def test_utility_scopes_damage_and_cost_by_zone():
    st = {"damage_zones": ["下流"], "cost_zones": ["下流"], "weights": {"damage": 1.0, "cost": 1.0}}
    ref = {"damage_zone": {"上流": 50.0, "下流": 100.0}, "households": 10, "livability": 50, "ecology": 50}
    m = {"damage_zone": {"上流": 0.0, "下流": 50.0}, "cost_zone": {"上流": 999.0, "下流": 10.0},
         "households": 5, "livability": 50, "ecology": 50}
    assert np.isclose(RB.utility(m, st, ref, budget=100.0), -0.5 - 0.1)    # upstream cost / damage ignored


class FakeEvaluator:
    """Analytic stand-in for the RRI evaluator: each measure removes a share
    of the damage of its zone; climate scales the damage."""
    zones = ["上流", "下流"]

    def __init__(self):
        self.variables = [
            CA.Variable("堤防・上流", "levee", "上流", ["なし", "+1 m", "+2 m"], [1, 2], [None, {"height": 1.0}, {"height": 2.0}]),
            CA.Variable("堤防・下流", "levee", "下流", ["なし", "+1 m", "+2 m"], [3], [None, {"height": 1.0}, {"height": 2.0}]),
            CA.Variable("遊水地1・下流", "retention", "下流", ["なし", "3 m"], [4, 5], [None, {"depth": 3.0}]),
            CA.Variable("田んぼダム・上流", "paddy_dam", "上流", ["なし", "全水田"], [6], [None, {}]),
            CA.Variable("住宅対策・下流", "relocate", "下流", ["なし", "嵩上げ", "移転"], [7],
                        [None, {"type": "raise", "height": 1.0}, {"type": "relocate", "dest": "new"}]),
        ]
        self.cost_of = [[0, 300, 700], [0, 400, 900], [0, 500], [0, 50], [0, 200, 800]]
        self.n_sim = 0

    def cost(self, x):
        by_zone = {z: 0.0 for z in self.zones}; by_type = {}
        for v, xi, cs in zip(self.variables, x, self.cost_of):
            by_zone[v.zone] += cs[int(xi)]; by_type[v.type] = by_type.get(v.type, 0) + cs[int(xi)]
        return {"total": float(sum(by_zone.values())), "by_zone": by_zone, "by_type": by_type}

    def cost_parts(self, x):
        ini = [0.8 * cs[int(xi)] for xi, cs in zip(x, self.cost_of)]
        return ini, [0.01 * c for c in ini]

    def simulate(self, x, sim, strict=True):
        self.n_sim += 1
        k = ({"current": 1.0, "2C": 1.3, "4C": 1.7}[sim["climate"]] if "climate" in sim else sim.get("rain_factor", 1.0) ** 2.5) \
            * sim.get("rain_gamma", 1.0)
        x = [int(v) for v in x]
        up = 100 * k * (1 - 0.3 * x[0] * sim.get("levee_effect", 1.0)) * (1 - 0.1 * x[3] * sim.get("paddy_effect", 1.0))
        down = 300 * k * (1 - 0.25 * x[1]) * (1 - 0.3 * x[2] * sim.get("retention_effect", 1.0)) * (1 - 0.35 * x[4])
        down *= 1 + 0.1 * x[0] * k                     # upstream levees shift the flood downstream
        hh = 1000 * k * (1 - 0.3 * x[4]) * (1 - 0.2 * x[1])
        return {"damage": up + down, "damage_zone": {"上流": up, "下流": down}, "households": hh,
                "livability": 50 - 2.0 * (x[4] == 2), "ecology": 60 - 1.5 * (x[0] + x[1]) + 1.0 * x[2],
                "inundated_km2": 10.0, "peak_q": 1000.0, "pop_hazard": 2.5 * hh}


def test_pipeline_with_a_fake_evaluator_writes_reproducible_results(tmp_path):
    cfg = CF.load({"name": "fake", "budget_oku": 1500.0, "optimisation": {"pop": 8, "gens": 3},
                   "zones": [["上流", "x"], ["下流", None]],
                   "stakeholders": {
                       "国": {"damage_zones": "all", "cost_zones": "all", "weights": {"damage": 0.6, "cost": 0.4}, "regret_threshold": 0.05},
                       "下流自治体": {"damage_zones": ["下流"], "cost_zones": ["下流"], "weights": {"damage": 0.8, "cost": 0.2}, "regret_threshold": 0.05}},
                   "scenarios": {"climates": ["current", "2C", "4C"], "n_per_climate": 2, "n_post": 4}, "sobol": {"n_base": 128}})
    res = RUN.run(cfg, tmp_path, evaluator=FakeEvaluator(), log=lambda *a: None)
    assert (tmp_path / "config.json").exists() and (tmp_path / "results.json").exists()
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["policies"][0]["label"] == "施策なし" and len(saved["policies"]) >= 2
    n_p, n_s = len(saved["policies"]), len(saved["scenarios"])
    assert n_s == 6 * 4
    R = np.array(saved["robustness"]["stakeholders"]["国"]["regret"])
    F = np.array(saved["robustness"]["feasible"])
    assert R.shape == (n_p, n_s) and (R[F] >= -1e-9).all()          # feasible policies never beat the best feasible
    for p in saved["pareto"]:
        assert saved["archive"][p]["cost"] <= 1500.0
    rec = saved["robustness"]["overall"]["minimax"]
    assert 0 <= rec < n_p
    # climate dominates the damage of the recommended policy in the fake model
    s = saved["sobol"]["被害額（総合推奨案）"]
    assert s["climate"]["ST"] > s["paddy_effect"]["ST"]
    # same config + seed -> same results
    res2 = RUN.run(cfg, tmp_path / "again", evaluator=FakeEvaluator(), log=lambda *a: None)
    assert res2["pareto"] == res["pareto"] and res2["robustness"]["overall"] == res["robustness"]["overall"]


def test_edits_scale_retention_and_override_the_housing_type():
    fe = FakeEvaluator()
    e = CA.edits(fe.variables, [0, 1, 1, 0, 2], {"retention_effect": 0.5})
    types = [x["type"] for x in e]
    assert types == ["levee", "retention", "relocate"]
    assert e[1]["depth"] == 1.5


def test_candidates_on_the_synthetic_basin():
    from tests.test_gui_engine import session as _session
    s = _session.__wrapped__() if hasattr(_session, "__wrapped__") else None
    if s is None:
        pytest.skip("fixture not callable")
    d = s.d
    meta = {"gauge_index": {"g": int(np.argmax(d.river.area_km2))}, "fdir": None}
    cfg = CF.load({"zones": [["全域", None]]})
    depth = np.where(s.assets.households > 0, 1.5, 0.0)
    V = CA.build(d, meta, s.hand, s.assets, s.class_names, depth, cfg)
    types = {v.type for v in V}
    assert {"paddy_dam", "relocate"} <= types
    for v in V:
        assert v.params[0] is None and len(v.params) == len(v.options) and v.cells
