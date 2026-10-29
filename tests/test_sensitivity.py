"""Sensitivity analysis (rri_sim.rdm.sensitivity) without RRI runs."""
import json

import numpy as np

from rri_sim.rdm import config as CF
from rri_sim.rdm import run as RUN
from rri_sim.rdm import sensitivity as SA
from tests.test_rdm import FakeEvaluator


def test_morris_ranks_a_linear_model_and_flags_interaction():
    k = 4
    trajs, delta = SA.morris_trajectories(k, 20, 4, np.random.default_rng(0))
    f = lambda u: 5 * u[0] + 1 * u[1] + 0 * u[2] + 3 * u[0] * u[3]  # noqa: E731
    mu, mus, sg = SA.morris_indices(trajs, delta, [np.array([f(u) for u in pts]) for pts, _, _ in trajs])
    assert mus[0] > mus[3] > mus[1] > mus[2] - 1e-12 and np.isclose(mus[2], 0)
    assert np.isclose(sg[1], 0, atol=1e-9) and sg[0] > 0 and sg[3] > 0     # interaction -> sigma > 0


def test_gp_surrogate_fits_a_smooth_function_and_reports_errors():
    rng = np.random.default_rng(1)
    f = lambda X: np.sin(3 * X[:, 0]) + X[:, 1] ** 2  # noqa: E731
    Xt, Xv = rng.random((25, 2)), rng.random((40, 2))
    gp = SA.GP(Xt, f(Xt))
    e = SA.error_stats(f(Xv), gp(Xv))
    assert e["r2"] > 0.97 and e["rel_rmse"] < 0.1


def test_sobol_sequence_estimates_of_the_ishigami_function():
    def ish(U):
        x = -np.pi + 2 * np.pi * U
        return np.sin(x[:, 0]) + 7 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])
    r = SA.sobol_indices(ish, 3, 14, 50, 0)
    assert np.allclose(r["S1"], [0.314, 0.442, 0.0], atol=0.03)
    assert np.allclose(r["ST"], [0.558, 0.442, 0.244], atol=0.03)
    for (lo, hi), v in zip(r["ST_ci"], r["ST"]):
        assert lo <= v <= hi


def test_preference_sweep_switches_the_recommendation():
    # policy 0: cheap, high damage; policy 1: costly, low damage
    mk = lambda d, c: {"damage": d, "cost": c, "households": 1.0, "livability": 50, "ecology": 50,  # noqa: E731
                       "damage_zone": {"A": d}, "cost_zone": {"A": c}}
    M = [[mk(100, 0), mk(120, 0)], [mk(40, 80), mk(50, 80)]]
    st = {"国": {"damage_zones": "all", "cost_zones": "all", "weights": {"damage": 0.5, "cost": 0.5}, "regret_threshold": 0.05}}
    out = SA.preference(M, st, M[0][0], 100.0, np.array([True, True]), 200, 11, np.random.default_rng(0))
    sw = out["stakeholders"]["国"]["sweeps"]["damage"]
    assert sw[0] == 0 and sw[-1] == 1                        # cost only -> do nothing; damage only -> the measure
    f = out["stakeholders"]["国"]["frequency"]
    assert np.isclose(sum(f), 1.0) and 0 < f[1] < 1


def test_sensitivity_pipeline_with_a_fake_evaluator(tmp_path):
    cfg = CF.load({"name": "fake", "budget_oku": 1500.0, "optimisation": {"pop": 8, "gens": 2},
                   "zones": [["上流", "x"], ["下流", None]],
                   "stakeholders": {
                       "国": {"damage_zones": "all", "cost_zones": "all", "weights": {"damage": 0.6, "cost": 0.4}, "regret_threshold": 0.05},
                       "下流自治体": {"damage_zones": ["下流"], "cost_zones": ["下流"], "weights": {"damage": 0.8, "cost": 0.2}, "regret_threshold": 0.05}},
                   "scenarios": {"climates": ["current", "2C", "4C"], "n_per_climate": 1, "n_post": 4}, "sobol": {"n_base": 64}})
    RUN.run(cfg, tmp_path, evaluator=FakeEvaluator(), log=lambda *a: None)
    ov = {"sensitivity": {"morris": {"r": 6, "levels": 4}, "surrogate": {"n_train": 14, "n_val": 6},
                          "sobol": {"log2_n": 9, "n_boot": 20}, "decision": {"n_samples": 512},
                          "preference": {"n_weight_samples": 100, "sweep_steps": 5}}}
    out = SA.run(tmp_path, ov, evaluator=FakeEvaluator(), log=lambda *a: None)
    saved = json.loads((tmp_path / "sensitivity.json").read_text())
    assert (tmp_path / "config_sensitivity.json").exists()
    # the fake model depends on rain_factor (strongly) and not on forest/retention effects of the analysed policies
    sc = saved["morris"]["score"]
    assert sc["rain_factor"] > sc["forest_effect"] and "rain_factor" in saved["morris"]["selected"]
    assert saved["surrogate"]["validation"] and saved["surrogate"]["worst_rel_rmse"] < 0.2
    s = saved["sobol"]["results"]["damage"]; ins = saved["sobol"]["inputs"]
    assert s["ST"][ins.index("rain_factor")] == max(s["ST"])
    for name, d in saved["decision"].items():
        assert np.isclose(sum(d["best_frequency"]), 1.0) and 0 <= d["rank_reversal_rate"] <= 1
    assert len(saved["preference"]["conflict"]["cross_max_regret"]) == 2
    assert out["tornado"]["damage"]["rows"][0]["param"] in ("rain_factor", "asset_factor")
