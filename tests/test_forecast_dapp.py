"""Adaptive DAPP with forecasts and learning (rri_sim.dapp.forecast_dapp) on the analytic physics stand-in."""
import numpy as np

from rri_sim.dapp import config as DCF
from rri_sim.dapp import forecast_dapp as FD
from rri_sim.roadmap import sim as SM
from rri_sim.roadmap import world as WD
from test_roadmap import FakePhysics


def small_cfg(**over):
    base = {"forecast": {"scenarios": {"n_eval": 12, "n_search": 6, "seeds": [3, 4]},
                         "initial": {"pop": 12, "gens": 3, "n_max": 4},
                         "policy": {"rho_grid": [0.8, 1.0], "lookahead_grid": [10, 30], "k_grid": [1], "wait_grid": [0, 1]}}}
    return DCF.load(DCF._merge(base, over))


def model(cfg=None):
    cfg = cfg or small_cfg()
    M = FD.Model(cfg, FakePhysics())
    ws, _, _ = FD.make_worlds(M.rc, cfg["forecast"]["scenarios"]["groups"], 8, 5)
    M.worlds = ws
    return cfg, M


def test_projection_grid_matches_the_surrogate_and_costs_discount_lead_time():
    cfg, M = model()
    g = tuple([2, 1, 2, 0, 0, 1, 0])
    G = M.proj.grids([g])
    y = M.phys.predict(M.L.raw(g)[None], {"rain_factor": np.array([1.2]), "sea_level_rise_m": np.array([0.3]),
                                          "rain_gamma": np.array([M.proj.gamma])})
    assert np.isclose(M.proj.at(G, np.array([1.2]), np.array([0.3]))[0, 0, 0], y["damage"][0], rtol=1e-6)
    lev = M.action_cost(0, 0, 2)                  # levee: in place after consensus + build (a step)
    assert lev["lead"] > 5 and lev["share"](2030, np.array([2031.0]))[0] == 0.0 and lev["share"](2030, np.array([2060.0]))[0] == 1.0
    assert lev["pv"](2050) < lev["pv"](2030)      # the same project later costs less in present value
    pump = M.action_cost(5, 0, 1)                 # pump: linear over its build years
    s = pump["share"](2030, np.array([2030 + pump["lead"] - 0.5 * WD.nominal_phases(M.rc, "pump_urban")["build"]]))[0]
    assert np.isclose(s, 0.5)


def test_controller_acts_on_the_forecast_only_and_waits_when_information_is_worth_it():
    cfg, M = model()
    none = tuple([0] * len(M.L.V))
    jobs = [(none, FD.Policy(rho, 10, 1, w), s) for rho in (0.5, 5.0) for w in (0, 1) for s in range(len(M.worlds))]
    sims = FD.simulate(M, jobs)
    acts = {(c.p.rho, c.p.wait): sum(e["kind"] == "act" for e in c.events) for st, c in sims}
    by = lambda rho, w: sum(sum(e["kind"] == "act" for e in c.events) for st, c in sims if c.p.rho == rho and c.p.wait == w)  # noqa: E731
    assert by(5.0, 0) == 0 and by(5.0, 1) == 0             # never triggered
    assert by(0.5, 0) > 0                                   # triggered: acts
    assert by(0.5, 1) <= by(0.5, 0)                         # waiting can only defer actions
    for st, c in sims:
        for e in c.events:
            if e["kind"] == "act":
                assert e["ratio"] > c.p.rho and e["ratio_after"] < e["ratio"]
    recs = [FD.record(M, st, c) for st, c in sims]
    assert all(np.isfinite(r["cost_pv"]) for r in recs)


def test_initial_actions_are_pareto_under_nominal_values_and_pipeline_runs(tmp_path):
    cfg = small_cfg()
    res = FD.run(cfg, out_dir=tmp_path, phys=FakePhysics(), log=lambda m: None)
    T = res["table"]
    assert T[0]["label"] == "施策なし" and all(r["initial_cost"] <= cfg["forecast"]["initial"]["max_cost_oku"] + 1e-9 for r in T)
    for r in T:
        cov = [f["coverage"] for f in r["front"]]
        cost = [f["mean_cost"] for f in r["front"]]
        assert cost == sorted(cost) and cov == sorted(cov)  # a front: more coverage costs more
        assert set(r["convergence"]["hv_eval"]) == {"3", "4"} and 0 <= r["convergence"]["jaccard"] <= 1
    assert (tmp_path / "results.json").exists()


def test_annual_expected_damage_by_return_period_and_gumbel_ratios():
    from rri_sim import frequency as FQ
    T = [2, 10, 100]
    assert np.isclose(FQ.ead([0.0, 100.0, 1000.0], T), (0 + 100) / 2 * (0.5 - 0.1) + (100 + 1000) / 2 * (0.1 - 0.01))
    rng = np.random.default_rng(0)
    xi, al = 100.0, 30.0
    x = xi - al * np.log(-np.log(rng.random(4000)))          # Gumbel sample
    a, b = FQ.gumbel_lmom(x)
    assert abs(a - xi) < 3 and abs(b - al) < 3
    r = FQ.quantile(a, b, [2, 100, 200]) / FQ.quantile(a, b, 100)
    assert r[0] < 1 and np.isclose(r[1], 1) and r[2] > 1
    # the controller's benefit: a measure lowers the expected annual damage over the forecast
    cfg, M = model()
    c = FD.Controller(M, tuple([0] * len(M.L.V)), FD.Policy(1.0, 10, 1, 0), 0)
    fc = M.worlds[0].forecast(2040)
    g0, g1 = tuple([0] * len(M.L.V)), tuple([2] + [0] * (len(M.L.V) - 1))
    E = c.expected_ead(M.proj.grids([g0, g1]), fc, np.array([2050.0]))
    assert E[1, 0] < E[0, 0] and E[1, 0] > 0
