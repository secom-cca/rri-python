"""DAPP analysis (rri_sim.dapp) with the analytic evaluator of tests/test_rdm.py."""
import json

import numpy as np

from rri_sim.dapp import config as CF
from rri_sim.dapp import model as MD
from rri_sim.dapp import run as RUN
from tests.test_rdm import FakeEvaluator


def cfg(**over):
    base = {"zones": [["上流", "x"], ["下流", None]],
            "stakeholders": {
                "国": {"damage_zones": "all", "cost_zones": "all", "weights": {"damage": 0.6, "cost": 0.4}, "regret_threshold": 0.05},
                "下流自治体": {"damage_zones": ["下流"], "cost_zones": ["下流"], "weights": {"damage": 0.8, "cost": 0.2}, "regret_threshold": 0.05}},
            "stakeholder_atp_ratio": {"国": 1.2, "下流自治体": 1.0}, "weight_samples": 200}
    base.update(over)
    return CF.load(base)


def ref(ev):
    return ev.simulate([0] * len(ev.variables), {"rain_factor": 1.0, "sea_level_rise_m": 0.0, "forest_effect": 1.0})


def test_climate_path_interpolates_and_actions_parse():
    c = cfg()
    assert MD.climate_at(c["climate_paths"]["4C"], 2075) == (np.interp(2075, [2050, 2100], [1.08, 1.20]), np.interp(2075, [2050, 2100], [0.2, 0.7]))
    V = FakeEvaluator().variables
    assert MD.parse_action("levee:1", V) == [(0, 1), (1, 1)]
    assert MD.parse_action("relocate@下流:2", V) == [(4, 2)]
    assert MD.parse_action("forest:1", V) == []


def test_lead_time_and_budget_deferral():
    ev = FakeEvaluator(); c = cfg(budget={"total_oku": 5000.0, "per_decade_oku": 500.0})
    spec = {"kind": "static", "schedule": [[2020, ["levee:1", "retention:1"]]]}
    r = MD.simulate("x", spec, "current", ev, c, ref(ev))
    P = r["periods"]
    assert P[0]["x"][0] == 0 and P[1]["x"][0] == 1             # levees (10 y lead) in place from the 2030s
    starts = [(e["year"], e["var"]) for e in r["log"] if e["event"] == "start"]
    # 0.8 x (300 + 400) = 560 > 500: the second levee waits for the next decade, retention (400) too
    assert starts[0][0] == 2020 and any(y == 2030 for y, _ in starts)
    assert any(e["event"] == "deferred" for e in r["log"])
    assert r["cum_capital"] <= 5000.0 and all(p["capital"] <= 500.0 + 1e-9 for p in P)


def test_adaptive_strategy_triggers_only_when_the_climate_worsens():
    ev = FakeEvaluator(); c = cfg()
    spec = {"kind": "adaptive", "initial": 0, "trigger_ratio": 1.1, "monitor_lag": 1,
            "pathway": [["levee:1"], ["retention:1"], ["relocate@下流:1"]]}
    cur = MD.simulate("a", spec, "current", ev, c, ref(ev))
    hot = MD.simulate("a", spec, "4C", ev, c, ref(ev))
    assert not [e for e in cur["log"] if e["event"] == "trigger"]
    trig = [e for e in hot["log"] if e["event"] == "trigger"]
    assert trig and trig[0]["observed_year"] < trig[0]["year"]          # decided on the lagged observation
    assert hot["atp"]["damage"] is None or hot["atp"]["damage"] >= 2040


def test_dapp_pipeline_with_a_fake_evaluator(tmp_path):
    c = cfg(name="fake_dapp")
    res = RUN.run(c, tmp_path, evaluator=FakeEvaluator(), log=lambda *a: None)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert (tmp_path / "config.json").exists()
    n = len(saved["strategies"]); assert n == 6 and saved["climates"] == ["current", "2C", "4C"]
    none = saved["records"]["無対策|4C"]
    assert none["cum_cost_pv"] == 0 and none["atp"]["damage"] is not None    # no measures fail under 4C
    early = saved["records"]["早期集中投資|4C"]
    assert early["cum_damage_pv"] < none["cum_damage_pv"]
    for sn, st in saved["stakeholders"].items():
        R = np.array(st["regret"]); assert R.shape == (n, 3) and (R >= -1e-9).all() and 0 <= st["minimax"] < n
        assert np.isclose(sum(saved["weight_sensitivity"][sn]), 1.0)
    seg = saved["pathways"]["段階的投資|current"]["segments"]
    assert seg and all(s["start"] <= s["complete"] <= s["end"] for s in seg)
    res2 = RUN.run(c, tmp_path / "again", evaluator=FakeEvaluator(), log=lambda *a: None)
    assert res2["overall_minimax"] == res["overall_minimax"]


# ------------------------------------------------------------------ Robust DAPP (rri_sim.dapp.robust, landuse) on the analytic physics of tests/test_roadmap.py
def robust_cfg(**over):
    base = {"robust": {"name": "fake_robust", "scenarios": {"n": 10}, "candidates": {"pop": 16, "gens": 4, "n_nodes": 6, "max_nodes": 40},
                       "initial": {"n_max": 5}}}
    return CF.load(CF._merge(base, {"robust": over}))


def robust_setup(**over):
    from rri_sim.dapp import robust as RB
    from tests.test_roadmap import FakePhysics
    c = robust_cfg(**over)
    rc, ph, ref, T, F, Z, vary, worlds = RB.setup(c, FakePhysics(), log=lambda *a: None)
    return c, rc, ph, ref, T, worlds


def test_transitions_keep_land_use_apart_and_respect_reversibility():
    c, rc, ph, ref, T, worlds = robust_setup()
    iv = {v["type"]: i for i, v in enumerate(ph.variables)}
    top = [len(L) - 1 for L in T.levels]
    none = tuple([0] * len(top))
    raise_ = tuple(1 if i == iv["relocate"] else 0 for i in range(len(top)))
    reloc = tuple(2 if i == iv["relocate"] else 0 for i in range(len(top)))
    paddy = tuple(top[i] if i == iv["paddy_dam"] else 0 for i in range(len(top)))
    forest = tuple(top[i] if i == iv["forest"] else 0 for i in range(len(top)))
    levee = tuple(top[i] if i == iv["levee"] else 0 for i in range(len(top)))
    M = T.matrices([none, raise_, reloc, paddy, forest, levee])
    assert M["ok"][1, 2] and not M["ok"][2, 1]                     # raise -> relocate, never back
    assert M["initial"][1, 2] > T.costs[iv["relocate"]][2][0] - T.costs[iv["relocate"]][1][0]   # + change cost of the raised houses
    assert M["ok"][3, 0] and not M["ok"][5, 0]                     # paddy dams can stop, levees are not removed
    assert M["ok"][4, 0] and M["conditional"][4, 0]                # young forest back to farmland: conditional
    assert M["lead"][0, 5] > M["lead"][0, 1]                        # a levee takes longer than raising houses
    # land use vs implementation: paddy dams keep farmland, forest converts it, relocation leaves a vacated site
    lu0, luf, lur = T.landuse(none), T.landuse(forest), T.landuse(reloc)
    up, down = "上流", "下流"
    assert T.landuse(paddy) == lu0 and luf[up]["森林（転換）"] > 0 and luf[up]["農地"] < lu0[up]["農地"]
    assert lur[down]["移転跡地"] > 0 and lur[down]["住宅地（浸水域）"] == 0
    # afforested paddies cannot hold paddy dams
    both = tuple(top[i] if i in (iv["paddy_dam"], iv["forest"]) else 0 for i in range(len(top)))
    assert T.paddy_available(both, up) < 1.0 and T.repair(both)[iv["paddy_dam"]] < top[iv["paddy_dam"]]


def test_state_scales_back_reversible_measures_and_reconverts_young_forest_only():
    from rri_sim.roadmap import sim as SM
    from rri_sim.roadmap import world as WD
    from rri_sim.roadmap.social import Social
    c, rc, ph, ref, T, worlds = robust_setup()
    iv = {v["type"]: i for i, v in enumerate(ph.variables)}
    w = WD.belief_world(rc)

    def state():
        st = SM.State(rc, ph, w, Social(rc, w.p["social"], {}))
        st.allow_decrease = True
        return st
    st = state()
    st.request(iv["paddy_dam"], 1.0); st.process(2020)
    st.request(iv["paddy_dam"], 0.0); st.process(2040)
    assert st.vec(iv["paddy_dam"], 2035)[0] > 0.9 and st.vec(iv["paddy_dam"], 2042)[0] == 0.0
    st.request(iv["levee"], 1); st.process(2020); st.request(iv["levee"], 0); st.process(2040)
    assert st.committed[iv["levee"]] == 1                           # not reversible: the request is ignored
    young, old = state(), state()
    for s_, t in ((young, 2030), (old, 2060)):
        s_.request(iv["forest"], 1.0); s_.process(2020)
        s_.request(iv["forest"], 0.0); s_.process(t)
    assert young.committed[iv["forest"]] == 0.0 and young.vec(iv["forest"], 2035)[1] < 1e-9
    assert any(e["event"] == "decrease" and e["cost"] > 0 for e in young.log)
    assert old.committed[iv["forest"]] == 1.0                       # older than max_age_years: stays forest
    assert not SM.State(rc, ph, w, Social(rc, w.p["social"], {})).can_decrease(iv["paddy_dam"], 2030)   # docs/13 behaviour unchanged


def test_robust_dapp_pipeline(tmp_path):
    from rri_sim.dapp import robust as RB
    from tests.test_roadmap import FakePhysics
    c = robust_cfg()
    res = RB.run(c, tmp_path, phys=FakePhysics(), log=lambda *a: None)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["mode"] == "robust" and (tmp_path / "network.npz").exists()
    ev = saved["evaluation"]
    assert ev["inits"][0] == 0 and len(ev["inits"]) <= 5 and ev["promising"]
    nS = len(ev["scenarios"])
    for r in ev["table"]:
        assert 0 <= r["coverage"] <= 1 and 0 <= r["coverage_perfect"] <= 1 and r["max_regret"] >= -1e-9
        assert 0 <= r["lockin"]["options"] <= 1 and 0 <= r["lockin"]["scenarios"] <= 1
        tree = ev["trees"][str(r["node"])]
        assert tree[0]["n"] == nS and sum(b["end"] for b in tree) == nS and all(b["end"] for b in tree if b["leaf"])
        assert abs(sum(b["end_met"] for b in tree) / nS - r["coverage"]) < 1e-9
    none = next(r for r in ev["table"] if r["node"] == 0)
    assert none["lockin"]["options"] == 0 and none["lockin"]["scenarios"] == 0 and none["lockin"]["sunk_oku"] == 0
    # moves only along the pathway options; observation-based decisions never use the scenario's truth
    S = saved["network"]["successors"]
    for rec in ev["records"]["adaptive"]:
        for a, b in zip(rec["events"], rec["events"][1:]):
            assert b["to"] in [o["to"] for o in S[str(a["to"])]["options"]]
            assert b["reason"] in ("observed", "projected") and b["obs_rain"] > 0.9
    assert all(e["reason"] in ("initial", "projected") for rec in ev["records"]["perfect"] for e in rec["events"])
    assert set(ev["unreachable"]) == {"adaptive", "perfect", "best_initial"}
    # the GUI panel: stricter targets cannot raise the coverage of the perfect-information pathways
    strict = RB.resimulate(tmp_path, targets={"damage_ratio": 0.6, "pop_ratio": 0.6}, phys=FakePhysics())
    loose = RB.resimulate(tmp_path, trigger={"ratio_of_threshold": 0.8}, phys=FakePhysics())
    assert [r["node"] for r in strict["evaluation"]["table"]] == ev["inits"]
    assert np.mean([r["coverage_perfect"] for r in strict["evaluation"]["table"]]) <= np.mean([r["coverage_perfect"] for r in ev["table"]]) + 1e-9
    assert loose["trigger"]["ratio_of_threshold"] == 0.8
    res2 = RB.run(c, tmp_path / "again", phys=FakePhysics(), log=lambda *a: None)
    assert res2["evaluation"]["table"] == res["evaluation"]["table"]


# ------------------------------------------------------------------ Multi-Actor Robust DAPP (multi.py, actors.py)
FAKE_ACTORS = {
    "国": {"measures": ["堤防・下流", "遊水地1・下流"], "budget": {"total_oku": 600.0, "per_decade_oku": 400.0}, "metric": "basin",
          "targets": {"damage_ratio": 1.2, "pop_ratio": 1.5}, "weights": {"damage": 0.5, "pop": 0.2, "cost": 0.3}},
    "上流自治体": {"measures": ["田んぼダム・上流", "森林化・上流"], "budget": {"total_oku": 100.0, "per_decade_oku": 60.0}, "metric": "上流",
              "targets": {"damage_ratio": 1.2}, "weights": {"damage": 0.6, "cost": 0.3, "ecology": 0.1}},
    "下流自治体": {"measures": ["住宅対策・下流", "排水ポンプ（下水道）・下流", "雨水貯留（各戸）・下流"], "budget": {"total_oku": 100.0, "per_decade_oku": 100.0},
              "metric": "下流", "targets": {"damage_ratio": 1.2}, "weights": {"damage": 0.6, "cost": 0.4}},
}


def multi_cfg(**over):
    base = {"multi": {"name": "fake_multi", "single_name": "fake_single", "actors": FAKE_ACTORS, "robust_overrides": {"scenarios": {"n": 8}},
                      "search": {"n_scenarios": 4, "pop": 6, "gens": 2, "basin_pop": 6, "basin_gens": 1}, "dependence_scenarios": 4}}
    return CF.load(CF._merge(base, {"multi": over}))


def multi_ctx(single=False, **over):
    from rri_sim.dapp import multi as MU
    from tests.test_roadmap import FakePhysics
    return MU.setup(multi_cfg(**over), single, FakePhysics(), log=lambda *a: None)[0]


def test_state_charges_each_payer_within_its_own_budget():
    from rri_sim.roadmap import sim as SM
    from rri_sim.roadmap import world as WD
    from rri_sim.roadmap.social import Social
    ctx = multi_ctx()
    assert set(ctx.names) == set(FAKE_ACTORS)                     # the actors of the settings replace the default ones
    iv = {v["name"]: i for i, v in enumerate(ctx.T.V)}
    rel, lev, pump = iv["住宅対策・下流"], iv["堤防・下流"], iv["排水ポンプ（下水道）・下流"]
    assert ctx.payer(rel) == {"下流自治体": 0.5, "国": 0.5} and ctx.payer(lev) == {"国": 1.0}     # subsidy of relocation
    w = WD.belief_world(ctx.rc)
    st = SM.State(ctx.rc, ctx.phys, w, Social(ctx.rc, w.p["social"], {}))
    st.budgets, st.payer = ctx.budgets, ctx.payer
    st.request(rel, 2); st.request(lev, 2); st.process(2020)       # relocation 120 (60 / 60); levee 450 > national decade cap after 60
    assert st.spent_by == {"下流自治体": 60.0, "国": 60.0} and [q[0] for q in st.queue] == [lev]
    st.process(2030)
    assert st.spent_by["国"] == 510.0 and not st.queue
    st.request(pump, 1); st.process(2040); st.process(2050)         # 60 + 80 > the downstream total of 100: waits for good
    assert [q[0] for q in st.queue] == [pump] and st.spent_by["下流自治体"] == 60.0
    C = (st.capital_var + st.maint_var) @ ctx.P.T                  # cost by payer adds up to the total
    assert np.isclose(C.sum(), (st.capital + st.maint).sum())


def test_strategy_codec_rules_and_conditional_controller():
    from rri_sim.dapp import actors as AC
    from rri_sim.dapp import multi as MU
    ctx = multi_ctx(strategy={"switch_from_year": 2030})
    up = ctx.by["上流自治体"]
    codec = AC.Codec(up, ctx.T, ctx.MA["strategy"])
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = codec.decode([rng.integers(0, n) for n in codec.sizes])
        assert codec.decode(codec.encode(s)) == s
    assert [lab for lab in up.labels if "再転換" in lab or "縮小" in lab]          # scaling back / reconversion are actions
    rank = AC.ranking(up, ctx.T, ctx.phys, ctx.rc, ctx.metric_fn())
    R = AC.rules(up, rank, ctx.MA["rules"])
    assert R["積極投資"].init == tuple(len(ctx.T.levels[i]) - 1 for i in up.vars) and R["投資抑制"].init == (0, 0)
    # paddy dams and forest at the top together: the land allows fewer paddy dams; switch at once to list B
    iv = {v["name"]: i for i, v in enumerate(ctx.T.V)}
    pd = iv["田んぼダム・上流"]
    B = tuple(k for k, (i, li) in enumerate(up.catalog) if i == iv["森林化・上流"] and li == 0)
    s = AC.Strategy(up.name, R["積極投資"].init, ((), B), switch=1.0, rho=0.5)
    (st, cs, opt), = MU.simulate(ctx, [({up.name: s}, 0)])
    ev = cs[0].events
    assert ev[0]["kind"] == "initial" and up.catalog[ev[0]["acts"][0]][0] == pd
    assert st.log[0]["level"] < 1.0 or any(e["event"] == "start" and e["var"] == pd and e["level"] < 1.0 for e in st.log)
    assert ev[1]["kind"] == "switch" and ev[1]["year"] == 2030          # forest above the decade cap waits until 2030, then B
    assert ev[2]["kind"] == "act" and ev[2]["branch"] == 1 and ev[2]["year"] >= ev[1]["year"]
    assert st.committed[iv["森林化・上流"]] == 0.0                     # young forest reconverted (within max_age_years)
    (_, cs2, _), = MU.simulate(multi_ctx(), [({up.name: s}, 0)])      # default: the rain trend is not trusted before 2050
    assert next(e["year"] for e in cs2[0].events if e["kind"] == "switch") >= 2050
    assert 0 <= opt["mid"][up.name]["lockin"] <= 1 and abs(sum(v for k, v in opt["end"][up.name].items() if k != "lockin") - 1) < 1e-9


def test_multi_actor_dapp_pipeline(tmp_path):
    from rri_sim.dapp import multi as MU
    from tests.test_roadmap import FakePhysics
    c = multi_cfg()
    res = MU.run(c, out_dir=tmp_path, phys=FakePhysics(), log=lambda *a: None)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["mode"] == "multi" and not saved["single"]
    names = [a["name"] for a in saved["actors"]]
    nS = len(saved["worlds"])
    comp = saved["comparison"]
    assert {"individual", "basin"} <= set(comp) and sum(k.startswith("rules:") for k in comp) == 3 ** len(names)
    for k, r in comp.items():
        assert 0 <= r["basin"]["coverage"] <= 1 and r["basin"]["max_regret"] >= -1e-9
        for n in names:
            x = r["actors"][n]
            assert 0 <= x["coverage"] <= 1 and x["max_regret"] >= -1e-9 and 0 <= x["lockin"] <= 1
    g = saved["gap"]
    assert np.isclose(g["basin_coverage"], comp["basin"]["basin"]["coverage"] - comp["individual"]["basin"]["coverage"])
    for conf in ("individual", "basin"):
        for n in names:
            tree = saved["trees"][conf][n]
            assert tree[0]["n"] == nS and sum(b["end"] for b in tree) == nS
            assert np.isclose(sum(b["end_met"] for b in tree) / nS, comp[conf]["actors"][n]["coverage"])
            assert all(set(b["reasons"]) <= {"observed", "projected", "switch"} for b in tree[1:])
        net = saved["networks"][conf]
        for y in net["years"]:
            assert sum(nd["n"] for nd in net["nodes"] if nd["year"] == y) == nS
        assert all(e["changes"] is not None for e in net["edges"])
    assert set(saved["unmet"]["individual"]) == {"all", *names}
    assert all(len(saved["dependence"][n]) == 3 ** (len(names) - 1) + 1 for n in names)
    assert set(saved["spillover"]["portfolios"]) == {"施策なし", "全主体", *names}
    assert [q["year"] for q in saved["signal_quality"]] == saved["years"][1:]
    res2 = MU.run(c, out_dir=tmp_path / "again", phys=FakePhysics(), log=lambda *a: None)
    assert res2["comparison"] == res["comparison"]
    # one basin actor with all the measures (the single-actor Robust DAPP on single measures)
    one = MU.run(c, single=True, out_dir=tmp_path / "single", phys=FakePhysics(), log=lambda *a: None)
    assert [a["name"] for a in one["actors"]] == ["流域"] and len(one["actors"][0]["measures"]) == 7
    assert one["gap"]["basin_coverage"] == 0 and sum(k.startswith("rules:") for k in one["comparison"]) == 3
