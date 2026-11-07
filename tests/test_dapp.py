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
    C = (st.capital_var + st.maint_var) @ ctx.Pc.T                 # cost by actor (with subsidies) adds up to the total
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


def test_per_decade_capacity_binds_and_single_sums_it():
    from rri_sim.dapp import multi as MU
    from rri_sim.roadmap import sim as SM
    from rri_sim.roadmap import world as WD
    from rri_sim.roadmap.social import Social
    free = {n: {**a, "budget": {"total_oku": None, "per_decade_oku": a["budget"]["per_decade_oku"]}} for n, a in FAKE_ACTORS.items()}
    ctx = multi_ctx(actors=free)
    iv = {v["name"]: i for i, v in enumerate(ctx.T.V)}
    w = WD.belief_world(ctx.rc)
    st = SM.State(ctx.rc, ctx.phys, w, Social(ctx.rc, w.p["social"], {}))
    st.budgets, st.payer = ctx.budgets, ctx.payer
    rel, lev, pump = iv["住宅対策・下流"], iv["堤防・下流"], iv["排水ポンプ（下水道）・下流"]
    st.request(rel, 2); st.request(lev, 2); st.process(2020)       # the levee (450) exceeds the national decade capacity after 60
    st.process(2030)
    st.request(pump, 1); st.process(2040)                           # no total limit: 60 + 80 > 100 does not block the pump
    assert not st.queue and st.spent_by["下流自治体"] == 140.0
    u = MU.budget_use(ctx, st)                                      # carry-over shows in the decade it happened
    assert u["deferred"]["国"][:2] == [1, 0] and u["commit"]["国"][:2] == [60.0, 450.0] and u["commit"]["下流自治体"][2] == 80.0
    # Single: one payer for all the measures with the summed per-decade capacity; the actors' groups and targets stay
    one = multi_ctx(single=True, actors=free)
    pool = one.pool["name"]
    assert one.budgets == {pool: {"total_oku": None, "per_decade_oku": 400.0 + 60.0 + 100.0}}
    assert all(one.payer(i) == {pool: 1.0} for a in one.actors for i in a.vars) and one.names == list(FAKE_ACTORS)
    assert np.allclose(one.Pc, ctx.Pc)                              # costs still attributed to the actors (with the subsidy)
    st = SM.State(one.rc, one.phys, w, Social(one.rc, w.p["social"], {}))
    st.budgets, st.payer = one.budgets, one.payer
    st.request(rel, 2); st.request(lev, 2); st.request(pump, 1); st.process(2020)   # 120 + 450 > 560: the levee waits, the pump fits
    assert [q[0] for q in st.queue] == [lev] and MU.budget_use(one, st)["deferred"] == {pool: [1] + [0] * 7}
    assert multi_ctx(single=True).budgets[pool]["total_oku"] == 800.0              # totals given: their sum


def test_multi_actor_dapp_pipeline(tmp_path):
    from rri_sim.dapp import multi as MU
    from tests.test_roadmap import FakePhysics
    c = multi_cfg()
    res = MU.run(c, out_dir=tmp_path, phys=FakePhysics(), log=lambda *a: None)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["mode"] == "multi"
    names = [a["name"] for a in saved["actors"]]
    nS = len(saved["worlds"])
    comp = saved["comparison"]
    assert {"individual", "joint", "single"} <= set(comp) and sum(k.startswith("rules:") for k in comp) == 3 ** len(names)
    for k, r in comp.items():
        assert 0 <= r["basin"]["coverage"] <= 1 and r["basin"]["max_regret"] >= -1e-9
        for n in names:
            x = r["actors"][n]
            assert 0 <= x["coverage"] <= 1 and x["max_regret"] >= -1e-9 and (x["lockin"] is None or 0 <= x["lockin"] <= 1)
    g = saved["gap"]
    cov = {k: comp[k]["basin"]["coverage"] for k in ("individual", "joint", "single")}
    assert np.isclose(g["coordination_gap"]["coverage"], cov["joint"] - cov["individual"])
    assert np.isclose(g["institutional_fragmentation_gap"]["coverage"], cov["single"] - cov["joint"])
    assert np.isclose(g["institutional_fragmentation_gap"]["mean_cost"], comp["single"]["basin"]["mean_cost"] - comp["joint"]["basin"]["mean_cost"])
    for k, (a_, b_) in (("coordination_gap", ("individual", "joint")), ("institutional_fragmentation_gap", ("joint", "single"))):
        lo, hi = g[k]["coverage_ci90"]
        assert lo <= g[k]["coverage"] + 1e-9 and g[k]["coverage"] <= hi + 1e-9
        assert np.isclose(g[k]["scenarios_gained"] - g[k]["scenarios_lost"], round(g[k]["coverage"] * nS))
    fr = saved["frontier"]
    for k in ("joint", "single"):
        assert sum(r["chosen"] for r in fr[k]) == 1
        ch = next(r for r in fr[k] if r["chosen"])
        assert np.isclose(ch["coverage"], cov[k]) and np.isclose(ch["mean_cost"], comp[k]["basin"]["mean_cost"])
    assert max(cov.values()) <= fr["any_case_coverage"] + 1e-9
    sa = saved["single_actor"]
    assert sa["budget"]["per_decade_oku"] == sum(a["budget"]["per_decade_oku"] for a in saved["actors"])
    assert list(comp["single"]["strategies"]) == names and set(saved["budget"]["single"]) == {sa["name"]}
    assert set(saved["budget"]["individual"]) == set(names)
    assert len(sa["catalog"]) == sum(len(a["catalog"]) for a in saved["actors"])
    for k in ("individual", "joint", "single"):
        for b in saved["budget"][k].values():
            assert len(b["mean"]) == len(saved["years"]) and all(0 <= x <= 1 for x in b["deferred_share"])
        for n in names + ([sa["name"]] if k == "single" else []):
            tree = saved["trees"][k][n]
            assert tree[0]["n"] == nS and sum(b["end"] for b in tree) == nS
            target = comp[k]["basin"]["coverage"] if n == sa["name"] else comp[k]["actors"][n]["coverage"]
            assert np.isclose(sum(b["end_met"] for b in tree) / nS, target)
            assert all(set(b["reasons"]) <= {"observed", "projected", "switch"} for b in tree[1:])
        net = saved["networks"][k]
        for y in net["years"]:
            assert sum(nd["n"] for nd in net["nodes"] if nd["year"] == y) == nS
        assert all(e["changes"] is not None for e in net["edges"])
        assert set(saved["unmet"][k]) == {"all", *names}
    assert all(len(saved["dependence"][n]) == 3 ** (len(names) - 1) + 1 for n in names)
    assert set(saved["spillover"]["portfolios"]) == {"施策なし", "全主体", *names}
    assert [q["year"] for q in saved["signal_quality"]] == saved["years"][1:]
    assert {"joint", "single", *names} <= set(saved["search"])
    res2 = MU.run(c, out_dir=tmp_path / "again", phys=FakePhysics(), log=lambda *a: None)
    assert res2["comparison"] == res["comparison"]


def test_gui_recalculation_settings_progress_and_summaries(tmp_path):
    from rri_sim.dapp import multi as MU
    from tests.test_roadmap import FakePhysics
    c = multi_cfg(gui={"sizes": {"tiny": {"robust_overrides": {"scenarios": {"n": 6}}, "dependence_scenarios": 3,
                                          "search": {"n_scenarios": 3, "pop": 4, "gens": 1, "basin_pop": 4, "basin_gens": 1}}}})
    inputs = {"actors": {"国": {"per_decade_oku": 300, "targets": {"damage_ratio": 1.3, "pop_ratio": 1.8}},
                         "上流自治体": {"per_decade_oku": 50, "weights": {"damage": 0.7, "cost": 0.3}},
                         "下流自治体": {"targets": {"damage_ratio": 1.1}}},
              "basin_weights": {"damage": 0.6, "cost": 0.4}}
    g = MU.gui_config(c, inputs, "tiny")
    MA = g["multi"]
    assert MA["actors"]["国"]["budget"]["per_decade_oku"] == 300 and MA["actors"]["下流自治体"]["targets"]["damage_ratio"] == 1.1
    assert MA["actors"]["上流自治体"]["weights"] == {"damage": 0.7, "cost": 0.3} and MA["search"]["n_scenarios"] == 3
    assert MA["name"].startswith("fake_multi_gui_") and MU.gui_config(c, inputs, "tiny")["multi"]["name"] == MA["name"]
    assert MU.gui_config(c, {"actors": {"国": {"per_decade_oku": 301}}}, "tiny")["multi"]["name"] != MA["name"]
    old = json.loads(json.dumps(c)); del old["multi"]["gui"]                     # a run saved before the GUI sizes existed
    assert MU.gui_config(old, {}, "fast")["multi"]["search"]["n_scenarios"] == CF.DEFAULT["multi"]["gui"]["sizes"]["fast"]["search"]["n_scenarios"]
    for bad in ({"actors": {"上流自治体": {"targets": {"pop_ratio": 1.5}}}},          # no exposed population of a zone
                {"actors": {"国": {"per_decade_oku": -1}}}, {"actors": {"誰か": {}}}):
        try:
            MU.gui_config(c, bad, "tiny")
            raise AssertionError(bad)
        except ValueError:
            pass
    ctx = MU.setup(g, False, FakePhysics(), lambda m: None)[0]
    assert ctx.R["targets"]["damage_ratio"] == 1.3 and ctx.R["targets"]["pop_ratio"] == 1.8    # the basin target follows 国
    one = MU.setup(g, True, FakePhysics(), lambda m: None)[0]
    assert one.pool["budget"]["per_decade_oku"] == 300 + 50 + 100                     # Single: the actors' sum (下流 unchanged)
    seen = []
    out = MU.recalculate(g, FakePhysics(), lambda f, m: seen.append(f), out_dir=tmp_path / MA["name"])
    assert out == {"name": MA["name"]} and seen == sorted(seen) and 0 <= seen[0] and seen[-1] < 1
    S = MU.summaries("fake_multi", tmp_path)
    assert [x["name"] for x in S] == [MA["name"]]
    x = S[0]
    assert x["size"] == "tiny" and x["base"] == "fake_multi" and x["single_per_decade_oku"] == 450
    assert x["actors"]["国"]["targets"] == {"damage_ratio": 1.3, "pop_ratio": 1.8}
    assert np.isclose(x["gap"]["coordination_gap"]["coverage"], x["coverage"]["joint"] - x["coverage"]["individual"])
    assert np.isclose(x["gap"]["institutional_fragmentation_gap"]["coverage"], x["coverage"]["single"] - x["coverage"]["joint"])


# ------------------------------------------------------------------ future option space (option_space.py, Robust DAPP)
def test_option_space_classifies_used_lost_and_conditional_options():
    from rri_sim.dapp import option_space as OS
    from rri_sim.dapp import robust as RB
    from rri_sim.roadmap import sim as SM
    from rri_sim.roadmap.social import Social

    def space(**over):
        c, rc, ph, ref, T, worlds = robust_setup(**over)
        iv = {v["type"]: i for i, v in enumerate(ph.variables)}
        top = [len(L) - 1 for L in T.levels]
        one = lambda **kv: tuple(kv.get(v["type"], 0) for v in ph.variables)  # noqa: E731
        nodes = [one(), one(relocate=1), one(relocate=2), one(paddy_dam=top[iv["paddy_dam"]]), one(forest=top[iv["forest"]]),
                 one(levee=1), one(levee=2), one(retention=1), T.repair(one(paddy_dam=top[iv["paddy_dam"]], forest=top[iv["forest"]]))]
        net = RB.Net(c, rc, ph, T, nodes, worlds, ref)
        return OS.OptionSpace(net), net, iv, T

    sp, net, iv, T = space()
    lab = {l: o for o, l in enumerate(sp.labels)}
    opt = lambda i, li: lab[f"{T.V[i]['name']} {T.label(i, li)}"]  # noqa: E731
    assert not sp.base[opt(iv["pump_urban"], 1)]                     # in no node of the network: not an option (never "lost")
    assert sp.base[opt(iv["paddy_dam"], 1)] and sp.base[opt(iv["paddy_dam"], 2)]      # 50 % is reached through the 100 % node

    def state(*reqs):
        st = SM.State(net.rc, net.phys, net.worlds[0], Social(net.rc, net.worlds[0].p["social"], {}))
        st.allow_decrease = True
        for t, i, lv in reqs:
            st.request(i, lv); st.process(t)
        return st
    r = iv["relocate"]
    s1 = sp.status(state((2020, r, 2)), 2, 2030)
    assert s1[opt(r, 2)] == OS.USED and s1[opt(r, 1)] == OS.CODE["irreversible_land_use"]      # relocated: no raise any more
    s2 = sp.status(state((2020, r, 1), (2030, r, 2)), 2, 2040)
    assert s2[opt(r, 1)] == OS.USED                                                            # raised first: used, not lost
    f, pd = iv["forest"], iv["paddy_dam"]
    st = state((2020, f, T.levels[f][-1]))
    young, old = st.can_decrease(f, 2030), st.can_decrease(f, 2050)
    assert young and not old
    assert sp.status(st, 4, 2030)[opt(pd, 2)] == OS.CONDITIONAL                               # reconvert the young forest first
    assert sp.status(st, 4, 2050)[opt(pd, 2)] == OS.CODE["land_use_conflict"]                 # past the reconversion deadline
    assert sp.deadline(st, T.V[f]["zone"]) is not None
    sp_fixed, *_ = space(model={"measures": {"forest": {"reconvert": {"allowed": False}}}})
    assert sp_fixed.status(st, 4, 2030)[opt(pd, 2)] == OS.CODE["irreversible_measure"]
    st = state((2020, iv["retention"], 1))                                                     # retention: farmland converted,
    s = sp.status(st, 7, 2030)                                                                 # no option of the space lost
    land = [OS.CODE[k] for k in ("irreversible_land_use", "land_use_conflict", "irreversible_measure")]
    assert s[opt(iv["retention"], 1)] == OS.USED and not np.isin(s, land).any()
    assert T.landuse(net.nodes[7])[T.V[iv["retention"]]["zone"]]["遊水地"] > 0
    assert sp.status(state(), 0, 2095)[opt(iv["levee"], 1)] == OS.CODE["lead_time"]
    st = state(); st.spent = net.rc["budget"]["total_oku"]
    assert sp.status(st, 0, 2030)[opt(iv["levee"], 2)] == OS.CODE["budget"]


def test_robust_dapp_reports_future_option_space(tmp_path):
    from rri_sim.dapp import robust as RB
    from tests.test_roadmap import FakePhysics
    res = RB.run(robust_cfg(), tmp_path, phys=FakePhysics(), log=lambda *a: None)
    ev = res["evaluation"]
    O = ev["option_space"]
    assert O["n_base"] > 0 and O["years"] == res["years"] and set(O["reasons"]) == set(O["labels"])
    for r in ev["table"]:
        d = O["inits"][str(r["node"])]
        ts = d["timeseries"]
        assert all(0 <= x <= 1 for x in ts["remaining"]["mean"]) and len(ts["cost_cum"]) == len(O["years"])
        tot = [ts["remaining"]["mean"][n] + ts["used"]["mean"][n] + sum(ts["loss"][q][n] for q in O["reasons"]) for n in range(len(O["years"]))]
        assert np.allclose(tot, 1.0, atol=1e-3)                              # every base option is feasible, used or lost (rounded)
        assert 0 <= r["remaining_options_mean"] <= 1 and 0 <= r["transition_feasibility"] <= 1
        assert sum(b["n"] for b in d["branches"].values()) >= len(ev["scenarios"])
    assert O["inits"]["0"]["timeseries"]["remaining"]["mean"][0] == 1.0      # nothing done in 2020: every option is open
    for f in ("option_space.json", "scorecard.csv", "timeseries.csv", "branches.csv"):
        assert (tmp_path / "option_space" / f).stat().st_size > 0


# ------------------------------------------------------------------ Adaptive DAPP (adaptive.py): decision rules per initial action
def test_adaptive_dapp_searches_rules_and_compares_with_robust(tmp_path):
    from rri_sim.dapp import adaptive as AD
    from rri_sim.dapp import option_space as OS
    from rri_sim.dapp import robust as RB
    from tests.test_roadmap import FakePhysics
    rc_ = robust_cfg()
    RB.run(rc_, tmp_path / "robust", phys=FakePhysics(), log=lambda *a: None)
    c = CF._merge(rc_, {"adaptive": {"name": "fake_adaptive", "inits": "all", "n_eval": 8,
                                     "search": {"n_scenarios": 4, "pop": 6, "gens": 1}}})
    P = CF.load(c)["adaptive"]["policy"]
    codec = AD.Codec(P)
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = codec.decode([rng.integers(0, n) for n in codec.sizes])
        assert codec.decode(codec.encode(p)) == p and any(p.monitor)
    res = AD.run(c, tmp_path / "ad", phys=FakePhysics(), log=lambda *a: None, robust_dir=tmp_path / "robust")
    saved = json.loads((tmp_path / "ad" / "results.json").read_text())
    assert saved["mode"] == "adaptive" and saved["n_eval"] == 8
    T = saved["table"]
    assert [r["node"] for r in T] == json.loads((tmp_path / "robust" / "results.json").read_text())["evaluation"]["inits"]
    assert any(r.get("pareto") for r in T) and sum(bool(r.get("recommended")) for r in T) == 1
    assert next(r for r in T if r.get("recommended")).get("pareto")
    for r in T:
        for m in ("adaptive", "robust"):
            x = r[m]
            assert 0 <= x["coverage"] <= 1 and x["max_regret"] >= -1e-9 and 0 <= x["remaining_mean"] <= 1
            assert all(v == 0 for v in x["options"]["loss"]["transition_unavailable"])     # state-based option space
        tree = saved["trees"][str(r["node"])]
        assert tree[0]["n"] == 8 and sum(b["end"] for b in tree) == 8
        assert np.isclose(sum(b["end_met"] for b in tree) / 8, r["adaptive"]["coverage"])
        pol = r["policy"]
        for rec in saved["records"][str(r["node"])]:                       # every decision follows the rule
            for e in rec["events"][1:]:
                why = e["reason"]
                assert why in pol["monitor"] or (why == "scale_back" and pol["scale_back"] > 0)
                if why in ("observed", "projected"):
                    assert (e["obs_ratio"] if why == "observed" else e["projected"]) > pol["rho"]
                    assert e["benefit"] > P["min_benefit"] and e["score"] > 0
                if why == "rain":
                    assert e["rain2100"] >= pol["rain"] and e["year"] >= P["rain_from_year"]
    assert len(saved["catalog"]) > len(saved["records"]) and saved["n_base"] > 0
    assert set(saved["reasons"]) == set(OS.REASONS)
    res2 = AD.run(c, tmp_path / "ad2", phys=FakePhysics(), log=lambda *a: None, robust_dir=tmp_path / "robust")
    assert [r["adaptive"]["coverage"] for r in res2["table"]] == [r["adaptive"]["coverage"] for r in res["table"]]


def test_adaptive_dapp_sensitivity_targets_and_information_quality(tmp_path):
    from rri_sim.dapp import adaptive as AD
    from rri_sim.dapp import robust as RB
    from tests.test_roadmap import FakePhysics
    rc_ = robust_cfg()
    RB.run(rc_, tmp_path / "robust", phys=FakePhysics(), log=lambda *a: None)
    c = CF._merge(rc_, {"adaptive": {"name": "fake_adaptive", "inits": "promising", "n_eval": 6,
                                     "search": {"n_scenarios": 3, "pop": 4, "gens": 1}}})
    base = AD.run(c, tmp_path / "fake_adaptive", phys=FakePhysics(), log=lambda *a: None, robust_dir=tmp_path / "robust")
    for bad in ({"targets": {"damage_ratio": 9}}, {"noise_damage": -1}, {"trend_window": 1}):
        try:
            AD.gui_config(c, bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    inp = {"targets": {"damage_ratio": 1.4, "pop_ratio": 2.0}, "noise_damage": 3.0, "noise_climate": 2.0, "trend_window": 5,
           "rain_from_year": 2060, "forecast_noise": 0.0}
    g = AD.gui_config(c, inp, "fixed")
    assert g["adaptive"]["fixed_from"] == "fake_adaptive" and g["adaptive"]["name"].startswith("fake_adaptive_gui_")
    assert AD.gui_config(c, inp, "fixed")["adaptive"]["name"] == g["adaptive"]["name"]
    net = RB.load_net(tmp_path / "robust", FakePhysics(), log=lambda m: None)[0]
    m0 = dict(net.rc["monitoring"])
    sens = AD.apply_sensitivity(net, g["adaptive"])                       # reaches the 11.9 run too (both methods)
    assert np.isclose(net.rc["monitoring"]["risk_noise"], 3 * m0["risk_noise"]) and np.isclose(net.rc["monitoring"]["rain_noise"], 2 * m0["rain_noise"])
    assert net.R["targets"]["damage_ratio"] == 1.4 and net.rc["regimes"]["robust"]["thresholds"]["pop_ratio"] == 2.0
    assert net.rc["detect"]["trend_window"] == 5 and sens["rain_from_year"] == 2060 and sens["mode"] == "fixed"
    g["adaptive"]["fixed_from_dir"] = str(tmp_path / "fake_adaptive")
    fx = AD.run(g, tmp_path / g["adaptive"]["name"], phys=FakePhysics(), log=lambda *a: None, robust_dir=tmp_path / "robust")
    assert [r["policy"] for r in fx["table"]] == [r["policy"] for r in base["table"]]   # the same rules, re-evaluated
    assert all(r["front"] == [] for r in fx["table"]) and fx["settings"]["targets"]["damage_ratio"] == 1.4
    # forecast error changes only the adaptive pathways (11.9 has its own projection); 0 changes nothing
    f0 = AD.gui_config(c, {}, "fixed"); f0["adaptive"]["fixed_from_dir"] = str(tmp_path / "fake_adaptive")
    f1 = AD.gui_config(c, {"forecast_noise": 1.0}, "fixed"); f1["adaptive"]["fixed_from_dir"] = str(tmp_path / "fake_adaptive")
    r0 = AD.run(f0, tmp_path / "f0", phys=FakePhysics(), log=lambda *a: None, robust_dir=tmp_path / "robust")
    r1 = AD.run(f1, tmp_path / "f1", phys=FakePhysics(), log=lambda *a: None, robust_dir=tmp_path / "robust")
    assert [r["adaptive"]["coverage"] for r in r0["table"]] == [r["adaptive"]["coverage"] for r in base["table"]]
    assert [r["robust"]["coverage"] for r in r1["table"]] == [r["robust"]["coverage"] for r in r0["table"]]
    ev0 = [e["year"] for x in r0["records"].values() for rr in x for e in rr["events"]]
    ev1 = [e["year"] for x in r1["records"].values() for rr in x for e in rr["events"]]
    assert ev0 != ev1 or [r["adaptive"]["mean_cost"] for r in r1["table"]] != [r["adaptive"]["mean_cost"] for r in r0["table"]]
    seen = []
    s2 = AD.gui_config(c, {"noise_damage": 0.0}, "search", "fast")
    s2["adaptive"]["search"] = {"n_scenarios": 3, "pop": 4, "gens": 1, "p_mut": 0.2}
    AD.recalculate(s2, FakePhysics(), lambda f, m: seen.append(f), out_dir=tmp_path / s2["adaptive"]["name"], robust_dir=tmp_path / "robust")
    assert seen == sorted(seen) and seen[-1] < 1
    S = AD.summaries("fake_adaptive", tmp_path)
    assert {x["mode"] for x in S} == {"search", "fixed"} and all(x["n_inits"] == len(base["table"]) for x in S)
    x = next(x for x in S if x["name"] == fx["name"])
    assert x["sensitivity"]["noise_damage"] == 3.0 and x["sensitivity"]["targets"]["damage_ratio"] == 1.4
