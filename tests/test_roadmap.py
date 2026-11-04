"""Roadmap analysis (rri_sim.roadmap) with an analytic physics stand-in (no RRI data needed)."""
import json

import numpy as np

from rri_sim.rdm import candidates as CA
from rri_sim.rdm import explore as EX
from rri_sim.roadmap import config as CF
from rri_sim.roadmap import metrics as MT
from rri_sim.roadmap import physics as PH
from rri_sim.roadmap import planner as PL
from rri_sim.roadmap import run as RUN
from rri_sim.roadmap import sim as SM
from rri_sim.roadmap import world as WD
from rri_sim.roadmap.social import Social

RANGES = {"rain_factor": (1.0, 1.3), "sea_level_rise_m": (0.0, 0.7), "rain_gamma": (0.8, 1.3), "levee_effect": (0.5, 1.0),
          "retention_effect": (0.5, 1.0), "paddy_effect": (0.3, 1.0), "forest_effect": (0.3, 1.0), "raise_effect": (0.5, 1.0)}


class FakePhysics(PH.Physics):
    """Each measure removes a share of its zone's damage; climate scales damage and exposed population."""
    zones = ["上流", "下流"]

    def __init__(self):
        self.variables = [
            {"name": "堤防・下流", "type": "levee", "zone": "下流", "options": ["なし", "+1 m", "+2 m"], "continuous": False, "n_cells": 5,
             "heights": [0.0, 1.0, 2.0]},
            {"name": "遊水地1・下流", "type": "retention", "zone": "下流", "options": ["なし", "3 m"], "continuous": False, "n_cells": 2},
            {"name": "田んぼダム・上流", "type": "paddy_dam", "zone": "上流", "options": ["0%", "100%"], "continuous": True, "n_cells": 40},
            {"name": "森林化・上流", "type": "forest", "zone": "上流", "options": ["0%", "100%"], "continuous": True, "n_cells": 10},
            {"name": "住宅対策・下流", "type": "relocate", "zone": "下流", "options": ["なし", "嵩上げ", "移転"], "continuous": False, "n_cells": 3},
            {"name": "排水ポンプ（下水道）・下流", "type": "pump_urban", "zone": "下流", "options": ["なし", "+10 mm/h"], "continuous": False, "n_cells": 2},
            {"name": "雨水貯留（各戸）・下流", "type": "storage_private", "zone": "下流", "options": ["0%", "100%"], "continuous": True, "n_cells": 2},
        ]
        self.layout = PH.layout_of(self.variables)
        self.ranges = RANGES
        self.keys = PH.OUT_KEYS + ("dz:上流", "dz:下流")
        self.cell_km2 = 1.0
        self.farm_km2 = 50.0
        self.table = [[0, 200, 450], [0, 150], None, None, [0, 30, 120], [0, 80], None]

    def cost_parts(self, i, level):
        if self.variables[i]["continuous"]:
            c = {"paddy_dam": (10.0, 0.5), "forest": (60.0, 5.0), "storage_private": (40.0, 0.2)}[self.variables[i]["type"]]
            return c[0] * level, c[1] * level
        return float(self.table[i][int(round(level))]), 0.0

    def predict(self, raw, env):
        F = self.features(raw, env)
        lev, ret, paddy, f_eff, f_land, rai, rel, pump, stor, rain, slr, gam = [F[:, j] for j in range(12)]
        k = (1.0 + 0.3 * rain) ** 2.5 * (1 + 0.2 * slr)
        up = 300 * k * (1 - 0.35 * paddy) * (1 - 0.3 * f_eff)
        down = 500 * k * (1 - 0.3 * lev) * (1 - 0.2 * ret) * (1 - 0.15 * paddy) * (1 - 0.2 * rai - 0.4 * rel) * (1 - 0.2 * pump) * (1 - 0.1 * stor)
        pop = 3000 * k ** 1.2 * (1 - 0.3 * rai - 0.6 * rel) * (1 - 0.1 * paddy)
        return {"damage": up + down, "dz:上流": up, "dz:下流": down, "pop_hazard": pop, "households": pop / 2.5,
                "livability": 60 - 2.0 * rel, "ecology": 60 - 1.5 * lev + 1.0 * f_land}


def cfg(**over):
    base = {"optimiser": {"pop": 10, "gens": 3, "n_belief": 2},
            "experiment": {"n_worlds": 4, "n_goals": 12, "n_continuations": 10, "n_jobs": 1, "morris": {"r": 1, "levels": 4},
                           "sobol": {"log2_n": 6, "n_boot": 10}, "delay": {"scales": [1.0, 2.0], "postpone": [2020, 2040], "n_worlds": 2},
                           "complementarity": {"n_worlds": 2}, "participation": {"n_worlds": 1}, "scope": {"n_worlds": 2}}}
    return CF.load(CF._merge(base, over))


def setup(c=None):
    c = c or cfg()
    ph = PH.with_institutions(FakePhysics(), c)
    return c, ph, RUN.reference(ph, c)


def var(ph, typ):
    return next(i for i, v in enumerate(ph.variables) if v["type"] == typ)


# ------------------------------------------------------------------ physical layer
def test_features_match_the_exploration_surrogate_inputs():
    V = [CA.Variable("堤防・下流", "levee", "下流", ["なし", "+1 m", "+2 m"], [], [None, {"height": 1.0}, {"height": 2.0}]),
         CA.Variable("遊水地1", "retention", "下流", ["なし", "3 m"], [], [None, {"depth": 3.0}]),
         CA.Variable("田んぼダム", "paddy_dam", "上流", ["0%", "100%"], [], [None, {}], True),
         CA.Variable("森林化", "forest", "上流", ["0%", "100%"], [], [None, {}], True),
         CA.Variable("住宅", "relocate", "下流", ["なし", "嵩上げ", "移転"], [], [None, {"type": "raise"}, {"type": "relocate"}])]
    P = {k: {"low": lo, "high": hi, "nominal": 1.0, "kind": "sim"} for k, (lo, hi) in RANGES.items()}
    sp = EX.Space(V, P)
    X = np.array([[2, 1, 0.4, 0.7, 1], [1, 0, 1.0, 0.0, 2]], float)
    U = {"rain_factor": np.array([1.1, 1.25]), "sea_level_rise_m": np.array([0.2, 0.5]), "rain_gamma": np.array([1.0, 1.1]),
         "levee_effect": np.array([0.9, 0.6]), "retention_effect": np.array([0.8, 0.8]), "paddy_effect": np.array([0.5, 0.9]),
         "forest_effect": np.array([0.4, 0.4]), "raise_effect": np.array([0.7, 0.7])}
    ph = FakePhysics()
    ph.variables = ph.variables[:5]
    ph.layout = PH.layout_of(ph.variables)
    # raw columns: levee h/hmax, retention, paddy rate, forest (rate x maturity 1, rate), housing (raised, relocated)
    raw = np.array([[1.0, 1, 0.4, 0.7, 0.7, 1, 0], [0.5, 0, 1.0, 0.0, 0.0, 0, 1]], float)
    assert np.allclose(ph.features(raw, U), sp.features(X, U))


def test_reference_and_agriculture():
    c, ph, ref = setup()
    assert ref["damage"] > 0 and ref["agriculture"] == 100.0
    raw = np.zeros((1, len(ph.layout))); raw[0, 4] = 1.0; raw[0, 1] = 1.0     # all forest land + retention site
    assert len(ph.variables) == 7 + 4 and len(ph.layout) == 9                # institutional measures have no raw columns
    assert np.isclose(ph.agriculture(raw)[0], 100 * (1 - (10 + 2) / 50))


# ------------------------------------------------------------------ social layer
def test_early_participation_speeds_consensus_and_raises_adoption():
    c = cfg()
    v = WD.belief_params(c)["social"]
    early = Social(c, v, {s: "early" for s in c["participation"]["stakeholders"]})
    staged = Social(c, v)
    late = Social(c, v, {s: "late" for s in c["participation"]["stakeholders"]})
    for S in (early, staged, late):
        S.engage(c["measures"]["paddy_dam"]["stakeholders"], 2040)
    t = 2040
    assert early.consensus_years("paddy_dam", t) < staged.consensus_years("paddy_dam", t) < late.consensus_years("paddy_dam", t)
    assert early.fail_prob("paddy_dam", t) < staged.fail_prob("paddy_dam", t) < late.fail_prob("paddy_dam", t)
    assert early.adoption("paddy_dam", t + 5) > staged.adoption("paddy_dam", t + 5) > late.adoption("paddy_dam", t + 5)
    assert early.coordination_cost(2025, []) > 0 and Social(c, v).coordination_cost(2025, []) == 0     # early pays from the start


# ------------------------------------------------------------------ implementation state
def test_phases_budget_and_maintenance():
    c, ph, ref = setup(cfg(budget={"total_oku": 1000.0, "per_decade_oku": 400.0}))
    w = WD.belief_world(c)
    acts = [{"var": 0, "level": 2, "year": 2020}, {"var": 2, "level": 1.0, "year": 2020}, {"var": 1, "level": 1, "year": 2020}]
    st = SM.run(w, PL.FixedController(acts), ph, c, ref)
    a_lev = next(a for a in st.actions if a.var == 0)
    done = a_lev.t_cons + a_lev.build
    assert st.vec(0, done - 0.1)[0] == 0.0 and st.vec(0, done + 0.1)[0] == 1.0          # in place after consensus + build
    a_pd = next(a for a in st.actions if a.var == 2)
    mid = a_pd.t_cons + a_pd.diffuse / 2
    assert np.isclose(st.vec(2, mid)[0], a_pd.final[0] / 2) and a_pd.final[0] <= a_pd.adopt + 1e-9   # diffusion to the adoption ceiling
    # the levee (450 > cap 400) takes the 2020 budget alone; paddy dams and the retention basin wait for 2030
    starts = {e["var"]: e["year"] for e in st.log if e["event"] == "start"}
    assert starts[0] == 2020 and starts[2] == 2030 and starts[1] == 2030
    assert any(e["event"] == "deferred" for e in st.log)
    assert st.capital.sum() <= 1000.0 + 1e-6 and st.maint.sum() > 0


def test_forest_effect_matures_slowly_and_upgrade_costs_a_change():
    c, ph, ref = setup()
    w = WD.belief_world(c)
    st = SM.run(w, PL.FixedController([{"var": 3, "level": 1.0, "year": 2020}, {"var": 4, "level": 1, "year": 2020},
                                       {"var": 4, "level": 2, "year": 2040}]), ph, c, ref)
    a = next(a for a in st.actions if a.var == 3)
    e1, land1 = st.vec(3, a.t_cons + a.build)
    e2, land2 = st.vec(3, 2100)
    assert land1 == land2 > 0 and e1 < e2 <= land2                               # planted, effect still growing
    up = next(e for e in st.log if e["event"] == "start" and e["var"] == 4 and e["level"] == 2)
    assert np.isclose(up["change_cost"], c["measures"]["raise"]["change_cost_fraction"] * 30.0)
    assert st.vec(4, 2100)[1] > 0                                                 # relocated share


def test_failed_consensus_releases_budget_and_dapp_retries():
    c, ph, ref = setup(cfg(participation={"fail_scale": {"nominal": 10.0, "low": 10.0, "high": 10.0}}))
    p = WD.belief_params(c); p["deterministic"] = False; p["seed"] = 3
    w = WD.World(p, c)
    road = [{"var": 1, "level": 1, "year": 2020}]
    st0 = SM.run(w, PL.controller(c, ph, ref, "P0", road), ph, c, ref)
    assert st0.actions[0].failed and st0.spent == 0.0 and st0.committed[1] == 0.0   # failed: nothing spent, level back to 0
    st1 = SM.run(w, PL.controller(c, ph, ref, "P1", road), ph, c, ref)
    assert len([a for a in st1.actions if a.var == 1]) > 1 and any(a.source == "retry" for a in st1.actions)


# ------------------------------------------------------------------ worlds
def test_world_paths_and_regime_mapping():
    c = cfg()
    F = WD.factors(c)
    z = WD.unit_nominal(F)
    w = WD.make_world(c, z, 1)
    b = WD.belief_world(c)
    assert np.isclose(w.rain(2100), b.rain(2100)) and np.isclose(w.rain(2040), b.rain(2040))
    names = list(F)
    z2 = z.copy(); z2[names.index("rain_2100")] = 1.0; z2[names.index("climate_shift_year")] = 0.0
    w2 = WD.make_world(c, z2, 1)
    assert np.isclose(w2.rain(2030), b.rain(2030)) and np.isclose(w2.rain(2100), 1.3) and w2.rain(2060) > b.rain(2060)
    z3 = z.copy(); z3[names.index("regime_kind")] = 0.99
    w3 = WD.make_world(c, z3, 1)
    assert len(w3.p["regimes"]) == 2 and w3.regime_at(2100) == F["regime_kind"]["categories"][-1]
    assert w.fail_u("levee", 0) == WD.make_world(c, z, 1).fail_u("levee", 0)            # common random numbers


# ------------------------------------------------------------------ planner and policies
def test_genome_roundtrip_and_optimiser_beats_doing_nothing():
    c, ph, ref = setup()
    sp = PL.Space(c, ph)
    acts = [{"var": 0, "level": 1, "year": 2020}, {"var": 0, "level": 2, "year": 2040}, {"var": 2, "level": 0.5, "year": 2030}]
    assert sp.decode(sp.encode(acts)) == sorted(acts, key=lambda a: (a["year"], a["var"]))
    road, info = PL.initial_roadmap(c, ph, ref, seed=1)
    U0, _ = PL.rollouts(c, ph, ref, [[]], SM.State(c, ph, WD.belief_world(c), Social(c, WD.belief_params(c)["social"])),
                        WD.belief_ensemble(c, WD.belief_params(c), 2, 1), 2020, [[2020, c["initial_regime"]]])
    assert road and info["U"] > U0[0]


def test_reoptimisation_keeps_what_is_done():
    c, ph, ref = setup()
    w = WD.belief_world(c)
    st = SM.run(w, PL.FixedController([{"var": 0, "level": 2, "year": 2020}]), ph, c, ref)
    acts, _ = PL.optimise(c, ph, ref, [w], [[2020, c["initial_regime"]]], t_now=2050, state=st.snapshots.get(2050) or st, seed=2)
    assert all(not (a["var"] == 0 and a["level"] <= 2) for a in acts) and all(a["year"] >= 2050 for a in acts)


def test_p2_updates_on_a_climate_surprise_and_p3_on_new_criteria():
    c, ph, ref = setup()
    F = WD.factors(c); names = list(F)
    z = WD.unit_nominal(F)
    z[names.index("rain_2100")] = 1.0; z[names.index("slr_2100")] = 1.0; z[names.index("climate_shift_year")] = 0.0
    z[names.index("regime_kind")] = 0.99; z[names.index("regime_year")] = 0.2
    w = WD.make_world(c, z, 7)
    road, _ = PL.initial_roadmap(c, ph, ref, seed=1)
    res = {}
    for n in c["policies"]:
        ctrl = PL.controller(c, ph, ref, n, road, seed=1)
        SM.run(w, ctrl, ph, c, ref)
        res[n] = ctrl
    assert not hasattr(res["P0"], "replans") and not res["P1"].replans
    k2 = {x["kind"] for r in res["P2"].replans for x in r["reasons"]}
    k3 = {x["kind"] for r in res["P3"].replans for x in r["reasons"]}
    assert "climate" in k2 and "criteria" not in k2 and "criteria" in k3
    r = next(r for r in res["P2"].replans if any(x["kind"] == "climate" for x in r["reasons"]))
    assert "観測" in MT.describe_replan(c, ph, r)


# ------------------------------------------------------------------ metrics
def test_prim_finds_the_box():
    rng = np.random.default_rng(0)
    X = rng.random((400, 3))
    y = (X[:, 0] > 0.7) & (X[:, 2] < 0.5)
    F = {f"f{i}": {"low": 0.0, "high": 10.0, "nominal": 5.0, "label": f"f{i}", "group": "g"} for i in range(3)}
    b = MT.prim_box(X, y, F, density=0.9)
    dims = {d["factor"]: d for d in b["box"]}
    assert b["density"] >= 0.9 and "f0" in dims and dims["f0"]["low"] > 6.0 and "f2" in dims and dims["f2"]["high"] < 6.0


def test_coverage_and_lockin_of_a_spent_budget():
    c, ph, ref = setup(cfg(budget={"total_oku": 700.0, "per_decade_oku": 700.0}))
    w = WD.belief_world(c)
    lib = MT.continuation_library(c, ph, 30, 1)
    goals = MT.sample_goals(c, 30, 1)
    blank = MT.coverage(c, ph, ref, w, MT.blank_state(c, ph, w, 2040, None), lib, goals, 2040)
    st = SM.run(w, PL.FixedController([{"var": 0, "level": 2, "year": 2020}, {"var": 1, "level": 1, "year": 2030}]), ph, c, ref,
                snapshots=[2040])
    cov = MT.coverage(c, ph, ref, w, st.snapshots[2040], lib, goals, 2040)
    assert blank.shape == cov.shape == (30,)
    assert (blank & ~cov).sum() >= 0 and cov.mean() <= 1.0


# ------------------------------------------------------------------ pipeline
def test_pipeline_with_the_fake_physics(tmp_path):
    c = cfg(name="fake_roadmap")
    res = RUN.run(c, tmp_path, phys=FakePhysics(), log=lambda *a: None)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert (tmp_path / "config.json").exists()
    S = saved["summary"]
    assert set(S["policies"]) == {"P0", "P1", "P2", "P3"} and S["n_worlds"] == 4
    assert all(r >= -1e-9 for r in S["robustness"]["mean_regret"])
    for n in S["policies"]:
        assert set(S["coverage"][n]) == {"2040", "2060"} and 0 <= S["lockin"][n]["2040"] <= 1
        assert "n_key_assumptions" in S["explain"]["indices"][n]
    assert "P3>P2" in S["explain"]["switch"] and saved["morris"]["indices"]["U:P0"]["mu_star"]
    assert set(saved["delay"]["scales"]) == {"1.0", "2.0"} and "P2@2040" in saved["delay"]["scales"]["1.0"]["postpone"]
    assert len(saved["complementarity"]["portfolios"]) == 2 ** 10 and "paddy_dam+storage" in saved["complementarity"]["interaction"]
    assert "全員 早期参加" in saved["participation"]["results"]
    S3 = saved["scope"]
    assert list(S3["results"]) == list(c["scopes"])
    assert all(r["tier"] == "国" for r in S3["roadmaps"]["国のみ"])
    assert len(S3["variables"]["国のみ"]) < len(S3["variables"]["国＋自治体"]) < len(S3["variables"]["国＋自治体＋地域（住民・農家・民間）"])
    w0 = saved["worlds"][0]["records"]["P3"]
    assert w0["periods"] and "engagement" in w0 and isinstance(w0["replans"], list)
    res2 = RUN.run(c, tmp_path / "again", phys=FakePhysics(), log=lambda *a: None, parts=("main",))
    assert np.allclose(res2["summary"]["robustness"]["mean_regret"], res["summary"]["robustness"]["mean_regret"])


# ------------------------------------------------------------------ inland flooding and institutional measures
def test_inland_measures_complete_and_reduce_damage():
    c, ph, ref = setup()
    w = WD.belief_world(c)
    ip, isp = var(ph, "pump_urban"), var(ph, "storage_private")
    st = SM.run(w, PL.FixedController([{"var": ip, "level": 1, "year": 2020}, {"var": isp, "level": 1.0, "year": 2020}]), ph, c, ref)
    a = next(a for a in st.actions if a.var == ip)
    assert st.vec(ip, a.t_cons + a.build - 0.1)[0] == 0.0 and st.vec(ip, a.t_cons + a.build + 0.1)[0] == 1.0
    b = next(a for a in st.actions if a.var == isp)
    assert 0 < st.vec(isp, 2100)[0] <= b.adopt + 1e-9 and b.adopt < 0.7          # private storage limited by participation
    st0 = SM.run(w, PL.FixedController([]), ph, c, ref)
    assert st.periods[-1]["damage"] < st0.periods[-1]["damage"]
    assert st.maint.sum() > 0


def test_institutional_measures_change_exposure_damage_and_risk():
    c = cfg(landuse={"asset_2100": {"nominal": 1.3, "low": 1.3, "high": 1.3}})
    c, ph, ref = setup(c)
    p = WD.belief_params(c); p["asset_pts"] = [[2020, 1.0], [2100, 1.3]]
    w = WD.World(p, c)
    st0 = SM.run(w, PL.FixedController([]), ph, c, ref)
    out = {}
    for k in ("zoning", "induce", "floodproof", "evacuation"):
        i = var(ph, k)
        lv = ph.variables[i]
        st = SM.run(w, PL.FixedController([{"var": i, "level": 1.0 if lv["continuous"] else 1, "year": 2020}]), ph, c, ref)
        out[k] = st.periods[-1]
        assert st.actions[0].kind == k and st.periods[-1]["inst"][k] > 0
    last0 = st0.periods[-1]
    assert out["zoning"]["damage"] < last0["damage"]                              # growth of assets blocked
    assert out["induce"]["pop_hazard"] < last0["pop_hazard"]
    assert out["floodproof"]["damage"] < last0["damage"] and np.isclose(out["floodproof"]["pop_hazard"], last0["pop_hazard"])
    assert np.isclose(out["evacuation"]["damage"], last0["damage"]) and out["evacuation"]["pop_risk"] < last0["pop_risk"]
    assert SM.State(c, ph, w, Social(c, w.p["social"])).cols[var(ph, "zoning")] == []
    assert PL.Space(c, ph).levels[var(ph, "floodproof")] == [0.0, 0.5, 1.0]


def test_allowed_measures_by_implementer_tier():
    c, ph, ref = setup()
    nat = PL.allowed_vars(c, ph, ["国"])
    assert {ph.variables[i]["type"] for i in nat} == {"levee", "retention"}
    loc = PL.allowed_vars(c, ph, ["国", "自治体"])
    assert "relocate" in {ph.variables[i]["type"] for i in loc} and "paddy_dam" not in {ph.variables[i]["type"] for i in loc}
    acts, _ = PL.initial_roadmap(c, ph, ref, seed=1, allowed=nat)
    assert all(a["var"] in nat for a in acts)
