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
