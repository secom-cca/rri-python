"""DAPP settings.  Plain JSON merged over DEFAULT (and over the RDM DEFAULT for
the basin, candidate measures, zones and stakeholders).  Every assumed value
carries status 根拠 (cited) / 仮定 (assumption) and its basis."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..rdm import config as RC

DEFAULT = {
    "name": "abukuma_dapp",
    "years": {"start": 2020, "end": 2100, "step": 10},
    "discount_rate": 0.04,
    # representative event: the basin's flood event (cfg["event"], 2019年台風19号) scaled by the decade's climate
    "representative_event": {"annual_probability": 0.01, "status": "仮定",
              "basis": "2019年台風19号級の洪水を年超過確率 1/100 の代表洪水とみなす。各10年の期待被害 = 10 × 確率 × 代表洪水の被害額"},
    # climate paths: [year, rain factor, sea-level rise m], linear in between
    "climate_paths": {
        "current": {"points": [[2020, 1.0, 0.0], [2100, 1.0, 0.0]], "status": "根拠", "basis": "現在気候（観測降雨のまま）"},
        "2C": {"points": [[2020, 1.0, 0.0], [2050, 1.06, 0.15], [2100, 1.10, 0.38]], "status": "仮定",
               "basis": "2100年値は国交省提言 2℃ 1.1・日本の気候変動2020 領域I 0.38 m（根拠）。途中の経路（2050年 1.06・0.15 m）は仮定"},
        "4C": {"points": [[2020, 1.0, 0.0], [2050, 1.08, 0.20], [2100, 1.20, 0.70]], "status": "仮定",
               "basis": "2100年値は国交省提言 4℃ 1.2・日本の気候変動2020 0.70 m（根拠）。途中の経路は仮定"},
    },
    # implementation: lead time (decision -> completion), effect ramp after completion, cost of changing an existing measure
    "measures": {
        "levee": {"lead_years": 10, "ramp_years": 0, "change_cost_fraction": 0.2, "status": "仮定", "basis": "築堤・嵩上げの用地取得〜完成 約10年"},
        "retention": {"lead_years": 10, "ramp_years": 0, "change_cost_fraction": 0.2, "status": "仮定", "basis": "遊水地の計画〜完成 約10年"},
        "paddy_dam": {"lead_years": 0, "ramp_years": 0, "change_cost_fraction": 0.0, "status": "仮定", "basis": "調整板の設置は数年以内（期内に効果）"},
        "forest": {"lead_years": 0, "ramp_years": 30, "change_cost_fraction": 0.0, "status": "仮定",
                   "basis": "植林後 30 年で効果が最大（効果係数を 0→1 に線形増加）"},
        "raise": {"lead_years": 5, "ramp_years": 0, "change_cost_fraction": 0.5, "status": "仮定", "basis": "住宅の嵩上げ 約5年。移転に切り替える場合は費用の半分が無駄になると仮定"},
        "relocate": {"lead_years": 10, "ramp_years": 0, "change_cost_fraction": 0.2, "status": "仮定", "basis": "移転先の造成・合意形成 約10年"},
    },
    "budget": {"total_oku": 2000.0, "per_decade_oku": 500.0},      # 初期費用（割引前）の上限
    # adaptation tipping points: thresholds relative to no measures under the current climate
    "atp": {"damage_ratio": 1.2, "pop_ratio": 1.2},
    # strategies: actions "type[@zone]:option" (option index of the RDM candidate variables; all zones if no @zone)
    "strategies": {
        "無対策": {"kind": "static", "schedule": []},
        "早期集中投資": {"kind": "static", "budget_cap_factor": 4.0,
                     "schedule": [[2020, ["levee:1", "retention:1", "paddy_dam:1", "relocate@下流:1"]]]},
        "段階的投資": {"kind": "static",
                   "schedule": [[2020, ["paddy_dam:1"]], [2030, ["levee:1"]], [2040, ["retention:1"]],
                                [2050, ["relocate@下流:1"]], [2060, ["levee:2"]]]},
        "自然基盤重視": {"kind": "static",
                     "schedule": [[2020, ["paddy_dam:1", "forest:1"]], [2030, ["retention:1"]], [2050, ["relocate@下流:2"]]]},
        "構造物重視": {"kind": "static",
                    "schedule": [[2020, ["levee:1"]], [2030, ["retention:1"]], [2040, ["levee:2"]]]},
        "状況適応型": {"kind": "adaptive", "initial": 1, "trigger_ratio": 1.0, "monitor_lag": 1,
                    "pathway": [["paddy_dam:1", "forest:1"], ["levee:1"], ["retention:1"], ["relocate@下流:1"],
                                ["levee:2"], ["relocate@下流:2", "relocate@上流:2"]]},
    },
    "quick_strategies": ["無対策", "早期集中投資", "段階的投資", "状況適応型"],
    "regret_threshold": 0.05,
    "weight_samples": 2000,
    "stakeholder_atp_ratio": {"国": 1.2, "上流自治体": 1.1, "下流自治体": 1.0, "住民": 1.0},
    # ---- Robust DAPP (robust.py, docs/11 11.9): pathways network on the exploration surrogate, observation-based triggers
    "robust": {
        "name": "abukuma_dapp_robust",
        "explore_run": "abukuma_explore_v2",          # surrogate + cost table (docs/12; roadmap --prepare once)
        # settings of the implementation model shared with docs/13 (rri_sim.roadmap.config): lead times (consensus + build /
        # diffuse / mature, uncertain phases {nominal, low, high}), change_cost_fraction, reversible, levels.  Overrides here.
        "model": {
            "years": {"start": 2020, "end": 2100, "decision_step": 10, "eval_step": 5},
            "budget": {"total_oku": 2000.0, "per_decade_oku": 500.0, "status": "仮定",
                       "basis": "初期費用（割引前）の総額と10年ごとの上限（docs/11 と同じ）。超過分は次の意思決定へ繰り越す"},
            "measures": {"forest": {"reconvert": {"allowed": True, "max_age_years": 10, "cost_fraction": 0.5, "status": "仮定",
                                                  "basis": "植栽後10年以内なら伐採して農地に戻せる（植栽費の半分を要する）。以後は森林として固定（保安林等）"}}},
            # the social process of docs/13 is not modelled here: no consensus failures, full adoption, no coordination cost
            "participation": {"fail_scale": {"nominal": 0.0, "low": 0.0, "high": 0.0},
                              "coord_cost_scale": {"nominal": 0.0, "low": 0.0, "high": 0.0}, "consensus_cost_oku_yr": 0.0,
                              "adoption": {k: [1.0, 1.0] for k in ("paddy_dam", "forest", "raise", "relocate", "storage_private",
                                                                  "floodproof", "evacuation")}},
        },
        # land use (distinct from the implementation state): which land a measure takes and whether it can return
        "landuse": {
            "forest_on_paddy_share": {"value": 0.5, "status": "仮定",
                                      "basis": "森林化の候補（HAND ≥ 20 m の水田・畑）のうち水田の割合。森林化した水田では田んぼダムができない"},
            "conversions": {
                "forest": {"from": "農地", "to": "森林", "irreversible": "植栽後 max_age_years を過ぎると農地に戻せない"},
                "retention": {"from": "農地", "to": "遊水地", "irreversible": "用地買収・地役権設定後は農地に戻さない（仮定）"},
                "relocate": {"from": "住宅地", "to": "移転跡地（災害危険区域）", "irreversible": "移転跡地は再び住宅地にしない（建築制限）"},
            },
        },
        # tolerated outcome (ATP): event damage / exposed population of the representative event relative to no measures,
        # current climate; checked in every evaluated period from from_year
        "targets": {"damage_ratio": 1.2, "pop_ratio": 1.5, "from_year": 2030, "status": "仮定",
                    "basis": "被害額は現状（対策なし・現在気候）の 1.2 倍まで（docs/11・13 の ATP と同じ）、浸水人口は 1.5 倍まで"
                             "（代理モデルの浸水人口は階段状で変動が大きい）。2020年代は施策の完成を待つ"},
        # scenarios: worlds of docs/13 (world.py) varying only these groups; the others stay at their nominal values
        "scenarios": {"n": 128, "groups": ["climate", "landuse", "effects", "delays"], "status": "仮定",
                      "basis": "気候（降雨・海面・強雨・想定から外れる年）、社会経済（浸水域の資産・人口）、施策効果、実施期間。"
                               "LHS の標本であり発生確率ではない"},
        # step 1: candidate portfolios (NSGA-II on the surrogate, no weights)
        "candidates": {"pop": 64, "gens": 30, "p_mut": 0.15, "eval_years": [2050, 2095], "n_nodes": 24, "max_nodes": 400},
        # steps 3-4: triggers of the observation-based pathways
        "trigger": {"ratio_of_threshold": 0.9, "use_observed": True, "use_projection": True, "horizon_years": 30, "safety": 1.0,
                    "status": "仮定",
                    "basis": "観測した被害の比が閾値 × ratio_of_threshold を超えるか、観測した気候の傾向から次の施策の完成時に閾値を超えると"
                             "予測されたら移行に着手する。移行先は horizon_years 先まで目標を満たすと予測される最も安い組合せ"},
        # pathway options: per node at most max_successors moves (non-dominated in move cost and tipping rain)
        "network": {"max_successors": 4},
        "initial": {"n_max": 16, "max_cost_oku": 500.0},      # initial portfolios: affordable within the first decade
        # utility weights for Regret only (the other indicators need no weights); thresholds = targets
        "weights": {"damage": 0.45, "cost": 0.25, "pop": 0.1, "ecology": 0.1, "livability": 0.05, "agriculture": 0.05},
        "regret_threshold": 0.05,
        "prim": {"alpha": 0.05, "min_mass": 0.05, "density": 0.8},
    },
    # ---- Multi-Actor Robust DAPP (multi.py, actors.py, docs/11 11.10): conditional strategies of single measures per actor.
    # The model, land use, scenarios, targets of the basin, weights and PRIM come from "robust" (robust_overrides here).
    "multi": {
        "name": "abukuma_dapp_multi",
        # Single: one decision maker for all the measures with the summed per-decade capacity (one pooled budget, no subsidy
        # routing).  It keeps the actors' measure groups as sub-strategies (own lists, switch and trigger on the group's
        # metric) and the Joint objectives, so that every Joint strategy is also a Single strategy: the difference between
        # the cases is the pooling of authority and capacity only
        "single": {"label": "流域（単一主体）"},
        "robust_overrides": {"scenarios": {"n": 64}},     # evaluation scenarios (the search uses search.n_scenarios of them)
        # actors: own measures (variables of the exploration by name), budget (initial cost, as sim.State), metric
        # ("basin" = damage and exposed population of the basin; a zone = event damage of that zone) and target (ratio to no
        # measures, current climate; ATP and trigger of the actor), utility weights (Regret).  Small model: 4 measures each.
        "actors": {
            "国": {"label": "国（河川管理者）", "measures": ["堤防・上流", "堤防・下流", "遊水地1・下流", "遊水地2・下流"],
                  "budget": {"total_oku": None, "per_decade_oku": 400.0}, "metric": "basin",
                  "targets": {"damage_ratio": 1.2, "pop_ratio": 1.5},
                  "weights": {"damage": 0.45, "pop": 0.15, "cost": 0.25, "ecology": 0.1, "agriculture": 0.05}},
            "上流自治体": {"label": "上流自治体", "measures": ["田んぼダム・上流", "森林化・上流", "住宅対策・上流", "雨水貯留（公共）・上流"],
                      "budget": {"total_oku": None, "per_decade_oku": 120.0}, "metric": "上流", "targets": {"damage_ratio": 1.2},
                      "weights": {"damage": 0.5, "cost": 0.3, "ecology": 0.1, "agriculture": 0.1}},
            "下流自治体": {"label": "下流自治体", "measures": ["田んぼダム・下流", "住宅対策・下流", "排水ポンプ（下水道）・下流", "雨水貯留（公共）・下流"],
                      "budget": {"total_oku": None, "per_decade_oku": 250.0}, "metric": "下流", "targets": {"damage_ratio": 1.2},
                      "weights": {"damage": 0.55, "cost": 0.35, "ecology": 0.05, "agriculture": 0.05}},
        },
        # budget: per_decade_oku = the actor's fiscal / implementation capacity in each decade (initial cost committed in the
        # decision year; the binding constraint, excess waits for the next decade).  total_oku = an optional limit on the total
        # over 2020-2100 (None = none; kept non-binding in the base case so that the cases differ in capacity, not in totals)
        "budget_status": {"status": "仮定", "basis": "10年ごとの財政・実施容量（国 400・上流 120・下流 250 億円; 単一主体は合計 770 億円）。総額の上限は置かない"},
        # cost sharing by variable type: the payer's share comes from its own budget (Stackelberg extension: the leader's variable)
        "subsidy": {"relocate": {"payer": "国", "share": 0.5, "status": "仮定",
                                 "basis": "住宅の嵩上げ・移転（防災集団移転促進事業など）の費用の半分を国が負担する"}},
        # exogenous decision rules of the other actors (uncertainty of their decisions)
        "rules": {"staged_rho": 0.9, "restrained_rho": 1.3, "status": "仮定",
                  "basis": "積極投資: 全施策を最初に着手（予算超過分は繰越）。段階投資: 費用対効果の順に、観測・予測が 0.9 × 目標を超えるたび1つ。"
                           "投資抑制: 観測した被害が 1.3 × 目標を超えるまで何もしない、以後リードタイムの短い順に1つずつ"},
        # strategy genome: up to seq_len actions in each of the lists A / B, grids of rho and of the switch (projected 2100 rain)
        # switch_from_year: the trend of the observed rain says little about 2100 before then (correlation with the true 2100
        # rain 0.0 in 2030-2040, 0.3 in 2050, 0.6 in 2060, 0.9 in 2080 over the scenarios; results.json signal_quality)
        "strategy": {"seq_len": 4, "rho_grid": [0.7, 0.8, 0.9, 1.0, 1.1], "switch_grid": [1.08, 1.12, 1.16, 1.20, 1.24],
                     "switch_from_year": 2050},
        # NSGA-II (explore.nsga_mixed): per actor pop x gens strategies, basin basin_pop x basin_gens joint strategies
        "search": {"n_scenarios": 48, "pop": 24, "gens": 10, "p_mut": 0.2, "basin_pop": 32, "basin_gens": 12},
        "choose": {"coverage_tolerance": 0.03},          # recommended = cheapest within this of the highest coverage
        "options_year": 2060,                            # future options / lock-in counted at this decision year
        "dependence_scenarios": 16,
        "network": {"max_states_per_year": 6},
        # GUI recalculation (POST /api/dapp/multi/recalculate): computing sizes merged over this block; "full" = as above
        "gui": {"sizes": {
            "fast": {"robust_overrides": {"scenarios": {"n": 32}}, "dependence_scenarios": 8,
                     "search": {"n_scenarios": 16, "pop": 12, "gens": 4, "basin_pop": 16, "basin_gens": 4}},
            "standard": {"robust_overrides": {"scenarios": {"n": 64}},
                         "search": {"n_scenarios": 32, "pop": 16, "gens": 6, "basin_pop": 24, "basin_gens": 8}},
            "full": {}}},
    },
    # ---- Adaptive DAPP (adaptive.py, docs/11 11.11): state-dependent decision rules over single-measure actions, searched
    # per initial action of a Robust DAPP run (its settings, scenarios and initial actions; compared with its pathways)
    "adaptive": {
        "name": "abukuma_dapp_adaptive",
        "robust_run": "abukuma_dapp_robust",          # 11.9 run: settings, evaluation scenarios, initial actions, comparison
        "inits": "all",                                 # "all" initial actions of that run or "promising"
        # policy genome (grids): which indicators are monitored, when to act, how the next action is chosen
        "policy": {
            "rho_grid": [0.7, 0.8, 0.9, 1.0, 1.1],      # act when the observed / projected ratio / target exceeds rho
            "rain_grid": [1.08, 1.12, 1.16, 1.20, 1.24],  # act when the projected 2100 rain (observed trend) reaches it (from rain_from_year)
            "rain_from_year": 2050,                     # the rain trend says little before (docs/11 11.10.1)
            "weight_grid": [0.0, 0.5, 1.0, 2.0],        # cost / irreversibility / lead-time weights of the action score
            "horizon_grid": [10, 20, 30],               # projected benefit of an action this many years after its completion
            "k_grid": [1, 2],                           # actions per decision
            "scale_back_grid": [0.0, 0.5, 0.7],         # scale back a reversible measure when observed and projected < this (0 = never)
            "min_benefit": 0.005,                       # an action must lower the projected ratio / target by at least this
            "cost_ref_oku": 100.0, "irr_penalty": 0.05, "lead_penalty": 0.05, "lead_ref_years": 20.0,
            "status": "仮定",
            "basis": "行動の得点＝予測した効果 ÷ (1 + w_cost × 初期費用 / 100億円) − w_irr × 0.05 × 不可逆 − w_lead × 0.05 × リードタイム/20年"},
        "search": {"n_scenarios": 24, "pop": 16, "gens": 6, "p_mut": 0.2},
        "choose": {"coverage_tolerance": 0.03},
        # sensitivity (GUI / config): targets {damage_ratio, pop_ratio, from_year}, noise_damage / noise_climate (x the observation
        # noise of docs/13 monitoring), trend_window (observations in a trend), forecast_noise (relative error of the projected
        # rain increase; adaptive projections only).  Empty = the 11.9 run's values
        "sensitivity": {},
        "gui": {"sizes": {"fast": {"search": {"n_scenarios": 12, "pop": 10, "gens": 3}},
                          "standard": {"search": {"n_scenarios": 24, "pop": 16, "gens": 6}}}},
    },
}


def _merge(a, b):
    return RC._merge(a, b)


def load(path_or_dict=None) -> dict:
    over = path_or_dict if isinstance(path_or_dict, dict) else (
        json.loads(Path(path_or_dict).read_text(encoding="utf-8")) if path_or_dict else {})
    base = RC.load({k: over[k] for k in ("basin", "res", "event", "seed", "zones", "candidates", "stakeholders") if k in over})
    cfg = _merge(base, DEFAULT)
    cfg = _merge(cfg, {k: v for k, v in over.items() if k not in ("strategies",)})
    if "strategies" in over:                  # replace, not merge: strategies are a user choice
        cfg["strategies"] = copy.deepcopy(over["strategies"])
    if "actors" in over.get("multi", {}):     # the actors of the Multi-Actor DAPP too
        cfg["multi"]["actors"] = copy.deepcopy(over["multi"]["actors"])
    return cfg


def save(cfg, path):
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


QUICK = {"name": "abukuma_dapp_quick"}
QUICK_ROBUST = {"robust": {"name": "abukuma_dapp_robust_quick", "scenarios": {"n": 32},
                           "candidates": {"pop": 32, "gens": 10, "n_nodes": 12, "max_nodes": 120}, "initial": {"n_max": 8}}}
QUICK_MULTI = {"multi": {"name": "abukuma_dapp_multi_quick",
                         "robust_overrides": {"scenarios": {"n": 16}}, "dependence_scenarios": 8,
                         "search": {"n_scenarios": 9, "pop": 10, "gens": 3, "basin_pop": 12, "basin_gens": 3}}}
QUICK_ADAPTIVE = {"adaptive": {"name": "abukuma_dapp_adaptive_quick", "inits": "promising", "n_eval": 32,
                               "search": {"n_scenarios": 8, "pop": 8, "gens": 2}}}
