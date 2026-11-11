"""RDM analysis settings.  A config is plain JSON (see DEFAULT); `load` fills
missing keys from DEFAULT so that a run directory always holds the complete,
reproducible settings (config.json)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT = {
    "name": "abukuma_trial",
    "basin": "abukuma_whole",
    "res": 1000.0,
    "event": "abukuma_2019",
    "seed": 1,
    "budget_oku": 2000.0,                     # 総事業費（現在価値, 億円）の上限
    # zones: nested catchments of gauges, upstream first; the last zone (gauge null) is the rest
    "zones": [["上流", "本宮"], ["中流", "伏黒"], ["下流", None]],
    "candidates": {
        # per zone; "overflow": river cells (>= min_area_km2) that overtop in the baseline (within margin_m),
        # "area": all of them (the earlier default with min_area_km2 300 = main rivers, which hardly overtop at 1000 m)
        "levee": {"min_area_km2": 5.0, "heights_m": [0, 1, 2], "select": "overflow", "margin_m": 0.2},
        # 遊水地: the planned 阿武隈川上流遊水地群 (鏡石町・玉川村・矢吹町, 3 basins as one option; 国交省 福島河川国道事務所
        # 「阿武隈川上流遊水地群の概要」: 総面積 約350 ha, 貯留量 約1,500万m³, 全面買収・掘削).  Earlier runs (explore v2 and before)
        # used 3 automatic sites (n_sites, depth_m, site_km2, hand_max_m)
        # other basins (or a domain without the site) fall back to the automatic sites (n_sites ...)
        "retention": {"n_sites": 3, "depth_m": 3.0, "site_km2": 2.0, "hand_max_m": 3.0, "min_river_km2": 300.0,
                      "sites": [{"basin": "abukuma_whole", "label": "阿武隈川上流遊水地群", "lonlat": [140.392, 37.228], "radius_km": 2.5,
                                 "area_ha": 350.0, "volume_m3": 1.5e7, "status": "計画値",
                                 "basis": "国交省 福島河川国道事務所 阿武隈川上流遊水地群の概要（鏡石町・玉川村・矢吹町）"}]},
        "paddy_dam": {},                                                      # per zone, all paddies
        # 森林化（農地の植林）は 1K-RRI では扱わない（山地・平野の面積が変わる）。森林は森林整備（既存森林の物理パラメータ）で扱う
        "forest": None,
        # 森林整備 of the existing forest, per zone and owner (国有林 KSJ A45 / 民有林等)
        "forest_mgmt": {"hand_min_m": 0.0, "split_owner": False, "min_cells": 3},
        # housing: "split" = separate variables 移転 (none / relocate) and 嵩上げ (none / heights_m);
        # "combined" (stored configs without this key) = one variable none / raise (raise_m) / relocate
        "housing": "split",
        "relocate": {"depth_min_m": 1.0, "raise_m": 1.0, "dest": "new"},
        "raise": {"depth_min_m": 1.0, "heights_m": [1.0]},
        # 内水対策 (流域治水施策集 #5 #6 #8–#10): on cells flooded >= depth_min_m in the baseline
        "pump_urban": {"mmh": 10.0, "hand_max_m": 5.0, "depth_min_m": 0.1, "min_cells": 3},
        "pump_agri": {"mmh": 3.0, "hand_max_m": 5.0, "depth_min_m": 0.1, "min_cells": 3},
        "storage_public": {"depth_mm": 30.0, "release_mmh": 2.0, "depth_min_m": 0.1, "min_cells": 3},
        "storage_private": {"depth_mm": 10.0, "release_mmh": 2.0, "depth_min_m": 0.1, "min_cells": 3},
    },
    "optimisation": {"pop": 12, "gens": 4, "p_mutation": 0.15},
    # scenario design: every climate x n_per_climate LHS samples of the simulated factors;
    # each simulated scenario is combined with n_post samples of the post-hoc factors
    "scenarios": {"climates": ["current", "2C", "4C"], "n_per_climate": 2, "n_post": 8},
    "uncertainties": {
        # simulated (each value needs RRI runs)
        "rain_gamma": {"low": 0.9, "high": 1.2, "nominal": 1.0, "kind": "sim",
                       "label": "降雨強度の強調 γ（総雨量一定で強雨を強める）"},
        # post-hoc (applied to stored results, no RRI run)
        "cost_factor": {"low": 0.8, "high": 1.5, "nominal": 1.0, "kind": "post", "label": "事業費係数"},
        "asset_factor": {"low": 0.8, "high": 1.2, "nominal": 1.0, "kind": "post", "label": "資産額（被害額）係数"},
    },
    # stakeholders: evaluation scope (zones), weights of normalised metrics, satisficing regret
    "stakeholders": {
    # (Livability / Ecology composites are not weighted: auxiliary indicators only; robustness.utility still
    #  accepts "livability" / "ecology" weights for compatibility)
        "国": {"damage_zones": "all", "cost_zones": "all",
               "weights": {"damage": 0.55, "cost": 0.45}, "regret_threshold": 0.05},
        "上流自治体": {"damage_zones": ["上流", "中流"], "cost_zones": ["上流", "中流"],
                   "weights": {"damage": 0.65, "cost": 0.35}, "regret_threshold": 0.05},
        "下流自治体": {"damage_zones": ["下流"], "cost_zones": ["下流"],
                   "weights": {"damage": 0.75, "cost": 0.25}, "regret_threshold": 0.05},
        "地域主体": {"damage_zones": "all", "cost_zones": [],
                 "weights": {"households": 0.7, "damage": 0.3}, "regret_threshold": 0.05},
    },
    "sobol": {"n_base": 512},
    # ---- sensitivity analysis (rri_sim.rdm.sensitivity, scripts/run_sensitivity.py; docs/10_sensitivity.md)
    # status: 根拠 = range from a cited source, 仮定 = analyst's assumption (to be replaced when data exist)
    "sensitivity": {
        "parameters": {
            "rain_factor": {"low": 1.0, "high": 1.3, "nominal": 1.0, "kind": "sim", "status": "根拠",
                            "label": "降雨倍率（評価期間の総雨量）",
                            "basis": "国交省「気候変動を踏まえた治水計画のあり方」提言（改訂版）の降雨量変化倍率 2℃ 1.1・4℃ 1.2"
                                     "（その他の地域）。上限 1.3 は 4℃ の予測幅を見込んだ仮定"},
            "sea_level_rise_m": {"low": 0.0, "high": 0.7, "nominal": 0.0, "kind": "sim", "status": "根拠",
                                 "label": "海面上昇 [m]（河口の潮位境界）",
                                 "basis": "日本の気候変動2020 領域I（北海道・東北）21世紀末 2℃ 0.38 m・4℃ 0.70 m"},
            "rain_gamma": {"low": 0.8, "high": 1.3, "nominal": 1.0, "kind": "sim", "status": "仮定",
                           "label": "降雨強度分布 γ（総雨量一定で強雨を強める/弱める）",
                           "basis": "幅は仮定。d4PDF 等の時間雨量分布の変化で置き換えること"},
            "cost_factor": {"low": 0.8, "high": 1.5, "nominal": 1.0, "kind": "post", "status": "仮定",
                            "label": "事業費係数", "basis": "公共事業の事業費の増減幅を想定した仮定"},
            "asset_factor": {"low": 0.8, "high": 1.2, "nominal": 1.0, "kind": "post", "status": "仮定",
                             "label": "資産額（被害額）係数", "basis": "資産評価単価・延床面積推計の誤差を想定した仮定（docs/03）"},
        },
        # policies analysed: "recommended" = no measures + the minimax-regret policies of the RDM run
        "policies": "recommended",
        "morris": {"r": 4, "levels": 4},                     # trajectories, grid levels
        "screening": {"top_k": 4, "min_rel_mu_star": 0.05},  # keep the top_k sim parameters (and any above the threshold)
        # LHS points (RRI runs per policy) for training / validation; anchor_nominal adds the nominal point
        # (often a corner of the ranges, where a GP would otherwise extrapolate) to the training set
        "surrogate": {"n_train": 16, "n_val": 6, "anchor_nominal": True},
        "sobol": {"log2_n": 12, "n_boot": 200},              # 2^log2_n Sobol-sequence base samples (A and B independent)
        "decision": {"n_samples": 4096},
        "preference": {"n_weight_samples": 2000, "sweep_steps": 21},
    },
    # ---- exploration without stakeholder weights (explore.py, docs/12): damage target instead of a budget,
    # paddy dams / afforestation as implementation rates, uncertainties = sensitivity.parameters
    "explore": {
        "name": "abukuma_explore",
        "target_ratio": 1.0,                  # 目標: 被害額 <= 現在気候・対策なしの被害額 x target_ratio
        "continuous": ["paddy_dam", "forest", "forest_mgmt", "forest_mgmt_national", "storage_private"],   # implementation rate 0-100 % of the candidate cells
        "cell_order": "random",               # which cells a partial rate takes ("random" = spatially uniform, seeded)
        "current_baseline": True,             # measures set against the current-climate baseline (no extra run per climate)
        "zero_share": 0.2,                    # share of training samples with a rate of exactly 0 (measure not used)
        "design": {"n_train": 240, "n_none": 16, "n_val": 30, "adaptive_rounds": 6, "adaptive_batch": 10, "pool": 4000},
        "ensemble": {"n": 512},               # uncertainty samples per policy on the surrogate
        # objectives: cost, expected damage, target achievement rate.  Sub-indicators (flooded / relocated
        # population, service access, natural habitat, habitat connectivity) and the composite Livability /
        # Ecology scores are reported for every policy but not optimised (they can be added here if wanted)
        "optimisation": {"pop": 64, "gens": 60, "p_mut": 0.15,
                         "objectives": ["cost", "damage_mean", "reliability"]},
        "verify": {"n_policies": 10, "n_scenarios": 12},
        "target_curve": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        "rain_bins": 5,
        # trade spaces by implementing actor (explore_scope.py, docs/12): measure type -> actor, or {actor: cost
        # share} for a measure several actors implement jointly (it needs all of them; cost split by share).
        # 流域治水施策集の実施主体 (docs/13 13.10).  `zonal` maps an actor to an actor per zone (上流 / 下流の自治体).
        "scopes": {
            "actors": {"levee": "国", "retention": "国", "forest_mgmt_national": "国",
                       "pump_urban": "自治体", "storage_public": "自治体", "relocate": "自治体", "raise": "自治体",
                       # 農業用排水機場: 国営・県営土地改良事業 + 土地改良区（負担割合は仮定）
                       "pump_agri": {"国": 0.6, "自治体": 0.25, "地域主体": 0.15},
                       # 民有林の森林整備: 造林補助（国・県）・森林環境譲与税（市町村）+ 森林所有者（負担割合は仮定）
                       "forest_mgmt": {"国": 0.3, "自治体": 0.4, "地域主体": 0.3},
                       "paddy_dam": "地域主体", "forest": "地域主体", "storage_private": "地域主体"},
            "zonal": {"自治体": {"上流": "上流自治体", "中流": "上流自治体", "下流": "下流自治体"}},
            "optimisation": {"pop": 64, "gens": 40, "p_mut": 0.15,
                             "objectives": ["cost", "damage_mean", "reliability"]},
            "max_points": 1500,               # stored per actor combination (non-dominated first, then a random sample)
        },
    },
    # ---- RDM on the exploration surrogate (rdm_surrogate.py, docs/09 9.8): no RRI runs, no budget constraint;
    # stakeholders / thresholds as above, uncertainties = sensitivity.parameters (the exploration's ranges)
    "surrogate_rdm": {
        "name": "abukuma_rdm_surrogate",
        "explore_run": "abukuma_explore_v2",
        "cost_scale_oku": 2000.0,             # normalises the cost term of the utilities (was the budget; no constraint)
        "n_scenarios": 512,                   # common scenarios = the exploration's uncertainty ensemble (Sobol sequence)
        # NSGA-II on the stakeholders' mean utilities (adds policies that are good for someone to the pool)
        "optimisation": {"pop": 64, "gens": 40, "p_mut": 0.15, "n_scenarios": 128},
        "n_display": 20,                      # policies shown in the Regret charts / table
        "archive_display": 6000,              # evaluated policies shown in the Pareto chart (Pareto ones first)
        "sobol": {"n_base": 256, "pool_per_stakeholder": 300},
        # scenario colour classes by the rain factor (MLIT 降雨量変化倍率: 2℃ 1.1, 4℃ 1.2)
        "climate_bins": [[1.05, "current", "現在気候相当（降雨倍率 < 1.05）"], [1.15, "2C", "2℃相当（1.05–1.15）"],
                         [9.99, "4C", "4℃相当以上（≥ 1.15）"]],
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("stakeholders",) else v
    return out


def load(path_or_dict=None) -> dict:
    if path_or_dict is None:
        over = {}
    elif isinstance(path_or_dict, dict):
        over = path_or_dict
    else:
        over = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    return _merge(DEFAULT, over)


def save(cfg: dict, path) -> None:
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


QUICK_SA = {"sensitivity": {"morris": {"r": 2, "levels": 4}, "surrogate": {"n_train": 10, "n_val": 5},
                            "sobol": {"log2_n": 11, "n_boot": 100}}}

QUICK_SUR = {"surrogate_rdm": {"name": "abukuma_rdm_surrogate_quick", "n_scenarios": 128,
                               "optimisation": {"pop": 16, "gens": 3, "n_scenarios": 32}, "sobol": {"n_base": 64, "pool_per_stakeholder": 50}}}

QUICK = {"name": "abukuma_quick", "optimisation": {"pop": 8, "gens": 2},
         "scenarios": {"climates": ["current", "2C", "4C"], "n_per_climate": 1, "n_post": 8},
         "sobol": {"n_base": 256}}

QUICK_EX = {"explore": {"name": "abukuma_explore_quick",
                        "design": {"n_train": 48, "n_none": 6, "n_val": 10, "adaptive_rounds": 1, "adaptive_batch": 8, "pool": 2000},
                        "optimisation": {"pop": 40, "gens": 30}, "verify": {"n_policies": 5, "n_scenarios": 6}}}
