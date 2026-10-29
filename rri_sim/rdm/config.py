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
        "levee": {"min_area_km2": 300.0, "heights_m": [0, 1, 2]},            # per zone, main rivers
        "retention": {"n_sites": 3, "depth_m": 3.0, "site_km2": 2.0, "hand_max_m": 3.0, "min_river_km2": 300.0},
        "paddy_dam": {},                                                      # per zone, all paddies
        "forest": {"hand_min_m": 20.0},                                       # per zone, upland farmland
        "relocate": {"depth_min_m": 1.0, "raise_m": 1.0, "dest": "new"},      # per zone: none / raise / relocate
    },
    "optimisation": {"pop": 12, "gens": 4, "p_mutation": 0.15},
    # scenario design: every climate x n_per_climate LHS samples of the simulated factors;
    # each simulated scenario is combined with n_post samples of the post-hoc factors
    "scenarios": {"climates": ["current", "2C", "4C"], "n_per_climate": 2, "n_post": 8},
    "uncertainties": {
        # simulated (each value needs RRI runs)
        "rain_gamma": {"low": 0.9, "high": 1.2, "nominal": 1.0, "kind": "sim",
                       "label": "降雨強度の強調 γ（総雨量一定で強雨を強める）"},
        "paddy_effect": {"low": 0.5, "high": 1.0, "nominal": 1.0, "kind": "sim",
                         "label": "田んぼダムの効果係数（貯留深 15 cm × 係数）"},
        "forest_effect": {"low": 0.5, "high": 1.0, "nominal": 1.0, "kind": "sim",
                          "label": "森林化の効果係数（土層厚・粗度の増分 × 係数）"},
        "retention_effect": {"low": 0.7, "high": 1.0, "nominal": 1.0, "kind": "sim",
                             "label": "遊水地の効果係数（有効貯留深 × 係数）"},
        # post-hoc (applied to stored results, no RRI run)
        "cost_factor": {"low": 0.8, "high": 1.5, "nominal": 1.0, "kind": "post", "label": "事業費係数"},
        "asset_factor": {"low": 0.8, "high": 1.2, "nominal": 1.0, "kind": "post", "label": "資産額（被害額）係数"},
    },
    # stakeholders: evaluation scope (zones), weights of normalised metrics, satisficing regret
    "stakeholders": {
        "国": {"damage_zones": "all", "cost_zones": "all",
               "weights": {"damage": 0.45, "cost": 0.35, "livability": 0.1, "ecology": 0.1}, "regret_threshold": 0.05},
        "上流自治体": {"damage_zones": ["上流", "中流"], "cost_zones": ["上流", "中流"],
                   "weights": {"damage": 0.4, "cost": 0.2, "livability": 0.25, "ecology": 0.15}, "regret_threshold": 0.05},
        "下流自治体": {"damage_zones": ["下流"], "cost_zones": ["下流"],
                   "weights": {"damage": 0.6, "cost": 0.2, "livability": 0.1, "ecology": 0.1}, "regret_threshold": 0.05},
        "住民": {"damage_zones": "all", "cost_zones": [],
               "weights": {"households": 0.6, "damage": 0.1, "livability": 0.3}, "regret_threshold": 0.05},
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
            "levee_effect": {"low": 0.5, "high": 1.0, "nominal": 1.0, "kind": "sim", "status": "仮定",
                             "label": "堤防の効果係数（嵩上げの実効高さ）", "basis": "施工区間の不連続・沈下・弱部を想定した仮定"},
            "retention_effect": {"low": 0.5, "high": 1.0, "nominal": 1.0, "kind": "sim", "status": "仮定",
                                 "label": "遊水地の効果係数（有効貯留深）", "basis": "越流堤の運用・堆砂を想定した仮定"},
            "paddy_effect": {"low": 0.3, "high": 1.0, "nominal": 1.0, "kind": "sim", "status": "仮定",
                             "label": "田んぼダムの効果係数（貯留深 15 cm × 係数）", "basis": "取組率・排水口管理のばらつきを想定した仮定"},
            "forest_effect": {"low": 0.3, "high": 1.0, "nominal": 1.0, "kind": "sim", "status": "仮定",
                              "label": "森林化の効果係数（土層厚・粗度の増分 × 係数）", "basis": "森林の成熟度（植栽後の年数）を想定した仮定"},
            "raise_effect": {"low": 0.5, "high": 1.0, "nominal": 1.0, "kind": "sim", "status": "仮定",
                             "label": "嵩上げの効果係数（床高の上げ幅 × 係数）", "basis": "対象世帯の実施率を想定した仮定"},
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
        "continuous": ["paddy_dam", "forest"],   # implementation rate 0-100 % of the candidate cells
        "cell_order": "random",               # which cells a partial rate takes ("random" = spatially uniform, seeded)
        "current_baseline": True,             # measures set against the current-climate baseline (no extra run per climate)
        "zero_share": 0.2,                    # share of training samples with a rate of exactly 0 (measure not used)
        "design": {"n_train": 240, "n_none": 16, "n_val": 30, "adaptive_rounds": 6, "adaptive_batch": 10, "pool": 4000},
        "ensemble": {"n": 512},               # uncertainty samples per policy on the surrogate
        "optimisation": {"pop": 64, "gens": 60, "p_mut": 0.15,
                         "objectives": ["cost", "reliability", "damage_mean", "ecology"]},
        "verify": {"n_policies": 10, "n_scenarios": 12},
        "target_curve": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        "rain_bins": 5,
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

QUICK = {"name": "abukuma_quick", "optimisation": {"pop": 8, "gens": 2},
         "scenarios": {"climates": ["current", "2C", "4C"], "n_per_climate": 1, "n_post": 8},
         "sobol": {"n_base": 256}}

QUICK_EX = {"explore": {"name": "abukuma_explore_quick",
                        "design": {"n_train": 48, "n_none": 6, "n_val": 10, "adaptive_rounds": 1, "adaptive_batch": 8, "pool": 2000},
                        "optimisation": {"pop": 40, "gens": 30}, "verify": {"n_policies": 5, "n_scenarios": 6}}}
