"""Roadmap settings.  Plain JSON merged over DEFAULT; a run directory keeps the
complete settings (config.json).  Assumed values carry status 根拠 (cited) /
仮定 (assumption) and their basis.  Uncertain values are {nominal, low, high}:
the planner believes `nominal`, a world draws a value in [low, high]
(world.FACTORS lists them)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..rdm import config as RC

DEFAULT = {
    "name": "abukuma_roadmap",
    "seed": 1,
    # physical layer: RRI-trained surrogate of an exploration run (docs/12) and its cost table (prepared once)
    "physics": {"explore_run": "abukuma_explore_v2"},
    "years": {"start": 2020, "end": 2100, "decision_step": 10, "eval_step": 5},
    "discount_rate": 0.04,
    "representative_event": {"annual_probability": 0.01, "status": "仮定",
                             "basis": "2019年台風19号級の洪水を年超過確率 1/100 の代表洪水とみなす（docs/11 と同じ近似）"},
    "budget": {"total_oku": 1500.0, "per_decade_oku": 500.0, "status": "仮定",
               "basis": "初期費用（割引前）の総額と10年ごとの上限。超過分は次の意思決定に繰り越す"},
    # ---- measures: phases decision -> consensus -> build (construction / planting) -> diffusion (adoption)
    # -> maturation; durations [years].  Uncertain phases are {nominal, low, high}.
    "measures": {
        "levee": {"tier": "国", "measure_no": "#1", "implementer": "河川管理者", "label": "堤防", "levels": [0, 1, 2], "staged": True, "consensus": 2.0,
                  "build": {"nominal": 8.0, "low": 6.0, "high": 12.0}, "diffuse": 0.0, "mature": 0.0,
                  "change_cost_fraction": 0.2, "maintenance_fraction": 0.005, "fail_prob": 0.05, "reversible": False,
                  "stakeholders": ["国", "自治体", "住民"], "status": "仮定",
                  "basis": "合意形成2年＋用地取得・築堤8年＝約10年（初期仮定）。維持費は初期費用の0.5%/年（仮定）"},
        "retention": {"tier": "国", "measure_no": "#1", "implementer": "河川管理者", "label": "遊水地", "levels": [0, 1], "consensus": 3.0,
                      "build": {"nominal": 7.0, "low": 2.0, "high": 12.0}, "diffuse": 0.0, "mature": 0.0,
                      "change_cost_fraction": 0.2, "maintenance_fraction": 0.005, "fail_prob": 0.3, "reversible": False,
                      "stakeholders": ["国", "自治体", "農家"], "status": "仮定",
                      "basis": "合意形成3年＋施工2〜12年＝5〜15年（初期仮定）。農地の買収・地役権の合意が要る"},
        "paddy_dam": {"tier": "地域", "measure_no": "#12", "implementer": "農業者", "label": "田んぼダム", "levels": [0.0, 0.5, 1.0], "consensus": 1.0, "build": 0.0,
                      "diffuse": {"nominal": 9.0, "low": 4.0, "high": 19.0}, "mature": 0.0,
                      "change_cost_fraction": 0.0, "maintenance_fraction": 0.0, "fail_prob": 0.15, "reversible": True,
                      "stakeholders": ["自治体", "農家"], "status": "仮定",
                      "basis": "協議1年＋普及4〜19年＝5〜20年（初期仮定）。実施率は参加率（social）で頭打ち"},
        "forest": {"tier": "地域", "measure_no": "#15", "implementer": "国・都道府県・市町村・森林所有者", "label": "森林化", "levels": [0.0, 0.5, 1.0], "consensus": 2.0, "build": 5.0, "diffuse": 0.0,
                   "mature": {"nominal": 30.0, "low": 20.0, "high": 40.0},
                   "change_cost_fraction": 0.0, "maintenance_fraction": 0.0, "fail_prob": 0.25, "reversible": False,
                   "stakeholders": ["自治体", "農家", "森林所有者"], "status": "仮定",
                   "basis": "植栽5年、成熟20〜40年で効果が最大（初期仮定）。農地の転換は戻さない（不可逆）"},
        "raise": {"tier": "自治体", "measure_no": "#19", "implementer": "市町村・都道府県（助成）", "label": "住宅嵩上げ", "consensus": 2.0, "build": 0.0,
                  "diffuse": {"nominal": 6.0, "low": 3.0, "high": 18.0}, "mature": 0.0,
                  "change_cost_fraction": 0.5, "maintenance_fraction": 0.0, "fail_prob": 0.1, "reversible": False,
                  "stakeholders": ["自治体", "住民"], "status": "仮定",
                  "basis": "合意2年＋実施3〜18年＝5〜20年（初期仮定）。移転に切り替えると嵩上げ費用の半分が無駄（仮定）"},
        "relocate": {"tier": "自治体", "measure_no": "#20", "implementer": "市町村（防災集団移転）", "label": "移転", "consensus": 3.0, "build": 0.0,
                     "diffuse": {"nominal": 9.0, "low": 2.0, "high": 17.0}, "mature": 0.0,
                     "change_cost_fraction": 0.2, "maintenance_fraction": 0.0, "fail_prob": 0.35, "reversible": False,
                     "stakeholders": ["国", "自治体", "住民"], "status": "仮定",
                     "basis": "合意3年＋移転2〜17年＝5〜20年（初期仮定）。防災集団移転は合意の失敗が多い（仮定）"},
        # ---- 内水対策（物理: RRI の排水能力・貯留, 流域治水施策集 #5 #6 #8–#10）
        "pump_urban": {"tier": "自治体", "label": "排水ポンプ（下水道）", "consensus": 2.0, "build": {"nominal": 7.0, "low": 4.0, "high": 12.0},
                       "diffuse": 0.0, "mature": 0.0, "change_cost_fraction": 0.0, "maintenance_fraction": 0.0, "fail_prob": 0.05,
                       "reversible": False, "stakeholders": ["自治体"], "measure_no": "#5", "implementer": "下水道管理者（市町村）",
                       "status": "仮定", "basis": "下水道の排水ポンプ場の増強。計画・施工 4〜12 年（仮定）"},
        "pump_agri": {"tier": "自治体", "label": "排水機場（農業）", "consensus": 2.0, "build": {"nominal": 7.0, "low": 4.0, "high": 12.0},
                      "diffuse": 0.0, "mature": 0.0, "change_cost_fraction": 0.0, "maintenance_fraction": 0.0, "fail_prob": 0.1,
                      "reversible": False, "stakeholders": ["国", "自治体", "農家"], "measure_no": "#6",
                      "implementer": "国・都道府県・農業水利施設管理者（土地改良区）", "status": "仮定",
                      "basis": "農業用の排水機場（土地改良事業）。受益農家の同意が要る"},
        "storage_public": {"tier": "自治体", "label": "雨水貯留浸透（公共）", "consensus": 2.0, "build": {"nominal": 6.0, "low": 3.0, "high": 10.0},
                           "diffuse": 0.0, "mature": 0.0, "change_cost_fraction": 0.0, "maintenance_fraction": 0.0, "fail_prob": 0.15,
                           "reversible": False, "stakeholders": ["自治体", "住民"], "measure_no": "#8 #9",
                           "implementer": "市町村・都道府県・下水道管理者", "status": "仮定",
                           "basis": "調整池・公共施設の貯留・下水道の貯留管。用地の合意が要る"},
        "storage_private": {"tier": "地域", "label": "雨水貯留浸透（各戸）", "levels": [0.0, 0.5, 1.0], "consensus": 1.0, "build": 0.0,
                            "diffuse": {"nominal": 10.0, "low": 5.0, "high": 20.0}, "mature": 0.0, "change_cost_fraction": 0.0,
                            "maintenance_fraction": 0.0, "fail_prob": 0.05, "reversible": True,
                            "stakeholders": ["民間事業者", "住民", "自治体"], "measure_no": "#10", "implementer": "民間事業者・個人（自治体が助成）",
                            "status": "仮定", "basis": "貯留タンク・浸透ます。実施率は参加率で頭打ち"},
        # ---- 制度的施策（社会・制度モデルのみ: 物理計算なし。被害額・資産・人口への効果は仮定）
        "zoning": {"tier": "自治体", "label": "浸水被害防止区域・災害危険区域", "institutional": True, "consensus": 3.0, "build": 0.0,
                   "diffuse": {"nominal": 10.0, "low": 5.0, "high": 20.0}, "mature": 0.0, "change_cost_fraction": 0.0,
                   "maintenance_fraction": 0.0, "fail_prob": 0.3, "reversible": False,
                   "stakeholders": ["自治体", "住民", "民間事業者"], "measure_no": "#17 #18", "implementer": "都道府県・市町村",
                   "cost_initial_oku": 2.0, "cost_annual_oku": 0.2, "effect": {"growth_block": 0.8},
                   "status": "仮定", "basis": "浸水域での新たな居住・資産の増加を 80% 抑える（区域指定・建築規制）"},
        "induce": {"tier": "自治体", "label": "居住誘導区域（防災指針）", "institutional": True, "consensus": 3.0, "build": 0.0,
                   "diffuse": {"nominal": 20.0, "low": 10.0, "high": 40.0}, "mature": 0.0, "change_cost_fraction": 0.0,
                   "maintenance_fraction": 0.0, "fail_prob": 0.25, "reversible": False, "stakeholders": ["自治体", "住民"],
                   "measure_no": "#22", "implementer": "市町村", "cost_initial_oku": 30.0, "cost_annual_oku": 1.0,
                   "effect": {"pop": 0.15, "asset": 0.10},
                   "status": "仮定", "basis": "浸水域の人口を 15%・資産を 10% 減らす（立地適正化計画による長期の誘導）"},
        "floodproof": {"tier": "地域", "label": "耐水化・止水板", "institutional": True, "levels": [0.0, 0.5, 1.0], "consensus": 1.0, "build": 0.0,
                       "diffuse": {"nominal": 10.0, "low": 5.0, "high": 20.0}, "mature": 0.0, "change_cost_fraction": 0.0,
                       "maintenance_fraction": 0.0, "fail_prob": 0.05, "reversible": False,
                       "stakeholders": ["民間事業者", "住民", "自治体"], "measure_no": "#29", "implementer": "市町村・都道府県・民間事業者",
                       "cost_initial_oku": 80.0, "cost_annual_oku": 0.0, "effect": {"damage": 0.15},
                       "status": "仮定", "basis": "実施した建物の被害を減らし、全体で被害額を最大 15% 減らす"},
        "evacuation": {"tier": "地域", "label": "ハザードマップ・避難", "institutional": True, "consensus": 1.0, "build": 0.0,
                       "diffuse": {"nominal": 5.0, "low": 2.0, "high": 10.0}, "mature": 0.0, "change_cost_fraction": 0.0,
                       "maintenance_fraction": 0.0, "fail_prob": 0.05, "reversible": False, "stakeholders": ["国", "自治体", "住民"],
                       "measure_no": "#26–#28", "implementer": "河川管理者・市町村・住民", "cost_initial_oku": 3.0, "cost_annual_oku": 1.0,
                       "effect": {"risk": 0.3},
                       "status": "仮定", "basis": "浸水域の住民の人的リスク（逃げ遅れ）を最大 30% 減らす。被害額は減らさない"},
    },
    # who can implement a measure (tier of its 実施主体 in 流域治水施策集): 国 (河川管理者), 自治体 (都道府県・市町村・
    # 下水道管理者・土地改良事業), 地域 (農業者・森林所有者・民間事業者・住民). A scope = the tiers taking part.
    "scopes": {"国のみ": ["国"], "国＋自治体": ["国", "自治体"], "国＋自治体＋地域（住民・農家・民間）": ["国", "自治体", "地域"]},
    # measure groups (complementarity, continuation library)
    "portfolio_groups": {"paddy_dam": ["paddy_dam"], "forest": ["forest"], "housing": ["relocate"], "retention": ["retention"],
                         "levee": ["levee"], "pump": ["pump_urban", "pump_agri"], "storage": ["storage_public", "storage_private"],
                         "landuse": ["zoning", "induce"], "floodproof": ["floodproof"], "evacuation": ["evacuation"]},
    # housing variables (住宅対策): option 1 = raise, 2 = relocate; staged = raise first, relocate later
    "housing_staged": True,
    # ---- SOCIAL / INSTITUTIONAL layer (social.py)
    "participation": {
        "stakeholders": {
            "国": {"coord_cost_oku_yr": 0.3, "role": "河川管理者・事業費"},
            "自治体": {"coord_cost_oku_yr": 0.5, "role": "土地利用・住宅対策・協議の場"},
            "農家": {"coord_cost_oku_yr": 0.3, "role": "田んぼダム・遊水地用地・農地転換"},
            "森林所有者": {"coord_cost_oku_yr": 0.2, "role": "森林化・森林管理"},
            "住民": {"coord_cost_oku_yr": 0.4, "role": "嵩上げ・移転・堤防用地・各戸貯留・避難"},
            "民間事業者": {"coord_cost_oku_yr": 0.2, "role": "各戸貯留・耐水化・土地利用規制"},
        },
        # early = engaged from the start year; staged = when the first measure that needs it is decided;
        # late = informed after the decision (trust deficit: slower, more failures, lower adoption)
        "mode": {"国": "staged", "自治体": "staged", "農家": "staged", "森林所有者": "staged", "住民": "staged", "民間事業者": "staged"},
        "engagement_tau_years": {"nominal": 8.0, "low": 4.0, "high": 15.0},          # e(t) = 1 - exp(-(t - t0) / tau)
        "consensus_reduction": {"nominal": 0.5, "low": 0.2, "high": 0.8},            # consensus x (1 - a e)
        "fail_scale": {"nominal": 1.0, "low": 0.5, "high": 2.0},                    # x measures.fail_prob
        "fail_engagement_reduction": 0.7,                                            # p x (1 - b e)
        "adoption": {"paddy_dam": [0.3, 0.9], "forest": [0.3, 0.8], "raise": [0.5, 0.95], "relocate": [0.4, 0.9],
                     "storage_private": [0.1, 0.6], "floodproof": [0.2, 0.7], "evacuation": [0.4, 0.9]},
        "adoption_shift": {"nominal": 0.0, "low": -0.15, "high": 0.15},
        "coord_cost_scale": {"nominal": 1.0, "low": 0.5, "high": 2.0},
        "consensus_cost_oku_yr": 0.5,                 # per measure and consensus year
        "late": {"consensus_factor": 1.5, "trust": 0.5, "fail_factor": 1.5, "conflict_cost_factor": 2.0},
        "status": "仮定",
        "basis": "参加の効果（合意期間・失敗確率・参加率）と調整費用の大きさはすべて仮定。地域の事例・聞き取りで置き換えること",
    },
    # ---- the world (world.py): climate, land use, measure effects, delays, social values, criteria regime
    "climate": {
        "belief": {"rain_2100": 1.10, "slr_2100": 0.38, "status": "根拠",
                   "basis": "計画時の想定＝2℃（国交省提言 降雨倍率 1.1、日本の気候変動2020 領域I 0.38 m）"},
        "rain_2100": {"nominal": 1.10, "low": 1.0, "high": 1.3, "status": "根拠",
                      "basis": "2℃ 1.1・4℃ 1.2（国交省提言）、上限 1.3 は予測幅を見込んだ仮定（代理モデルの学習範囲）"},
        "slr_2100": {"nominal": 0.38, "low": 0.2, "high": 0.7, "status": "根拠", "basis": "日本の気候変動2020 領域I 2℃ 0.38・4℃ 0.70 m"},
        "shift_year": {"nominal": 2050, "low": 2030, "high": 2070, "status": "仮定",
                       "basis": "その年まで想定（2℃）経路、以後 2100年値へ直線（気候の『想定外』の時期）"},
        "rain_gamma": {"nominal": 1.0, "low": 0.9, "high": 1.2, "status": "仮定", "basis": "降雨強度の強調（docs/10）"},
    },
    "landuse": {
        "asset_2100": {"nominal": 1.0, "low": 0.8, "high": 1.3, "status": "仮定",
                       "basis": "浸水域の資産（被害額）係数。低地の開発で増加・撤退で減少"},
        "pop_2100": {"nominal": 0.8, "low": 0.6, "high": 1.05, "status": "仮定",
                     "basis": "浸水域の人口係数。計画時の想定は人口減少 0.8（仮定）"},
        "shift_year": {"nominal": 2050, "low": 2030, "high": 2080, "status": "仮定", "basis": "資産の変化が始まる年"},
    },
    # measure effects (RRI effect coefficients, docs/10): planner's belief = nominal; technology changes
    # the NbS / operation effects (tech_types) by tech_gain over 10 years from tech_year
    "effects": {
        "levee": {"nominal": 0.9, "low": 0.5, "high": 1.0}, "retention": {"nominal": 0.85, "low": 0.5, "high": 1.0},
        "paddy_dam": {"nominal": 0.6, "low": 0.3, "high": 1.0}, "forest": {"nominal": 0.6, "low": 0.3, "high": 1.0},
        "raise": {"nominal": 0.8, "low": 0.5, "high": 1.0},
        "tech_types": ["paddy_dam", "forest", "retention"],
        "tech_gain": {"nominal": 0.0, "low": -0.2, "high": 0.3, "status": "仮定",
                      "basis": "技術進歩（自動調整板・森林管理・遊水地の運用）で +、劣化・堆砂で −"},
        "tech_year": {"nominal": 2050, "low": 2030, "high": 2080},
        "status": "仮定", "basis": "効果係数の幅は docs/10 の感度分析の範囲（代理モデルの学習範囲）",
    },
    # ---- evaluation criteria (regimes): weights of normalised terms and tolerated ratios (ATP thresholds)
    "regimes": {
        "現行": {"weights": {"damage": 0.45, "cost": 0.25, "pop": 0.1, "ecology": 0.1, "livability": 0.05, "agriculture": 0.05},
                 "thresholds": {"damage_ratio": 1.2, "pop_ratio": 1.2}},
        "環境・農業重視": {"weights": {"damage": 0.3, "cost": 0.2, "pop": 0.1, "ecology": 0.2, "livability": 0.05, "agriculture": 0.15},
                     "thresholds": {"damage_ratio": 1.2, "pop_ratio": 1.2}},
        "安全重視": {"weights": {"damage": 0.45, "cost": 0.15, "pop": 0.25, "ecology": 0.05, "livability": 0.05, "agriculture": 0.05},
                 "thresholds": {"damage_ratio": 0.8, "pop_ratio": 0.8}},
    },
    "initial_regime": "現行",
    "regime_change": {"year": {"nominal": 2060, "low": 2030, "high": 2090}, "p_none": 0.25, "status": "仮定",
                      "basis": "評価基準（重み・許容閾値）がその年に別のレジームへ変わる。確率 p_none で変わらない"},
    "threshold_penalty": 0.3,         # utility penalty x share of periods above a tolerated ratio
    # ---- monitoring and detection (P1 triggers, P2/P3 updates)
    "monitoring": {"risk_noise": 0.05, "rain_noise": 0.015, "slr_noise": 0.02, "effect_noise": 0.06, "exposure_noise": 0.02,
                   "regime_lag_years": 5, "status": "仮定", "basis": "観測の誤差（相対・絶対）と、評価基準の変化が政策に反映されるまでの遅れ"},
    "detect": {"rain_tol": 0.025, "slr_tol": 0.06, "effect_tol": 0.12, "exposure_tol": 0.08, "delay_tol": 0.3, "trend_window": 3},
    "trigger": {"ratio_of_threshold": 0.9, "anticipate": True},
    # ---- policies compared
    "policies": {
        "P0": {"label": "P0 固定計画", "kind": "fixed"},
        "P1": {"label": "P1 従来型DAPP", "kind": "dapp", "update_model": False, "update_criteria": False},
        "P2": {"label": "P2 仮定更新DAPP", "kind": "dapp", "update_model": True, "update_criteria": False},
        "P3": {"label": "P3 仮定・基準更新DAPP", "kind": "dapp", "update_model": True, "update_criteria": True},
    },
    "replan": {"max": 4},
    # roadmap optimisation (GA on the surrogate, from the inherited state)
    "optimiser": {"pop": 24, "gens": 12, "p_mut": 0.15, "n_belief": 3,
                  "belief_spread": {"rain_2100": 0.05, "effect": 0.1, "delay": 0.25}},
    # ---- experiments
    "experiment": {
        "n_worlds": 64, "oracle": True, "regret_threshold": 0.05,
        "checkpoints": [2040, 2060], "n_goals": 40, "n_continuations": 48, "goal_lag_years": 20,
        "goals": {"damage_ratio": [1.0, 1.6], "pop_ratio": [0.8, 2.0], "ecology_delta": [-1.2, 0.2], "agriculture_min": [88.0, 100.0]},
        "morris": {"r": 3, "levels": 4},
        "sobol": {"log2_n": 11, "n_boot": 100},
        "prim": {"alpha": 0.05, "min_mass": 0.05, "density": 0.8},
        "delay": {"scales": [0.5, 1.0, 1.5, 2.0], "postpone": [2020, 2030, 2040], "n_worlds": 16},
        "complementarity": {"n_worlds": 32},
        "participation": {"modes": ["early", "staged", "late"], "n_worlds": 16},
        "scope": {"n_worlds": 32},
        "verify_rri": 0,
        "n_jobs": 0,
    },
}

QUICK = {"name": "abukuma_roadmap_quick",
         "optimiser": {"pop": 12, "gens": 5, "n_belief": 2},
         "experiment": {"n_worlds": 8, "n_goals": 24, "n_continuations": 24, "morris": {"r": 1, "levels": 4},
                        "sobol": {"log2_n": 9, "n_boot": 50},
                        "delay": {"scales": [0.5, 1.0, 2.0], "postpone": [2020, 2040], "n_worlds": 6},
                        "complementarity": {"n_worlds": 8}, "participation": {"n_worlds": 6}, "scope": {"n_worlds": 8}}}


def _merge(a, b):
    return RC._merge(a, b)


def load(path_or_dict=None) -> dict:
    over = path_or_dict if isinstance(path_or_dict, dict) else (
        json.loads(Path(path_or_dict).read_text(encoding="utf-8")) if path_or_dict else {})
    cfg = _merge(DEFAULT, {k: v for k, v in over.items() if k not in ("regimes", "policies")})
    for k in ("regimes", "policies"):                # replace, not merge: user choices
        if k in over:
            cfg[k] = copy.deepcopy(over[k])
    return cfg


def save(cfg, path):
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def uval(spec):
    """Nominal value of a number or a {nominal, low, high} spec."""
    return float(spec["nominal"]) if isinstance(spec, dict) else float(spec)
