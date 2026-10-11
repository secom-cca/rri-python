"""Project cost estimates for flood-management measures.

All unit costs are *editable defaults* (GUI: 単価設定) with their basis given
in `UNIT_COSTS[...]["basis"]`.  Several are assumptions where no public
standard unit cost exists; they must be replaced by project-specific
estimates for decision making.

Present value follows 治水経済調査マニュアル（案）: evaluation period 50 years,
social discount rate 4 %  ->  annuity factor  (1 - 1.04^-50) / 0.04 = 21.48.
"""
from __future__ import annotations

import copy

import numpy as np

DISCOUNT = 0.04
YEARS = 50
ANNUITY = (1 - (1 + DISCOUNT) ** -YEARS) / DISCOUNT

UNIT_COSTS = {
    "levee_fill_yen_m3": {"value": 8000.0, "label": "築堤 盛土単価（材料・運搬・締固め・法面）", "unit": "円/m³",
                          "basis": "仮定。掘削単価(319–777円/m³, 大和川保田遊水地 工事設計書 R6)に盛土材・運搬・締固め・法面保護を加算"},
    "levee_fixed_yen_m": {"value": 50000.0, "label": "築堤 付帯工（護岸・天端舗装等）", "unit": "円/m", "basis": "仮定"},
    "levee_crest_m": {"value": 5.0, "label": "堤防天端幅", "unit": "m", "basis": "河川管理施設等構造令の目安"},
    "levee_slope": {"value": 2.0, "label": "法勾配（1:n）", "unit": "-", "basis": "構造令の標準 2割"},
    "land_yen_m2": {"value": 5000.0, "label": "用地費（農地・山林）", "unit": "円/m²", "basis": "仮定（地方部の農地価格の目安）"},
    "excavation_yen_m3": {"value": 1500.0, "label": "遊水地 掘削・運搬・処分", "unit": "円/m³",
                          "basis": "掘削319–777円/m³（工事設計書）＋運搬・残土処分を仮定"},
    "weir_yen_m": {"value": 1.0e6, "label": "越流堤", "unit": "円/m", "basis": "仮定"},
    "paddy_dam_init_yen_10a": {"value": 5000.0, "label": "田んぼダム 調整板設置", "unit": "円/10a", "basis": "仮定（調整板 数千円/枚、10aに1枚）"},
    "paddy_dam_annual_yen_10a": {"value": 400.0, "label": "田んぼダム 維持（年）", "unit": "円/10a/年",
                                 "basis": "多面的機能支払交付金 田んぼダム加算 400円/10a（香川県資料）"},
    "afforest_yen_ha": {"value": 1.5e6, "label": "植林・保育", "unit": "円/ha", "basis": "仮定（植栽・下刈り等の初期費用）"},
    "crop_loss_yen_ha_yr": {"value": 0.3 * 4.75 * 187000, "label": "森林化による営農所得の逸失（年）", "unit": "円/ha/年",
                            "basis": "水稲 平年収量4.75 t/ha × 米価187千円/t（資産評価単価R7）× 所得率0.3（仮定）"},
    "relocate_yen_household": {"value": 2.0e7, "label": "移転（1世帯）", "unit": "円/世帯",
                               "basis": "防災集団移転促進事業の事例 1,036–1,680万円/戸（2005）を物価補正した仮定値"},
    "relocate_yen_employee": {"value": 3.0e6, "label": "事業所移転（従業者1人）", "unit": "円/人", "basis": "仮定"},
    "raise_floor_yen_household_m": {"value": 6.0e6, "label": "住宅嵩上げ（1世帯・1 m）", "unit": "円/世帯/m", "basis": "仮定（曳家・ジャッキアップ工法）"},
    "raise_fill_yen_m3": {"value": 5000.0, "label": "宅地盛土", "unit": "円/m³", "basis": "仮定"},
}


def default_unit_costs() -> dict:
    return {k: v["value"] for k, v in UNIT_COSTS.items()}


def estimate(edits_applied: list[dict], unit: dict | None = None) -> dict:
    """Cost of applied measures.

    edits_applied: list of dicts produced by the scenario engine with the
    physical quantities of each measure (keys documented in `quantities`).
    Returns {"items": [...], "initial_yen", "annual_yen", "pv_yen"}.
    """
    u = default_unit_costs()
    u.update(unit or {})
    items = []
    for e in edits_applied:
        t = e["type"]
        q = e.get("q", {})
        init = annual = 0.0
        if t == "levee":
            h = q["height_m"]
            area = h * (u["levee_crest_m"] + u["levee_slope"] * h)          # m²/m
            foot = u["levee_crest_m"] + 2 * u["levee_slope"] * h            # m
            L = q["length_m"]
            init = L * (area * u["levee_fill_yen_m3"] + u["levee_fixed_yen_m"] + foot * u["land_yen_m2"])
        elif t == "retention":
            init = (q["area_m2"] * q["depth_m"] * u["excavation_yen_m3"] + q["area_m2"] * u["land_yen_m2"]
                    + q["weir_length_m"] * u["weir_yen_m"]
                    + q["households"] * u["relocate_yen_household"])
        elif t == "paddy_dam":
            a10 = q["area_m2"] / 1000.0
            init = a10 * u["paddy_dam_init_yen_10a"]
            annual = a10 * u["paddy_dam_annual_yen_10a"]
        elif t == "forest":
            ha = q["area_m2"] / 1e4
            init = ha * u["afforest_yen_ha"]
            annual = q["farm_area_m2"] / 1e4 * u["crop_loss_yen_ha_yr"]
        elif t == "relocate":
            init = q["households"] * u["relocate_yen_household"] + q["employees"] * u["relocate_yen_employee"]
        elif t == "raise":
            if q.get("terrain"):
                init = q["area_m2"] * q["height_m"] * u["raise_fill_yen_m3"]
            else:
                init = q["households"] * q["height_m"] * u["raise_floor_yen_household_m"]
        pv = init + annual * ANNUITY
        items.append({"type": t, "initial_yen": init, "annual_yen": annual, "pv_yen": pv, "q": q})
    return {"items": items, "initial_yen": sum(i["initial_yen"] for i in items),
            "annual_yen": sum(i["annual_yen"] for i in items),
            "pv_yen": sum(i["pv_yen"] for i in items), "annuity_factor": ANNUITY,
            "unit_costs": copy.deepcopy(u)}
