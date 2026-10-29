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
    return cfg


def save(cfg, path):
    Path(path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


QUICK = {"name": "abukuma_dapp_quick"}
