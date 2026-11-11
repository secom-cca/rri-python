"""Single-measure effects on the surrogate (explore v2) for the DAPP paper outline -> measure_effects.json.
    .venv/bin/python docs/paper/dapp/measure_effects.py"""
import json
from pathlib import Path
import numpy as np
from rri_sim.roadmap.physics import SurrogatePhysics

import sys
EXP = sys.argv[1] if len(sys.argv) > 1 else "abukuma_explore_v2"
ph = SurrogatePhysics(EXP, log=lambda *a: None)
OUT = Path(__file__).parent / ("measure_effects.json" if EXP.endswith("v2") else f"measure_effects_{EXP.split('_')[-1]}.json")
nom = {"levee_effect": 0.9, "retention_effect": 0.85, "paddy_effect": 0.6, "forest_effect": 0.6, "raise_effect": 0.8}
clim = {"現在": (1.0, 0.0), "2℃": (1.1, 0.38), "4℃": (1.2, 0.54)}
cols = {}
for c, (i, eff) in enumerate(ph.layout):
    cols.setdefault(i, []).append((c, eff))


def raw_of(i, lv):
    v = ph.variables[i]; r = np.zeros(len(ph.layout)); c = [x[0] for x in cols[i]]
    if v["type"] == "levee":
        r[c[0]] = v["heights"][lv] / max(v["heights"])
    elif v["type"] == "relocate":
        r[c[1 if lv == 2 else 0]] = 1.0
    elif v["type"] == "forest":
        r[c[0]] = r[c[1]] = lv
    else:
        r[c[0]] = lv
    return r


rows = []
for i, v in enumerate(ph.variables):
    levels = ([1, 2] if v["type"] in ("levee", "relocate") and len(v["options"]) == 3 else [1.0] if not v["continuous"] else [1.0])
    for lv in levels:
        raw = np.vstack([np.zeros(len(ph.layout)), raw_of(i, lv)])
        res = {}
        for k, (rf, slr) in clim.items():
            env = {"rain_factor": np.full(2, rf), "sea_level_rise_m": np.full(2, slr), "rain_gamma": np.ones(2)}
            env.update({kk: np.full(2, vv) for kk, vv in nom.items()})
            o = ph.predict(raw, env)
            res[k] = {"damage": [float(o["damage"][0]), float(o["damage"][1])], "pop": [float(o["pop_hazard"][0]), float(o["pop_hazard"][1])],
                      "ecology": float(o["ecology"][1] - o["ecology"][0]), "livability": float(o["livability"][1] - o["livability"][0])}
        ini, ann = ph.cost_parts(i, lv)
        opt = v["options"][int(lv)] if not v["continuous"] else "100%"
        rows.append({"name": v["name"], "type": v["type"], "zone": v["zone"], "level": opt, "initial_oku": ini, "annual_oku": ann, **res})
OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
for r in rows:
    d0, d1 = r["現在"]["damage"]; e0, e1 = r["4℃"]["damage"]
    print(f'{r["name"]:22s} {r["level"]:14s} 初期 {r["initial_oku"]:6.1f} 年 {r["annual_oku"]:5.2f}  被害 現在 {d1-d0:+7.1f}  4℃ {e1-e0:+7.1f}  人口 {r["現在"]["pop"][1]-r["現在"]["pop"][0]:+7.0f}  Eco {r["現在"]["ecology"]:+.2f} Liv {r["現在"]["livability"]:+.2f}')
