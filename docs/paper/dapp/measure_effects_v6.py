"""Single-measure effects on the v6 surrogate (1K-RRI, no effect coefficients) for the paper -> measure_effects_v6.json.
Per variable and level: initial / annual cost, change in event damage (representative event) under the current climate and
4 deg C (rain x1.2, sea level +0.70 m), change in the exposed population, and change in the annual expected damage
(return periods 1/2-1/200, rri_sim.frequency).

    .venv/bin/python docs/paper/dapp/measure_effects_v6.py
"""
import json
from pathlib import Path

import numpy as np

from rri_sim import frequency as FQ
from rri_sim.dapp import config as CF
from rri_sim.roadmap.physics import SurrogatePhysics

ph = SurrogatePhysics("abukuma_explore_v6", log=lambda *a: None)
fq = CF.load()["forecast"]["frequency"]
T, rT = np.array(fq["T"], float), np.array(fq["ratio"], float)
cols = {}
for c, (i, eff) in enumerate(ph.layout):
    cols.setdefault(i, []).append(c)
CLIM = {"current": (1.0, 0.0), "4C": (1.2, 0.70)}


def raw_of(i, lv):
    v = ph.variables[i]; r = np.zeros(len(ph.layout)); c = cols[i]
    if lv == 0:
        return r
    if v["type"] == "levee":
        r[c[0]] = v["heights"][lv] / max(v["heights"])
    elif v["type"] == "relocate":
        r[c[1 if lv == 2 else 0]] = 1.0
    elif v["continuous"]:
        r[c[0]] = lv
    else:
        r[c[0]] = lv / max(len(v["options"]) - 1, 1)
    return r


def outcomes(raw):
    out = {}
    for k, (rf, slr) in CLIM.items():
        y = ph.predict(raw[None], {"rain_factor": np.array([rf]), "sea_level_rise_m": np.array([slr]), "rain_gamma": np.array([1.0])})
        out[k] = (float(y["damage"][0]), float(y["pop_hazard"][0]))
        n = len(T)
        yT = ph.predict(np.repeat(raw[None], n, 0), {"rain_factor": rf * rT, "sea_level_rise_m": np.full(n, slr), "rain_gamma": np.ones(n)})
        out[k + "_ead"] = float(FQ.ead(yT["damage"], T))
    return out


base = outcomes(np.zeros(len(ph.layout)))
rows = []
for i, v in enumerate(ph.variables):
    levels = [1.0] if v["continuous"] else list(range(1, len(v["options"])))
    for lv in levels:
        o = outcomes(raw_of(i, lv))
        ini, ann = ph.cost_parts(i, lv)
        rows.append({"name": v["name"], "type": v["type"], "zone": v["zone"], "level": v["options"][-1] if v["continuous"] else v["options"][lv],
                     "initial": ini, "annual": ann,
                     "d_cur": o["current"][0] - base["current"][0], "d_4c": o["4C"][0] - base["4C"][0],
                     "p_cur": o["current"][1] - base["current"][1],
                     "ead_cur": o["current_ead"] - base["current_ead"], "ead_4c": o["4C_ead"] - base["4C_ead"]})
out = Path(__file__).parent / "measure_effects_v6.json"
out.write_text(json.dumps({"base": base, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(base, ensure_ascii=False))
for r in rows:
    print(f"{r['name']:24s} {r['level']:28s} ini {r['initial']:6.1f} ann {r['annual']:5.2f} | dD {r['d_cur']:7.1f} / {r['d_4c']:7.1f} | dP {r['p_cur']:7.0f} | dEAD {r['ead_cur']:6.1f} / {r['ead_4c']:6.1f}")
