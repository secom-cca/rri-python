"""RRI check that every measure of the exploration acts, and that its effect coefficient scales it.

For each decision variable of the exploration (default abukuma_explore_v3): the largest option alone (other measures
none) with the effect coefficient 1.0 and, for measures that have one, 0.5; levees also under 4℃.  RRI damage /
exposed population vs no measures, and the surrogate's prediction of the same states.

    .venv/bin/python scripts/verify_measures_v3.py [--explore abukuma_explore_v3]
Output: data/processed/roadmap/_physics/<explore>/verify_measures.json
"""
import argparse
import json

import numpy as np

from rri_sim.roadmap.physics import PHYS_DIR, SurrogatePhysics, rri_evaluator

ap = argparse.ArgumentParser()
ap.add_argument("--explore", default="abukuma_explore_v3")
a = ap.parse_args()
log = lambda m: print(m, flush=True)  # noqa: E731
ph = SurrogatePhysics(a.explore, log=log)
ev = rri_evaluator(a.explore, log)
V = ev.variables
COEF = {"levee": "levee_effect", "retention": "retention_effect", "paddy_dam": "paddy_effect", "forest": "forest_effect"}
CLIM = {"現在気候": {"rain_factor": 1.0, "sea_level_rise_m": 0.0, "rain_gamma": 1.0},
        "4℃": {"rain_factor": 1.2, "sea_level_rise_m": 0.54, "rain_gamma": 1.0}}
ONE = {"levee_effect": 1.0, "retention_effect": 1.0, "paddy_effect": 1.0, "forest_effect": 1.0, "raise_effect": 1.0}
cols = {}
for c, (i, eff) in enumerate(ph.layout):
    cols.setdefault(i, []).append(c)


def raw_of(i, opt):
    """Surrogate raw columns of variable i at option opt (rate 1.0 for continuous variables)."""
    v, r = ph.variables[i], np.zeros(len(ph.layout))
    c = cols[i]
    if v["type"] == "levee":
        r[c[0]] = opt / (len(v["options"]) - 1)
    elif v["type"] == "relocate":
        r[c[1 if opt == 2 else 0]] = 1.0
    elif v["type"] == "forest":
        r[c[0]] = r[c[1]] = 1.0
    elif v["continuous"]:
        r[c[0]] = 1.0
    else:
        r[c[0]] = opt / max(len(v["options"]) - 1, 1)
    return r


def run(x, sim):
    y = ev.simulate(x, sim)
    return y["damage"], y["pop_hazard"]


def sur(raw, sim):
    p = ph.predict(raw[None], {k: np.atleast_1d(v) for k, v in sim.items()})
    return float(p["damage"][0]), float(p["pop_hazard"][0])


rows = []
base = {cn: run([0] * len(V), {**c, **ONE}) for cn, c in CLIM.items()}
sbase = {cn: sur(np.zeros(len(ph.layout)), {**c, **ONE}) for cn, c in CLIM.items()}
for i, v in enumerate(V):
    opts = [1, 2] if v.type == "relocate" and len(v.options) == 3 else [len(v.options) - 1 if not v.continuous else 1.0]
    for opt in opts:
        kind = "raise" if v.type == "relocate" and opt == 1 else v.type
        coef = "raise_effect" if kind == "raise" else COEF.get(v.type)
        x = [0] * len(V); x[i] = opt
        for cn in (["現在気候", "4℃"] if v.type == "levee" else ["現在気候"]):
            for k in ([1.0, 0.5] if coef else [1.0]):
                sim = {**CLIM[cn], **ONE, **({coef: k} if coef else {})}
                d, p = run(x, sim)
                sd, sp = sur(raw_of(i, opt), sim)
                rows.append({"name": v.name, "type": kind, "option": v.options[int(opt)] if not v.continuous else "100%",
                             "climate": cn, "coef": coef, "k": k, "rri_d": d - base[cn][0], "rri_p": p - base[cn][1],
                             "sur_d": sd - sbase[cn][0], "sur_p": sp - sbase[cn][1]})
                r = rows[-1]
                log(f"  {v.name} {r['option']} {cn} {coef or '係数なし'}={k}: 被害 RRI {r['rri_d']:+.1f} / 代理 {r['sur_d']:+.1f} 億円、"
                    f"浸水人口 RRI {r['rri_p']:+.0f} / 代理 {r['sur_p']:+.0f}")
out = PHYS_DIR / a.explore / "verify_measures.json"
out.write_text(json.dumps({"base": base, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"保存: {out}")
