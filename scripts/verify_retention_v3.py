"""RRI check of the 遊水地 (阿武隈川上流遊水地群) effect on the v3 surrogate.

For initial actions of the v3 Adaptive DAPP run, with and without the retention basin group, under current / 2℃ / 4℃
climate (nominal effect coefficients): RRI damage vs surrogate damage and the retention effect (with - without).

    .venv/bin/python scripts/verify_retention_v3.py [--inits 0,1,4,62,92]
Output: data/processed/dapp/abukuma_dapp_robust_v3/verify_retention.json
"""
import argparse
import json

import numpy as np

from rri_sim.dapp import robust as RB
from rri_sim.roadmap.physics import rri_evaluator

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="abukuma_dapp_robust_v3")
ap.add_argument("--inits", default="0,1,4,62,92")
a = ap.parse_args()
log = lambda m: print(m, flush=True)  # noqa: E731
out = RB.DAPP_DIR / a.run
net, *_ = RB.load_net(out, log=log)
ph, T = net.phys, net.T
rv = next(i for i, v in enumerate(T.V) if v["type"] == "retention")
log(f"遊水地の変数: {T.V[rv]['name']}（水準 {T.levels[rv]}）")
eff = {"levee_effect": 0.9, "retention_effect": 0.85, "paddy_effect": 0.6, "forest_effect": 0.6, "raise_effect": 0.8}
clim = {"現在気候": (1.0, 0.0), "2℃": (1.1, 0.38), "4℃": (1.2, 0.54)}
rri = rri_evaluator(net.R["explore_run"], log)
rows = []
for i in [int(x) for x in a.inits.split(",")]:
    g0 = list(net.nodes[i])
    for with_ret in (0, 1):
        g = g0.copy(); g[rv] = with_ret
        raw = T.raw(g)
        for cn, (rf, slr) in clim.items():
            env = {"rain_factor": rf, "sea_level_rise_m": slr, "rain_gamma": 1.0, **eff}
            x, sim = ph.to_rri(raw, env)
            y = rri.simulate(x, sim)
            p = ph.predict(raw[None], {k: np.atleast_1d(v) for k, v in env.items()})
            rows.append({"init": i, "retention": with_ret, "climate": cn, "x": x, "sim": sim,
                         "rri_damage": y["damage"], "sur_damage": float(p["damage"][0]),
                         "rri_pop": y["pop_hazard"], "sur_pop": float(p["pop_hazard"][0]), "rri_peak_q": y.get("peak_q")})
            log(f"  初期行動 #{i} 遊水地 {'あり' if with_ret else 'なし'} {cn}: RRI {y['damage']:.1f} / 代理 {p['damage'][0]:.1f} 億円")
eff_rows = []
for i in dict.fromkeys(r["init"] for r in rows):
    for cn in clim:
        r0 = next(r for r in rows if r["init"] == i and r["climate"] == cn and not r["retention"])
        r1 = next(r for r in rows if r["init"] == i and r["climate"] == cn and r["retention"])
        eff_rows.append({"init": i, "climate": cn, "rri": r1["rri_damage"] - r0["rri_damage"], "sur": r1["sur_damage"] - r0["sur_damage"],
                         "rri_pop": r1["rri_pop"] - r0["rri_pop"], "sur_pop": r1["sur_pop"] - r0["sur_pop"]})
o = np.array([r["rri_damage"] for r in rows]); q = np.array([r["sur_damage"] for r in rows])
summary = {"n": len(rows), "r2": float(1 - ((o - q) ** 2).sum() / ((o - o.mean()) ** 2).sum()),
           "rel_rmse": float(np.sqrt(((o - q) ** 2).mean()) / o.mean()),
           "effect_rri_mean": float(np.mean([e["rri"] for e in eff_rows])), "effect_sur_mean": float(np.mean([e["sur"] for e in eff_rows]))}
(out / "verify_retention.json").write_text(json.dumps({"states": rows, "effects": eff_rows, "summary": summary, "effects_nominal": eff},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
log("遊水地の効果（あり − なし, 被害額 億円）: 初期行動・気候ごとに RRI / 代理")
for e in eff_rows:
    log(f"  #{e['init']} {e['climate']}: RRI {e['rri']:+.1f} / 代理 {e['sur']:+.1f}（浸水人口 RRI {e['rri_pop']:+.0f} / 代理 {e['sur_pop']:+.0f}）")
log(f"全状態: R² {summary['r2']:.3f}・相対 RMSE {summary['rel_rmse']:.1%}、効果の平均 RRI {summary['effect_rri_mean']:+.1f} / 代理 {summary['effect_sur_mean']:+.1f} 億円")
