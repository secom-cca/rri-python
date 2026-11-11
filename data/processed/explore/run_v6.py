"""Exploration v6: exploration v5 (1K-RRI, no effect coefficients) extended to events of other return periods, for the
annual expected damage (治水経済調査マニュアル).  The event of return period T is the representative event's rainfall x r(T)
(rri_sim.frequency; 1/2 .. 1/200 -> 0.41 .. 1.10) x the climate's rain factor, so the surrogate must cover rain factors
~0.4 - 1.45; v5 covers 1.0 - 1.3.  v5's design points are kept; new LHS points (measures, rain factor 0.35 - 1.5, sea level,
rain intensity) and no-measure points at all rain factors are added, the surrogate refitted on the union and validated on
new points.  Writes data/processed/explore/abukuma_explore_v6/results.json (and copies v5's cost table).

    .venv/bin/python data/processed/explore/run_v6.py
"""
import copy
import json
import shutil
import time

import numpy as np

from rri_sim.data import PROCESSED
from rri_sim.rdm import explore as EX
from rri_sim.rdm import uncertainty as UQ
from rri_sim.rdm.run import setup_rri
from rri_sim.rdm.explore import Surrogate
from rri_sim.roadmap.physics import PHYS_DIR

N_NEW, N_NONE, N_VAL = 160, 40, 30
RAIN = (0.35, 1.5)
log = lambda m: print(m, flush=True)  # noqa: E731
t0 = time.time()
src = PROCESSED / "explore" / "abukuma_explore_v5"
res = json.loads((src / "results.json").read_text(encoding="utf-8"))
cfg = copy.deepcopy(res["config"])
cfg["explore"]["name"] = "abukuma_explore_v6"
P = cfg["sensitivity"]["parameters"]
P["rain_factor"] = dict(P["rain_factor"], low=RAIN[0], high=RAIN[1], status="仮定",
                        label="降雨倍率（代表洪水の総雨量に対する比: 確率規模 × 気候変動）",
                        basis="確率規模 1/2〜1/200（代表洪水 = 1/100 の 0.41〜1.10 倍, rri_sim.frequency）× 気候の降雨倍率 1.0〜1.3")
out = PROCESSED / "explore" / cfg["explore"]["name"]
out.mkdir(parents=True, exist_ok=True)
(out / "config_explore.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
ev, _ = setup_rri(cfg, out / "evaluations.jsonl", log)
ev.gextra = {"baseline": "current"}
EX.make_continuous(ev.variables, cfg["explore"], cfg["seed"])
sp = EX.Space(ev.variables, P)
assert [v.name for v in sp.V] == [v["name"] for v in res["variables"]], "variables differ from v5"
rng = np.random.default_rng(cfg["seed"] + 606)
nV, nU = len(sp.V), len(sp.unames)
d5 = res["design"]
rows_x, rows_u, rows_y = list(d5["x"]), list(d5["u"]), list(d5["y"])
keys = list(rows_y[0])


def design(n, zero_x=False):
    Z = UQ.lhs(n, nV + nU, rng)
    X = np.zeros((n, nV)) if zero_x else sp.x_from_unit(Z[:, :nV], cfg["explore"]["zero_share"], rng)
    Uv = sp.u_values(Z[:, nV:])
    return X, [{k: float(Uv[k][i]) for k in sp.unames} for i in range(n)]


def run(Xs, Us, tag, store=True):
    ys = []
    for j, (x, u) in enumerate(zip(Xs, Us)):
        m = ev.simulate(list(x), sp.sim_dict(u))
        y = EX.outputs_of(m)
        ys.append(y)
        if store:
            rows_x.append(list(map(float, x))); rows_u.append(dict(u)); rows_y.append(y)
        if (j + 1) % 10 == 0 or j + 1 == len(Xs):
            log(f"  {tag} {j + 1}/{len(Xs)}（RRI 計 {ev.n_sim} 回, {time.time() - t0:.0f} s）")
    return ys


log(f"v5 の学習点 {len(rows_y)} + 新しい点 {N_NEW}（施策と降雨倍率 {RAIN[0]}〜{RAIN[1]}）+ 対策なし {N_NONE} + 検証 {N_VAL}")
run(*design(N_NEW), "学習")
run(*design(N_NONE, zero_x=True), "対策なし")
Fx = sp.features(np.array(rows_x, float), {k: np.array([u[k] for u in rows_u]) for k in sp.unames})
log(f"代理モデルの学習（{len(rows_y)} 点, 入力 {Fx.shape[1]} 次元）")
S = Surrogate(Fx, {k: [y[k] for y in rows_y] for k in keys}, cfg["seed"])
Xv, Uv = design(N_VAL)
yv = run(Xv, Uv, "検証", store=False)
Fv = sp.features(np.array(Xv, float), {k: np.array([u[k] for u in Uv]) for k in sp.unames})
validation = {}
for k in keys:
    yk = np.array([y[k] for y in yv])
    validation[k] = {"chosen": S.kind[k], "loo_rmse": S.loo[k], **EX.error_stats(yk, S.predict(Fv, k)),
                     "y_val": yk.tolist(), "yhat_val": S.predict(Fv, k).tolist(),
                     "rain_val": [u["rain_factor"] for u in Uv]}
log("代理モデルの検証: " + ", ".join(f"{k} R² {validation[k]['r2']:.2f}/相対RMSE {validation[k]['rel_rmse']:.1%}"
                                     for k in ("damage", "pop_hazard")))
new = dict(res)
new.update({"name": cfg["explore"]["name"], "config": cfg, "uncertainties": {n: {k: P[n][k] for k in ("low", "high", "nominal", "kind", "status", "label", "basis") if k in P[n]} for n in P},
            "design": {"n_train": len(rows_y), "x": rows_x, "u": rows_u, "y": rows_y},
            "surrogate": dict(res["surrogate"], validation=validation), "extends": "abukuma_explore_v5",
            "elapsed_s": time.time() - t0})
(out / "results.json").write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
(PHYS_DIR / cfg["explore"]["name"]).mkdir(parents=True, exist_ok=True)
shutil.copy(PHYS_DIR / "abukuma_explore_v5" / "cost_parts.json", PHYS_DIR / cfg["explore"]["name"] / "cost_parts.json")
log(f"完了: {out / 'results.json'}（{time.time() - t0:.0f} s, RRI {ev.n_sim} 回）")
