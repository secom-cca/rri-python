"""RDM pipeline: candidates -> NSGA-II (nominal scenario) -> stress test on the
common scenario set -> robustness (regret) -> Sobol sensitivity -> results.

    from rri_sim.rdm import run, config
    run.run(config.load("my.json"), "data/processed/rdm/my_run")

Everything needed to reproduce a run is written to the run directory:
config.json, candidates.json, evaluations.jsonl (simulation cache, re-used on
re-runs), results.json (read by the GUI page /rdm).
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np

from . import candidates as CA
from . import config as CF
from . import nsga2
from . import robustness as RB
from . import uncertainty as UQ
from ..data import PROCESSED

RDM_DIR = PROCESSED / "rdm"


def _b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def setup_rri(cfg, cache_path, log=print):
    from ..gui.engine import Engine
    from .evaluate import RRIEvaluator
    s = Engine().session(cfg["basin"], cfg["res"], lambda f, m: log(f"[準備] {m}"))
    log("対策なし（現在気候）を計算中")
    base = s.run(cfg["event"])
    zm = CA.zone_masks(s.d, s.meta, cfg["zones"])
    variables = CA.build(s.d, s.meta, s.hand, s.assets, s.class_names, base["_depth"], cfg)
    ev = RRIEvaluator(s, cfg, variables, zm, cache_path, log)
    d, r = s.d, s.d.river
    zone_id = np.zeros(d.shape, np.uint8)
    for k, m in enumerate(zm.values()):
        zone_id[m] = k + 1
    mapinfo = {"nx": d.shape[1], "ny": d.shape[0], "dx": d.dx, "mask": _b64(d.mask, "u1"), "zone_id": _b64(zone_id, "u1"),
               "z": _b64(np.where(d.mask, d.z, np.nan), "<f4"), "zones": list(zm),
               "river": {"ri": r.ri.tolist(), "rj": r.rj.tolist(), "dn": r.dn.tolist(), "area": r.area_km2.round(1).tolist()},
               "gauges": {g: int(k) for g, k in s.meta["gauge_index"].items()},
               "baseline_depth_cm": _b64(np.clip(np.nan_to_num(base["_depth"]) * 100, 0, 65535), "<u2")}
    return ev, mapinfo


def run(cfg, out_dir=None, evaluator=None, mapinfo=None, log=print):
    t0 = time.time()
    cfg = CF.load(cfg)
    out = Path(out_dir) if out_dir else RDM_DIR / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    CF.save(cfg, out / "config.json")
    rng = np.random.default_rng(cfg["seed"])
    if evaluator is None:
        evaluator, mapinfo = setup_rri(cfg, out / "evaluations.jsonl", log)
    ev, V = evaluator, evaluator.variables
    (out / "candidates.json").write_text(json.dumps([v.to_dict() for v in V], ensure_ascii=False), encoding="utf-8")
    log(f"候補施策 {len(V)} 変数: " + ", ".join(f"{v.name}({len(v.options)})" for v in V))
    budget = cfg["budget_oku"]
    sims, posts, full = UQ.design(cfg, rng)
    nominal = sims[0]
    n_opt = [len(v.options) for v in V]
    zero = [0] * len(V)

    # ---- 1. multi-objective optimisation under the nominal scenario
    def objectives(x):
        c = ev.cost(x)
        viol = max(0.0, c["total"] - budget)
        if viol > 0:
            return [1e12, c["total"], 0.0, 0.0], viol
        m = ev.simulate(x, nominal)
        return [m["damage"], c["total"], -m["livability"], -m["ecology"]], 0.0

    def repair(x):
        x = np.array(x, int)
        while ev.cost(x)["total"] > budget and x.any():
            i = rng.choice(np.flatnonzero(x))
            x[i] -= 1
        return x

    op = cfg["optimisation"]
    archive, final = nsga2.run(n_opt, objectives, op["pop"], op["gens"], rng, op["p_mutation"], init=[zero], repair=repair, log=log)
    F = np.array([a["f"] for a in archive]); Vv = np.array([a["v"] for a in archive])
    par = np.flatnonzero(nsga2.pareto_mask(F, Vv)).tolist()
    log(f"Pareto 最適解 {len(par)} 案（評価 {len(archive)} 案）")

    # ---- 2. stress test: Pareto policies + no-measure on the common scenario set
    pol_idx = list(dict.fromkeys([archive.index(next(a for a in archive if a["x"] == zero))] + par))
    policies = [archive[i]["x"] for i in pol_idx]
    log(f"頑健性評価: {len(policies)} 案 × 気候・係数シナリオ {len(sims)}（事後係数込み {len(full)} シナリオ）")
    simres = [[ev.simulate(x, s) for s in sims] for x in policies]
    costs = [ev.cost(x) for x in policies]
    matrix = [[UQ.apply_post(simres[p][f["sim"]], costs[p], posts[f["post"]]) for f in full] for p in range(len(policies))]
    ref = UQ.apply_post(simres[0][0], costs[0], posts[0])
    eligible = np.array([c["total"] <= budget for c in costs])
    rob = RB.evaluate(matrix, cfg["stakeholders"], ref, budget, eligible)

    # ---- 3. Sobol sensitivity on a surrogate (simulated factors) + exact post-hoc factors
    U = cfg["uncertainties"]
    simf = [k for k, u in U.items() if u["kind"] == "sim"]
    postf = [k for k, u in U.items() if u["kind"] == "post"]
    factors = ["climate"] + simf + postf
    Xs = np.array([UQ.sim_features(s, cfg) for s in sims])
    keys = ["damage", "households", "livability", "ecology"]
    sur = []
    for p in range(len(policies)):
        mods = {k: UQ.Ridge(Xs, [simres[p][i][k] for i in range(len(sims))]) for k in keys}
        for z in mapinfo["zones"] if mapinfo else simres[p][0]["damage_zone"]:
            mods["dz:" + z] = UQ.Ridge(Xs, [simres[p][i]["damage_zone"][z] for i in range(len(sims))])
        sur.append(mods)
    zones = list(simres[0][0]["damage_zone"])

    def predict(X):
        """(N, k) unit-cube samples -> metrics[p][n] (dicts)."""
        Xsim = X[:, :1 + len(simf)]
        postv = {k: U[k]["low"] + X[:, 1 + len(simf) + i] * (U[k]["high"] - U[k]["low"]) for i, k in enumerate(postf)}
        res = []
        for p in range(len(policies)):
            pr = {k: sur[p][k](Xsim) for k in keys}
            dz = {z: np.maximum(sur[p]["dz:" + z](Xsim), 0) * postv.get("asset_factor", 1.0) for z in zones}
            cf = postv.get("cost_factor", 1.0)
            res.append({"damage_zone": dz, "households": np.maximum(pr["households"], 0), "livability": pr["livability"],
                        "ecology": pr["ecology"], "cost": costs[p]["total"] * cf,
                        "cost_zone": {z: v * cf for z, v in costs[p]["by_zone"].items()},
                        "damage": np.maximum(pr["damage"], 0) * postv.get("asset_factor", 1.0)})
        return res

    def target_fn(kind, st_name=None):
        rec = rob["overall"]["minimax"] if st_name is None else rob["stakeholders"][st_name]["minimax"]

        def f(X):
            P = predict(X)
            if kind == "damage":
                return P[rec]["damage"]
            st = cfg["stakeholders"][st_name]
            Ut = np.array([RB.utility(P[p], st, ref, budget) for p in range(len(policies))])
            feas = np.array([np.broadcast_to(P[p]["cost"] <= budget, X.shape[:1]) for p in range(len(policies))])
            return RB.regret(Ut, feas)[rec]
        return f

    sob = {}
    n_base = cfg["sobol"]["n_base"]
    targets = [("damage", None, "被害額（総合推奨案）")] + [("regret", n, f"Regret・{n}（{n}の推奨案）") for n in cfg["stakeholders"]]
    for kind, st_name, label in targets:
        S1, ST = UQ.saltelli(target_fn(kind, st_name), len(factors), n_base, np.random.default_rng(cfg["seed"] + 7))
        sob[label] = {f: {"S1": float(a), "ST": float(b)} for f, a, b in zip(factors, S1, ST)}

    # ---- 4. stakeholder-specific Pareto sets over the archive (their own scope, nominal scenario)
    st_pareto = {}
    feas_arch = [i for i, a in enumerate(archive) if a["v"] <= 0]
    for name, st in cfg["stakeholders"].items():
        G = []
        for i in feas_arch:
            x = archive[i]["x"]
            m = UQ.apply_post(ev.simulate(x, nominal), ev.cost(x), posts[0])
            G.append([RB._scope(m["damage_zone"], st["damage_zones"]),
                      RB._scope(m["cost_zone"], st["cost_zones"]) if st["cost_zones"] else 0.0,
                      -m["livability"], -m["ecology"], m["households"]])
        G = np.array(G)
        use = [k for k, w in zip(["damage", "cost", "livability", "ecology", "households"], range(5))
               if st["weights"].get(k, 0) > 0]
        cols = [["damage", "cost", "livability", "ecology", "households"].index(k) for k in use]
        mk = nsga2.pareto_mask(G[:, cols]) if len(G) else np.zeros(0, bool)
        st_pareto[name] = [feas_arch[i] for i in np.flatnonzero(mk)]

    nominal_metrics = []
    for i, a in enumerate(archive):
        c = ev.cost(a["x"])
        row = {"x": a["x"], "gen": a["gen"], "feasible": a["v"] <= 0, "cost": c["total"], "cost_by_type": c["by_type"],
               "cost_zone": c["by_zone"]}
        if a["v"] <= 0:
            m = ev.simulate(a["x"], nominal)
            row.update(damage=m["damage"], damage_zone=m["damage_zone"], households=m["households"],
                       livability=m["livability"], ecology=m["ecology"], peak_q=m.get("peak_q"))
        nominal_metrics.append(row)
    res = {
        "name": cfg["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0,
        "n_simulations": getattr(ev, "n_sim", None), "config": cfg, "map": mapinfo,
        "variables": [{"name": v.name, "type": v.type, "zone": v.zone, "options": v.options, "cells": v.cells} for v in V],
        "archive": nominal_metrics, "pareto": par, "stakeholder_pareto": st_pareto,
        "policies": [{"archive": pol_idx[p], "x": policies[p], "label": "施策なし" if p == 0 else f"案{p}",
                      "desc": CA.describe(V, policies[p]) or ["施策なし"]} for p in range(len(policies))],
        "sims": sims, "posts": posts, "scenarios": full,
        "matrix": {k: [[matrix[p][s][k] for s in range(len(full))] for p in range(len(policies))]
                   for k in ("damage", "cost", "households", "livability", "ecology")},
        "robustness": rob, "sobol": sob, "sobol_factors": {f: (U[f]["label"] if f in U else "気候（現在→2℃→4℃）") for f in factors},
    }
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    log(f"完了: {out / 'results.json'}（{time.time() - t0:.0f} s, シミュレーション {res['n_simulations']} 回）")
    return res
