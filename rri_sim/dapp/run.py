"""DAPP pipeline: every strategy x climate path -> pathways, ATPs, scorecard,
stakeholder regret / robustness, weight and threshold sensitivity.

    from rri_sim.dapp import run, config
    run.run(config.load(), "data/processed/dapp/abukuma_dapp")

Writes config.json, evaluations.jsonl (simulation cache) and results.json
(read by the GUI page /dapp) to the run directory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config as CF
from . import model as MD
from ..data import PROCESSED
from ..rdm import robustness as RB

DAPP_DIR = PROCESSED / "dapp"


def _scope(d, zones):
    return RB._scope(d, zones)


def run(cfg, out_dir=None, evaluator=None, mapinfo=None, quick=False, log=print):
    t0 = time.time()
    cfg = CF.load(cfg)
    out = Path(out_dir) if out_dir else DAPP_DIR / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    CF.save(cfg, out / "config.json")
    if evaluator is None:
        from ..rdm.run import setup_rri
        evaluator, mapinfo = setup_rri(cfg, out / "evaluations.jsonl", log)
    ev = evaluator
    V = ev.variables
    zero = [0] * len(V)
    ref = ev.simulate(zero, {"rain_factor": 1.0, "sea_level_rise_m": 0.0, "forest_effect": 1.0})
    names = [n for n in cfg["strategies"] if not quick or n in cfg["quick_strategies"]]
    clims = list(cfg["climate_paths"])
    log(f"DAPP: 戦略 {len(names)} × 気候経路 {len(clims)} × {len(MD.years(cfg))} 期（候補施策 {len(V)} 変数）")
    recs = {}
    for c in clims:
        for n in names:
            recs[(n, c)] = MD.simulate(n, cfg["strategies"][n], c, ev, cfg, ref)
            r = recs[(n, c)]
            log(f"  {n} / {c}: 累積被害(PV) {r['cum_damage_pv']:.0f} 億円, 累積費用(PV) {r['cum_cost_pv']:.0f} 億円, ATP {r['atp']}")

    # ---- stakeholder evaluation (utility of cumulative / mean metrics; ref = no measures, current climate)
    def metric(r):
        dz = {}
        for p in r["periods"]:
            for z, v in p["exp_damage_zone_pv"].items():
                dz[z] = dz.get(z, 0.0) + v
        cz = {}
        for p in r["periods"]:
            for z, v in p["cost_zone_pv"].items():
                cz[z] = cz.get(z, 0.0) + v
        return {"damage_zone": dz, "cost_zone": cz, "households": r["mean_households"],
                "livability": r["mean_livability"], "ecology": r["mean_ecology"]}
    base = recs[("無対策", "current")] if ("無対策", "current") in recs else recs[(names[0], clims[0])]
    refm = metric(base)
    budget = cfg["budget"]["total_oku"]
    S = cfg["stakeholders"]
    thr_ratio = cfg["stakeholder_atp_ratio"]

    def st_atp(r, st, ratio):
        for p in r["periods"]:
            d = _scope(p["event_damage_zone"], st["damage_zones"])
            d0 = _scope(ref["damage_zone"], st["damage_zones"])
            if d > ratio * max(d0, 1e-9):
                return p["year"]
        return None
    stake = {}
    rel = np.zeros((len(S), len(names)))
    for j, (sn, st) in enumerate(S.items()):
        U = np.array([[RB.utility(metric(recs[(n, c)]), st, refm, budget) for c in clims] for n in names])
        R = RB.regret(U, np.ones_like(U, bool))
        ratio = thr_ratio.get(sn, cfg["atp"]["damage_ratio"])
        atp = [[st_atp(recs[(n, c)], st, ratio) for c in clims] for n in names]
        ok = np.array([[a is None for a in row] for row in atp])
        sat = ((R <= cfg["regret_threshold"]) & ok).mean(1)
        stake[sn] = {"utility": U.tolist(), "regret": R.tolist(), "max_regret": R.max(1).tolist(), "mean_regret": R.mean(1).tolist(),
                     "satisficing": sat.tolist(), "atp_ratio": ratio, "atp": atp, "minimax": int(np.argmin(R.max(1)))}
        rel[j] = R.max(1) / cfg["regret_threshold"]
    overall = int(np.argmin(rel.max(0)))

    # ---- weight sensitivity: uniform random weights over each stakeholder's metrics -> minimax strategy
    rng = np.random.default_rng(cfg["seed"] + 21)
    keys = ["damage", "cost", "households", "livability", "ecology"]
    wsens = {}
    for sn, st in S.items():
        act = [k for k in keys if st["weights"].get(k, 0) > 0]
        T = {k: np.array([[RB.utility(metric(recs[(n, c)]), dict(st, weights={k: 1.0}), refm, budget) for c in clims] for n in names]) for k in act}
        freq = np.zeros(len(names))
        for w in rng.dirichlet(np.ones(len(act)), cfg["weight_samples"]):
            Uw = sum(wk * T[k] for k, wk in zip(act, w))
            freq[int(np.argmin(RB.regret(Uw, np.ones_like(Uw, bool)).max(1)))] += 1
        wsens[sn] = (freq / cfg["weight_samples"]).tolist()
    # ---- threshold sensitivity: damage ATP year vs tolerated ratio
    grid = [round(x, 2) for x in np.arange(1.0, 1.81, 0.1)]
    tsens = {c: {n: [next((p["year"] for p in recs[(n, c)]["periods"] if p["damage_ratio"] > g), None) for g in grid] for n in names} for c in clims}

    # ---- pathways map data: action segments (start -> complete -> end)
    pathways = {}
    for (n, c), r in recs.items():
        segs = []
        for e in r["log"]:
            if e["event"] != "start":
                continue
            for s in segs:                       # an upgrade / change ends the previous segment of the variable
                if s["var"] == e["var"] and s["end"] is None:
                    s["end"] = e["complete"]
            segs.append({"var": e["var"], "option": e["option"], "kind": e["kind"], "start": e["year"], "complete": e["complete"], "end": None})
        for s in segs:
            s["end"] = s["end"] or cfg["years"]["end"]
        pathways[f"{n}|{c}"] = {"segments": segs, "triggers": [e for e in r["log"] if e["event"] == "trigger"],
                                "deferred": [e for e in r["log"] if e["event"] == "deferred"], "atp": r["atp"]}

    res = {"name": cfg["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0,
           "n_simulations": getattr(ev, "n_sim", None), "config": cfg, "map": mapinfo,
           "variables": [{"name": v.name, "type": v.type, "zone": v.zone, "options": v.options} for v in V],
           "strategies": names, "climates": clims, "years": MD.years(cfg),
           "reference": {"damage": ref["damage"], "pop_hazard": ref["pop_hazard"], "damage_zone": ref["damage_zone"]},
           "records": {f"{n}|{c}": recs[(n, c)] for (n, c) in recs},
           "stakeholders": stake, "overall_minimax": overall, "max_relative_regret": rel.max(0).tolist(),
           "weight_sensitivity": wsens, "threshold_sensitivity": {"grid": grid, "atp_year": tsens}, "pathways": pathways,
           "approximation": ("各10年を代表洪水1回（{}、その10年の中央年の気候で降雨を拡大）で代表し、年超過確率 {} を掛けて期待被害とする。"
                             "洪水の頻度分布・年々の変動・複数洪水の連続は表さない。").format(cfg["event"], cfg["representative_event"]["annual_probability"])}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"完了: {out / 'results.json'}（{time.time() - t0:.0f} s, RRI {res['n_simulations']} 回）")
    return res
