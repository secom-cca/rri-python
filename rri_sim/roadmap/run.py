"""Roadmap pipeline: initial roadmap -> worlds x policies (P0-P3 + oracle) -> robustness, updateability,
future option coverage, lock-in -> explainability (Sobol on a GP of the world factors, Morris, PRIM,
switch conditions, update reasons) -> experiments (delays / early decision, complementarity,
participation) -> results.json (GUI /roadmap).

    from rri_sim.roadmap import run, config
    run.run(config.load(), "data/processed/roadmap/abukuma_roadmap")
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from . import config as CF
from . import experiments as EXP
from . import metrics as MT
from . import planner as PL
from . import sim as SM
from . import world as WD
from ..data import PROCESSED

ROADMAP_DIR = PROCESSED / "roadmap"
PARTS = ("main", "morris", "delay", "complementarity", "participation", "scope")


def reference(phys, cfg):
    """No measures, current climate (rain factor 1, no sea-level rise), planning effects."""
    env = WD.belief_world(cfg).env(cfg["years"]["start"])
    env.update(rain_factor=1.0, sea_level_rise_m=0.0)
    ref = phys.reference(env)
    ref["agriculture"] = 100.0
    return ref


def variants(cfg):
    S = list(cfg["participation"]["stakeholders"])
    out = {"全員 早期参加": {s: "early" for s in S}, "全員 段階参加": {s: "staged" for s in S}, "全員 事後参加": {s: "late" for s in S}}
    for s in S:
        out[f"{s}のみ事後"] = {t: ("late" if t == s else "staged") for t in S}
    for s in S:
        out[f"{s}のみ早期"] = {t: ("early" if t == s else "staged") for t in S}
    return out


def _pool(tasks, cfg, phys, ref, shared, log, n_jobs):
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    out = []
    t0 = time.time()
    if n_jobs == 1 or len(tasks) <= 1:
        EXP.init_worker(cfg, phys, ref, shared)
        for k, t in enumerate(tasks):
            out.append(EXP.job(t))
            if (k + 1) % 20 == 0 or k + 1 == len(tasks):
                log(f"  {k + 1}/{len(tasks)} 件（{time.time() - t0:.0f} s）")
        return out
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(n_jobs, initializer=EXP.init_worker, initargs=(cfg, phys, ref, shared)) as ex:
        futs = [ex.submit(EXP.job, t) for t in tasks]
        for k, f in enumerate(as_completed(futs)):
            out.append(f.result())
            if (k + 1) % 20 == 0 or k + 1 == len(tasks):
                log(f"  {k + 1}/{len(tasks)} 件（{time.time() - t0:.0f} s, {n_jobs} 並列）")
    return out


def _sobol(Z, y, F, cfg, seed):
    from ..rdm.sensitivity import GP, Linear, loo_rmse, sobol_indices
    so = cfg["experiment"]["sobol"]
    y = np.asarray(y, float)
    if np.std(y) < 1e-12:
        k = len(F)
        return {"S1": [0.0] * k, "ST": [0.0] * k, "model": "constant", "loo": None, "groups": {}, "n_key": 0, "order": []}
    # GP (closed-form leave-one-out, one fit) or ridge, whichever has the smaller LOO error
    gp = GP(Z, y, restarts=2, seed=seed)
    loo = {"gp": float(np.sqrt(np.mean(gp.loo_residuals() ** 2))), "linear": loo_rmse(Linear, Z, y)}
    kind = min(loo, key=loo.get)
    model = gp if kind == "gp" else Linear(Z, y)
    s = sobol_indices(lambda X: model(X), len(F), so["log2_n"], so["n_boot"], seed)
    names = list(F)
    st = np.maximum(np.array(s["ST"]), 0)
    order = np.argsort(-st)
    cum = np.cumsum(st[order]) / max(st.sum(), 1e-12)
    groups = {}
    for n, v in zip(names, st):
        g = F[n]["group"]
        groups[g] = groups.get(g, 0.0) + float(v)
    return {**s, "model": kind, "loo": loo, "groups": groups, "order": [names[i] for i in order],
            "n_key": int(np.searchsorted(cum, 0.8) + 1), "_model": model}


def _pd_crossing(model, Z, j, grid=21):
    """Partial dependence of a model on factor j and the unit values where it changes sign."""
    xs = np.linspace(0, 1, grid)
    pd = []
    for x in xs:
        Zx = Z.copy(); Zx[:, j] = x
        pd.append(float(np.mean(model(Zx))))
    pd = np.array(pd)
    cross = [float(xs[i] + (xs[i + 1] - xs[i]) * pd[i] / (pd[i] - pd[i + 1])) for i in range(grid - 1) if pd[i] * pd[i + 1] < 0]
    return xs.tolist(), pd.tolist(), cross


def summarize_main(mains, cfg, F, log):
    names = list(cfg["policies"])
    ex = cfg["experiment"]
    thr = ex["regret_threshold"]
    mains = sorted(mains, key=lambda m: m["i"])
    Z = np.array([m["z"] for m in mains])
    R = np.array([[m["records"][n]["regret"] for n in names] for m in mains])
    U = np.array([[m["records"][n]["U"] for n in names] for m in mains])
    atp_none = np.array([[m["records"][n]["atp"] is None for n in names] for m in mains])
    out = {"policies": names, "n_worlds": len(mains)}
    out["robustness"] = {"mean_regret": R.mean(0).tolist(), "max_regret": R.max(0).tolist(), "p90_regret": np.quantile(R, 0.9, 0).tolist(),
                         "satisficing": ((R <= thr) & atp_none).mean(0).tolist(), "regret_ok": (R <= thr).mean(0).tolist(),
                         "no_atp": atp_none.mean(0).tolist(), "mean_U": U.mean(0).tolist(),
                         "minimax": names[int(np.argmin(R.max(0)))], "best_mean": names[int(np.argmin(R.mean(0)))],
                         "U_initial_criteria": np.mean([[m["records"][n]["U_initial_criteria"] for n in names] for m in mains], 0).tolist()}
    # updateability
    up = {}
    cost1 = np.array([m["records"]["P1"]["cost_pv"] for m in mains]) if "P1" in names else None
    for j, n in enumerate(names):
        evs = [e for m in mains for e in m["records"][n]["update"]["events"]]
        kinds = sorted({e["kind"] for e in evs})
        by = {}
        for k in kinds:
            E = [e for e in evs if e["kind"] == k]
            det = [e for e in E if not e["censored"]]
            resp = [e["response"] for e in det if e["response"] is not None]
            by[k] = {"n": len(E), "detected": len(det) / max(len(E), 1),
                     "delay": float(np.mean([e["delay"] for e in det])) if det else None,
                     "delay_censored": float(np.mean([e["delay"] for e in E])) if E else None,
                     "response": float(np.mean(resp)) if resp else None}
        rp = [m["records"][n]["regret_post"] for m in mains if m["records"][n]["regret_post"] is not None]
        U_ = [m["records"][n]["update"] for m in mains]
        att = [u["attribution"] for u in U_ if u["attribution"] is not None]
        up[n] = {"by_kind": by, "regret_post": float(np.mean(rp)) if rp else None,
                 "change_cost_pv": float(np.mean([m["records"][n]["change_pv"] for m in mains])),
                 "extra_cost_vs_P1": None if cost1 is None else float(np.mean(np.array([m["records"][n]["cost_pv"] for m in mains]) - cost1)),
                 "coord_pv": float(np.mean([m["records"][n]["coord_pv"] for m in mains])),
                 "n_replans": float(np.mean([u["n_replans"] for u in U_])), "false_alarms": float(np.mean([u["false_alarms"] for u in U_])),
                 "attribution": float(np.mean(att)) if att else None, "churn": float(np.mean([u["churn"] for u in U_]))}
    out["updateability"] = up
    checks = [str(T) for T in ex["checkpoints"]]
    out["coverage"] = {n: {T: float(np.mean([m["records"][n]["coverage"][T] for m in mains])) for T in checks} for n in names}
    out["blank_coverage"] = {T: float(np.mean([m["blank_coverage"][T] for m in mains])) for T in checks}
    out["lockin"] = {n: {T: float(np.mean([m["records"][n]["lockin"][T] for m in mains])) for T in checks} for n in names}
    out["gained"] = {n: {T: float(np.mean([m["records"][n]["gained"][T] for m in mains])) for T in checks} for n in names}
    out["table"] = {n: {k: float(np.mean([m["records"][n][k] for m in mains]))
                        for k in ("ead_pv", "cost_pv", "capital_pv", "maint_pv", "coord_pv", "change_pv", "violation_share",
                                  "mean_ecology", "mean_agriculture", "mean_livability", "failures", "deferred")} for n in names}
    # ---- explainability
    log("説明可能性: GP 代理モデル上の Sobol 指数・PRIM・推奨の切替条件")
    seed = cfg["seed"]
    sob = {}
    for j, n in enumerate(names):
        sob[f"regret:{n}"] = _sobol(Z, R[:, j], F, cfg, seed + j)
    pairs = [(a, b) for a, b in (("P0", "P1"), ("P1", "P2"), ("P2", "P3"), ("P1", "P3")) if a in names and b in names]
    switch = {}
    for a, b in pairs:
        d = U[:, names.index(b)] - U[:, names.index(a)]
        s = _sobol(Z, d, F, cfg, seed + 50)
        sob[f"dU:{b}-{a}"] = s
        top = []
        if "_model" in s:
            for f in s["order"][:3]:
                jf = list(F).index(f)
                xs, pd, cr = _pd_crossing(s["_model"], Z, jf)
                fv = F[f]
                top.append({"factor": f, "label": fv["label"], "unit": xs, "pd": pd,
                            "values": [fv["low"] + x * (fv["high"] - fv["low"]) for x in xs],
                            "crossing": [fv["low"] + x * (fv["high"] - fv["low"]) for x in cr]})
        pr = cfg["experiment"]["prim"]
        switch[f"{b}>{a}"] = {"share": float((d > 0).mean()), "mean_dU": float(d.mean()),
                              "box": MT.prim_box(Z, d > 0, F, pr["alpha"], pr["min_mass"], pr["density"]), "partial_dependence": top}
    for s in sob.values():
        s.pop("_model", None)
    vul = {}
    pr = cfg["experiment"]["prim"]
    for j, n in enumerate(names):
        y = R[:, j] > thr
        vul[n] = MT.prim_box(Z, y, F, pr["alpha"], pr["min_mass"], pr["density"])
    best_pol = np.argmin(R, 1)
    best_box = {n: MT.prim_box(Z, best_pol == j, F, pr["alpha"], pr["min_mass"], pr["density"]) for j, n in enumerate(names)}
    reasons = {}
    for n in names:
        cnt = {}
        for m in mains:
            for r in m["records"][n]["replans"]:
                for x in r["reasons"]:
                    cnt[x["kind"]] = cnt.get(x["kind"], 0) + 1
        reasons[n] = cnt
    out["explain"] = {"sobol": sob, "switch": switch, "vulnerability": vul, "best_box": best_box, "reasons": reasons,
                      "indices": {n: {"n_key_assumptions": sob[f"regret:{n}"]["n_key"],
                                      "vulnerability_box_dims": len(vul[n]["box"]),
                                      "vulnerability_density": vul[n]["density"], "vulnerability_coverage": vul[n]["coverage"],
                                      "attribution": up[n]["attribution"], "churn": up[n]["churn"]} for n in names}}
    return out


def summarize_morris(jobs, trajs, delta, cfg, F):
    from ..rdm.sensitivity import morris_indices
    names = list(cfg["policies"])
    byt = {}
    for j in jobs:
        byt.setdefault(j["traj"], {})[j["step"]] = j["U"]
    Ys = {n: [np.array([byt[t][s][n] for s in range(len(trajs[t][0]))]) for t in range(len(trajs))] for n in names}
    out = {}
    for n in names:
        mu, mus, sg = morris_indices(trajs, delta, Ys[n])
        out[f"U:{n}"] = {"mu": mu.tolist(), "mu_star": mus.tolist(), "sigma": sg.tolist()}
    for a, b in (("P1", "P2"), ("P1", "P3"), ("P0", "P1")):
        if a in names and b in names:
            mu, mus, sg = morris_indices(trajs, delta, [yb - ya for ya, yb in zip(Ys[a], Ys[b])])
            out[f"dU:{b}-{a}"] = {"mu": mu.tolist(), "mu_star": mus.tolist(), "sigma": sg.tolist()}
    return {"factors": list(F), "n_trajectories": len(trajs), "indices": out}


def run(cfg=None, out_dir=None, phys=None, log=print, parts=PARTS):
    t0 = time.time()
    cfg = CF.load(cfg)
    out = Path(out_dir) if out_dir else ROADMAP_DIR / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    CF.save(cfg, out / "config.json")
    if phys is None:
        from .physics import SurrogatePhysics
        phys = SurrogatePhysics(cfg["physics"]["explore_run"], log)
    from .physics import with_institutions
    phys = with_institutions(phys, cfg)
    ref = reference(phys, cfg)
    F = WD.factors(cfg)
    k = len(F)
    ex = cfg["experiment"]
    seed = cfg["seed"]
    rng = np.random.default_rng(seed + 501)
    from ..rdm.uncertainty import lhs
    log(f"ロードマップ: 施策 {len(phys.variables)} 変数 × 世界の不確実性 {k} 因子 × 方策 {list(cfg['policies'])}")
    roadmaps, infos = {}, {}
    roadmaps["main"], infos["main"] = PL.initial_roadmap(cfg, phys, ref, seed=seed)
    log(f"初期ロードマップ（{cfg['years']['start']}年に最適化, 期待効用 {infos['main']['U']:.4f}）: "
        + ", ".join(f"{a['year']} {phys.variables[a['var']]['name']}={a['level']}" for a in roadmaps["main"]))
    lib = MT.continuation_library(cfg, phys, ex["n_continuations"], seed)
    goals = MT.sample_goals(cfg, ex["n_goals"], seed)
    N = ex["n_worlds"]
    Z = lhs(N, k, rng)
    Z[0] = WD.unit_nominal(F)
    tasks = []
    if "main" in parts:
        tasks += [{"kind": "main", "i": i, "z": Z[i], "seed": seed * 1000 + i} for i in range(N)]
    trajs, delta = None, None
    if "morris" in parts and ex["morris"]["r"] > 0:
        from ..rdm.sensitivity import morris_trajectories
        trajs, delta = morris_trajectories(k, ex["morris"]["r"], ex["morris"]["levels"], np.random.default_rng(seed + 503))
        tasks += [{"kind": "morris", "traj": t, "step": s, "z": pts[s], "seed": seed * 1000 + 700 + t}
                  for t, (pts, _, _) in enumerate(trajs) for s in range(len(pts))]
    if "delay" in parts:
        de = ex["delay"]
        for s in de["scales"]:
            roadmaps[f"delay:{s}"], _ = PL.initial_roadmap(cfg, phys, ref, seed=seed, delay_scale=s)
            for y in de["postpone"]:
                roadmaps[f"delay:{s}:postpone:{y}"], _ = PL.initial_roadmap(cfg, phys, ref, seed=seed, delay_scale=s, earliest=y)
            tasks += [{"kind": "delay", "scale": s, "i": i, "z": Z[i], "seed": seed * 1000 + i} for i in range(min(de["n_worlds"], N))]
    ports = EXP.portfolios(cfg, phys)
    if "complementarity" in parts:
        tasks += [{"kind": "complementarity", "i": i, "z": Z[i], "seed": seed * 1000 + i} for i in range(min(ex["complementarity"]["n_worlds"], N))]
    V = variants(cfg)
    if "participation" in parts:
        for v, modes in V.items():
            tasks += [{"kind": "participation", "variant": v, "i": i, "z": Z[i], "seed": seed * 1000 + i, "checkpoint": ex["checkpoints"][0]}
                      for i in range(min(ex["participation"]["n_worlds"], N))]
    scope_vars = {}
    if "scope" in parts:
        for sc, tiers in cfg["scopes"].items():
            scope_vars[sc] = PL.allowed_vars(cfg, phys, tiers)
            roadmaps[f"scope:{sc}"], _ = PL.initial_roadmap(cfg, phys, ref, seed=seed, allowed=scope_vars[sc])
            tasks += [{"kind": "scope", "scope": sc, "i": i, "z": Z[i], "seed": seed * 1000 + i}
                      for i in range(min(ex["scope"]["n_worlds"], N))]
    log(f"計算: {len(tasks)} 件（主実験 {N} 世界・Morris・時間遅れ・相互補完性・参加）")
    shared = {"roadmaps": roadmaps, "lib": lib, "goals": goals, "portfolios": ports, "variants": V, "scope_vars": scope_vars}
    jobs = _pool(tasks, cfg, phys, ref, shared, log, ex.get("n_jobs", 0))
    by = {}
    for j in jobs:
        by.setdefault(j["kind"], []).append(j)
    res = {"name": cfg["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg,
           "physics": {"kind": type(phys).__name__, "explore_run": getattr(phys, "explore_name", None),
                       "validation": getattr(phys, "validation", None), "verification": getattr(phys, "verification", None),
                       "ranges": getattr(phys, "ranges", None)},
           "variables": phys.variables, "zones": phys.zones, "reference": ref,
           "factors": F, "goals": goals, "continuations": [c["portfolio"] for c in lib],
           "roadmap": {"actions": roadmaps["main"], "U": infos["main"]["U"], "n_eval": infos["main"]["n_eval"]},
           "roadmaps": {k_: v for k_, v in roadmaps.items()}, "years": SM.decision_years(cfg),
           "eval_starts": SM.eval_starts(cfg)}
    if "main" in by:
        res["worlds"] = sorted(by["main"], key=lambda m: m["i"])
        res["summary"] = summarize_main(by["main"], cfg, F, log)
    if "morris" in by:
        res["morris"] = summarize_morris(by["morris"], trajs, delta, cfg, F)
    if "delay" in by:
        res["delay"] = EXP.summarize_delay(by["delay"], cfg)
    if "complementarity" in by:
        res["complementarity"] = EXP.summarize_complementarity(by["complementarity"], ports)
    if "participation" in by:
        res["participation"] = EXP.summarize_participation(by["participation"], cfg, V)
    if "scope" in by:
        res["scope"] = EXP.summarize_scope(by["scope"], cfg, list(cfg["scopes"]), roadmaps, phys)
        res["scope"]["variables"] = {sc: [phys.variables[i]["name"] for i in v] for sc, v in scope_vars.items()}
    res["elapsed_s"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=_json), encoding="utf-8")
    log(f"完了: {out / 'results.json'}（{res['elapsed_s']:.0f} s）")
    return res


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------ one world (GUI: parameters changed on the page)
def simulate_one(phys, cfg_over=None, values=None, modes=None, policies=None, unit=None):
    """Initial roadmap + P0-P3 in one world given by factor values or unit positions (missing = nominal);
    for /api/roadmap/simulate."""
    cfg = CF.load(cfg_over or {})
    from .physics import with_institutions
    phys = with_institutions(phys, cfg)
    if policies:
        cfg["policies"] = {k: v for k, v in cfg["policies"].items() if k in policies}
    ref = reference(phys, cfg)
    F = WD.factors(cfg)
    z = WD.unit_nominal(F)
    for i, n in enumerate(F):
        if unit and n in unit:
            z[i] = float(np.clip(unit[n], 0, 1))
        elif values and n in values:
            f = F[n]
            z[i] = float(np.clip((float(values[n]) - f["low"]) / ((f["high"] - f["low"]) or 1.0), 0, 1))
    road, info = PL.initial_roadmap(cfg, phys, ref, modes=modes, seed=cfg["seed"])
    w = WD.make_world(cfg, z, cfg["seed"] * 1000)
    sims = EXP.policies_in_world(cfg, phys, ref, w, road, cfg["seed"], modes=modes)
    recs = {}
    for n, (st, ctrl) in sims.items():
        sc = SM.score(st, cfg, ref, w.p["regimes"])
        recs[n] = EXP.record(cfg, phys, ref, st, ctrl, sc)
    best = max(r["U"] for r in recs.values())
    for r in recs.values():
        r["regret"] = best - r["U"]
    return {"roadmap": {"actions": road, "U": info["U"]}, "records": recs, "regimes": w.p["regimes"],
            "events": MT.change_events(cfg, w), "factors": {n: F[n]["low"] + zi * (F[n]["high"] - F[n]["low"]) for n, zi in zip(F, z)},
            "world": {"rain_pts": w.p["rain_pts"], "slr_pts": w.p["slr_pts"], "asset_pts": w.p["asset_pts"], "pop_pts": w.p["pop_pts"]},
            "reference": ref, "variables": phys.variables, "years": SM.decision_years(cfg), "config": cfg}


# ------------------------------------------------------------------ RRI re-evaluation of selected states
def verify_rri(res_path, n_states=8, log=print):
    """Re-run RRI for states met in the main experiment (worlds with the lowest / highest 2100 rain,
    policies P0 / P3, mid and late periods) and compare the event damage with the surrogate."""
    from .physics import SurrogatePhysics, rri_evaluator
    res = json.loads(Path(res_path).read_text(encoding="utf-8"))
    cfg = res["config"]
    from .physics import with_institutions
    phys = with_institutions(SurrogatePhysics(cfg["physics"]["explore_run"], log), cfg)
    ref = reference(phys, cfg)
    ws = res["worlds"]
    order = sorted(ws, key=lambda m: m["world"]["rain_pts"][-1][1])
    picks = [order[0], order[len(order) // 2], order[-1]]
    ev = rri_evaluator(cfg["physics"]["explore_run"], log)
    out = []
    for m in picks:
        w = WD.make_world(cfg, np.array(m["z"]), m["seed"])
        road = res["roadmaps"]["main"]
        for name in [p for p in ("P0", "P3") if p in cfg["policies"]]:
            st = SM.run(w, PL.controller(cfg, phys, ref, name, road, seed=m["seed"]), phys, cfg, ref)
            for p in [st.periods[len(st.periods) // 2], st.periods[-1]]:
                if len(out) >= n_states:
                    break
                x, sim = phys.to_rri(p["raw"], p["env"])
                y = ev.simulate(x, sim)
                pred = phys.predict(p["raw"][None], {k: np.atleast_1d(v) for k, v in p["env"].items()})["damage"][0]
                out.append({"world": m["i"], "policy": name, "t": p["t"], "x": x, "sim": sim, "rri": y["damage"], "surrogate": float(pred)})
                log(f"  世界 {m['i']} {name} {p['t']}年: RRI {y['damage']:.0f} / 代理 {pred:.0f} 億円")
    from ..rdm.sensitivity import error_stats
    o = np.array([r["rri"] for r in out]); q = np.array([r["surrogate"] for r in out])
    res["verification_rri"] = {"states": out, **(error_stats(o, q) if len(out) > 1 else {})}
    Path(res_path).write_text(json.dumps(res, ensure_ascii=False, default=_json), encoding="utf-8")
    return res["verification_rri"]
