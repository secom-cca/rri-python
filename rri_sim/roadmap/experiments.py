"""Per-world jobs of the roadmap analysis (run in worker processes) and their summaries.

    main            P0-P3 + the perfect-information roadmap (oracle) in one world: utility, regret,
                    updateability, future option coverage and lock-in at the checkpoints
    morris          P0-P3 in one point of a Morris trajectory (utilities only)
    delay           all phase durations x scale: P0-P3, and P2 that may not act before a later year
                    (value of an early decision = U(start now) - U(postponed))
    complementarity fixed portfolios of measure types (all subsets), decided at the start, budget relaxed
    scope           P0-P3 with the measures of a scope of implementers only (国のみ / 国＋自治体 / 国＋自治体＋地域);
                    the initial roadmap and the re-planning use those measures only
    participation   P0-P3 with other participation modes (early / staged / late, one stakeholder late or early),
                    same initial roadmap (the social effect alone; P2 / P3 re-plan with the engagement they have)
"""
from __future__ import annotations

import copy
import itertools

import numpy as np

from . import metrics as MT
from . import planner as PL
from . import sim as SM
from . import world as WD

G = {}       # worker globals: cfg, phys, ref, roadmaps, lib, goals


def init_worker(cfg, phys, ref, shared):
    G.update(cfg=cfg, phys=phys, ref=ref, **shared)


def job(task):
    return {"main": main_job, "morris": morris_job, "delay": delay_job, "complementarity": complementarity_job,
            "participation": participation_job, "scope": scope_job}[task["kind"]](task)


# ------------------------------------------------------------------ compact records (GUI)
def record(cfg, phys, ref, st, ctrl, sc):
    per = [{"t": p["t"], "damage_ratio": round(p["damage"] / ref["damage"], 4), "pop_ratio": round(p["pop_risk"] / ref["pop_hazard"], 4),
            "pop_exposed_ratio": round(p["pop_hazard"] / ref["pop_hazard"], 4),
            "ecology": round(p["ecology"], 3), "livability": round(p["livability"], 3), "agriculture": round(p["agriculture"], 3),
            "rain": round(p["env"]["rain_factor"], 4), "slr": round(p["env"]["sea_level_rise_m"], 3),
            "asset": round(p["asset"], 3), "pop": round(p["pop"], 3)} for p in st.periods]
    acts = [{"var": a.var, "kind": a.kind, "level": a.level, "decide": a.t_decide, "cons_end": round(a.t_cons, 2),
             "complete": round(st.complete_year(a), 2), "failed": a.failed, "adopt": round(a.adopt, 3), "source": a.source,
             "build": round(a.build, 2), "diffuse": round(a.diffuse, 2), "mature": round(a.mature, 2), "cost": round(a.cost, 2),
             "change_cost": round(a.change_cost, 2)} for a in st.actions]
    years = SM.decision_years(cfg) + [cfg["years"]["end"]]
    eng = {s: [round(st.social.engagement(s, y), 3) for y in years] for s in cfg["participation"]["stakeholders"]}
    out = {**{k: v for k, v in sc.items()}, "periods": per, "actions": acts, "engagement": eng,
           "modes": dict(st.social.modes), "since": dict(st.social.since),
           "cost_series": {"capital": np.round(st.capital, 3).tolist(), "maint": np.round(st.maint, 3).tolist(),
                           "coord": np.round(st.coord, 3).tolist()},
           "deferred_log": [e for e in st.log if e["event"] == "deferred"]}
    if hasattr(ctrl, "replans"):
        out["replans"] = [dict(r, text=MT.describe_replan(cfg, phys, r)) for r in ctrl.replans]
        out["triggers"] = ctrl.triggers
    else:
        out["replans"], out["triggers"] = [], []
    return out


def policies_in_world(cfg, phys, ref, w, road, seed, modes=None, snapshots=(), allowed=None):
    out = {}
    for name in cfg["policies"]:
        ctrl = PL.controller(cfg, phys, ref, name, road, modes, seed=seed, allowed=allowed)
        st = SM.run(w, ctrl, phys, cfg, ref, snapshots=snapshots)
        out[name] = (st, ctrl)
    return out


def main_job(task):
    cfg, phys, ref = G["cfg"], G["phys"], G["ref"]
    ex = cfg["experiment"]
    y0 = cfg["years"]["start"]
    w = WD.make_world(cfg, task["z"], task["seed"])
    road = G["roadmaps"]["main"]
    checks = ex["checkpoints"]
    sims = policies_in_world(cfg, phys, ref, w, road, task["seed"], snapshots=checks)
    recs, U = {}, {}
    for name, (st, ctrl) in sims.items():
        sc = SM.score(st, cfg, ref, w.p["regimes"])
        sc["U_initial_criteria"] = SM.score(st, cfg, ref, [[y0, cfg["initial_regime"]]])["U"]
        recs[name] = record(cfg, phys, ref, st, ctrl, sc)
        U[name] = sc["U"]
    # perfect-information roadmap (knows the world, failures included); seeded with the realised schedules
    oracle = None
    if ex.get("oracle", True):
        init = [road] + [[{"var": a.var, "level": a.level, "year": int(a.t_decide)} for a in st.actions if not a.failed]
                         for st, _ in sims.values()]
        acts, info = PL.optimise(cfg, phys, ref, [w], w.p["regimes"], init=init, seed=task["seed"])
        st_o = SM.run(w, PL.FixedController(acts), phys, cfg, ref)
        oracle = {"actions": acts, "U": SM.score(st_o, cfg, ref, w.p["regimes"])["U"], "state": st_o}
    best = max(list(U.values()) + ([oracle["U"]] if oracle else []))
    events = MT.change_events(cfg, w)
    t_star = None
    if events:
        e0 = min(e["year"] for e in events)
        t_star = next((y for y in SM.decision_years(cfg) if y >= e0), None)
    post = {}
    if t_star is not None:
        for name, (st, _) in sims.items():
            post[name] = SM.score(st, cfg, ref, w.p["regimes"], t_from=t_star)["U"]
        if oracle:
            post["oracle"] = SM.score(oracle["state"], cfg, ref, w.p["regimes"], t_from=t_star)["U"]
    # future option coverage and lock-in
    lib, goals = G["lib"], G["goals"]
    blank = {T: MT.coverage(cfg, phys, ref, w, MT.blank_state(cfg, phys, w, T, None), lib, goals, T) for T in checks}
    for name, (st, ctrl) in sims.items():
        r = recs[name]
        r["regret"] = best - U[name]
        r["regret_post"] = (max(post.values()) - post[name]) if post else None
        r["update"] = MT.updateability(cfg, events, getattr(ctrl, "replans", []), st, w)
        r["coverage"], r["lockin"], r["gained"] = {}, {}, {}
        for T in checks:
            cov = MT.coverage(cfg, phys, ref, w, st.snapshots[T], lib, goals, T)
            b = blank[T]
            r["coverage"][str(T)] = float(cov.mean())
            r["lockin"][str(T)] = float((b & ~cov).sum() / max(b.sum(), 1))
            r["gained"][str(T)] = float((cov & ~b).mean())
    return {"kind": "main", "i": task["i"], "seed": task["seed"], "z": list(map(float, task["z"])),
            "regimes": w.p["regimes"], "events": events, "t_star": t_star, "records": recs,
            "oracle": None if not oracle else {"U": oracle["U"], "actions": oracle["actions"]},
            "best": best, "blank_coverage": {str(T): float(b.mean()) for T, b in blank.items()},
            "world": {"rain_pts": w.p["rain_pts"], "slr_pts": w.p["slr_pts"], "asset_pts": w.p["asset_pts"], "pop_pts": w.p["pop_pts"],
                      "eff0": w.p["eff0"], "tech_gain": w.p["tech_gain"], "tech_year": w.p["tech_year"]}}


def morris_job(task):
    cfg, phys, ref = G["cfg"], G["phys"], G["ref"]
    w = WD.make_world(cfg, task["z"], task["seed"])
    sims = policies_in_world(cfg, phys, ref, w, G["roadmaps"]["main"], task["seed"])
    return {"kind": "morris", "traj": task["traj"], "step": task["step"],
            "U": {n: SM.score(st, cfg, ref, w.p["regimes"])["U"] for n, (st, _) in sims.items()}}


def delay_job(task):
    cfg, phys, ref = G["cfg"], G["phys"], G["ref"]
    s = task["scale"]
    w = WD.make_world(cfg, task["z"], task["seed"], delay_scale=s)
    sims = policies_in_world(cfg, phys, ref, w, G["roadmaps"][f"delay:{s}"], task["seed"])
    U = {n: SM.score(st, cfg, ref, w.p["regimes"])["U"] for n, (st, _) in sims.items()}
    V = {n: SM.score(st, cfg, ref, w.p["regimes"])["violation_share"] for n, (st, _) in sims.items()}
    st0 = SM.run(w, PL.FixedController([]), phys, cfg, ref)
    U["none"] = SM.score(st0, cfg, ref, w.p["regimes"])["U"]
    V["none"] = SM.score(st0, cfg, ref, w.p["regimes"])["violation_share"]
    post = {}
    for y in cfg["experiment"]["delay"]["postpone"]:
        road = G["roadmaps"][f"delay:{s}:postpone:{y}"]
        for name in ("P0", "P2"):
            if name not in cfg["policies"]:
                continue
            ctrl = PL.Postponed(PL.controller(cfg, phys, ref, name, road, seed=task["seed"]), y)
            st = SM.run(w, ctrl, phys, cfg, ref)
            sc = SM.score(st, cfg, ref, w.p["regimes"])
            post[f"{name}@{y}"] = {"U": sc["U"], "violation_share": sc["violation_share"], "ead_pv": sc["ead_pv"], "cost_pv": sc["cost_pv"]}
    return {"kind": "delay", "scale": s, "i": task["i"], "U": U, "violation": V, "postpone": post}


def portfolios(cfg, phys):
    """Every subset of the measure groups (portfolio_groups present in the model), each at its top
    level in every zone, decided at the start."""
    groups = {g: ts for g, ts in cfg["portfolio_groups"].items() if any(v["type"] in ts for v in phys.variables)}
    types = list(groups)
    sp = PL.Space(cfg, phys)
    out = []
    for r in range(len(types) + 1):
        for comb in itertools.combinations(types, r):
            acts = [{"var": i, "level": sp.levels[i][-1], "year": cfg["years"]["start"]}
                    for i, v in enumerate(phys.variables) if any(v["type"] in groups[g] for g in comb)]
            out.append({"types": list(comb), "actions": acts})
    return out


def complementarity_job(task):
    cfg, phys, ref = G["cfg"], G["phys"], G["ref"]
    c2 = copy.deepcopy(cfg)
    c2["budget"] = {"total_oku": 1e9, "per_decade_oku": 1e9}        # isolate physics and delays from the budget
    w = WD.make_world(c2, task["z"], task["seed"])
    states = [SM.run(w, PL.FixedController(p["actions"]), phys, c2, ref, batch=True) for p in G["portfolios"]]
    SM.evaluate(states, phys)
    out = []
    mid = 2050
    for st in states:
        sc = SM.score(st, c2, ref, w.p["regimes"])
        dr = np.array([p["damage"] / ref["damage"] for p in st.periods])
        pr = np.array([p["pop_risk"] / ref["pop_hazard"] for p in st.periods])
        t = np.array([p["t"] for p in st.periods])
        out.append({"U": sc["U"], "violation_share": sc["violation_share"], "cost_pv": sc["cost_pv"], "ead_pv": sc["ead_pv"],
                    "dr_early": float(dr[t < mid].mean()), "dr_late": float(dr[t >= mid].mean()),
                    "pr_early": float(pr[t < mid].mean()), "pr_late": float(pr[t >= mid].mean()),
                    "dr_series": np.round(dr, 4).tolist(), "ecology": sc["mean_ecology"], "agriculture": sc["mean_agriculture"]})
    return {"kind": "complementarity", "i": task["i"], "portfolios": out}


def participation_job(task):
    cfg, phys, ref = G["cfg"], G["phys"], G["ref"]
    v = task["variant"]
    w = WD.make_world(cfg, task["z"], task["seed"])
    T = task["checkpoint"]
    sims = policies_in_world(cfg, phys, ref, w, G["roadmaps"]["main"], task["seed"], modes=G["variants"][v], snapshots=[T])
    out = {}
    events = MT.change_events(cfg, w)
    for name, (st, ctrl) in sims.items():
        sc = SM.score(st, cfg, ref, w.p["regimes"])
        cov = MT.coverage(cfg, phys, ref, w, st.snapshots[T], G["lib"], G["goals"], T)
        up = MT.updateability(cfg, events, getattr(ctrl, "replans", []), st, w)
        resp = [e["response"] for e in up["events"] if e.get("response") is not None]
        out[name] = {"U": sc["U"], "violation_share": sc["violation_share"], "failures": sc["failures"], "coord_pv": sc["coord_pv"],
                     "cost_pv": sc["cost_pv"], "ead_pv": sc["ead_pv"], "coverage": float(cov.mean()),
                     "response": float(np.mean(resp)) if resp else None,
                     "consensus_years": float(np.mean([a.t_cons - a.t_decide for a in st.actions])) if st.actions else None,
                     "adoption": _mean_or_none([a.adopt for a in st.actions if a.kind in SM.ADOPTION_KINDS])}
    return {"kind": "participation", "variant": v, "i": task["i"], "policies": out}


def scope_job(task):
    cfg, phys, ref = G["cfg"], G["phys"], G["ref"]
    sc = task["scope"]
    w = WD.make_world(cfg, task["z"], task["seed"])
    sims = policies_in_world(cfg, phys, ref, w, G["roadmaps"][f"scope:{sc}"], task["seed"], allowed=G["scope_vars"][sc])
    out = {}
    for name, (st, _) in sims.items():
        r = SM.score(st, cfg, ref, w.p["regimes"])
        tiers = {}
        for a in st.actions:
            if not a.failed:
                t = cfg["measures"][a.kind].get("tier", "国")
                tiers[t] = tiers.get(t, 0.0) + a.cost
        out[name] = {"U": r["U"], "ead_pv": r["ead_pv"], "cost_pv": r["cost_pv"], "violation_share": r["violation_share"],
                     "mean_pop_risk": r["mean_pop_risk"], "mean_ecology": r["mean_ecology"], "failures": r["failures"],
                     "damage_2100": st.periods[-1]["damage"] / ref["damage"], "pop_2100": st.periods[-1]["pop_risk"] / ref["pop_hazard"],
                     "cost_by_tier": tiers}
    return {"kind": "scope", "scope": sc, "i": task["i"], "policies": out}


def summarize_scope(jobs, cfg, scopes, roadmaps, phys):
    names = list(cfg["policies"])
    idx = sorted({j["i"] for j in jobs})
    best = {i: max(j["policies"][n]["U"] for j in jobs if j["i"] == i for n in names) for i in idx}
    keys = ["U", "ead_pv", "cost_pv", "violation_share", "mean_pop_risk", "mean_ecology", "failures", "damage_2100", "pop_2100"]
    res = {}
    for sc in scopes:
        J = [j for j in jobs if j["scope"] == sc]
        row = {}
        for n in names:
            d = {k: float(np.mean([j["policies"][n][k] for j in J])) for k in keys}
            d["regret"] = float(np.mean([best[j["i"]] - j["policies"][n]["U"] for j in J]))
            tiers = {}
            for j in J:
                for t, c in j["policies"][n]["cost_by_tier"].items():
                    tiers[t] = tiers.get(t, 0.0) + c / len(J)
            d["cost_by_tier"] = tiers
            row[n] = d
        res[sc] = row
    V = phys.variables
    plans = {sc: [{"year": a["year"], "name": V[a["var"]]["name"], "level": a["level"],
                   "kind": SM.kind_of(V[a["var"]], a["level"]), "tier": cfg["measures"][SM.kind_of(V[a["var"]], a["level"])].get("tier")}
                  for a in roadmaps[f"scope:{sc}"]] for sc in scopes}
    return {"policies": names, "scopes": scopes, "results": res, "roadmaps": plans, "n_worlds": len(idx)}


def _mean_or_none(x):
    return float(np.mean(x)) if len(x) else None


# ------------------------------------------------------------------ summaries
def summarize_complementarity(jobs, ports):
    n = len(ports)
    keys = ["U", "violation_share", "cost_pv", "ead_pv", "dr_early", "dr_late", "pr_early", "pr_late"]
    mean = {k: [float(np.mean([j["portfolios"][p][k] for j in jobs])) for p in range(n)] for k in keys}
    idx = {tuple(sorted(p["types"])): i for i, p in enumerate(ports)}
    types = [p["types"][0] for p in ports if len(p["types"]) == 1]
    none = idx[()]
    inter = {}
    for a, b in itertools.combinations(types, 2):
        ia, ib, iab = idx[(a,)], idx[(b,)], idx[tuple(sorted([a, b]))]
        row = {}
        for k in ("dr_early", "dr_late", "pr_early", "pr_late", "violation_share"):
            m = mean[k]
            # reduction R(S) = M(none) - M(S); interaction = R(AB) - R(A) - R(B)  (> 0: complementary)
            R = lambda i: m[none] - m[i]  # noqa: E731
            row[k] = R(iab) - R(ia) - R(ib)
        row["U"] = (mean["U"][iab] - mean["U"][none]) - (mean["U"][ia] - mean["U"][none]) - (mean["U"][ib] - mean["U"][none])
        inter[f"{a}+{b}"] = row
    series = [np.mean([j["portfolios"][p]["dr_series"] for j in jobs], 0).round(4).tolist() for p in range(n)]
    # exact Shapley values of each measure type in the 'portfolio game' v(S) = M(none) - M(S) (reductions), and
    # the Shapley interaction index of each pair (> 0: complementary, < 0: substitutes)
    from math import factorial
    m = len(types)
    shap, shap_int = {}, {}
    for k in ("U", "dr_early", "dr_late", "pr_early", "pr_late", "violation_share"):
        sign = 1.0 if k == "U" else -1.0
        val = lambda S: sign * (mean[k][idx[tuple(sorted(S))]] - mean[k][none])  # noqa: E731
        phi = {}
        for a in types:
            rest = [t for t in types if t != a]
            phi[a] = sum(factorial(len(S)) * factorial(m - len(S) - 1) / factorial(m) * (val(S + (a,)) - val(S))
                         for r in range(m) for S in itertools.combinations(rest, r))
        shap[k] = phi
        I = {}
        for a, b in itertools.combinations(types, 2):
            rest = [t for t in types if t not in (a, b)]
            I[f"{a}+{b}"] = sum(factorial(len(S)) * factorial(m - len(S) - 2) / factorial(m - 1)
                                * (val(S + (a, b)) - val(S + (a,)) - val(S + (b,)) + val(S))
                                for r in range(m - 1) for S in itertools.combinations(rest, r))
        shap_int[k] = I
    return {"portfolios": [p["types"] for p in ports], "mean": mean, "interaction": inter, "shapley": shap,
            "shapley_interaction": shap_int, "dr_series": series, "n_worlds": len(jobs), "types": types}


def summarize_delay(jobs, cfg):
    """Per delay scale: utility / regret of the policies; value of an early decision = U(start now) -
    U(no action before the postponement year), absolute and as a share of the plan's value
    U(plan) - U(nothing); extra years above a tolerated ratio caused by the postponement."""
    out = {}
    names = list(cfg["policies"])
    span = cfg["years"]["end"] - cfg["years"]["start"]
    for s in sorted({j["scale"] for j in jobs}):
        J = [j for j in jobs if j["scale"] == s]
        Um = np.array([[j["U"][n] for n in names] for j in J])
        best = Um.max(1, keepdims=True)
        none = np.array([j["U"]["none"] for j in J])
        row = {"U": Um.mean(0).tolist(), "regret": (best - Um).mean(0).tolist(), "max_regret": (best - Um).max(0).tolist(),
               "violation": np.mean([[j["violation"][n] for n in names] for j in J], 0).tolist(),
               "U_none": float(none.mean()), "violation_none": float(np.mean([j["violation"]["none"] for j in J])),
               "plan_value": (Um - none[:, None]).mean(0).tolist(), "postpone": {}}
        for k in J[0]["postpone"]:
            pol, y = k.split("@")
            base = np.array([j["U"][pol] for j in J])
            pu = np.array([j["postpone"][k]["U"] for j in J])
            val = base - none
            row["postpone"][k] = {"U": float(pu.mean()), "value_of_early": float((base - pu).mean()),
                                  "value_of_early_p90": float(np.quantile(base - pu, 0.9)),
                                  "share_of_plan_value": float((base - pu).mean() / max(val.mean(), 1e-9)),
                                  "violation": float(np.mean([j["postpone"][k]["violation_share"] for j in J])),
                                  "extra_violation_years": float(span * np.mean([j["postpone"][k]["violation_share"] - j["violation"][pol] for j in J]))}
        out[str(s)] = row
    return {"policies": names, "scales": out}


def summarize_participation(jobs, cfg, variants):
    names = list(cfg["policies"])
    keys = ["U", "violation_share", "failures", "coord_pv", "cost_pv", "ead_pv", "coverage", "response", "consensus_years", "adoption"]
    idx = sorted({j["i"] for j in jobs})
    best = {i: max(j["policies"][n]["U"] for j in jobs if j["i"] == i for n in names) for i in idx}
    out = {}
    for v in variants:
        J = [j for j in jobs if j["variant"] == v]
        row = {}
        for n in names:
            d = {}
            for k in keys:
                vals = [j["policies"][n][k] for j in J if j["policies"][n][k] is not None]
                d[k] = float(np.mean(vals)) if vals else None
            d["regret"] = float(np.mean([best[j["i"]] - j["policies"][n]["U"] for j in J]))
            row[n] = d
        out[v] = row
    return {"policies": names, "variants": {v: variants[v] for v in variants}, "results": out}
