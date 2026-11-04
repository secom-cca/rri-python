"""Evaluation of roadmaps: robustness / regret, updateability, future option coverage, lock-in,
explainability (PRIM scenario discovery, switch conditions, update reasons).

Regret (per world): best utility among the policies and the perfect-information roadmap (the oracle,
optimised knowing the world) minus the policy's utility, under the criteria that actually hold
(regime changes included).

Future Option Coverage at a checkpoint T: share of alternative goals g (sampled tolerated damage /
exposed-population ratios, Ecology change, farmland kept) that some continuation (a library of
portfolios started at T, simulated from the policy's state at T in the same world) meets in every
period from T + goal_lag_years to the end, within the remaining budget.
Lock-in at T: share of the goals reachable from a blank state at T (nothing built, full budget) that
are not reachable from the policy's state (options lost by commitments: budget spent, irreversible
measures, raised houses, farmland converted).
"""
from __future__ import annotations

import itertools

import numpy as np

from . import planner as PL
from . import sim as SM
from . import world as WD


# ------------------------------------------------------------------ goals and continuations
def sample_goals(cfg, n, seed):
    from ..rdm.uncertainty import lhs
    G = cfg["experiment"]["goals"]
    keys = list(G)
    U = lhs(n, len(keys), np.random.default_rng(seed + 401))
    return [{k: float(G[k][0] + u * (G[k][1] - G[k][0])) for k, u in zip(keys, row)} for row in U]


def continuation_library(cfg, phys, n, seed):
    """Portfolios by measure group (portfolio_groups; the same level in every zone): for each group
    nothing / the middle level / the top level (two choices where a group has only on/off); a
    deterministic subset of n including nothing and everything."""
    sp = PL.Space(cfg, phys)
    groups = {g: ts for g, ts in cfg["portfolio_groups"].items()
              if any(v["type"] in ts for v in phys.variables)}
    nlev = {g: max(len(sp.levels[i]) for i, v in enumerate(phys.variables) if v["type"] in ts) for g, ts in groups.items()}
    names = list(groups)
    combos = list(itertools.product(*[range(min(nlev[g], 3)) for g in names]))
    rng = np.random.default_rng(seed + 409)
    first, last = combos[0], combos[-1]
    rest = [combos[i] for i in rng.permutation(len(combos))[:4 * n] if combos[i] not in (first, last)]
    pick = [first, last] + rest[:max(n - 2, 0)]
    lib = []
    for c in pick:
        sel = dict(zip(names, c))
        acts = []
        for i, v in enumerate(phys.variables):
            g = next((g for g in names if v["type"] in groups[g]), None)
            k = sel.get(g, 0) if g else 0
            if k:
                L = sp.levels[i]
                acts.append({"var": i, "level": L[-1] if k == min(nlev[g], 3) - 1 else L[min(k, len(L) - 1)]})
        lib.append({"portfolio": {g: int(k) for g, k in sel.items()}, "actions": acts})
    return lib


def reachable(states, goals, ref, t_goal):
    """(n_states, n_goals) bool: the state's periods from t_goal meet the goal (and nothing waits for budget)."""
    out = np.zeros((len(states), len(goals)), bool)
    for s, st in enumerate(states):
        P = [p for p in st.periods if p["t"] >= t_goal]
        if not P or st.queue:
            continue
        rd = max(p["damage"] for p in P) / ref["damage"]
        rp = max(p["pop_risk"] for p in P) / ref["pop_hazard"]
        eco = min(p["ecology"] for p in P) - ref["ecology"]
        agr = min(p["agriculture"] for p in P)
        for g, G in enumerate(goals):
            out[s, g] = rd <= G["damage_ratio"] and rp <= G["pop_ratio"] and eco >= G["ecology_delta"] and agr >= G["agriculture_min"]
    return out


def coverage(cfg, phys, ref, world, snap, lib, goals, t_check):
    """Goals reachable from `snap` (state at t_check) by some continuation; returns bool (n_goals,)."""
    states = []
    for c in lib:
        acts = [dict(a, year=t_check) for a in c["actions"]]
        states.append(SM.run(world, PL.FixedController(acts, snap.social.modes), phys, cfg, ref, state=snap.copy(),
                             t_from=t_check, batch=True))
    SM.evaluate(states, phys)
    R = reachable(states, goals, ref, t_check + cfg["experiment"]["goal_lag_years"])
    return R.any(0)


def blank_state(cfg, phys, world, t_check, modes):
    from .social import Social
    st = SM.State(cfg, phys, world, Social(cfg, world.p["social"], modes))
    st.accrued_to = t_check
    return st


# ------------------------------------------------------------------ updateability
def change_events(cfg, world):
    """Deviations of the world from the planner's initial belief: [{kind, year, what, magnitude}].
    A wrong initial assumption (effect, delay, population trend) counts from the start year,
    a later change (climate shift, technology, land use, criteria) from the year it begins."""
    p, b = world.p, WD.belief_params(cfg)
    y0 = float(cfg["years"]["start"])
    tol = cfg["detect"]
    out = []
    if abs(p["rain_pts"][-1][1] - b["rain_pts"][-1][1]) > 0.02 or abs(p["slr_pts"][-1][1] - b["slr_pts"][-1][1]) > 0.05:
        out.append({"kind": "climate", "year": float(p["rain_pts"][-2][0]), "what": "rain_2100",
                    "magnitude": p["rain_pts"][-1][1] - b["rain_pts"][-1][1]})
    for k, e in p["eff0"].items():
        if abs(e - b["eff0"][k]) > tol["effect_tol"]:
            out.append({"kind": "effect", "year": y0, "what": k, "magnitude": e - b["eff0"][k]})
    if abs(p["tech_gain"]) > 0.05:
        out.append({"kind": "effect", "year": float(p["tech_year"]), "what": "tech", "magnitude": p["tech_gain"]})
    if abs(p["asset_pts"][-1][1] - b["asset_pts"][-1][1]) > 0.05:
        out.append({"kind": "landuse", "year": float(p["asset_pts"][1][0]), "what": "asset", "magnitude": p["asset_pts"][-1][1] - 1.0})
    if abs(p["pop_pts"][-1][1] - b["pop_pts"][-1][1]) > 0.05:
        out.append({"kind": "landuse", "year": y0, "what": "pop", "magnitude": p["pop_pts"][-1][1] - b["pop_pts"][-1][1]})
    for k, ph in p["phases"].items():
        for name in ("build", "diffuse", "mature"):
            bv = b["phases"][k][name]
            if bv > 0 and abs(ph[name] / bv - 1) > tol["delay_tol"]:
                out.append({"kind": "delay", "year": y0, "what": f"{k}.{name}", "magnitude": ph[name] / bv - 1})
    if len(p["regimes"]) > 1:
        out.append({"kind": "criteria", "year": float(p["regimes"][1][0]), "what": p["regimes"][1][1], "magnitude": 1.0})
    return out


def updateability(cfg, events, replans, st, world):
    """Detection / response delays per true change, false alarms, plan churn."""
    end = cfg["years"]["end"]
    det = []
    for ev in [e for e in events if e["kind"] != "delay"]:       # delays are learnt from completed measures only
        hit = next((r for r in replans if r["year"] >= ev["year"] - 1e-9 and any(x["kind"] == ev["kind"] for x in r["reasons"])), None)
        if hit is None:
            det.append({**ev, "detected": None, "delay": end - ev["year"], "censored": True, "response": None})
            continue
        resp = [a for a in st.actions if a.source == "replan" and a.t_decide >= hit["year"] and not a.failed]
        done = min((st.complete_year(a) for a in resp), default=None)
        det.append({**ev, "detected": hit["year"], "delay": hit["year"] - ev["year"], "censored": False,
                    "response": None if done is None else done - ev["year"]})
    false = 0
    for r in replans:
        for x in r["reasons"]:
            if not any(e["kind"] == x["kind"] and e["year"] <= r["year"] + 1e-9 for e in events):
                false += 1
    n_reasons = sum(len(r["reasons"]) for r in replans)
    churn = [len(r.get("diff", [])) for r in replans if not r.get("skipped")]
    return {"events": det, "false_alarms": false, "n_reasons": n_reasons,
            "attribution": None if not n_reasons else 1 - false / n_reasons,
            "churn": float(np.mean(churn)) if churn else 0.0, "n_replans": len(churn)}


# ------------------------------------------------------------------ PRIM
def prim(X, y, alpha=0.05, min_mass=0.05, names=None):
    """Patient Rule Induction Method (Friedman & Fisher 1999), peeling only.
    X (n, k) in [0, 1], y bool.  Returns the peeling trajectory (coverage, density, mass, box)."""
    X = np.asarray(X, float); y = np.asarray(y, bool)
    n, k = X.shape
    inside = np.ones(n, bool)
    box = np.array([[0.0, 1.0]] * k)
    total = max(y.sum(), 1)
    traj = [{"coverage": float(y[inside].sum() / total), "density": float(y[inside].mean()) if inside.any() else 0.0,
             "mass": 1.0, "box": box.tolist()}]
    while inside.mean() > min_mass:
        best = None
        for j in range(k):
            xj = X[inside, j]
            for side in (0, 1):
                q = np.quantile(xj, alpha if side == 0 else 1 - alpha)
                keep = inside & ((X[:, j] >= q) if side == 0 else (X[:, j] <= q))
                if keep.sum() == inside.sum() or keep.sum() < 2:
                    continue
                d = y[keep].mean()
                if best is None or d > best[0]:
                    best = (d, j, side, q, keep)
        if best is None:
            break
        d, j, side, q, keep = best
        if d <= traj[-1]["density"] + 1e-12 and len(traj) > 1:
            break
        inside = keep
        box[j, side] = q
        traj.append({"coverage": float(y[inside].sum() / total), "density": float(d), "mass": float(inside.mean()),
                     "box": box.tolist()})
    return traj


def prim_box(X, y, F, alpha=0.05, min_mass=0.05, density=0.8):
    """The first box on the peeling trajectory with density >= `density` (else the densest), described
    by its restricted dimensions in physical units."""
    y = np.asarray(y, bool)
    if y.sum() == 0 or y.all():
        return {"n_cases": int(y.sum()), "n": len(y), "box": [], "coverage": float(y.all()), "density": float(y.mean()), "mass": 1.0,
                "trajectory": []}
    tr = prim(X, y, alpha, min_mass)
    cand = [t for t in tr if t["density"] >= density]
    b = cand[0] if cand else max(tr, key=lambda t: (t["density"], t["coverage"]))
    names = list(F)
    dims = []
    for j, (lo, hi) in enumerate(b["box"]):
        if lo > 1e-9 or hi < 1 - 1e-9:
            f = F[names[j]]
            dims.append({"factor": names[j], "label": f["label"], "group": f["group"],
                         "low": f["low"] + lo * (f["high"] - f["low"]), "high": f["low"] + hi * (f["high"] - f["low"]),
                         "unit_low": lo, "unit_high": hi})
    return {"n_cases": int(y.sum()), "n": len(y), "box": dims, "coverage": b["coverage"], "density": b["density"], "mass": b["mass"],
            "trajectory": [{k: t[k] for k in ("coverage", "density", "mass")} for t in tr]}


# ------------------------------------------------------------------ explanation texts
def describe_replan(cfg, phys, r):
    """One sentence per update: why (observed vs believed) and what changed."""
    V = phys.variables
    why = []
    for x in r["reasons"]:
        if x["kind"] == "criteria":
            why.append(f"評価基準が「{x['believed']}」から「{x['observed']}」に変化")
        elif "new_2100" in x and x["kind"] == "climate":
            why.append(f"{x['what']}の観測 {x['observed']:.3f}（想定 {x['believed']:.3f}）→ 2100年の想定を {x['new_2100']:.2f} に更新")
        elif x["kind"] == "landuse":
            why.append(f"{x['what']}の観測 {x['observed']:.2f}（想定 {x['believed']:.2f}）")
        else:
            why.append(f"{x['what']}の観測 {x['observed']:.2f}（想定 {x['believed']:.2f}）")
    what = []
    for d in r.get("diff", []):
        nm = V[d["var"]]["name"]
        if d["change"] == "added":
            what.append(f"{nm} を追加（{d['year']}年）")
        elif d["change"] == "dropped":
            what.append(f"{nm} を中止")
        elif d["change"] == "level":
            what.append(f"{nm} の水準 {d['from']}→{d['to']}")
        else:
            what.append(f"{nm} を{'前倒し' if d['change'] == 'advanced' else '先送り'}（{d['from']}→{d['to']}年）")
    if r.get("skipped"):
        return f"{r['year']}年: " + "、".join(why) + "（再計画の上限に達したため更新せず）"
    gain = "" if r.get("U_old") is None else f"（更新後の想定での効用 {r['U_old']:.3f} → {r['U_new']:.3f}）"
    return f"{r['year']}年: " + "、".join(why) + " → " + ("、".join(what) if what else "ロードマップは変更なし") + gain
