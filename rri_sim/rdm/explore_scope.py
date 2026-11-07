"""Trade spaces by implementing actor (国 / 上流自治体 / 下流自治体 / 地域主体) on the exploration surrogate.

    python scripts/run_explore.py --scopes abukuma_explore_v2

Every decision variable belongs to an actor (`explore.scopes.actors`: measure type -> 国 / 自治体 /
地域主体; `zonal` maps 自治体 to the municipalities of the variable's zone, e.g. 上流 -> 上流自治体; a
list form appends the zone, 自治体・上流) or to several actors jointly ({actor: cost share}, e.g. 農業用
排水機場 or 民有林の森林整備): a joint variable needs all of them and its cost is split by share.  For
every non-empty combination of actors a mixed NSGA-II runs on the surrogate with the measures outside
the combination fixed at 0.  Every
evaluated policy is kept with the set of actors it needs, so the trade space of a combination is all
policies whose actors are in it (including those found for smaller combinations and the exploration's
own archive).  No RRI runs: the surrogate is refit from the stored design (cached) and the costs come
from the exploration's cost table (or the roadmap's cost_parts.json).

Writes data/processed/explore/<name>/scopes.json (GUI /explore, docs/12 12.8).
"""
from __future__ import annotations

import base64
import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import config as CF
from .explore import EXPLORE_DIR, Space, nsga_mixed, space_from_results, surrogate_from_results
from .candidates import superseded_raise
from .evaluate import xkey
from .sensitivity import GP

ANNUITY = (1 - 1.04 ** -50) / 0.04      # 治水経済調査マニュアル: 50 years at 4 % (costs.py)
KEYS = ("damage", "pop_hazard", "households", "livability", "ecology")
SUB_KEYS = ("relocated_households", "relocated_population", "service_access", "natural_habitat_km2", "habitat_connectivity")


def shares_of(variables, sc):
    """{actor: cost share} per variable (zonal actors resolved by the variable's zone) and the ordered actor list."""
    zonal = sc.get("zonal", ())

    def name(a, z):
        if isinstance(zonal, dict) and a in zonal:
            return zonal[a].get(z, f"{a}・{z}")
        return f"{a}・{z}" if a in zonal else a
    out = []
    for v in variables:
        a = sc["actors"].get(v["type"], "地域主体")
        sh = {a: 1.0} if isinstance(a, str) else {k: float(w) for k, w in a.items()}
        tot = sum(sh.values()) or 1.0
        res = {}
        for k, w in sh.items():
            n = name(k, v["zone"])
            res[n] = res.get(n, 0.0) + w / tot
        out.append(res)
    base = []
    for a in sc["actors"].values():
        base += [a] if isinstance(a, str) else list(a)
    base = list(dict.fromkeys(base))
    root = lambda a: next((k for k, m in zonal.items() if a in m.values()), a.split("・")[0]) if isinstance(zonal, dict) \
        else a.split("・")[0]  # noqa: E731
    seen = list(dict.fromkeys(n for sh in out for n in sh))
    order = sorted(seen, key=lambda a: (base.index(root(a)) if root(a) in base else 99, a))
    return out, order


def actors_of(variables, sc):
    """Lead actor per variable (largest cost share) and the ordered actor list."""
    sh, order = shares_of(variables, sc)
    return [max(s, key=s.get) for s in sh], order


def cost_rows(res):
    """Per variable: (values, total present value [億円]) for options / a rate grid."""
    if res.get("cost_tables"):
        return [(np.array(t["values"], float), np.array(t["total"], float)) for t in res["cost_tables"]]
    from ..roadmap.physics import PHYS_DIR
    cp = PHYS_DIR / res["name"] / "cost_parts.json"
    if not cp.exists():
        raise FileNotFoundError(f"{cp}: no cost table in results.json; run `python scripts/run_roadmap.py --prepare` once")
    tab = json.loads(cp.read_text(encoding="utf-8"))["table"]
    return [(np.array(t["values"], float), np.array(t["initial"]) + ANNUITY * np.array(t["annual"])) for t in tab]


def _gp_mean(m, F):
    """Surrogate output by matrix products (same as GP.__call__, faster for large batches)."""
    if not isinstance(m, GP):
        return m(F)
    from ..roadmap.physics import _fast_gp
    return _fast_gp(m, F)


def nondominated(F):
    """Mask of the non-dominated rows of F (minimised), in chunks."""
    F = np.asarray(F, float)
    keep = np.ones(len(F), bool)
    for s in range(0, len(F), 256):
        a = F[s:s + 256]
        dom = ((F[None] <= a[:, None]).all(2) & (F[None] < a[:, None]).any(2)).any(1)
        keep[s:s + 256] = ~dom
    return keep


def run(name, cfg=None, log=print, out_dir=None):
    t0 = time.time()
    out = Path(out_dir) if out_dir else EXPLORE_DIR / name
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    ex = CF._merge(CF.DEFAULT["explore"], res["config"]["explore"])
    if cfg:
        ex = CF._merge(ex, cfg.get("explore", {}))
    sc = ex["scopes"]
    sp = space_from_results(res)
    V = res["variables"]
    nV = len(V)
    zones = res["zones"]
    S = surrogate_from_results(res, tuple(k for k in KEYS + SUB_KEYS if k in res["design"]["y"][0]) + tuple("dz:" + z for z in zones),
                               [out, EXPLORE_DIR.parent / "roadmap" / "_physics" / name], log)
    costs = cost_rows(res)
    shares, actors = shares_of(V, sc)
    actor = [max(s_, key=s_.get) for s_ in shares]
    amask = np.array([sum(1 << actors.index(a) for a in s_) for s_ in shares])
    log(f"実施主体: {actors}（{', '.join(f'{a} {sum(a in x for x in shares)} 変数' for a in actors)}; "
        f"共同: {[v['name'] for v, x in zip(V, shares) if len(x) > 1]}）")
    keys = tuple(k for k in KEYS + SUB_KEYS if k in res["design"]["y"][0])     # older designs lack the sub-indicators

    E = {k: np.array(v, float) for k, v in res["ensemble"].items()}
    ne = len(next(iter(E.values())))
    af, D_ref = E.get("asset_factor", np.ones(ne)), res["reference"]["damage"]
    tc = np.array(ex["target_curve"], float)
    edges = np.array(res["rain_edges"])
    nb = len(edges) - 1
    bins = np.clip(np.searchsorted(edges, E.get("rain_factor", np.ones(ne)), side="right") - 1, 0, nb - 1)
    groups = [np.ones(ne, bool)] + [bins == b for b in range(nb)]
    it = int(np.argmin(np.abs(tc - ex["target_ratio"])))

    def metrics(xs):
        X = np.repeat(np.array(xs, float), ne, axis=0)
        U = {k: np.tile(v, len(xs)) for k, v in E.items()}
        Fe = sp.features(X, U)
        P = {}
        for k in keys:
            t = _gp_mean(S.models[k], Fe)
            P[k] = (np.maximum(np.expm1(t), 0) if S._is_log(k) else t).reshape(len(xs), ne)
        ratio = P["damage"] / D_ref
        ok = ratio[:, None, :] <= tc[None, :, None] + 1e-9                         # (n, targets, ne)
        rel = np.stack([ok[:, :, g].mean(2) if g.any() else np.zeros(ok.shape[:2]) for g in groups], 2)  # (n, targets, 1 + nb)
        d = P["damage"] * af
        cv = np.array([[float(np.interp(x[i], *costs[i])) if V[i]["continuous"] else float(costs[i][1][int(x[i])])
                        for i in range(nV)] for x in xs])
        for i, j in superseded_raise(V):                         # relocation active -> raising costs nothing
            cv[:, j] *= np.array([float(x[i]) == 0 for x in xs])
        return {"cost_var": cv, "cost": cv.sum(1), "rel": rel, "damage_mean": d.mean(1), "damage_p90": np.quantile(d, 0.9, axis=1),
                "pop": P["pop_hazard"].mean(1), "households": P["households"].mean(1),
                "livability": P["livability"].mean(1), "ecology": P["ecology"].mean(1),
                **{k: P[k].mean(1) for k in SUB_KEYS if k in P}}

    obj = sc["optimisation"]["objectives"]
    sign = {"cost": 1, "reliability": -1, "damage_mean": 1, "damage_p90": 1, "pop": 1, "ecology": -1, "livability": -1,
            "relocated_households": 1, "relocated_population": 1, "service_access": -1, "natural_habitat_km2": -1,
            "habitat_connectivity": -1}
    okeys = [k for k in obj if k != "reliability"]

    def fvec(m):
        cols = [m["rel"][:, it, 0] if k == "reliability" else m[k] for k in obj]
        return np.column_stack([sign[k] * np.asarray(c) for k, c in zip(obj, cols)])

    # ---- policies: exploration archive + one NSGA-II per actor combination
    pts, index = [], {}

    def add(xs, m=None):
        new = [list(t) for t in dict.fromkeys(tuple(xkey(x)) for x in xs) if t not in index]
        if new:
            m = metrics(new)
            for j, x in enumerate(new):
                index[tuple(x)] = len(pts)
                pts.append({"x": x, **{k: (v[j] if np.ndim(v) > 1 else float(v[j])) for k, v in m.items()}})
        return [index[tuple(xkey(x))] for x in xs]

    for s in range(0, len(res["archive"]), 256):
        add([a["x"] for a in res["archive"][s:s + 256]])
    log(f"探索の評価済み {len(pts)} 案を再評価（{time.time() - t0:.0f} s）")
    combos = [c for r in range(1, len(actors) + 1) for c in itertools.combinations(range(len(actors)), r)]
    op = sc["optimisation"]
    scopes = []
    for c in combos:
        m = sum(1 << k for k in c)
        idx = [i for i in range(nV) if amask[i] & m]
        sub = Space([sp.V[i] for i in idx], sp.P)

        def full(xs, idx=idx):
            out_ = []
            for x in xs:
                y = [0] * nV
                for i, a in zip(idx, x):
                    y[i] = a
                out_.append(y)
            return out_

        def evaluate(xs, full=full):
            ii = add(full(xs))
            return fvec({k: np.array([pts[i][k] for i in ii]) for k in okeys}
                        | {"rel": np.array([pts[i]["rel"] for i in ii])}).tolist()
        n0 = len(pts)
        rng = np.random.default_rng(res["config"]["seed"] + 101 + m)
        top = [1.0 if v.continuous else len(v.options) - 1 for v in sub.V]
        nsga_mixed(sub, evaluate, op["pop"], op["gens"], rng, op["p_mut"], init=[[0] * len(idx), top], log=lambda s: None)
        label = "＋".join(actors[k] for k in c)
        scopes.append({"mask": m, "actors": [actors[k] for k in c], "label": label, "n_vars": len(idx), "n_new": len(pts) - n0})
        log(f"  {label}: 変数 {len(idx)}・新しい案 {len(pts) - n0}（計 {len(pts)}, {time.time() - t0:.0f} s）")

    # ---- what to store: per combination the non-dominated policies first, then a random sample
    need = np.array([int(np.bitwise_or.reduce(amask[np.array(p["x"]) > 0])) if any(np.array(p["x"]) > 0) else 0 for p in pts])
    Fall = fvec({k: np.array([p[k] for p in pts]) for k in okeys}
                | {"rel": np.array([p["rel"] for p in pts])})
    keep = set(np.flatnonzero(need == 0).tolist())
    rng = np.random.default_rng(res["config"]["seed"] + 211)
    cap = sc["max_points"]
    for s_ in scopes:
        inside = np.flatnonzero((need & ~s_["mask"]) == 0)
        nd = inside[nondominated(Fall[inside])]
        if len(nd) > cap:
            nd = nd[np.argsort(Fall[nd, 0])][np.linspace(0, len(nd) - 1, cap).astype(int)]
        rest = np.setdiff1d(inside, nd)
        pick = rng.choice(rest, min(len(rest), max(cap - len(nd), cap // 3)), replace=False) if len(rest) else rest
        keep |= set(nd.tolist()) | set(pick.tolist())
        s_["n_points"] = int(len(inside))
        s_["n_nondominated"] = int(len(nd))
    keep = sorted(keep, key=lambda i: pts[i]["cost"])
    P = [pts[i] for i in keep]
    rel = np.round(np.array([p["rel"] for p in P]) * 250).astype(np.uint8)
    ca = np.array([[sum(p["cost_var"][i] * shares[i].get(a, 0.0) for i in range(nV)) for a in actors] for p in P])
    r2 = lambda a, d=2: np.round(np.asarray(a, float), d).tolist()  # noqa: E731
    doc = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0,
           "actors": actors, "actor_of": actor, "actor_shares": shares, "objectives": obj, "target_curve": tc.tolist(), "rain_edges": edges.tolist(),
           "reference": {"damage": D_ref, "livability": res["reference"]["livability"], "ecology": res["reference"]["ecology"],
                         "pop_hazard": res["reference"]["pop_hazard"]},
           "scopes": scopes, "n_evaluated": len(pts),
           "points": {"n": len(P), "x": [[round(float(a), 2) for a in p["x"]] for p in P], "need": need[keep].tolist(),
                      "cost": r2([p["cost"] for p in P], 1), "cost_actor": r2(ca, 1),
                      "damage_mean": r2([p["damage_mean"] for p in P], 1), "damage_p90": r2([p["damage_p90"] for p in P], 1),
                      "pop": r2([p["pop"] for p in P], 0), "households": r2([p["households"] for p in P], 0),
                      "livability": r2([p["livability"] for p in P], 3), "ecology": r2([p["ecology"] for p in P], 3),
                      **{k: r2([p[k] for p in P], 4) for k in SUB_KEYS if k in P[0]},
                      "rel_shape": list(rel.shape[1:]), "rel_scale": 250,
                      "rel": base64.b64encode(rel.tobytes()).decode()}}
    (out / "scopes.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    log(f"完了: {out / 'scopes.json'}（{len(P)} 案を保存 / 評価 {len(pts)} 案, {time.time() - t0:.0f} s）")
    return doc
