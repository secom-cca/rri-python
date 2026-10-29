"""Exploration without stakeholder weights (MORDM-style) on a joint surrogate.

    from rri_sim.rdm import explore, config
    explore.run(config.load(), "data/processed/explore/abukuma_explore")

Instead of a budget constraint and weighted utilities, every policy is described
by raw metrics over the uncertainty: the cost (by zone / type), the share of
scenarios in which the damage stays below the target (`target_ratio` x the
damage of the current climate without measures), the damage distribution by
zone, flooded population, Livability and Ecology.  Paddy dams and afforestation
are implementation rates (0-100 % of the candidate cells per zone); levees,
retention basins and housing measures keep their options (candidates.py).

1. design      RRI runs at a joint LHS of (measures, uncertainties) + no-measure runs
               + adaptive runs where the damage is close to the target (largest
               chance of misclassifying the target, |mu - T| / sigma smallest)
2. surrogate   per output a GP (ARD) or ridge model on physically scaled features
               (levee height x levee_effect, paddy rate x paddy_effect, ...),
               chosen by leave-one-out error; validated on an independent LHS
3. optimise    mixed integer / continuous NSGA-II on the surrogate; each policy is
               evaluated over a common ensemble of uncertainty samples
4. verify      a spread of Pareto policies x new scenarios with RRI: prediction error
               and agreement of the target classification

Writes config_explore.json, evaluations.jsonl (RRI cache) and results.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config as CF
from . import nsga2 as NS
from . import uncertainty as UQ
from .evaluate import xkey
from .sensitivity import GP, Linear, error_stats
from ..data import PROCESSED

EXPLORE_DIR = PROCESSED / "explore"
LOG_KEYS = ("damage", "pop_hazard", "households")       # modelled as log(1 + y)


# ------------------------------------------------------------------ decisions and uncertainties
def make_continuous(variables, ex, seed):
    """Paddy dams / afforestation -> implementation rates; fixes the order in which a rate takes cells."""
    rng = np.random.default_rng(seed + 31)
    for v in variables:
        if v.type in ex["continuous"]:
            v.continuous = True
            if ex.get("cell_order", "random") == "random":
                v.cells = [v.cells[i] for i in rng.permutation(len(v.cells))]
            v.options = ["0%", "100%"]
    return variables


class Space:
    """Decision genes (integer option or rate) and uncertainty parameters in unit coordinates."""

    def __init__(self, variables, params):
        self.V = variables
        self.P = params
        self.unames = list(params)
        self.sim_names = [n for n in self.unames if params[n]["kind"] == "sim"]
        self.n_opt = np.array([0 if v.continuous else len(v.options) for v in variables])
        self.cont = np.array([v.continuous for v in variables])

    def x_from_unit(self, Z, zero_share=0.0, rng=None):
        X = np.empty_like(Z)
        X[:, self.cont] = np.round(Z[:, self.cont], 2)
        if zero_share and rng is not None:
            off = (rng.random(Z.shape) < zero_share) & self.cont[None]
            X[off] = 0.0
        d = ~self.cont
        X[:, d] = np.minimum((Z[:, d] * self.n_opt[d]).astype(int), self.n_opt[d] - 1)
        return X

    def u_values(self, Zu):
        """unit -> {name: values}"""
        return {n: self.P[n]["low"] + Zu[:, i] * (self.P[n]["high"] - self.P[n]["low"]) for i, n in enumerate(self.unames)}

    def nominal_u(self):
        return {n: self.P[n]["nominal"] for n in self.unames}

    def sim_dict(self, u):
        return {n: round(float(u[n]), 5) for n in self.sim_names}

    def features(self, X, U):
        """Physically scaled surrogate inputs, ~[0, 1]: effect coefficients multiply the measure
        they scale (a coefficient has no influence where the measure is absent)."""
        X = np.asarray(X, float)

        def g(n):
            return np.broadcast_to(np.asarray(U.get(n, self.P[n]["nominal"] if n in self.P else 1.0), float), (len(X),))
        cols = []
        for i, v in enumerate(self.V):
            xi = X[:, i]
            if v.type == "levee":
                hs = np.array([0.0 if p is None else p["height"] for p in v.params])
                cols.append(hs[xi.astype(int)] / max(hs.max(), 1e-9) * g("levee_effect"))
            elif v.type == "retention":
                cols.append((xi > 0) * g("retention_effect"))
            elif v.type == "relocate":
                kinds = [None if p is None else p.get("type") for p in v.params]
                k = np.array(kinds, object)[xi.astype(int)]
                cols.append((k == "raise") * g("raise_effect"))
                cols.append((k == "relocate").astype(float))
            elif v.type == "paddy_dam":
                cols.append(xi * g("paddy_effect"))
            elif v.type == "forest":
                cols.append(xi * g("forest_effect"))
                cols.append(xi)                     # land use itself (Ecology, crop loss)
            else:
                cols.append(xi / max(len(v.options) - 1, 1))
        for n in ("rain_factor", "sea_level_rise_m", "rain_gamma"):
            if n in self.P:
                lo, hi = self.P[n]["low"], self.P[n]["high"]
                cols.append((g(n) - lo) / ((hi - lo) or 1.0))
        return np.column_stack(cols)


# ------------------------------------------------------------------ cost tables (no simulation)
def cost_tables(ev, space, grid=11):
    """Per variable: cost by option (integer) or on a rate grid (continuous); costs of separate
    edits are additive (engine.estimate), rates are interpolated linearly."""
    zero = [0] * len(space.V)
    tabs = []
    for i, v in enumerate(space.V):
        vals = np.linspace(0, 1, grid) if v.continuous else np.arange(len(v.options))
        rows = []
        for a in vals:
            x = list(zero); x[i] = float(a) if v.continuous else int(a)
            c = ev.cost(x)
            rows.append({"total": c["total"], "by_zone": c["by_zone"], "by_type": c["by_type"]})
        tabs.append({"values": vals.tolist(), "rows": rows})
    return tabs


def cost_of(x, tabs, space):
    tot, zone, typ = 0.0, {}, {}
    for i, v in enumerate(space.V):
        t = tabs[i]
        if v.continuous:
            vals = np.array(t["values"]); j = int(np.clip(np.searchsorted(vals, x[i]) - 1, 0, len(vals) - 2))
            w = (x[i] - vals[j]) / (vals[j + 1] - vals[j])
            a, b = t["rows"][j], t["rows"][j + 1]
            mix = lambda p, q: (1 - w) * p + w * q  # noqa: E731
            tot += mix(a["total"], b["total"])
            for z in set(a["by_zone"]) | set(b["by_zone"]):
                zone[z] = zone.get(z, 0.0) + mix(a["by_zone"].get(z, 0.0), b["by_zone"].get(z, 0.0))
            for k in set(a["by_type"]) | set(b["by_type"]):
                typ[k] = typ.get(k, 0.0) + mix(a["by_type"].get(k, 0.0), b["by_type"].get(k, 0.0))
        else:
            r = t["rows"][int(x[i])]
            tot += r["total"]
            for z, c in r["by_zone"].items():
                zone[z] = zone.get(z, 0.0) + c
            for k, c in r["by_type"].items():
                typ[k] = typ.get(k, 0.0) + c
    return {"total": tot, "by_zone": zone, "by_type": typ}


# ------------------------------------------------------------------ surrogate
def _ridge_loo(F, y):
    err = []
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        err.append(float(Linear(F[k], y[k])(F[i:i + 1])[0]) - y[i])
    return float(np.sqrt(np.mean(np.square(err))))


class Surrogate:
    """One model per output key; damage-like outputs on a log(1 + y) scale."""

    def __init__(self, F, Y, seed=0, keys=None):
        self.models, self.kind, self.loo = {}, {}, {}
        for k in keys or Y:
            y = np.asarray(Y[k], float)
            t = np.log1p(np.maximum(y, 0)) if self._is_log(k) else y
            if np.std(t) == 0:
                self.models[k], self.kind[k], self.loo[k] = Linear(F, t), "linear", {"gp": 0.0, "linear": 0.0}
                continue
            gp = GP(F, t, restarts=3, seed=seed)
            loo = {"gp": float(np.sqrt(np.mean(gp.loo_residuals() ** 2))), "linear": _ridge_loo(F, t)}
            kind = min(loo, key=loo.get)
            self.models[k], self.kind[k], self.loo[k] = (gp if kind == "gp" else Linear(F, t)), kind, loo

    @staticmethod
    def _is_log(k):
        return k in LOG_KEYS or k.startswith("dz:")

    def predict(self, F, k, transformed=False):
        t = self.models[k](F)
        return t if transformed or not self._is_log(k) else np.maximum(np.expm1(t), 0)

    def std(self, F, k):
        m = self.models[k]
        return m.std(F) if isinstance(m, GP) else np.full(len(F), 1e-6)


def outputs_of(m):
    out = {k: m[k] for k in ("damage", "pop_hazard", "households", "livability", "ecology")}
    out.update({"dz:" + z: v for z, v in m["damage_zone"].items()})
    return out


# ------------------------------------------------------------------ mixed NSGA-II
def nsga_mixed(space, evaluate_batch, pop, gens, rng, p_mut=0.15, init=(), log=print):
    """Integer genes: uniform crossover + random reset; rate genes: blend crossover + Gaussian
    mutation (sd 0.15), rounded to 0.01.  evaluate_batch(list of x) -> objective rows (minimised)."""
    nV = len(space.V)
    archive, seen = [], {}

    def add(xs, gen):
        new = []
        for x in xs:
            t = tuple(xkey(x))
            if t not in seen:
                seen[t] = len(archive) + len(new)
                new.append(list(t))
        if new:
            F = evaluate_batch(new)
            for x, f in zip(new, F):
                archive.append({"x": x, "f": [float(a) for a in f], "gen": gen})
        return [seen[tuple(xkey(x))] for x in xs]

    xs = [list(x) for x in init]
    while len(xs) < pop:
        xs.append(space.x_from_unit(rng.random((1, nV)), 0.2, rng)[0].tolist())
    P = list(dict.fromkeys(add(xs, 0)))
    for g in range(1, gens + 1):
        F = np.array([archive[i]["f"] for i in P])
        rank, cd = np.zeros(len(P), int), np.zeros(len(P))
        for r, fr in enumerate(NS.nondominated_sort(F, np.zeros(len(P)))):
            rank[fr] = r; cd[fr] = NS.crowding(F[fr])

        def pick():
            a, b = rng.integers(0, len(P), 2)
            return P[a] if (rank[a], -cd[a]) < (rank[b], -cd[b]) else P[b]
        kids = []
        for _ in range(pop):
            a, b = np.array(archive[pick()]["x"], float), np.array(archive[pick()]["x"], float)
            c = np.where(rng.random(nV) < 0.5, a, b)
            beta = rng.uniform(-0.25, 1.25, nV)
            c[space.cont] = (a + beta * (b - a))[space.cont]
            mut = rng.random(nV) < p_mut
            d = mut & ~space.cont
            c[d] = rng.integers(0, space.n_opt[d])
            m = mut & space.cont
            c[m] += rng.normal(0, 0.15, m.sum())
            c[space.cont] = np.round(np.clip(c[space.cont], 0, 1), 2)
            kids.append(c.tolist())
        K = add(kids, g)
        U = list(dict.fromkeys(P + K))
        F = np.array([archive[i]["f"] for i in U])
        newP = []
        for fr in NS.nondominated_sort(F, np.zeros(len(U))):
            if len(newP) + len(fr) <= pop:
                newP += [U[i] for i in fr]
            else:
                c = NS.crowding(F[fr])
                newP += [U[fr[i]] for i in np.argsort(-c)[:pop - len(newP)]]
                break
        P = newP
        if g % 10 == 0 or g == gens:
            log(f"  NSGA-II 世代 {g}/{gens}: 評価 {len(archive)} 案")
    return archive


# ------------------------------------------------------------------ pipeline
def run(cfg, out_dir=None, evaluator=None, mapinfo=None, log=print):
    t0 = time.time()
    cfg = CF.load(cfg)
    ex = cfg["explore"]
    out = Path(out_dir) if out_dir else EXPLORE_DIR / ex["name"]
    out.mkdir(parents=True, exist_ok=True)
    CF.save(cfg, out / "config_explore.json")
    rng = np.random.default_rng(cfg["seed"] + 77)
    if evaluator is None:
        from .run import setup_rri
        evaluator, mapinfo = setup_rri(cfg, out / "evaluations.jsonl", log)
        if ex.get("current_baseline", True):
            evaluator.gextra = {"baseline": "current"}
    ev = evaluator
    make_continuous(ev.variables, ex, cfg["seed"])
    params = cfg["sensitivity"]["parameters"]
    sp = Space(ev.variables, params)
    nV, nU = len(sp.V), len(sp.unames)
    zero = [0] * nV
    nom = sp.nominal_u()
    log(f"探索: 施策 {nV} 変数（実施率: {[v.name for v in sp.V if v.continuous]}）× 不確実性 {nU} 個")

    def sim(x, u):
        return ev.simulate(x, sp.sim_dict(u))

    # reference: current climate, no measures
    ref = sim(zero, nom)
    D_ref = ref["damage"]
    Dz_ref = dict(ref["damage_zone"])
    T = ex["target_ratio"] * D_ref
    log(f"目標: 被害額 ≤ {ex['target_ratio']:.2f} × {D_ref:.0f} = {T:.0f} 億円（現在気候・対策なし）")
    tabs = cost_tables(ev, sp)

    # ---- 1. design
    de = ex["design"]
    rows_x, rows_u, rows_y = [], [], []

    def run_points(Xs, Us, tag):
        for j, (x, u) in enumerate(zip(Xs, Us)):
            m = sim(list(x), u)
            rows_x.append(list(x)); rows_u.append(dict(u)); rows_y.append(outputs_of(m))
            if (j + 1) % 10 == 0 or j + 1 == len(Xs):
                log(f"  {tag} {j + 1}/{len(Xs)}（RRI 計 {getattr(ev, 'n_sim', 0)} 回, {time.time() - t0:.0f} s）")

    def design(n, zero_x=False):
        Z = UQ.lhs(n, nV + nU, rng)
        X = np.zeros((n, nV)) if zero_x else sp.x_from_unit(Z[:, :nV], ex["zero_share"], rng)
        Uv = sp.u_values(Z[:, nV:])
        return X, [{k: float(Uv[k][i]) for k in sp.unames} for i in range(n)]
    rows_x.append(zero); rows_u.append(dict(nom)); rows_y.append(outputs_of(ref))
    log(f"学習データ: LHS {de['n_train']} 点 + 対策なし {de['n_none']} 点")
    run_points(*design(de["n_train"]), "学習")
    run_points(*design(de["n_none"], zero_x=True), "対策なし")
    keys = list(rows_y[0])

    def feats(xs, us):
        U = {k: np.array([u[k] for u in us]) for k in sp.unames}
        return sp.features(np.array(xs, float), U)

    # adaptive: points whose damage is most likely misclassified w.r.t. the target
    for r in range(de["adaptive_rounds"]):
        S = Surrogate(feats(rows_x, rows_u), {"damage": [y["damage"] for y in rows_y]}, cfg["seed"], ["damage"])
        Xp, Up = design(de["pool"])
        Fp = feats(Xp, Up)
        mu, sd = S.predict(Fp, "damage", transformed=True), S.std(Fp, "damage")
        crit = np.abs(mu - np.log1p(T)) / np.maximum(sd, 1e-6)
        cand = np.argsort(crit)[:max(5 * de["adaptive_batch"], 1)]
        pick = [int(cand[0])]
        while len(pick) < min(de["adaptive_batch"], len(cand)):       # spread the batch (maximin in feature space)
            dmin = np.min(np.linalg.norm(Fp[cand][:, None] - Fp[pick][None], axis=2), 1)
            pick.append(int(cand[int(np.argmax(dmin))]))
        log(f"適応サンプリング {r + 1}/{de['adaptive_rounds']}: |μ−T|/σ の小さい {len(pick)} 点")
        run_points(Xp[pick], [Up[i] for i in pick], "追加")

    # ---- 2. surrogate + independent validation
    Ftr = feats(rows_x, rows_u)
    log(f"代理モデルの学習（{len(rows_y)} 点, 入力 {Ftr.shape[1]} 次元, 出力 {len(keys)}）")
    S = Surrogate(Ftr, {k: [y[k] for y in rows_y] for k in keys}, cfg["seed"])
    Xv, Uv = design(de["n_val"])
    yv = [outputs_of(sim(list(x), u)) for x, u in zip(Xv, Uv)]
    Fv = feats(Xv, Uv)
    validation = {}
    for k in keys:
        yk = np.array([y[k] for y in yv])
        validation[k] = {"chosen": S.kind[k], "loo_rmse": S.loo[k], **error_stats(yk, S.predict(Fv, k)),
                         "y_val": yk.tolist(), "yhat_val": S.predict(Fv, k).tolist()}
    yd = np.array([y["damage"] for y in yv]); pdm = S.predict(Fv, "damage")
    validation["target_agreement"] = float(np.mean((yd <= T) == (pdm <= T)))
    log("代理モデルの検証: " + ", ".join(f"{k} R² {validation[k]['r2'] if validation[k]['r2'] is not None else float('nan'):.2f}"
                                          f"/相対RMSE {validation[k]['rel_rmse']:.1%}" for k in ("damage", "pop_hazard", "ecology"))
        + f", 目標判定の一致率 {validation['target_agreement']:.0%}")

    # ---- 3. robust evaluation on a common uncertainty ensemble + NSGA-II
    ne = ex["ensemble"]["n"]
    from scipy.stats import qmc
    Zu = qmc.Sobol(d=nU, scramble=True, seed=cfg["seed"] + 13).random(ne)
    Uens = sp.u_values(Zu)
    af = Uens.get("asset_factor", np.ones(ne)); cf = Uens.get("cost_factor", np.ones(ne))
    zones = list(Dz_ref)

    def predict_many(xs):
        """policies x ensemble -> dict of (n_policy, ne) arrays (damage without the asset factor)."""
        X = np.repeat(np.array(xs, float), ne, axis=0)
        U = {k: np.tile(v, len(xs)) for k, v in Uens.items()}
        F = sp.features(X, U)
        return {k: S.predict(F, k).reshape(len(xs), ne) for k in keys}

    obj_names = ex["optimisation"]["objectives"]

    def objectives(xs):
        Pm = predict_many(xs)
        rel = (Pm["damage"] <= T).mean(1)
        rows = []
        for i, x in enumerate(xs):
            o = {"cost": cost_of(x, tabs, sp)["total"], "reliability": -rel[i],
                 "damage_mean": float((Pm["damage"][i] * af).mean()), "ecology": -float(Pm["ecology"][i].mean()),
                 "livability": -float(Pm["livability"][i].mean()), "pop_mean": float(Pm["pop_hazard"][i].mean())}
            rows.append([o[k] for k in obj_names])
        return rows
    op = ex["optimisation"]
    full = [1.0 if v.continuous else len(v.options) - 1 for v in sp.V]
    log(f"NSGA-II（代理モデル上, {op['pop']} 個体 × {op['gens']} 世代, 各案 {ne} シナリオ）: 目的 {obj_names}")
    archive = nsga_mixed(sp, objectives, op["pop"], op["gens"], rng, op["p_mut"], init=[zero, full], log=log)
    F = np.array([a["f"] for a in archive])
    pm = NS.pareto_mask(F)
    par = [i for i in np.flatnonzero(pm)]
    if 0 not in par:
        par = [0] + par                                   # keep "no measures" as the reference row
    log(f"Pareto 解 {len(par)} 案（評価 {len(archive)} 案）")

    # ---- raw metrics of the Pareto policies
    from .candidates import describe
    xs = [archive[i]["x"] for i in par]
    Pm = predict_many(xs)
    rf = Uens.get("rain_factor", np.ones(ne))
    nb = ex["rain_bins"]
    lo, hi = params["rain_factor"]["low"], params["rain_factor"]["high"]
    edges = np.linspace(lo, hi, nb + 1)
    bins = np.clip(np.searchsorted(edges, rf, side="right") - 1, 0, nb - 1)
    q = lambda a, p: float(np.quantile(a, p))  # noqa: E731
    policies = []
    for k, (i, x) in enumerate(zip(par, xs)):
        c = cost_of(x, tabs, sp)
        d, ratio = Pm["damage"][k], Pm["damage"][k] / D_ref
        policies.append({
            "archive": int(i), "x": x, "desc": describe(sp.V, x) or ["施策なし"],
            "cost": c["total"], "cost_zone": c["by_zone"], "cost_type": c["by_type"],
            "cost_range": [c["total"] * float(cf.min()), c["total"] * float(cf.max())],
            "reliability": float((d <= T).mean()),
            "reliability_zone": {z: float((Pm["dz:" + z][k] <= ex["target_ratio"] * Dz_ref[z]).mean()) for z in zones},
            "reliability_by_rain": [float((d[bins == b] <= T).mean()) if (bins == b).any() else None for b in range(nb)],
            "reliability_curve": [float((ratio <= t).mean()) for t in ex["target_curve"]],
            "damage": {"mean": float((d * af).mean()), "p10": q(d * af, .1), "p50": q(d * af, .5), "p90": q(d * af, .9)},
            "damage_zone": {z: {"mean": float((Pm["dz:" + z][k] * af).mean()), "p90": q(Pm["dz:" + z][k] * af, .9)} for z in zones},
            "pop_hazard": {"mean": float(Pm["pop_hazard"][k].mean()), "p90": q(Pm["pop_hazard"][k], .9)},
            "households": float(Pm["households"][k].mean()),
            "livability": float(Pm["livability"][k].mean()), "ecology": float(Pm["ecology"][k].mean()),
            "ratio_samples": np.round(ratio, 3).tolist()})

    # ---- 4. verification with RRI
    vf = ex["verify"]
    order = sorted(range(1, len(policies)), key=lambda k: policies[k]["cost"])
    sel = [0] + [order[int(round(j))] for j in np.linspace(0, len(order) - 1, max(vf["n_policies"] - 1, 0))] if order else [0]
    sel = list(dict.fromkeys(sel))
    Zs = UQ.lhs(vf["n_scenarios"], nU, rng)
    Us = sp.u_values(Zs)
    us = [{k: float(Us[k][i]) for k in sp.unames} for i in range(vf["n_scenarios"])]
    log(f"RRI による検証: {len(sel)} 案 × {len(us)} シナリオ")
    ver = []
    for k in sel:
        x = policies[k]["x"]
        Fx = feats([x] * len(us), us)
        pred = S.predict(Fx, "damage")
        obs = np.array([sim(x, u)["damage"] for u in us])
        ver.append({"policy": k, "obs": obs.tolist(), "pred": pred.tolist(),
                    "reliability_rri": float((obs <= T).mean()), "reliability_sur": float((pred <= T).mean()),
                    "agreement": float(((obs <= T) == (pred <= T)).mean()), **error_stats(obs, pred)})
    allo = np.concatenate([v["obs"] for v in ver]); allp = np.concatenate([v["pred"] for v in ver])
    verification = {"policies": ver, "scenarios": us, **error_stats(allo, allp),
                    "agreement": float(((allo <= T) == (allp <= T)).mean())}
    log(f"検証: 被害額 R² {verification['r2']:.2f}・相対 RMSE {verification['rel_rmse']:.1%}、目標判定の一致率 {verification['agreement']:.0%}")

    res = {"name": ex["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0,
           "n_simulations": getattr(ev, "n_sim", None), "config": cfg, "map": mapinfo,
           "variables": [{"name": v.name, "type": v.type, "zone": v.zone, "options": v.options, "continuous": v.continuous,
                          "n_cells": len(v.cells)} for v in sp.V],
           "uncertainties": {n: {k: params[n][k] for k in ("low", "high", "nominal", "kind", "status", "label", "basis") if k in params[n]}
                             for n in sp.unames},
           "reference": {"damage": D_ref, "damage_zone": Dz_ref, "target": T, "target_ratio": ex["target_ratio"],
                         "pop_hazard": ref["pop_hazard"], "livability": ref["livability"], "ecology": ref["ecology"]},
           "zones": zones, "target_curve": ex["target_curve"], "rain_edges": edges.tolist(),
           "ensemble": {k: np.round(v, 4).tolist() for k, v in Uens.items()},
           "design": {"n_train": len(rows_y), "x": rows_x, "u": rows_u, "y": rows_y},
           "surrogate": {"features": Ftr.shape[1], "validation": validation},
           "objectives": obj_names, "archive": [{"x": a["x"], "f": a["f"]} for a in archive],
           "pareto": [int(i) for i in par], "policies": policies, "verification": verification}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"完了: {out / 'results.json'}（{time.time() - t0:.0f} s, RRI {res['n_simulations']} 回）")
    return res
