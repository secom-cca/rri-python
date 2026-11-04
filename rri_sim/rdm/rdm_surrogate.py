"""Robust Decision Making on the exploration surrogate (docs/09 9.8): no RRI runs, no budget constraint.

    python scripts/run_rdm.py --surrogate                   # settings: config.py "surrogate_rdm"

The policies are those evaluated by the exploration (explore.py) and its trade spaces by actor
(explore_scope.py) plus a mixed NSGA-II on the stakeholders' mean utilities.  Every policy is
evaluated on a common set of uncertainty scenarios (the exploration's ensemble: rain factor, sea
level, rain intensity gamma, effect coefficients, cost and asset factors) with the surrogate, the
stakeholder utilities are those of robustness.py with the cost normalised by `cost_scale_oku`
(the former budget; nothing is infeasible), and the Regret of a policy in a scenario is measured
against the best policy of the whole pool in that scenario.  Sobol indices of the recommended
policies' damage / Regret are estimated directly on the surrogate (Saltelli), with the best policy
taken from a reduced pool (the best policies of each stakeholder).

Writes data/processed/rdm/<name>/results.json in the format of run.py, so the GUI (/rdm) shows it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config as CF
from .explore import EXPLORE_DIR, nsga_mixed, space_from_results, surrogate_from_results
from .explore_scope import _gp_mean, cost_rows, nondominated
from .evaluate import xkey
from .run import RDM_DIR

KEYS = ("households", "livability", "ecology")


class Model:
    """Surrogate outputs of policies x scenarios, costs by zone, stakeholder utilities."""

    def __init__(self, res, sur, costs, cfg):
        self.res, self.sur, self.cfg = res, sur, cfg
        self.sp = space_from_results(res)
        self.V = res["variables"]
        self.zones = res["zones"]
        self.P = self.sp.P
        self.cvals = costs
        self.vzone = [v["zone"] for v in self.V]

    def cost_zone(self, xs):
        out = np.zeros((len(xs), len(self.zones)))
        for p, x in enumerate(xs):
            for i, v in enumerate(self.V):
                c = float(np.interp(x[i], *self.cvals[i])) if v["continuous"] else float(self.cvals[i][1][int(x[i])])
                out[p, self.zones.index(self.vzone[i])] += c
        return out

    def predict(self, xs, U, chunk=96):
        """xs: n policies; U: {factor: (S,)} -> {key: (n, S)} (damage by zone without the asset factor)."""
        S = len(next(iter(U.values())))
        out = {k: np.empty((len(xs), S), np.float32) for k in KEYS + tuple("dz:" + z for z in self.zones)}
        for s in range(0, len(xs), chunk):
            xb = np.array(xs[s:s + chunk], float)
            X = np.repeat(xb, S, axis=0)
            Ut = {k: np.tile(v, len(xb)) for k, v in U.items()}
            F = self.sp.features(X, Ut)
            for k in out:
                t = _gp_mean(self.sur.models[k], F)
                y = np.maximum(np.expm1(t), 0) if self.sur._is_log(k) else t
                out[k][s:s + len(xb)] = y.reshape(len(xb), S)
        return out

    def utilities(self, xs, U, ref, sts, scale, cz=None):
        """{stakeholder: (n, S)} utilities (robustness.utility, vectorised)."""
        M = self.predict(xs, U)
        cz = self.cost_zone(xs) if cz is None else cz
        a = np.asarray(U.get("asset_factor", 1.0), float)[None]
        c = np.asarray(U.get("cost_factor", 1.0), float)[None]
        out = {}
        for name, st in sts.items():
            w = st["weights"]
            zs = self.zones if st["damage_zones"] == "all" else st["damage_zones"]
            D = sum(M["dz:" + z].astype(float) for z in zs) * a
            Dr = max(sum(ref["damage_zone"][z] for z in zs), 1e-9)
            cs = self.zones if st["cost_zones"] == "all" else (st["cost_zones"] or [])
            C = (cz[:, [self.zones.index(z) for z in cs]].sum(1)[:, None] * c) if cs else 0.0
            out[name] = (-w.get("damage", 0) * D / Dr - w.get("cost", 0) * C / scale
                         - w.get("households", 0) * M["households"] / max(ref["households"], 1e-9)
                         + w.get("livability", 0) * (M["livability"] - ref["livability"]) / 10.0
                         + w.get("ecology", 0) * (M["ecology"] - ref["ecology"]) / 10.0).astype(np.float32)
        return out, M


def _scenarios(sp, res, n):
    E = {k: np.array(v, float)[:n] for k, v in res["ensemble"].items()}
    for k in sp.unames:
        E.setdefault(k, np.full(n, sp.P[k]["nominal"]))
    return E


def run(cfg=None, log=print, out_dir=None, evaluator=None):
    t0 = time.time()
    cfg = CF.load(cfg)
    sc = cfg["surrogate_rdm"]
    name = sc["name"]
    out = Path(out_dir) if out_dir else RDM_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    CF.save(cfg, out / "config.json")
    ex_dir = Path(sc.get("explore_dir") or EXPLORE_DIR / sc["explore_run"])
    res = json.loads((ex_dir / "results.json").read_text(encoding="utf-8"))
    sur = surrogate_from_results(res, KEYS + ("damage", "pop_hazard") + tuple("dz:" + z for z in res["zones"]),
                                 [ex_dir, EXPLORE_DIR.parent / "roadmap" / "_physics" / ex_dir.name], log)
    mdl = Model(res, sur, cost_rows(res), cfg)
    sp, V, zones = mdl.sp, mdl.V, mdl.zones
    nV = len(V)
    sts = cfg["stakeholders"]
    scale = sc["cost_scale_oku"]
    zero = [0] * nV
    nom = {k: np.array([sp.P[k]["nominal"]]) for k in sp.unames}
    M0 = mdl.predict([zero], nom)
    ref = {"damage_zone": {z: float(M0["dz:" + z][0, 0]) for z in zones}, "households": float(M0["households"][0, 0]),
           "livability": float(M0["livability"][0, 0]), "ecology": float(M0["ecology"][0, 0])}
    E = _scenarios(sp, res, sc["n_scenarios"])
    S = len(E["rain_factor"])
    log(f"代理モデル RDM: 施策 {nV} 変数・共通シナリオ {S}（{ex_dir.name} の不確実性 {len(sp.unames)} 個）・予算制約なし")

    # ---- 1. policy pool: exploration archive + actor trade spaces + NSGA-II on the stakeholders' mean utilities
    pool = {}
    for a in res["archive"]:
        pool.setdefault(tuple(xkey(a["x"])), "explore")
    scp = ex_dir / "scopes.json"
    if scp.exists():
        for x in json.loads(scp.read_text(encoding="utf-8"))["points"]["x"]:
            pool.setdefault(tuple(xkey(x)), "scopes")
    pool.setdefault(tuple(zero), "none")
    op = sc["optimisation"]
    Eo = {k: v[:op["n_scenarios"]] for k, v in E.items()}

    def objectives(xs):
        U, _ = mdl.utilities(xs, Eo, ref, sts, scale)
        return np.column_stack([-U[n].mean(1) for n in sts]).tolist()
    rng = np.random.default_rng(cfg["seed"] + 301)
    full = [1.0 if v.continuous else len(v.options) - 1 for v in sp.V]
    arch = nsga_mixed(sp, objectives, op["pop"], op["gens"], rng, op["p_mut"], init=[zero, full], log=log)
    for a in arch:
        pool.setdefault(tuple(xkey(a["x"])), "nsga")
    X = [list(x) for x in pool]
    src = list(pool.values())
    log(f"評価する案 {len(X)}（探索 {src.count('explore')}・主体別 {src.count('scopes')}・効用の NSGA-II {src.count('nsga')}）, {time.time() - t0:.0f} s")

    # ---- 2. all policies x common scenarios: utilities, regret against the best of the pool
    cz = mdl.cost_zone(X)
    U, M = {n: np.empty((len(X), S), np.float32) for n in sts}, {}
    for s in range(0, len(X), 1024):
        u, m = mdl.utilities(X[s:s + 1024], E, ref, sts, scale, cz[s:s + 1024])
        for n in sts:
            U[n][s:s + 1024] = u[n]
        dmg = sum(m["dz:" + z] for z in zones) * E["asset_factor"][None]
        for k, v in (("damage_mean", dmg.mean(1)), ("damage_p90", np.quantile(dmg, 0.9, axis=1))):
            M.setdefault(k, []).append(v)
        log(f"  {min(s + 1024, len(X))}/{len(X)} 案（{time.time() - t0:.0f} s）")
    M = {k: np.concatenate(v) for k, v in M.items()}
    R, summ = {}, {}
    for n, st in sts.items():
        best = U[n].max(0)
        R[n] = best[None] - U[n]
        summ[n] = {"max": R[n].max(1), "mean": R[n].mean(1), "p90": np.quantile(R[n], 0.9, axis=1),
                   "satisficing": (R[n] <= st["regret_threshold"]).mean(1)}
    rel = np.max([summ[n]["max"] / max(st["regret_threshold"], 1e-9) for n, st in sts.items()], axis=0)
    i_none = X.index(zero)
    rec = {n: int(np.argmin(summ[n]["max"])) for n in sts}
    ov = int(np.argmin(rel))
    log("Minimax Regret: " + "、".join(f"{n} 最大 Regret {summ[n]['max'][rec[n]]:.3f}" for n in sts)
        + f"／総合 最大相対 Regret {rel[ov]:.2f}")

    # ---- 3. nominal metrics, Pareto sets (nominal scenario), policies shown
    Mn = mdl.predict(X, nom)
    nm = {"damage_zone": np.column_stack([Mn["dz:" + z][:, 0] for z in zones]).astype(float),
          "households": Mn["households"][:, 0].astype(float), "livability": Mn["livability"][:, 0].astype(float),
          "ecology": Mn["ecology"][:, 0].astype(float), "cost": cz.sum(1)}
    nm["damage"] = nm["damage_zone"].sum(1)
    G = np.column_stack([nm["damage"], nm["cost"], -nm["livability"], -nm["ecology"]])
    par = np.flatnonzero(nondominated(G))

    def st_cols(st):
        zs = zones if st["damage_zones"] == "all" else st["damage_zones"]
        cs = zones if st["cost_zones"] == "all" else (st["cost_zones"] or [])
        cols = {"damage": nm["damage_zone"][:, [zones.index(z) for z in zs]].sum(1),
                "cost": cz[:, [zones.index(z) for z in cs]].sum(1) if cs else np.zeros(len(X)),
                "livability": -nm["livability"], "ecology": -nm["ecology"], "households": nm["households"]}
        return np.column_stack([cols[k] for k in cols if st["weights"].get(k, 0) > 0])
    st_par = {n: np.flatnonzero(nondominated(st_cols(st))) for n, st in sts.items()}
    order = np.argsort(rel)
    disp = list(dict.fromkeys([i_none, ov, *rec.values()]))
    for i in order:
        if len(disp) >= sc["n_display"]:
            break
        if int(i) not in disp:
            disp.append(int(i))
    keep = list(dict.fromkeys(disp + par.tolist() + [int(i) for n in sts for i in st_par[n]]))
    if len(keep) < sc["archive_display"]:
        rest = np.setdiff1d(np.arange(len(X)), keep)
        keep += rng.choice(rest, min(len(rest), sc["archive_display"] - len(keep)), replace=False).tolist()
    keep = keep[:max(sc["archive_display"], len(disp))]
    pos = {i: k for k, i in enumerate(keep)}

    # ---- 4. Sobol indices on the surrogate (damage of the overall recommendation, Regret of each stakeholder's)
    so = sc["sobol"]
    red = list(dict.fromkeys(disp + [int(i) for n in sts for i in np.argsort(-U[n].mean(1))[:so["pool_per_stakeholder"]]]))
    cover = {n: float(np.mean(U[n][red].max(0) >= U[n].max(0) - 1e-6)) for n in sts}
    names = sp.unames
    k = len(names)
    rs = np.random.default_rng(cfg["seed"] + 7)
    A, B = rs.random((so["n_base"], k)), rs.random((so["n_base"], k))
    mats = [A, B] + [np.where(np.arange(k)[None] == i, B, A) for i in range(k)]
    Z = np.concatenate(mats)
    Uz = {nm_: sp.P[nm_]["low"] + Z[:, j] * (sp.P[nm_]["high"] - sp.P[nm_]["low"]) for j, nm_ in enumerate(names)}
    Xr = [X[i] for i in red]
    ur, mr = mdl.utilities(Xr, Uz, ref, sts, scale, cz[red])
    targets = {"被害額（総合推奨案）": sum(mr["dz:" + z][red.index(ov)] for z in zones) * Uz["asset_factor"]}
    for n in sts:
        targets[f"Regret・{n}（{n}の推奨案）"] = ur[n].max(0) - ur[n][red.index(rec[n])]
    sob = {}
    nb = so["n_base"]
    for lab, y in targets.items():
        y = np.asarray(y, float)
        fA, fB = y[:nb], y[nb:2 * nb]
        var = np.var(np.r_[fA, fB])
        sob[lab] = {}
        for i, f in enumerate(names):
            fABi = y[(2 + i) * nb:(3 + i) * nb]
            sob[lab][f] = {"S1": float(np.mean(fB * (fABi - fA)) / var) if var > 0 else 0.0,
                           "ST": float(0.5 * np.mean((fA - fABi) ** 2) / var) if var > 0 else 0.0}
    log(f"Sobol（代理モデル上, N={nb}, 縮小した候補 {len(red)} 案; 最良案の一致率 "
        + "、".join(f"{n} {cover[n]:.0%}" for n in sts) + "）")

    # ---- 5. results in the format of run.py (+ surrogate information)
    mapinfo, cells = _map(cfg, res, out, evaluator, log)
    bins = sc["climate_bins"]
    cl = [next(b[1] for b in bins if r < b[0]) for r in E["rain_factor"]]
    sims = [{"climate": cl[s], **{f: round(float(E[f][s]), 4) for f in names}} for s in range(S)]
    lab = {i: ("施策なし" if i == i_none else f"案{j}") for j, i in enumerate(disp)}
    zd = lambda row: {z: float(v) for z, v in zip(zones, row)}  # noqa: E731
    archive = [{"x": X[i], "gen": 0, "feasible": True, "source": src[i], "cost": float(nm["cost"][i]), "cost_zone": zd(cz[i]),
                "cost_by_type": {}, "damage": float(nm["damage"][i]), "damage_zone": zd(nm["damage_zone"][i]),
                "households": float(nm["households"][i]), "livability": float(nm["livability"][i]), "ecology": float(nm["ecology"][i]),
                "damage_mean": float(M["damage_mean"][i]), "damage_p90": float(M["damage_p90"][i])} for i in keep]
    from .candidates import describe
    out_res = {
        "name": name, "mode": "surrogate", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0,
        "n_simulations": 0, "config": {**cfg, "budget_oku": None}, "map": mapinfo,
        "variables": [{"name": v["name"], "type": v["type"], "zone": v["zone"], "options": v["options"], "continuous": v["continuous"],
                       "cells": cells[i] if cells else []} for i, v in enumerate(V)],
        "archive": archive, "pareto": [pos[i] for i in par if i in pos],
        "stakeholder_pareto": {n: [pos[i] for i in st_par[n] if i in pos] for n in sts},
        "parallel": [pos[i] for i in disp],
        "policies": [{"archive": pos[i], "x": X[i], "label": lab[i], "desc": describe(sp.V, X[i]) or ["施策なし"]} for i in disp],
        "sims": sims, "posts": [{}], "scenarios": [{"sim": s, "post": 0} for s in range(S)],
        "matrix": {}, "robustness": {
            "feasible": [[True] * S for _ in disp],
            "stakeholders": {n: {"regret": np.round(R[n][disp], 4).tolist(), "utility": np.round(U[n][disp], 4).tolist(),
                                 "summary": {k: [float(v) for v in summ[n][k][disp]] for k in summ[n]} | {"feasible": [1.0] * len(disp)},
                                 "minimax": disp.index(rec[n])} for n in sts},
            "overall": {"minimax": disp.index(ov), "max_relative_regret": [float(rel[i]) for i in disp]}},
        "sobol": sob, "sobol_factors": {f: sp.P[f].get("label", f) for f in names},
        "surrogate": {"explore_run": ex_dir.name, "n_policies": len(X), "n_scenarios": S, "cost_scale_oku": scale,
                      "sources": {k: src.count(k) for k in ("explore", "scopes", "nsga", "none")},
                      "validation": {k: {kk: v.get(kk) for kk in ("chosen", "r2", "rel_rmse")} for k, v in res["surrogate"]["validation"].items()
                                     if isinstance(v, dict)},
                      "verification": {k: res["verification"].get(k) for k in ("r2", "rel_rmse", "agreement")},
                      "sobol_pool": len(red), "sobol_best_coverage": cover, "climate_bins": bins,
                      "uncertainties": {f: {kk: sp.P[f].get(kk) for kk in ("low", "high", "nominal", "kind", "label")} for f in names}},
    }
    (out / "results.json").write_text(json.dumps(out_res, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"完了: {out / 'results.json'}（{time.time() - t0:.0f} s, RRI なし）")
    return out_res


def _map(cfg, res, out, evaluator, log):
    """Map of the basin and the candidate cells of each variable (cached; one baseline run of RRI if missing)."""
    cp = out / "candidates.json"
    if cp.exists():
        d = json.loads(cp.read_text(encoding="utf-8"))
        return res.get("map"), d["cells"]
    if evaluator is None:
        try:
            from .run import setup_rri
            ex = res["config"]
            evaluator, _ = setup_rri(ex, None, log)
        except Exception as e:                 # the map is optional
            log(f"配置図の候補セルを作れなかった（{e}）")
            return res.get("map"), None
    cells = {v.name: list(map(int, v.cells)) for v in evaluator.variables}
    cl = [cells.get(v["name"], []) for v in res["variables"]]
    cp.write_text(json.dumps({"cells": cl}), encoding="utf-8")
    return res.get("map"), cl
