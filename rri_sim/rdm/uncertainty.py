"""Scenario design, post-hoc factors, surrogate model and Sobol indices.

Common scenario set (all policies are compared on the same scenarios):
    simulated scenarios  = climates x n_per_climate; the first of each climate
                           at nominal factors, the rest Latin hypercube samples
                           of the simulated factors (rain_gamma, *_effect)
    full scenarios       = every simulated scenario x n_post LHS samples of the
                           post-hoc factors (cost_factor, asset_factor)

Sobol indices (Saltelli 2010 for S1, Jansen for ST) are computed on a cheap
surrogate: per policy and output, ridge regression on the simulated factors
(climate coded 0 / 0.5 / 1 = current / 2C / 4C) fitted to the simulated
scenarios, with the post-hoc factors applied exactly.  With few simulated
scenarios the simulated factors other than climate are only weakly
identified (shrunk towards no effect); the run records the design so that
this is visible.
"""
from __future__ import annotations

import numpy as np

CLIMATE_CODE = {"current": 0.0, "2C": 0.5, "4C": 1.0}


def lhs(n, k, rng):
    """n x k Latin hypercube sample in [0, 1)."""
    u = (np.argsort(rng.random((k, n)), axis=1).T + rng.random((n, k))) / n
    return u


def design(cfg, rng):
    U = cfg["uncertainties"]
    simf = [k for k, u in U.items() if u["kind"] == "sim"]
    postf = [k for k, u in U.items() if u["kind"] == "post"]
    sc = cfg["scenarios"]
    sims = []
    for clim in sc["climates"]:
        sims.append({"climate": clim, **{k: U[k]["nominal"] for k in simf}})
        n = sc["n_per_climate"] - 1
        if n > 0:
            u = lhs(n, len(simf), rng)
            for row in u:
                sims.append({"climate": clim, **{k: round(U[k]["low"] + r * (U[k]["high"] - U[k]["low"]), 4) for k, r in zip(simf, row)}})
    u = lhs(sc["n_post"], len(postf), rng)
    posts = [{k: round(U[k]["low"] + r * (U[k]["high"] - U[k]["low"]), 4) for k, r in zip(postf, row)} for row in u]
    if posts:
        posts[0] = {k: U[k]["nominal"] for k in postf}          # include the nominal post factors
    full = [{"sim": i, "post": j} for i in range(len(sims)) for j in range(len(posts))]
    return sims, posts, full


def apply_post(m, cost, post):
    """Metrics of a simulated result with post-hoc factors applied."""
    a, c = post.get("asset_factor", 1.0), post.get("cost_factor", 1.0)
    return {"damage": m["damage"] * a, "damage_zone": {z: v * a for z, v in m["damage_zone"].items()},
            "households": m["households"], "livability": m["livability"], "ecology": m["ecology"],
            "cost": cost["total"] * c, "cost_zone": {z: v * c for z, v in cost["by_zone"].items()},
            "peak_q": m.get("peak_q"), "inundated_km2": m.get("inundated_km2")}


def sim_features(sim, cfg):
    U = cfg["uncertainties"]
    f = [CLIMATE_CODE.get(sim["climate"], 0.0)]
    for k, u in U.items():
        if u["kind"] == "sim":
            span = (u["high"] - u["low"]) or 1.0
            f.append((sim[k] - u["low"]) / span)
    return np.array(f)


class Ridge:
    """y ~ a + b . x (x in [0,1]^k), ridge penalty on b; exact when well determined."""

    def __init__(self, X, y, lam=1e-3):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mx = X.mean(0); self.my = y.mean()
        Xc = X - self.mx
        k = X.shape[1]
        self.b = np.linalg.solve(Xc.T @ Xc + lam * len(y) * np.eye(k), Xc.T @ (y - self.my)) if len(y) > 1 else np.zeros(k)

    def __call__(self, X):
        return self.my + (np.asarray(X, float) - self.mx) @ self.b


def saltelli(func, k, n, rng):
    """First-order (Saltelli 2010) and total (Jansen 1999) Sobol indices of
    func: (N, k) in [0,1]^k -> (N,) ; returns (S1, ST) arrays of length k."""
    A = rng.random((n, k)); B = rng.random((n, k))
    fA, fB = func(A), func(B)
    var = np.var(np.r_[fA, fB])
    S1, ST = np.zeros(k), np.zeros(k)
    if var <= 0:
        return S1, ST
    for i in range(k):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        fABi = func(ABi)
        S1[i] = np.mean(fB * (fABi - fA)) / var
        ST[i] = 0.5 * np.mean((fA - fABi) ** 2) / var
    return S1, ST
