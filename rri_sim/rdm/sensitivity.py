"""Sensitivity analysis on top of an RDM run (docs/10_sensitivity.md).

Stages (all settings in cfg["sensitivity"]; parameter ranges carry status 根拠/仮定):

1. Morris screening (RRI runs): r one-at-a-time trajectories on a p-level grid
   over all parameters; outputs of the overall recommended policy (damage,
   浸水人口, Livability, Ecology, damage reduction vs no measures).  mu*, sigma
   normalised by the output's standard deviation.  Post-hoc parameters
   (cost_factor, asset_factor) need no extra run.
2. Training / validation design (RRI runs): LHS over the screened simulated
   parameters (the others at nominal) for every analysed policy.
3. Surrogate: Gaussian process (ARD squared-exponential, hyper-parameters by
   maximum marginal likelihood) per policy and output; prediction error on the
   independent validation LHS is reported (and compared with linear ridge).
4. Sobol indices on the surrogate: A and B from one scrambled Sobol sequence of
   dimension 2k (independent inputs), S1 (Saltelli 2010), ST (Jansen 1999),
   bootstrap confidence intervals; outputs incl. stakeholder regret.
5. Decision sensitivity (surrogate samples): best policy per scenario and
   stakeholder, selection frequency, rank reversal, regret variation; Pareto
   membership frequency over the RDM stress-test scenarios (real runs).
6. Preference sensitivity (RDM stress-test matrix, real runs): random weights
   (Dirichlet) and one-weight sweeps -> minimax-regret policy; cross-regret
   between stakeholders (conflict) and policies acceptable to all (common benefit).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import candidates as CA
from . import config as CF
from . import nsga2
from . import robustness as RB
from . import uncertainty as UQ

OUTPUTS = {"damage": "被害額 [億円]", "pop_hazard": "浸水人口（0.5 m以上）[人]", "livability": "Livability",
           "ecology": "Ecology", "reduction": "被害軽減額 [億円]（対策なし比）"}


# ------------------------------------------------------------------ parameters
def params(cfg):
    P = cfg["sensitivity"]["parameters"]
    names = list(P)
    sim = [n for n in names if P[n]["kind"] == "sim"]
    post = [n for n in names if P[n]["kind"] == "post"]
    return P, names, sim, post


def from_unit(u, names, P):
    return {n: float(P[n]["low"] + ui * (P[n]["high"] - P[n]["low"])) for n, ui in zip(names, u)}


def nominal_unit(names, P):
    return np.array([(P[n]["nominal"] - P[n]["low"]) / ((P[n]["high"] - P[n]["low"]) or 1.0) for n in names])


def sim_dict(values, sim_names, P):
    """Simulated-scenario dict for the evaluator (rounded for cache keys)."""
    return {n: round(float(values.get(n, P[n]["nominal"])), 5) for n in sim_names}


# ------------------------------------------------------------------ Morris
def morris_trajectories(k, r, levels, rng):
    """r trajectories of k+1 points on the grid {0, 1/(p-1), ..., 1}; step
    Delta = p / (2 (p-1)).  Returns list of (points (k+1, k), order, signs)."""
    delta = levels / (2.0 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    out = []
    for _ in range(r):
        x = rng.choice(grid[grid <= 1 - delta + 1e-9], size=k)
        order = rng.permutation(k)
        pts, signs = [x.copy()], []
        for i in order:
            x = x.copy()
            if x[i] + delta <= 1 + 1e-9:
                x[i] += delta; signs.append(1.0)
            else:
                x[i] -= delta; signs.append(-1.0)
            pts.append(x)
        out.append((np.array(pts), order, np.array(signs)))
    return out, delta


def morris_indices(trajs, delta, Y):
    """Y: list of arrays (k+1,) per trajectory.  Returns mu, mu_star, sigma (k,)."""
    k = trajs[0][0].shape[1]
    EE = [[] for _ in range(k)]
    for (pts, order, signs), y in zip(trajs, Y):
        for step, i in enumerate(order):
            EE[i].append((y[step + 1] - y[step]) / (signs[step] * delta))
    EE = [np.array(e) for e in EE]
    return (np.array([e.mean() for e in EE]), np.array([np.abs(e).mean() for e in EE]),
            np.array([e.std(ddof=1) if len(e) > 1 else 0.0 for e in EE]))


# ------------------------------------------------------------------ surrogate
class GP:
    """Gaussian process regression, ARD squared-exponential kernel + noise,
    hyper-parameters by maximum log marginal likelihood (L-BFGS, restarts)."""

    def __init__(self, X, y, restarts=4, seed=0):
        from scipy.optimize import minimize
        self.X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu, self.sd = y.mean(), (y.std() or 1.0)
        self.y = (y - self.mu) / self.sd
        n, k = self.X.shape
        rng = np.random.default_rng(seed)

        def nll(th):
            K = self._K(self.X, self.X, th) + (np.exp(2 * th[-1]) + 1e-8) * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            return 0.5 * self.y @ a + np.log(np.diag(L)).sum()
        bounds = [(np.log(0.05), np.log(20.0))] * k + [(np.log(0.1), np.log(10.0)), (np.log(1e-4), np.log(1.0))]
        best = None
        for t in range(restarts):
            th0 = np.r_[np.log(rng.uniform(0.2, 2.0, k)), 0.0, np.log(0.05)] if t else np.r_[np.zeros(k), 0.0, np.log(0.05)]
            r = minimize(nll, th0, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        self.th = best.x
        K = self._K(self.X, self.X, self.th) + (np.exp(2 * self.th[-1]) + 1e-8) * np.eye(n)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    @staticmethod
    def _K(A, B, th):
        k = A.shape[1]
        ls, sf = np.exp(th[:k]), np.exp(th[k])
        d = (A[:, None, :] - B[None, :, :]) / ls
        return sf ** 2 * np.exp(-0.5 * (d ** 2).sum(-1))

    def __call__(self, Xn):
        Xn = np.atleast_2d(np.asarray(Xn, float))
        out = np.empty(len(Xn))
        for s in range(0, len(Xn), 4096):            # chunks keep the kernel matrix small
            out[s:s + 4096] = self._K(Xn[s:s + 4096], self.X, self.th) @ self.alpha
        return self.mu + self.sd * out

    def std(self, Xn):
        """Predictive standard deviation of the latent function."""
        Xn = np.atleast_2d(np.asarray(Xn, float))
        sf2 = np.exp(2 * self.th[self.X.shape[1]])
        out = np.empty(len(Xn))
        for s in range(0, len(Xn), 4096):
            v = np.linalg.solve(self.L, self._K(self.X, Xn[s:s + 4096], self.th))
            out[s:s + 4096] = np.sqrt(np.maximum(sf2 - (v ** 2).sum(0), 1e-12))
        return self.sd * out

    def loo_residuals(self):
        """Closed-form leave-one-out residuals (hyper-parameters fixed; Rasmussen & Williams 5.4.2)."""
        Ki = np.linalg.solve(self.L.T, np.linalg.solve(self.L, np.eye(len(self.y))))
        return self.sd * self.alpha / np.diag(Ki)

    @property
    def lengthscales(self):
        return np.exp(self.th[:self.X.shape[1]]).tolist()


class Linear:
    """Ridge-linear surrogate with the GP interface."""
    def __init__(self, X, y):
        self.m = UQ.Ridge(X, y)

    def __call__(self, Xn):
        return self.m(np.atleast_2d(Xn))

    lengthscales = None


def loo_rmse(make, X, y):
    """Leave-one-out RMSE of a surrogate constructor on the training data (model selection
    without touching the validation set)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    err = []
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        err.append(float(make(X[k], y[k])(X[i:i + 1])[0]) - y[i])
    return float(np.sqrt(np.mean(np.square(err))))


def fit_surrogate(X, y, seed=0):
    """GP or linear, whichever has the smaller leave-one-out error on the training points."""
    mk = {"gp": lambda A, b: GP(A, b, restarts=2, seed=seed), "linear": Linear}
    if np.std(y) == 0:
        return Linear(X, y), "linear", {"gp": 0.0, "linear": 0.0}
    loo = {k: loo_rmse(f, X, y) for k, f in mk.items()}
    kind = min(loo, key=loo.get)
    return (GP(X, y, seed=seed) if kind == "gp" else Linear(X, y)), kind, loo


def error_stats(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = ((y - y.mean()) ** 2).sum()
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"r2": float(1 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else None, "rmse": rmse,
            "rel_rmse": rmse / max(abs(y.mean()), 1e-9), "max_abs": float(np.max(np.abs(y - yhat)))}


# ------------------------------------------------------------------ Sobol
def sobol_indices(func, k, log2_n, n_boot, seed):
    """S1 / ST with bootstrap 95 % intervals.  A, B = two halves of one scrambled
    Sobol sequence in 2k dimensions (independent uniform inputs)."""
    from scipy.stats import qmc
    U = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random_base2(m=log2_n)
    A, B = U[:, :k], U[:, k:]
    fA, fB = func(A), func(B)
    fAB = []
    for i in range(k):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        fAB.append(func(ABi))
    n = len(fA)

    def est(idx):
        a, b = fA[idx], fB[idx]
        var = np.var(np.r_[a, b])
        if var <= 0:
            return np.zeros(k), np.zeros(k)
        s1 = np.array([np.mean(b * (f[idx] - a)) / var for f in fAB])
        st = np.array([0.5 * np.mean((a - f[idx]) ** 2) / var for f in fAB])
        return s1, st
    S1, ST = est(np.arange(n))
    rng = np.random.default_rng(seed + 1)
    boots = [est(rng.integers(0, n, n)) for _ in range(n_boot)]
    b1 = np.array([b[0] for b in boots]); bt = np.array([b[1] for b in boots])
    return {"S1": S1.tolist(), "ST": ST.tolist(),
            "S1_ci": np.quantile(b1, [0.025, 0.975], axis=0).T.tolist(), "ST_ci": np.quantile(bt, [0.025, 0.975], axis=0).T.tolist(),
            "var": float(np.var(np.r_[fA, fB])), "n": int(n)}


# ------------------------------------------------------------------ preference sensitivity
KEYS = ["damage", "cost", "households", "livability", "ecology"]


def terms(matrix, st, ref, budget):
    """Utility = sum_k w_k T_k (robustness.utility is linear in the weights): T_k (n_policy, n_scenario)."""
    n_p, n_s = len(matrix), len(matrix[0])
    T = {}
    for k in KEYS:
        st1 = dict(st, weights={k: 1.0})
        T[k] = np.array([[RB.utility(matrix[p][s], st1, ref, budget) for s in range(n_s)] for p in range(n_p)])
    feas = np.array([[matrix[p][s]["cost"] <= budget * (1 + 1e-9) for s in range(n_s)] for p in range(n_p)])
    return T, feas


def recommend_w(T, feas, w, eligible):
    U = sum(wk * T[k] for k, wk in w.items())
    R = RB.regret(U, feas)
    return RB.minimax(R, eligible), R


def preference(matrix, stakeholders, ref, budget, eligible, n_samples, steps, rng):
    out = {"keys": KEYS, "stakeholders": {}}
    names = list(stakeholders)
    TT = {n: terms(matrix, stakeholders[n], ref, budget) for n in names}
    base_rec, base_R = {}, {}
    for name in names:
        st = stakeholders[name]
        T, feas = TT[name]
        act = [k for k in KEYS if st["weights"].get(k, 0) > 0]
        base_rec[name], base_R[name] = recommend_w(T, feas, {k: st["weights"][k] for k in act}, eligible)
        # random weights over the stakeholder's own metrics (Dirichlet(1) = uniform on the simplex)
        W = rng.dirichlet(np.ones(len(act)), n_samples)
        freq = np.zeros(len(matrix))
        for w in W:
            freq[recommend_w(T, feas, dict(zip(act, w)), eligible)[0]] += 1
        # one-weight sweeps: w_k from 0 to 1, the others keep their base proportions
        sweeps = {}
        for k in act:
            others = {o: st["weights"][o] for o in act if o != k}
            tot = sum(others.values())
            seq = []
            for wk in np.linspace(0, 1, steps):
                w = {k: wk, **{o: (1 - wk) * (v / tot if tot else 0.0) for o, v in others.items()}}
                seq.append(recommend_w(T, feas, w, eligible)[0])
            sweeps[k] = seq
        out["stakeholders"][name] = {"active": act, "base_weights": {k: st["weights"][k] for k in act},
                                     "base_recommendation": base_rec[name],
                                     "frequency": (freq / n_samples).tolist(), "sweeps": sweeps,
                                     "sweep_values": np.linspace(0, 1, steps).tolist()}
    # conflict: max regret of stakeholder j (rows) under stakeholder k's recommendation (columns)
    cross = [[float(base_R[j][base_rec[k]].max()) for k in names] for j in names]
    # common benefit over joint weight draws (each stakeholder's weights drawn independently):
    # acceptable to all = max regret <= own threshold for everybody; top3 of all = in everybody's 3 best
    acc = np.zeros(len(matrix)); top3 = np.zeros(len(matrix)); agree = 0
    for _ in range(n_samples):
        ok = np.ones(len(matrix), bool); tp = np.ones(len(matrix), bool); rs = set()
        for name in names:
            T, feas = TT[name]
            act = out["stakeholders"][name]["active"]
            p, R = recommend_w(T, feas, dict(zip(act, rng.dirichlet(np.ones(len(act))))), eligible)
            mx = np.where(eligible, R.max(1), np.inf)
            ok &= mx <= stakeholders[name]["regret_threshold"]
            tp &= np.isin(np.arange(len(matrix)), np.argsort(mx)[:3])
            rs.add(p)
        acc += ok; top3 += tp; agree += len(rs) == 1
    out["conflict"] = {"names": names, "cross_max_regret": cross, "base_recommendation": base_rec}
    out["common"] = {"acceptable_to_all": (acc / n_samples).tolist(), "top3_of_all": (top3 / n_samples).tolist(),
                     "agreement_rate": agree / n_samples, "n_joint": n_samples}
    return out


def pareto_change(R):
    """Pareto membership of the RDM policies in every stress-test scenario (real runs)."""
    M = R["matrix"]
    n_p, n_s = len(M["damage"]), len(M["damage"][0])
    mem = np.zeros((n_p, n_s), bool)
    for s in range(n_s):
        F = np.array([[M["damage"][p][s], M["cost"][p][s], -M["livability"][p][s], -M["ecology"][p][s]] for p in range(n_p)])
        mem[:, s] = nsga2.pareto_mask(F)
    clim = [R["sims"][sc["sim"]].get("climate", "") for sc in R["scenarios"]]
    by_clim, fronts = {}, {}
    for c in dict.fromkeys(clim):
        cols = [i for i, x in enumerate(clim) if x == c]
        by_clim[c] = mem[:, cols].mean(1).tolist()
        i = cols[0]                                   # first scenario of the climate = nominal post factors
        fronts[c] = {"points": [[M["cost"][p][i], M["damage"][p][i]] for p in range(n_p)], "pareto": mem[:, i].astype(int).tolist()}
    return {"membership": mem.mean(1).tolist(), "by_climate": by_clim, "fronts_first_post": fronts,
            "membership_by_scenario": mem.astype(int).tolist()}


# ------------------------------------------------------------------ pipeline
def run(rdm_dir, overrides=None, evaluator=None, log=print):
    t0 = time.time()
    rdm_dir = Path(rdm_dir)
    R = json.loads((rdm_dir / "results.json").read_text(encoding="utf-8"))
    cfg = CF._merge(CF.load(R["config"]), overrides or {})
    sa = cfg["sensitivity"]
    CF.save(cfg, rdm_dir / "config_sensitivity.json")
    if evaluator is None:
        from .run import setup_rri
        evaluator, _ = setup_rri(cfg, rdm_dir / "evaluations.jsonl", log)
        saved = json.loads((rdm_dir / "candidates.json").read_text(encoding="utf-8"))
        if [v["cells"] for v in saved] != [v.cells for v in evaluator.variables]:
            raise RuntimeError("候補施策が RDM 実行時と一致しません（モデル・設定が変わった可能性）")
    ev = evaluator
    P, names, simn, postn = params(cfg)
    budget = cfg["budget_oku"]
    rob = R["robustness"]
    pol = R["policies"]
    if sa["policies"] == "recommended":
        idx = list(dict.fromkeys([0, rob["overall"]["minimax"]] + [s["minimax"] for s in rob["stakeholders"].values()]))
    elif sa["policies"] == "all":
        idx = list(range(len(pol)))
    else:
        idx = list(sa["policies"])
    X = [pol[i]["x"] for i in idx]
    labels = [pol[i]["label"] for i in idx]
    rec = idx.index(rob["overall"]["minimax"])
    costs = [ev.cost(x) for x in X]
    nom = {n: P[n]["nominal"] for n in names}
    rng = np.random.default_rng(cfg["seed"] + 101)

    def metrics(p, values):
        m = ev.simulate(X[p], sim_dict(values, simn, P))
        return UQ.apply_post(m, costs[p], {k: values[k] for k in postn}) | {"pop_hazard": m["pop_hazard"]}

    # ---- 1. Morris screening (recommended policy and no measures)
    mo = sa["morris"]
    trajs, delta = morris_trajectories(len(names), mo["r"], mo["levels"], rng)
    log(f"Morris: {len(names)} パラメータ × {mo['r']} 軌道（{len(trajs) * (len(names) + 1)} 点、"
        f"RRI は計算パラメータが変わる点のみ）")
    Ym = {o: [] for o in OUTPUTS}
    for pts, order, signs in trajs:
        ys = {o: [] for o in OUTPUTS}
        for u in pts:
            v = from_unit(u, names, P)
            m1, m0 = metrics(rec, v), metrics(0, v)
            for o in ("damage", "pop_hazard", "livability", "ecology"):
                ys[o].append(m1[o])
            ys["reduction"].append(m0["damage"] - m1["damage"])
        for o in OUTPUTS:
            Ym[o].append(np.array(ys[o]))
    morris = {}
    for o in OUTPUTS:
        mu, mus, sg = morris_indices(trajs, delta, Ym[o])
        sd = np.concatenate(Ym[o]).std() or 1.0
        morris[o] = {"mu": mu.tolist(), "mu_star": mus.tolist(), "sigma": sg.tolist(),
                     "rel_mu_star": (mus / sd).tolist(), "rel_sigma": (sg / sd).tolist()}
    score = {n: max(morris[o]["rel_mu_star"][i] for o in OUTPUTS) for i, n in enumerate(names)}
    ranked = sorted(simn, key=lambda n: -score[n])
    sc = sa["screening"]
    sel = [n for n in ranked if score[n] >= sc["min_rel_mu_star"]][:sc["top_k"]] or ranked[:1]
    log("Morris スクリーニング（相対 μ* の最大）: " + ", ".join(f"{n} {score[n]:.2f}" for n in ranked) + f" → 選択 {sel}")

    # ---- 2. training / validation designs over the screened simulated parameters
    su = sa["surrogate"]
    dims = sel + postn                            # Sobol / decision inputs
    Ltr = UQ.lhs(su["n_train"], len(sel), rng); Lva = UQ.lhs(su["n_val"], len(sel), rng)
    if su.get("anchor_nominal", True):
        Ltr = np.vstack([nominal_unit(sel, P)[None], Ltr])

    def run_design(L, tag):
        rows = []
        for j, u in enumerate(L):
            v = dict(nom); v.update(from_unit(u, sel, P))
            res = []
            for p in range(len(X)):
                m = ev.simulate(X[p], sim_dict(v, simn, P))
                res.append({k: m[k] for k in ("damage", "pop_hazard", "livability", "ecology", "households")}
                           | {"damage_zone": m["damage_zone"]})
            rows.append(res)
            log(f"  {tag} {j + 1}/{len(L)}: {', '.join(f'{n}={v[n]:.3f}' for n in sel)}")
        return rows
    log(f"代理モデルの学習 {len(Ltr)} 点・検証 {su['n_val']} 点 × {len(X)} 案（{labels}）")
    tr, va = run_design(Ltr, "学習"), run_design(Lva, "検証")

    # ---- 3. surrogates + validation
    zones = list(tr[0][0]["damage_zone"])
    keys = ["damage", "pop_hazard", "livability", "ecology", "households"] + ["dz:" + z for z in zones]
    get = lambda row, k: row["damage_zone"][k[3:]] if k.startswith("dz:") else row[k]  # noqa: E731
    sur, val = [], {}
    for p in range(len(X)):
        mods = {}
        for k in keys:
            ytr = [get(tr[j][p], k) for j in range(len(tr))]
            yva = [get(va[j][p], k) for j in range(len(va))]
            model, kind, loo = fit_surrogate(Ltr, ytr, cfg["seed"])
            mods[k] = model
            if k in ("damage", "pop_hazard", "livability", "ecology"):
                gp, lin = (model if kind == "gp" else GP(Ltr, ytr, seed=cfg["seed"])), Linear(Ltr, ytr)
                val.setdefault(labels[p], {})[k] = {"chosen": kind, "loo_rmse": loo, "selected": error_stats(yva, model(Lva)),
                                                    "gp": error_stats(yva, gp(Lva)), "linear": error_stats(yva, lin(Lva)),
                                                    "lengthscales": gp.lengthscales, "y_val": yva, "yhat_val": model(Lva).tolist()}
        sur.append(mods)
    worst = max(val[labels[p]][k]["selected"]["rel_rmse"] for p in range(len(X)) for k in ("damage", "pop_hazard"))
    log(f"代理モデルの検証: 被害額・浸水人口の相対 RMSE 最大 {worst:.1%}")

    U = cfg["stakeholders"]
    ref0 = UQ.apply_post(ev.simulate(X[0], sim_dict(nom, simn, P)), costs[0], {k: nom[k] for k in postn})

    def predict(Z):
        """(N, len(dims)) unit samples -> per policy metric arrays."""
        Zs = Z[:, :len(sel)]
        pv = {n: P[n]["low"] + Z[:, len(sel) + i] * (P[n]["high"] - P[n]["low"]) for i, n in enumerate(postn)}
        a, cf = pv.get("asset_factor", 1.0), pv.get("cost_factor", 1.0)
        out = []
        for p in range(len(X)):
            mods = sur[p]
            dz = {z: np.maximum(mods["dz:" + z](Zs), 0) * a for z in zones}
            out.append({"damage": np.maximum(mods["damage"](Zs), 0) * a, "pop_hazard": np.maximum(mods["pop_hazard"](Zs), 0),
                        "livability": mods["livability"](Zs), "ecology": mods["ecology"](Zs),
                        "households": np.maximum(mods["households"](Zs), 0), "damage_zone": dz,
                        "cost": costs[p]["total"] * cf, "cost_zone": {z: c * cf for z, c in costs[p]["by_zone"].items()}})
        return out

    st_rec = {n: (idx.index(s["minimax"]) if s["minimax"] in idx else rec) for n, s in rob["stakeholders"].items()}

    def utilities(Pm, n_rows):
        out = {}
        for name, st in U.items():
            Ut = np.array([RB.utility(Pm[p], st, ref0, budget) for p in range(len(X))])
            feas = np.array([np.broadcast_to(np.asarray(Pm[p]["cost"]) <= budget * (1 + 1e-9), (n_rows,)) for p in range(len(X))])
            out[name] = (np.broadcast_to(Ut, (len(X), n_rows)), feas)
        return out

    # ---- 4. Sobol indices (surrogate; independent inputs from one 2k-dim Sobol sequence)
    so = sa["sobol"]
    targets = {o: (lambda o: lambda Z: predict(Z)[rec][o])(o) for o in ("damage", "pop_hazard", "livability", "ecology")}
    targets["reduction"] = lambda Z: (lambda Pm: Pm[0]["damage"] - Pm[rec]["damage"])(predict(Z))
    for name in U:
        def f(Z, name=name):
            Ut, feas = utilities(predict(Z), len(Z))[name]
            return RB.regret(Ut, feas)[st_rec[name]]
        targets[f"regret:{name}"] = f
    sobol = {}
    for t, fn in targets.items():
        sobol[t] = sobol_indices(fn, len(dims), so["log2_n"], so["n_boot"], cfg["seed"] + 5)
        log(f"Sobol {t}: " + ", ".join(f"{d} S1 {a:.2f}/ST {b:.2f}" for d, a, b in zip(dims, sobol[t]["S1"], sobol[t]["ST"])))

    # tornado (surrogate, one at a time; unscreened parameters: Morris says negligible)
    znom = nominal_unit(dims, P)
    base_pred = {o: float(np.ravel(targets[o](znom[None]))[0]) for o in targets}
    tornado = {}
    for o in targets:
        rows = []
        for i, d in enumerate(dims):
            lo, hi = znom.copy(), znom.copy(); lo[i], hi[i] = 0.0, 1.0
            rows.append({"param": d, "low": float(np.ravel(targets[o](lo[None]))[0]), "high": float(np.ravel(targets[o](hi[None]))[0])})
        tornado[o] = {"base": base_pred[o], "rows": sorted(rows, key=lambda r: -abs(r["high"] - r["low"]))}

    # ---- 5. decision sensitivity (surrogate samples)
    from scipy.stats import qmc, kendalltau
    nd = sa["decision"]["n_samples"]
    Z = qmc.Sobol(d=len(dims), scramble=True, seed=cfg["seed"] + 9).random(nd)
    Pm = predict(Z)
    ut = utilities(Pm, nd)
    Pex = [metrics(p, nom) for p in range(len(X))]     # nominal ranking from the RRI runs (cached), not the surrogate
    decision = {}
    for name in U:
        Ut, feas = ut[name]
        Un = np.array([float(RB.utility(Pex[p], U[name], ref0, budget)) for p in range(len(X))])
        best = np.where(feas.any(0), np.argmax(np.where(feas, Ut, -np.inf), 0), np.argmax(Ut, 0))
        freq = np.bincount(best, minlength=len(X)) / nd
        nbest = int(np.argmax(Un))
        tau = [kendalltau(Un, Ut[:, j]).statistic for j in range(0, nd, max(1, nd // 512))]
        Rg = RB.regret(Ut, feas)[st_rec[name]]
        # best policy as a function of each input (5 bins)
        cond = {}
        for i, d in enumerate(dims):
            bins = np.minimum((Z[:, i] * 5).astype(int), 4)
            cond[d] = [(np.bincount(best[bins == b], minlength=len(X)) / max((bins == b).sum(), 1)).tolist() for b in range(5)]
        decision[name] = {"best_frequency": freq.tolist(), "nominal_best": nbest, "recommended": st_rec[name],
                          "rank_reversal_rate": float((best != nbest).mean()),
                          "kendall_tau_mean": float(np.nanmean(tau)), "kendall_tau_q10": float(np.nanquantile(tau, 0.1)),
                          "regret_quantiles": np.quantile(Rg, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                          "regret_hist": np.histogram(Rg, bins=20)[0].tolist(), "regret_edges": np.histogram(Rg, bins=20)[1].tolist(),
                          "best_by_param_bin": cond}

    # parameter-metric relation: simulated points and 1-D surrogate curves (others nominal)
    relation = {"sel": sel, "points": [{"u": Ltr.tolist() + Lva.tolist(),
                                        "y": {k: [r[p][k] for r in tr + va] for k in ("damage", "pop_hazard", "livability", "ecology")}}
                                       for p in range(len(X))], "curves": {}}
    for i, d in enumerate(sel):
        g = np.linspace(0, 1, 21); Zc = np.tile(znom, (21, 1)); Zc[:, i] = g
        Pc = predict(Zc)
        relation["curves"][d] = {k: [Pc[p][k].tolist() for p in range(len(X))] for k in ("damage", "pop_hazard", "livability", "ecology")}

    # ---- 6. Pareto change (RDM stress test, real runs) and preference sensitivity
    par = pareto_change(R)
    M = R["matrix"]
    n_s = len(R["scenarios"])
    full = [[{"damage": M["damage"][p][s], "cost": M["cost"][p][s], "households": M["households"][p][s],
              "livability": M["livability"][p][s], "ecology": M["ecology"][p][s],
              "damage_zone": None, "cost_zone": None} for s in range(n_s)] for p in range(len(pol))]
    # zone damages/costs of the RDM matrix: rebuild from the cached simulations
    for p, pp in enumerate(pol):
        c = ev.cost(pp["x"])
        for s, scn in enumerate(R["scenarios"]):
            m = ev.simulate(pp["x"], R["sims"][scn["sim"]], strict=False)
            full[p][s] = UQ.apply_post(m, c, R["posts"][scn["post"]])
    ref = full[0][0]
    eligible = np.array([ev.cost(pp["x"])["total"] <= budget for pp in pol])
    pref = preference(full, U, ref, budget, eligible, sa["preference"]["n_weight_samples"], sa["preference"]["sweep_steps"],
                      np.random.default_rng(cfg["seed"] + 13))

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0, "n_simulations": getattr(ev, "n_sim", None),
           "parameters": P, "names": names, "sim_params": simn, "post_params": postn, "policies": labels, "policy_index": idx,
           "recommended": rec, "stakeholder_recommended": st_rec,
           "morris": {"outputs": OUTPUTS, "indices": morris, "score": score, "selected": sel, "delta": delta,
                      "n_points": len(trajs) * (len(names) + 1), "trajectories": [t[0].tolist() for t in trajs]},
           "surrogate": {"kind": "Gaussian process (ARD SE) または線形（学習点の1個抜き交差検証で案・出力ごとに選択）", "inputs": sel, "n_train": len(Ltr), "anchor_nominal": bool(su.get("anchor_nominal", True)), "n_val": su["n_val"],
                         "validation": val, "worst_rel_rmse": worst,
                         "note": "Sobol・意思決定の感度は代理モデル上の推定。未選択の計算パラメータは名目値に固定（Morris で影響小）。"},
           "sobol": {"inputs": dims, "results": sobol, "design": f"scrambled Sobol sequence, 2^{so['log2_n']} x {2 * len(dims)} (A|B independent)"},
           "tornado": tornado, "decision": decision, "relation": relation, "pareto_change": par,
           "pareto_policies": [p["label"] for p in pol], "preference": pref}
    (rdm_dir / "sensitivity.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log(f"完了: {rdm_dir / 'sensitivity.json'}（{time.time() - t0:.0f} s, RRI {out['n_simulations']} 回）")
    return out
