"""Robust DAPP: robust initial measures and the adaptive pathways that branch from them (docs/11 11.9).

    python scripts/run_dapp.py --robust [--quick] [--verify 6]

Everything runs on the exploration surrogate (docs/12) through the implementation model of docs/13
(rri_sim.roadmap: worlds, sim.State with lead times / budget / change costs, observation); RRI only
re-checks representative states (--verify).

  1. candidates   NSGA-II (explore.nsga_mixed) over the portfolio levels: cost, reliability (share of the
                  scenarios meeting the targets at candidates.eval_years with the portfolio in place), damage,
                  Ecology, Livability - no weights.  Seeded with the exploration's Pareto set.
  2. transitions  landuse.Transitions: implementation state vs land use, allowed moves, change costs,
                  lead times, reversibility / reconversion, land fit.  Nodes = chosen Pareto portfolios,
                  no measures and the pairwise unions (so that one can always add another portfolio's measures).
  3. ATP          per node and scenario: first evaluated period (from targets.from_year) where the event
                  damage or the exposed population of the node in place exceeds the target.  Trigger: start the
                  move when the observed ratio exceeds ratio_of_threshold x target, or when the current node is
                  projected (observed climate / exposure trend, planning assumptions otherwise) to exceed it
                  before the next move would be complete (its lead time) plus one decision step.
  4. pathways     every initial node x scenario: decisions every decision_step years on what is observable
                  (sim.observe: noisy rain / sea level / exposure, the last event's damage); the move goes to the
                  cheapest successor projected to meet the target for horizon_years after completion (else the
                  best available, marked "insufficient").  The perfect-information benchmark uses the true
                  outcomes and lead times of the scenario instead (greedy, horizon to the end).
  5. evaluation   per initial node: coverage (share of the scenario sample met by the observation-based
                  pathway; perfect-information coverage for comparison), Regret (utility with the configured
                  weights, best of all pathways in the scenario), cumulative cost, lock-in (useful portfolios no
                  longer reachable, scenarios lost against starting with nothing, sunk irreversible cost and land),
                  promising = non-dominated in (coverage, max Regret, mean cost, lock-in).

Coverage is a share of a scenario sample (LHS over plausible ranges), not a probability.
Writes data/processed/dapp/<robust.name>/{config.json, network.npz, results.json} (GUI /dapp).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import config as CF
from .landuse import Transitions
from ..data import PROCESSED
from ..rdm import explore as EX
from ..rdm.explore_scope import nondominated
from ..rdm.uncertainty import lhs
from ..roadmap import config as RCF
from ..roadmap import metrics as MT
from ..roadmap import planner as PL
from ..roadmap import run as RR
from ..roadmap import sim as SM
from ..roadmap import world as WD
from ..roadmap.social import Social

DAPP_DIR = PROCESSED / "dapp"


# ------------------------------------------------------------------ settings and scenarios
def model_cfg(cfg, R=None):
    """Settings of the implementation model (docs/13) with the Robust DAPP overrides, targets as the criteria
    (R: the settings block, default cfg["robust"]; the Multi-Actor DAPP passes its merged block)."""
    R = R or cfg["robust"]
    rc = RCF.load(R["model"])
    rc["seed"] = cfg["seed"]
    rc["discount_rate"] = cfg["discount_rate"]
    rc["representative_event"] = cfg["representative_event"]
    rc["physics"] = {"explore_run": R["explore_run"]}
    T = R["targets"]
    rc["regimes"] = {"robust": {"weights": dict(R["weights"]), "thresholds": {"damage_ratio": T["damage_ratio"], "pop_ratio": T["pop_ratio"]}}}
    rc["initial_regime"] = "robust"
    rc["regime_change"] = dict(rc["regime_change"], p_none=1.0)
    return rc


def scenarios(rc, R, seed):
    """Worlds of docs/13 varying only the groups of robust.scenarios (LHS; scenario 0 = nominal)."""
    F = WD.factors(rc)
    names = list(F)
    n = int(R["scenarios"]["n"])
    z0 = WD.unit_nominal(F)
    Z = np.tile(z0, (n, 1))
    vary = [j for j, k in enumerate(names) if F[k]["group"] in R["scenarios"]["groups"]]
    Z[:, vary] = lhs(n, len(vary), np.random.default_rng(seed + 601))
    Z[0] = z0
    return F, Z, vary, [WD.make_world(rc, z, seed * 1000 + s) for s, z in enumerate(Z)]


def env_table(worlds, mids):
    keys = list(worlds[0].env(mids[0]))
    E = {k: np.array([[w.env(m)[k] for m in mids] for w in worlds]) for k in keys}
    return E, np.array([[w.asset(m) for m in mids] for w in worlds]), np.array([[w.pop(m) for m in mids] for w in worlds])


def outside_range(phys, E):
    """Share of the scenario-period values outside the surrogate's training range (clipped, never extrapolated)."""
    out = {}
    for k, v in E.items():
        if k in phys.ranges:
            lo, hi = phys.ranges[k]
            out[k] = float(((v < lo - 1e-9) | (v > hi + 1e-9)).mean())
    return out


def node_outcomes(phys, T, nodes, E, asset, popf, ref, chunk=16):
    """Nodes in place (mature) under every scenario x period: damage / exposed-population ratios to the
    reference, Ecology, Livability; arrays (N, S, K)."""
    raws = np.array([T.raw(g) for g in nodes])
    S, K = asset.shape
    out = {k: np.zeros((len(nodes), S, K), np.float32) for k in ("rd", "rp", "eco", "liv")}
    flat = {k: v.ravel() for k, v in E.items()}
    for a in range(0, len(nodes), chunk):
        b = min(a + chunk, len(nodes))
        y = phys.predict(np.repeat(raws[a:b], S * K, axis=0), {k: np.tile(v, b - a) for k, v in flat.items()})
        sh = (b - a, S, K)
        out["rd"][a:b] = y["damage"].reshape(sh) * asset[None] / max(ref["damage"], 1e-9)
        out["rp"][a:b] = y["pop_hazard"].reshape(sh) * popf[None] / max(ref["pop_hazard"], 1e-9)
        out["eco"][a:b] = y["ecology"].reshape(sh)
        out["liv"][a:b] = y["livability"].reshape(sh)
    return out


def annuity(rc, t0=None):
    y = rc["years"]
    t0 = y["start"] if t0 is None else t0
    return float(sum((1 + rc["discount_rate"]) ** -(k - t0) for k in range(int(t0), int(y["end"]))))


# ------------------------------------------------------------------ step 1: candidate portfolios
def snap(T, x):
    """An exploration decision vector (rates / option indices) -> level indices."""
    g = []
    for i, v in enumerate(T.V):
        L = T.levels[i]
        if v["continuous"]:
            g.append(int(np.argmin([abs(float(x[i]) - lv) for lv in L])))
        else:
            g.append(int(min(max(int(round(float(x[i]))), 0), len(L) - 1)))
    return T.repair(g)


def search(phys, T, rc, R, worlds, ref, seed, log, cap=None, memo=None):
    """NSGA-II over the levels; with `cap` only portfolios whose initial cost fits it (the affordable first moves).
    Returns all evaluated portfolios, their objectives and the non-dominated ones (within the cap)."""
    C, tg = R["candidates"], R["targets"]
    E, a, p = env_table(worlds, [float(y) for y in C["eval_years"]])
    A = annuity(rc)
    sp = EX.Space([SimpleNamespace(continuous=False, options=L, type=v["type"]) for v, L in zip(T.V, T.levels)], {})
    memo = {} if memo is None else memo
    over = (lambda g: T.cost_pv(g, A)[1] > cap + 1e-9) if cap is not None else (lambda g: False)

    def evaluate(xs):
        G = [T.repair(x) for x in xs]
        new = [g for g in dict.fromkeys(G) if g not in memo]
        if new:
            o = node_outcomes(phys, T, new, E, a, p, ref)
            ok = ((o["rd"] <= tg["damage_ratio"]) & (o["rp"] <= tg["pop_ratio"])).all(2)
            for j, g in enumerate(new):
                memo[g] = [T.cost_pv(g, A)[0], -float(ok[j].mean()), float(o["rd"][j, :, -1].mean()),
                           -float(o["eco"][j, :, -1].mean()), -float(o["liv"][j, :, -1].mean())]
        return [[f + 1e6 for f in memo[g]] if over(g) else memo[g] for g in G]
    init = [[0] * len(T.V), [len(L) - 1 for L in T.levels]]
    src = PROCESSED / "explore" / R["explore_run"] / "results.json"
    if src.exists():
        ex = json.loads(src.read_text(encoding="utf-8"))
        if [v["name"] for v in ex["variables"]] == [v["name"] for v in T.V]:
            par = [ex["archive"][i]["x"] for i in ex["pareto"]]
            seen = list(dict.fromkeys(snap(T, x) for x in par))
            seen = [g for g in seen if not over(g)]
            idx = np.linspace(0, len(seen) - 1, min(len(seen), C["pop"] - 2)).astype(int) if seen else []
            init += [list(seen[i]) for i in idx]
    if cap is not None:
        init = [g for g in init if not over(T.repair(g))]
    EX.nsga_mixed(sp, evaluate, C["pop"], C["gens"], np.random.default_rng(seed + 607 + (cap is not None)), C["p_mut"], init=init, log=log)
    G = list(memo)
    F = np.array([memo[g] for g in G])
    inside = np.array([not over(g) for g in G])
    nd = np.flatnonzero(inside)[nondominated(F[inside])]
    log(f"  候補の探索{'（初期費用 ≤ ' + format(cap, '.0f') + ' 億円）' if cap is not None else ''}: 評価 {len(G)} 案・非劣解 {len(nd)} 案")
    return G, F, nd


def choose_nodes(T, G, F, nd, C, first=()):
    """n_nodes non-dominated portfolios spread over the cost (plus the best of each objective), as many from the
    affordable first moves (`first`), no measures, and the pairwise unions (max level per variable, refitted to the land)."""
    def spread(ix):
        ix = np.asarray(ix, int)
        if not len(ix):
            return []
        order = ix[np.argsort(F[ix, 0])]
        return list(order[np.linspace(0, len(order) - 1, min(C["n_nodes"], len(order))).astype(int)])
    pick = spread(nd) + spread(first)
    pick += [nd[np.argmin(F[nd, j])] for j in range(F.shape[1])]
    base = list(dict.fromkeys([tuple([0] * len(T.V))] + [G[i] for i in pick]))
    joins = []
    for a in range(1, len(base)):
        for b in range(a + 1, len(base)):
            j = T.repair(np.maximum(base[a], base[b]))
            joins.append(j)
    joins = [j for j in dict.fromkeys(joins) if j not in base]
    A = annuity(T.cfg)
    joins.sort(key=lambda g: T.cost_pv(g, A)[0])
    nodes = (base + joins)[:C["max_nodes"]]
    return nodes, {g: ("pareto" if g in base[1:] else ("none" if not any(g) else "union")) for g in nodes}


# ------------------------------------------------------------------ the network (steps 2-3) and its tables
class Net:
    """Nodes, transition matrices, true outcome tables (scenario x period) and the planner's projection tables."""

    def __init__(self, cfg, rc, phys, T, nodes, worlds, ref, tables=None):
        self.cfg, self.rc, self.phys, self.T, self.worlds, self.ref = cfg, rc, phys, T, worlds, ref
        self.R = cfg["robust"]
        self.nodes = [tuple(int(a) for a in g) for g in nodes]
        y = rc["years"]
        self.y0, self.y1, self.step, self.estep = y["start"], y["end"], y["decision_step"], y["eval_step"]
        self.starts = np.array(SM.eval_starts(rc), float)
        self.mids = self.starts + self.estep / 2
        self.mat = T.matrices(self.nodes)
        A = annuity(rc)
        self.cost = np.array([T.cost_pv(g, A) for g in self.nodes])           # PV, initial, annual
        if tables is None:
            E, a, p = env_table(worlds, list(self.mids))
            o = node_outcomes(phys, T, self.nodes, E, a, p, ref)
            self.rd, self.rp = o["rd"], o["rp"]
            self.range_clip = outside_range(phys, E)
            self._projection()
        else:
            for k, v in tables.items():
                setattr(self, k, v)
        # scenario lead times of every measure kind (true phases of the world)
        self.lead_k = {k: np.array([PL.lead_years(w, k) for w in worlds]) for k in T.kinds}

    def _projection(self):
        """Ratios of the nodes in place under the planning assumptions (nominal effects, gamma) on a rain x sea-level
        grid inside the surrogate's training range; the controller interpolates its observed-trend projection."""
        lo, hi = self.phys.ranges["rain_factor"]
        so, sh = self.phys.ranges["sea_level_rise_m"]
        self.rg, self.sg = np.linspace(lo, hi, 13), np.linspace(so, sh, 8)
        env = WD.belief_world(self.rc).env(self.y0)
        rr, ss = np.meshgrid(self.rg, self.sg, indexing="ij")
        E = {k: np.full((rr.size, 1), v) for k, v in env.items()}
        E["rain_factor"], E["sea_level_rise_m"] = rr.reshape(-1, 1), ss.reshape(-1, 1)
        one = np.ones((rr.size, 1))
        o = node_outcomes(self.phys, self.T, self.nodes, E, one, one, self.ref)
        self.Pd = o["rd"].reshape(len(self.nodes), *rr.shape)
        self.Pp = o["rp"].reshape(len(self.nodes), *rr.shape)

    def tables(self):
        return {k: getattr(self, k) for k in ("rd", "rp", "Pd", "Pp", "rg", "sg")}

    def tipping(self, targets=None):
        """(N,) rain factor at which each node in place stops meeting the targets under the planning assumptions
        (sea level along the planning path); inf if not within the training range."""
        tg = targets or self.R["targets"]
        b = self.rc["climate"]["belief"]
        r = np.linspace(self.rg[0], self.rg[-1], 61)
        s = np.clip(b["slr_2100"] * (r - 1.0) / max(b["rain_2100"] - 1.0, 1e-9), self.sg[0], self.sg[-1])
        idx = np.arange(len(self.nodes))
        q = np.maximum(self.interp(self.Pd, idx, r, s) / tg["damage_ratio"], self.interp(self.Pp, idx, r, s) / tg["pop_ratio"])
        bad = q > 1.0
        return np.where(bad.any(1), r[np.argmax(bad, axis=1)], np.inf)

    def successors(self, targets=None):
        """Pathway options per node: among the allowed moves to a more robust node (higher tipping rain), those
        non-dominated in (move cost, tipping rain), at most network.max_successors spread over the tipping rain."""
        tg = targets or self.R["targets"]
        key = (tg["damage_ratio"], tg["pop_ratio"])
        if getattr(self, "_succ_key", None) == key:
            return self._succ
        tip = self.tipping(tg)
        k = int(self.R["network"]["max_successors"])
        M = self.mat
        S = np.zeros_like(M["ok"])
        for c in range(len(self.nodes)):
            ys = np.flatnonzero(M["ok"][c] & (tip > tip[c] + 1e-9))
            if not len(ys):
                continue
            t = np.where(np.isinf(tip[ys]), 99.0, tip[ys])
            nd = ys[nondominated(np.column_stack([M["initial"][c, ys], -t]))]
            nd = nd[np.argsort(np.where(np.isinf(tip[nd]), 99.0, tip[nd]))]
            if len(nd) > k:
                nd = nd[np.unique(np.linspace(0, len(nd) - 1, k).round().astype(int))]
            S[c, nd] = True
        self._succ_key, self._succ, self.tip = key, S, tip
        return S

    def interp(self, P, idx, r, s):
        """Bilinear interpolation of P[idx] (M, nr, ns) at the points (r, s) (L,) -> (M, L); clipped to the grid."""
        def frac(x, g):
            f = (np.clip(x, g[0], g[-1]) - g[0]) / (g[1] - g[0])
            i = np.minimum(f.astype(int), len(g) - 2)
            return i, f - i
        i, wi = frac(np.asarray(r, float), self.rg)
        j, wj = frac(np.asarray(s, float), self.sg)
        Q = P[idx]
        return ((1 - wi) * (1 - wj) * Q[:, i, j] + wi * (1 - wj) * Q[:, i + 1, j]
                + (1 - wi) * wj * Q[:, i, j + 1] + wi * wj * Q[:, i + 1, j + 1])

    def atp(self, targets=None):
        """(N, S) first period start year exceeding the targets with the node in place (NaN = never before the end)."""
        tg = targets or self.R["targets"]
        v = ((self.rd > tg["damage_ratio"]) | (self.rp > tg["pop_ratio"])) & (self.starts >= tg["from_year"])[None, None]
        first = np.where(v.any(2), self.starts[np.argmax(v, axis=2)], np.nan)
        return first

    def edge_lead(self, c, ys, s=None):
        if s is None:
            return self.mat["lead"][c, ys]
        L = np.zeros(len(ys))
        for k, m in self.mat["kinds"].items():
            L = np.maximum(L, m[c, ys] * self.lead_k[k][s])
        return np.where((L == 0) & self.mat["conditional"][c, ys], 1.0, L)


# ------------------------------------------------------------------ step 4: pathways (observation-based and perfect information)
class Pathway:
    needs_obs = True
    modes = {}

    def __init__(self, net, init, s, oracle=False, trig=None, targets=None):
        self.net, self.init, self.cur, self.s, self.oracle = net, init, init, s, oracle
        self.trig = trig or net.R["trigger"]
        self.tg = targets or net.R["targets"]
        self.hist = {}
        self.events = []

    def decide(self, t, st, obs):
        net = self.net
        if t == net.y0:
            self._move(t, st, 0, self.init)
            self.events.append({"year": t, "from": None, "to": self.init, "reason": "initial"})
            return
        tm = t - net.estep / 2
        for k in ("rain", "slr", "asset", "pop"):
            self.hist.setdefault(k, []).append((tm, float(obs[k])))
        if st.queue:                                      # the last move still waits for budget
            return
        ch = self.choose(t, st, obs)
        if ch:
            self._move(t, st, self.cur, ch["to"])
            self.events.append({"year": t, "from": self.cur, **ch})
            self.cur = ch["to"]

    def _move(self, t, st, a, b):
        ga, gb = self.net.nodes[a], self.net.nodes[b]
        for i in range(len(gb)):
            if gb[i] != ga[i]:
                st.request(i, self.net.T.levels[i][gb[i]], "trigger" if a else "plan")

    def _projected(self, idx, taus):
        """Ratio / target of nodes idx at times taus (M, L): observed trends (planner) or the true scenario (oracle)."""
        net, tg = self.net, self.tg
        if self.oracle:
            k = np.clip(np.searchsorted(net.mids, taus) , 0, len(net.mids) - 1)
            return np.maximum(net.rd[idx][:, self.s, k] / tg["damage_ratio"], net.rp[idx][:, self.s, k] / tg["pop_ratio"])
        y0 = net.y0
        proj = {}
        for k, v0 in (("rain", 1.0), ("slr", 0.0), ("asset", 1.0), ("pop", 1.0)):
            tl, vl, slope = PL.DAPPController._trend(self.hist[k], y0, v0, net.rc["detect"].get("trend_window", 3))
            proj[k] = vl + slope * (np.asarray(taus) - tl)
        d = net.interp(net.Pd, idx, proj["rain"], proj["slr"]) * np.maximum(proj["asset"], 0)
        p = net.interp(net.Pp, idx, proj["rain"], proj["slr"]) * np.maximum(proj["pop"], 0)
        return np.maximum(d / tg["damage_ratio"], p / tg["pop_ratio"])

    def choose(self, t, st, obs):
        net, c, tr, tg = self.net, self.cur, self.trig, self.tg
        M = net.mat
        ys = np.flatnonzero(net.successors(tg)[c])
        if not len(ys):
            return None
        cond = M["conditional"][c, ys]
        if cond.any():                                    # forest reconversion only while the forest is young
            fv = [i for i, v in enumerate(net.T.V) if v["type"] == "forest"]
            allow = {i: st.can_decrease(i, t) for i in fv}
            G = np.array(net.nodes)
            okc = np.array([all(allow[i] or G[y, i] >= G[c, i] for i in fv) for y in ys[cond]])
            keep = np.ones(len(ys), bool); keep[np.flatnonzero(cond)[~okc]] = False
            ys = ys[keep]
        left = net.rc["budget"]["total_oku"] - st.spent
        ys = ys[M["initial"][c, ys] <= left + 1e-9]
        if not len(ys):
            return None
        lead = net.edge_lead(c, ys, self.s if self.oracle else None)
        taus = net.mids[net.mids >= max(t, tg["from_year"])]
        if not len(taus):
            return None
        H = np.inf if self.oracle else tr["horizon_years"]
        r = self._projected(ys, taus)
        win = (taus[None] >= (t + lead)[:, None]) & (taus[None] <= (t + lead + H)[:, None])
        score = np.where(win.any(1), np.where(win, r, -np.inf).max(1), np.inf)
        A = annuity(net.rc, t)
        cost = M["initial"][c, ys] + M["annual"][c, ys] * A
        safety = 1.0 if self.oracle else tr["safety"]
        good = score <= safety
        j = int(np.argmin(np.where(good, cost, np.inf))) if good.any() else int(np.lexsort((cost, score))[0])
        y, L = int(ys[j]), float(lead[j])
        # trigger: the current portfolio would exceed the target before the move is complete (+ one decision step)
        near = taus[taus <= t + L + net.step]
        rc_ = self._projected(np.array([c]), near)[0] if len(near) else np.array([0.0])
        rho = 1.0 if self.oracle else tr["ratio_of_threshold"]
        r_obs = max(obs["damage_ratio"] / tg["damage_ratio"], obs["pop_ratio"] / tg["pop_ratio"]) if "damage_ratio" in obs else 0.0
        fire_p = (self.oracle or tr.get("use_projection", True)) and float(rc_.max()) > rho
        fire_o = (not self.oracle) and tr.get("use_observed", True) and t >= tg["from_year"] and r_obs > rho
        if not (fire_p or fire_o):
            return None
        cur_score = self._projected(np.array([c]), taus[(taus >= t + L) & (taus <= t + L + H)])[0] if np.isfinite(score[j]) else None
        if cur_score is not None and len(cur_score) and score[j] >= float(cur_score.max()) - 1e-9:
            return None                                    # nothing better is available
        return {"to": y, "reason": "observed" if fire_o else "projected", "obs_rain": round(float(obs["rain"]), 4),
                "obs_ratio": round(float(r_obs), 4), "projected": round(float(rc_.max()), 4), "lead": round(L, 1),
                "expected": round(float(score[j]), 4) if np.isfinite(score[j]) else None, "insufficient": bool(not good[j]), "cost": round(float(M["initial"][c, y]), 1)}


def simulate(net, inits, scen, oracle=False, trig=None, targets=None):
    """All initial nodes x scenarios in lock step (one surrogate batch per decision)."""
    rc, phys = net.rc, net.phys
    states, ctrls = [], []
    for i in inits:
        for s in scen:
            w = net.worlds[s]
            st = SM.State(rc, phys, w, Social(rc, w.p["social"], {}))
            st.allow_decrease = True
            states.append(st)
            ctrls.append(Pathway(net, i, s, oracle, trig, targets))
    for t in SM.decision_years(rc):
        for st, c in zip(states, ctrls):
            st.settle(t)
            c.decide(t, st, SM.observe(st, t, net.ref, rc))
            st.process(t)
            st.add_periods(t, t + net.step)
        SM.evaluate(states, phys)
        for st in states:
            st.accrue(t + net.step)
    return states, ctrls


def record(net, st, ctrl, targets=None):
    tg = targets or net.R["targets"]
    ref = net.ref
    sc = SM.score(st, net.rc, ref, [[net.y0, "robust"]])
    viol = [p["t"] for p in st.periods if p["t"] >= tg["from_year"]
            and (p["damage"] / ref["damage"] > tg["damage_ratio"] or p["pop_risk"] / ref["pop_hazard"] > tg["pop_ratio"])]
    return {"init": ctrl.init, "s": ctrl.s, "met": not viol, "first_violation": viol[0] if viol else None, "U": sc["U"],
            "cost_pv": sc["cost_pv"], "ead_pv": sc["ead_pv"], "spent": st.spent, "final": ctrl.cur, "events": ctrl.events,
            "deferred": sum(1 for e in st.log if e["event"] == "deferred"),
            "series": [[p["t"], round(p["damage"] / ref["damage"], 3), round(p["pop_risk"] / ref["pop_hazard"], 3)] for p in st.periods]}


# ------------------------------------------------------------------ step 5: evaluation of the initial nodes
def initial_nodes(net, R):
    """No measures plus up to n_max - 1 affordable first moves (initial cost <= max_cost_oku), non-dominated in
    (initial cost, scenarios never reaching the ATP, median ATP year), spread over the cost."""
    M = net.mat
    atp = net.atp()
    never = np.isnan(atp).mean(1)
    med = np.nanmedian(np.where(np.isnan(atp), net.y1, atp), axis=1)
    cand = [n for n in range(1, len(net.nodes)) if M["ok"][0, n] and M["initial"][0, n] <= R["initial"]["max_cost_oku"] + 1e-9]
    if not cand:
        return [0]
    F = np.column_stack([M["initial"][0, cand], -never[cand], -med[cand]])
    k = R["initial"]["n_max"] - 1
    pick, left = [], np.arange(len(cand))
    while len(pick) < k and len(left):                 # successive non-dominated fronts, each spread over the cost
        nd = left[nondominated(F[left])]
        nd = nd[np.argsort(F[nd, 0])]
        need = k - len(pick)
        pick += [cand[i] for i in (nd[np.linspace(0, len(nd) - 1, need).astype(int)] if len(nd) > need else nd)]
        left = np.setdiff1d(left, nd)
    return [0] + sorted(dict.fromkeys(pick), key=lambda n: M["initial"][0, n])


def lockin(net, inits, ora, scen, targets=None):
    """Options lost by starting with node i instead of nothing.  For every useful portfolio y (meeting the targets to the
    end, in place, in some scenario; affordable from nothing) the closest protection from i is z = y + i (higher level
    per variable, refitted to the land): y is lost when z cannot be reached within the total budget or the land no longer
    allows y's measures; the premium is the extra initial cost of z over y (sunk / changed measures).  Also the share of
    scenarios that the perfect-information pathway meets from nothing but not from i."""
    T, M, B = net.T, net.mat, net.rc["budget"]["total_oku"]
    useful = [y for y in np.flatnonzero(np.isnan(net.atp(targets)[:, scen]).any(1)) if M["initial"][0, y] <= B + 1e-9]
    met0 = ora[0]
    out = {}
    for i in inits:
        gi = net.nodes[i]
        lost, prem = 0, []
        for y in useful:
            gy = net.nodes[y]
            z = T.repair(np.maximum(gy, gi))
            fits = all(z[k] >= gy[k] for k in range(len(z)))
            c = T.change_cost(gi, z)
            total = M["initial"][0, i] + (c or 0.0)
            if not fits or c is None or total > B + 1e-9:
                lost += 1
            else:
                prem.append(total - M["initial"][0, y])
        out[i] = {"options": lost / max(len(useful), 1), "n_lost": lost, "n_useful": len(useful),
                  "premium_oku": float(np.mean(prem)) if prem else None,
                  "scenarios": float(np.mean(met0 & ~ora[i])), **T.irreversible(gi)}
    return out


def pathway_tree(net, recs):
    """Branching map of one initial node: a trie over the sequences of nodes (the timing of a move varies by scenario:
    years = the decision years, summarised as median / range); per branch the scenarios, the observed signals at the
    decision, the share meeting the targets, and how many pathways end there."""
    tree = [{"id": 0, "parent": None, "years": [net.y0], "node": recs[0]["init"], "n": 0, "met": 0, "scen": [], "obs_rain": [],
             "obs_ratio": [], "reasons": {}, "insufficient": 0}]
    child = {}
    for r in recs:
        k = 0
        tree[0]["n"] += 1; tree[0]["met"] += r["met"]; tree[0]["scen"].append(r["s"])
        for e in r["events"][1:]:
            key = (k, e["to"])
            if key not in child:
                child[key] = len(tree)
                tree.append({"id": len(tree), "parent": k, "years": [], "node": e["to"], "n": 0, "met": 0, "scen": [], "obs_rain": [],
                             "obs_ratio": [], "reasons": {}, "insufficient": 0, "lead": e["lead"], "cost": e["cost"]})
            k = child[key]
            b = tree[k]
            b["years"].append(e["year"])
            b["n"] += 1; b["met"] += r["met"]; b["scen"].append(r["s"])
            b["obs_rain"].append(e["obs_rain"]); b["obs_ratio"].append(e["obs_ratio"])
            b["reasons"][e["reason"]] = b["reasons"].get(e["reason"], 0) + 1
            b["insufficient"] += e["insufficient"]
        tree[k]["end"] = tree[k].get("end", 0) + 1
        tree[k]["end_met"] = tree[k].get("end_met", 0) + r["met"]
    for b in tree:
        b.setdefault("end", 0); b.setdefault("end_met", 0)
        ys = b.pop("years")
        b["year"] = int(np.median(ys)) if len(ys) % 2 else int(sorted(ys)[len(ys) // 2 - 1])
        b["year_range"] = [int(min(ys)), int(max(ys))]
        b["year_counts"] = {str(y): ys.count(y) for y in sorted(set(ys))}
        for k in ("obs_rain", "obs_ratio"):
            b[k] = [round(min(b[k]), 3), round(max(b[k]), 3)] if b[k] else None
        b["leaf"] = not any(c["parent"] == b["id"] for c in tree)
        b["met_share"] = b["met"] / max(b["n"], 1)
    return tree


def evaluate(net, inits, scen, trig=None, targets=None, F=None, Z=None, vary=None, log=print):
    """Steps 4-5 for the given initial nodes and scenarios (also used by the GUI's trigger panel)."""
    t0 = time.time()
    R = net.R
    tg = targets or R["targets"]
    sa, ca = simulate(net, inits, scen, False, trig, tg)
    so, co = simulate(net, inits, scen, True, trig, tg)
    A = [record(net, s, c, tg) for s, c in zip(sa, ca)]
    O = [record(net, s, c, tg) for s, c in zip(so, co)]
    nS = len(scen)
    U = np.array([r["U"] for r in A]).reshape(len(inits), nS)
    Uo = np.array([r["U"] for r in O]).reshape(len(inits), nS)
    met = np.array([r["met"] for r in A]).reshape(len(inits), nS)
    meto = np.array([r["met"] for r in O]).reshape(len(inits), nS)
    best = np.maximum(U.max(0), Uo.max(0))
    reg = best[None] - U
    cost = np.array([r["cost_pv"] for r in A]).reshape(len(inits), nS)
    ora = {i: meto[k] for k, i in enumerate(inits)}
    lk = lockin(net, inits, ora, scen, tg) if 0 in inits else {i: {} for i in inits}
    atp = net.atp(tg)[:, scen]
    rows = []
    for k, i in enumerate(inits):
        a = atp[i]
        rows.append({"node": i, "coverage": float(met[k].mean()), "coverage_perfect": float(meto[k].mean()),
                     "max_regret": float(reg[k].max()), "mean_regret": float(reg[k].mean()), "p90_regret": float(np.quantile(reg[k], 0.9)),
                     "satisficing": float((met[k] & (reg[k] <= R["regret_threshold"])).mean()),
                     "mean_cost": float(cost[k].mean()), "p90_cost": float(np.quantile(cost[k], 0.9)),
                     "mean_ead": float(np.mean([A[k * nS + j]["ead_pv"] for j in range(nS)])),
                     "initial_cost": float(net.mat["initial"][0, i]), "atp_never": float(np.isnan(a).mean()),
                     "atp_p10": float(np.nanquantile(np.where(np.isnan(a), net.y1, a), 0.1)),
                     "atp_median": float(np.nanmedian(np.where(np.isnan(a), net.y1, a))),
                     "n_moves": float(np.mean([len(A[k * nS + j]["events"]) - 1 for j in range(nS)])),
                     "lockin": lk[i]})
    Fm = np.array([[-r["coverage"], r["max_regret"], r["mean_cost"], r["lockin"].get("options", 0.0)] for r in rows])
    prom = np.flatnonzero(nondominated(Fm))
    for j in prom:
        rows[j]["promising"] = True
    trees = {str(i): pathway_tree(net, A[k * nS:(k + 1) * nS]) for k, i in enumerate(inits)}
    unc = {"adaptive": ~met.any(0), "perfect": ~meto.any(0)}
    out = {"inits": inits, "scenarios": list(map(int, scen)), "table": rows, "promising": [inits[j] for j in prom],
           "trees": trees, "regret": reg.tolist(), "met": met.astype(int).tolist(), "met_perfect": meto.astype(int).tolist(),
           "uncovered": {k: v.astype(int).tolist() for k, v in unc.items()},
           "records": {"adaptive": [_slim(r) for r in A], "perfect": [_slim(r) for r in O]}, "elapsed_s": time.time() - t0}
    if F is not None:
        Fv = {k: F[k] for j, k in enumerate(F) if j in vary}
        X = Z[np.ix_(scen, vary)]
        pr = R["prim"]
        out["unreachable"] = {k: MT.prim_box(X, v, Fv, pr["alpha"], pr["min_mass"], pr["density"]) for k, v in unc.items()}
        best_i = max(rows, key=lambda r: (r.get("promising", False), r["coverage"], -r["max_regret"]))["node"]
        kb = inits.index(best_i)
        out["unreachable"]["best_initial"] = {"node": best_i, **MT.prim_box(X, ~met[kb], Fv, pr["alpha"], pr["min_mass"], pr["density"])}
    log(f"  経路: 初期施策 {len(inits)} × シナリオ {nS}（観測に基づく経路・完全情報）{time.time() - t0:.0f} s")
    return out


def _slim(r):
    return {k: r[k] for k in ("init", "s", "met", "first_violation", "U", "cost_pv", "ead_pv", "final", "events", "deferred", "series")}


# ------------------------------------------------------------------ pipeline
def setup(cfg, phys=None, log=print, R=None):
    R = R or cfg["robust"]
    rc = model_cfg(cfg, R)
    if phys is None:
        from ..roadmap.physics import SurrogatePhysics
        phys = SurrogatePhysics(R["explore_run"], log)
    ref = RR.reference(phys, rc)
    T = Transitions(rc, phys, R["landuse"])
    F, Z, vary, worlds = scenarios(rc, R, cfg["seed"])
    return rc, phys, ref, T, F, Z, vary, worlds


def node_info(net, i):
    T, g = net.T, net.nodes[i]
    return {"id": i, "levels": list(g), "implementation": T.implementation(g), "landuse": T.landuse(g),
            "cost_pv": float(net.cost[i, 0]), "initial": float(net.cost[i, 1]), "annual": float(net.cost[i, 2]),
            "label": " + ".join(f"{d['name']} {d['label']}" for d in T.implementation(g)) or "施策なし"}


def tipping_rain(net, i):
    """Rain factor (sea level by the planning path) at which node i in place stops meeting the targets, planning effects."""
    tg = net.R["targets"]
    b = net.rc["climate"]["belief"]
    r = np.linspace(net.rg[0], net.rg[-1], 61)
    s = np.clip(b["slr_2100"] * (r - 1.0) / max(b["rain_2100"] - 1.0, 1e-9), net.sg[0], net.sg[-1])
    q = np.maximum(net.interp(net.Pd, [i], r, s)[0] / tg["damage_ratio"], net.interp(net.Pp, [i], r, s)[0] / tg["pop_ratio"])
    bad = np.flatnonzero(q > 1.0)
    return None if not len(bad) else float(r[bad[0]])


def run(cfg=None, out_dir=None, phys=None, log=print):
    t0 = time.time()
    cfg = CF.load(cfg)
    R = cfg["robust"]
    out = Path(out_dir) if out_dir else DAPP_DIR / R["name"]
    out.mkdir(parents=True, exist_ok=True)
    CF.save(cfg, out / "config.json")
    rc, phys, ref, T, F, Z, vary, worlds = setup(cfg, phys, log)
    log(f"Robust DAPP: 施策 {len(T.V)} 変数・シナリオ {len(worlds)}（{', '.join(R['scenarios']['groups'])}）")
    log("1. 施策候補の探索（代理モデル・NSGA-II, 重みなし）")
    memo = {}
    G, Fo, nd = search(phys, T, rc, R, worlds, ref, cfg["seed"], log, memo=memo)
    G, Fo, nd2 = search(phys, T, rc, R, worlds, ref, cfg["seed"], log, cap=R["initial"]["max_cost_oku"], memo=memo)
    nd = np.flatnonzero(nondominated(Fo))
    nodes, origin = choose_nodes(T, G, Fo, nd, R["candidates"], first=nd2)
    log(f"2. 移行可能性: ノード {len(nodes)}（Pareto {sum(o == 'pareto' for o in origin.values())}・和集合・施策なし）")
    net = Net(cfg, rc, phys, T, nodes, worlds, ref)
    log(f"   可能な移行 {int(net.mat['ok'].sum())}（条件付き {int(net.mat['conditional'].sum())}）・経路の選択肢 {int(net.successors().sum())}")
    log("3. ATP（ノードごと・シナリオごと）")
    atp = net.atp()
    inits = initial_nodes(net, R)
    log(f"4–5. 適応経路と初期施策の評価: 初期施策の候補 {len(inits)}")
    ev = evaluate(net, inits, list(range(len(worlds))), F=F, Z=Z, vary=vary, log=log)
    used = sorted({b["node"] for tr in ev["trees"].values() for b in tr} | set(inits))
    pairs = sorted({(b["node"], c["node"]) for tr in ev["trees"].values() for c in tr for b in tr if c["parent"] == b["id"]})
    res = {"mode": "robust", "name": R["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg,
           "variables": phys.variables, "zones": phys.zones, "reference": ref, "years": SM.decision_years(rc),
           "eval_starts": net.starts.tolist(), "factors": F, "varied": [list(F)[j] for j in vary], "Z": Z.tolist(),
           "worlds": [{"rain_pts": w.p["rain_pts"], "slr_pts": w.p["slr_pts"], "asset_pts": w.p["asset_pts"], "pop_pts": w.p["pop_pts"]}
                      for w in worlds],
           "physics": {"explore_run": R["explore_run"], "validation": getattr(phys, "validation", None),
                       "verification": getattr(phys, "verification", None), "ranges": phys.ranges, "outside_range": net.range_clip},
           "model": {k: rc[k] for k in ("years", "budget", "measures")},
           "candidates": {"n_evaluated": len(G), "n_nondominated": int(len(nd)),
                          "objectives": ["cost_pv", "reliability", "damage_ratio_end", "ecology_end", "livability_end"],
                          "front": [{"levels": list(G[i]), "f": [round(float(x), 4) for x in Fo[i]], "node": nodes.index(G[i]) if G[i] in nodes else None}
                                    for i in nd]},
           "network": {"n_nodes": len(nodes), "n_edges": int(net.mat["ok"].sum()), "n_conditional": int(net.mat["conditional"].sum()),
                       "n_options": int(net.successors().sum()), "max_successors": R["network"]["max_successors"],
                       "origin": [origin[g] for g in nodes], "cost_pv": net.cost[:, 0].round(1).tolist(),
                       "initial_from_none": net.mat["initial"][0].round(1).tolist(),
                       "nodes": {str(i): node_info(net, i) for i in used},
                       "edges": [{"from": a, "to": b, "cost": round(float(net.mat["initial"][a, b]), 1), "lead": round(float(net.mat["lead"][a, b]), 1),
                                  "conditional": bool(net.mat["conditional"][a, b]), "changes": T.describe_edge(net.nodes[a], net.nodes[b])}
                                 for a, b in pairs],
                       "successors": {str(i): _successors(net, i) for i in used}},
           "atp": {str(i): {"years": [None if np.isnan(x) else int(x) for x in atp[i]], "tipping_rain": tipping_rain(net, i)} for i in used},
           "evaluation": ev}
    np.savez_compressed(out / "network.npz", nodes=np.array(nodes, int), **net.tables())
    res["elapsed_s"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=RR._json), encoding="utf-8")
    log(f"完了: {out / 'results.json'}（{res['elapsed_s']:.0f} s）")
    return res


def _successors(net, i, targets=None):
    """Moves available from node i: counts and why the others are blocked (GUI: transition constraints)."""
    M, G = net.mat, np.array(net.nodes)
    blocked = {"irreversible": 0, "land": 0}
    for y in range(len(net.nodes)):
        if y == i or M["ok"][i, y]:
            continue
        down = [k for k in range(G.shape[1]) if G[y, k] < G[i, k]]
        if down:
            blocked["irreversible"] += 1
        else:
            blocked["land"] += 1
    ys = np.flatnonzero(net.successors(targets)[i])
    tip = net.tip
    return {"n": int(M["ok"][i].sum()), "n_conditional": int(M["conditional"][i].sum()), "blocked": blocked,
            "tipping_rain": None if np.isinf(tip[i]) else float(tip[i]),
            "options": [{"to": int(y), "cost": round(float(M["initial"][i, y]), 1), "lead": round(float(M["lead"][i, y]), 1),
                         "tipping_rain": None if np.isinf(tip[y]) else float(tip[y]), "changes": net.T.describe_edge(net.nodes[i], net.nodes[y])}
                        for y in ys]}


def load_net(out_dir, phys=None, log=print):
    """The saved network of a run (GUI trigger panel: re-run steps 4-5 without steps 1-3)."""
    out = Path(out_dir)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    rc, phys, ref, T, F, Z, vary, worlds = setup(cfg, phys, log)
    d = np.load(out / "network.npz")
    net = Net(cfg, rc, phys, T, [tuple(g) for g in d["nodes"]], worlds, ref, tables={k: d[k] for k in ("rd", "rp", "Pd", "Pp", "rg", "sg")})
    net.range_clip = {}
    return net, F, Z, vary


def resimulate(out_dir, trigger=None, targets=None, n_scenarios=None, phys=None):
    """Steps 4-5 again with changed trigger settings / targets (surrogate only; for /api/dapp/robust/simulate)."""
    net, F, Z, vary = load_net(out_dir, phys, log=lambda m: None)
    res = json.loads((Path(out_dir) / "results.json").read_text(encoding="utf-8"))
    inits = res["evaluation"]["inits"]
    trig = CF._merge(net.R["trigger"], trigger or {})
    tg = CF._merge(net.R["targets"], targets or {})
    n = len(net.worlds) if not n_scenarios else min(int(n_scenarios), len(net.worlds))
    ev = evaluate(net, inits, list(range(n)), trig, tg, F=F, Z=Z, vary=vary, log=lambda m: None)
    used = sorted({b["node"] for tr in ev["trees"].values() for b in tr} | set(inits))
    pairs = sorted({(b["node"], c["node"]) for tr in ev["trees"].values() for c in tr for b in tr if c["parent"] == b["id"]})
    atp = net.atp(tg)
    return {"evaluation": ev, "trigger": trig, "targets": tg,
            "nodes": {str(i): node_info(net, i) for i in used}, "successors": {str(i): _successors(net, i, tg) for i in used},
            "edges": [{"from": a, "to": b, "cost": round(float(net.mat["initial"][a, b]), 1), "lead": round(float(net.mat["lead"][a, b]), 1),
                       "conditional": bool(net.mat["conditional"][a, b]), "changes": net.T.describe_edge(net.nodes[a], net.nodes[b])}
                      for a, b in pairs],
            "atp": {str(i): {"years": [None if np.isnan(x) else int(x) for x in atp[i]], "tipping_rain": tipping_rain(net, i)} for i in used}}


# ------------------------------------------------------------------ RRI check of representative pathways and branch points
def verify_rri(out_dir, n_states=6, log=print):
    """Re-run RRI for states on the representative pathways of the two promising initial nodes with the highest coverage,
    in the scenarios with the lowest / median / highest 2100 rain, at the first move and at the end."""
    from ..roadmap.physics import rri_evaluator
    from ..rdm.sensitivity import error_stats
    out = Path(out_dir)
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    net, F, Z, vary = load_net(out, log=log)
    ev = res["evaluation"]
    rows_ = sorted((r for r in ev["table"] if r.get("promising") and r["node"] != 0), key=lambda r: (-r["coverage"], r["max_regret"]))
    picks = [r["node"] for r in rows_][:2] or ev["inits"][1:2]
    rain = np.array([w["rain_pts"][-1][1] for w in res["worlds"]])
    rri = rri_evaluator(net.R["explore_run"], log)
    rows = []
    for i in picks:
        recs = [r for r in ev["records"]["adaptive"] if r["init"] == i]
        ss = sorted((r["s"] for r in recs), key=lambda s: rain[s])
        for s in dict.fromkeys([ss[0], ss[len(ss) // 2], ss[-1]]):
            states, ctrls = simulate(net, [i], [s])
            st = states[0]
            moves = [e["year"] for e in ctrls[0].events[1:]]
            when = [min((p for p in st.periods if p["t"] >= (moves[0] if moves else 2040)), key=lambda p: p["t"]), st.periods[-1]]
            for p in when:
                if len(rows) >= n_states:
                    break
                x, sim = net.phys.to_rri(p["raw"], p["env"])
                y = rri.simulate(x, sim)
                pred = net.phys.predict(p["raw"][None], {k: np.atleast_1d(v) for k, v in p["env"].items()})["damage"][0]
                rows.append({"init": i, "s": int(s), "t": p["t"], "x": x, "sim": sim, "rri": y["damage"], "surrogate": float(pred),
                             "branch": bool(moves) and p is when[0]})
                log(f"  初期施策 {i} シナリオ {s} {p['t']}年: RRI {y['damage']:.0f} / 代理 {pred:.0f} 億円")
    o = np.array([r["rri"] for r in rows]); q = np.array([r["surrogate"] for r in rows])
    res["verification_rri"] = {"states": rows, **(error_stats(o, q) if len(rows) > 1 else {})}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=RR._json), encoding="utf-8")
    return res["verification_rri"]
