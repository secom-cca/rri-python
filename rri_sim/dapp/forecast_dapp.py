"""Adaptive DAPP with forecasts and learning (docs/11 11.12).

    python scripts/run_forecast_dapp.py [--quick]

Physics: the 1K-RRI surrogate of an exploration without effect coefficients (explore v5).  Implementation model: docs/13
(roadmap.sim.State: lead times, budget and deferral, change costs; levees / retention in place when built, every other
measure growing linearly over its build / diffusion years).  Worlds (roadmap.world): the true 2100 climate and its path, the
forecast of the 2100 climate at every time (Gaussian learning: the forecast s.d. shrinks as information accumulates), rain
intensity, exposure, implementation periods.  No observation error.

  1. initial actions  portfolios evaluated under the planner's nominal values only (belief world, implemented from the start
                      year, nothing added later): Pareto-efficient in (initial cost, PV of the annual expected damage
                      (return periods 1/2-1/200, forecast.frequency), mean exposed-
                      population ratio) by NSGA-II; up to initial.n_max of them spread over the front, plus "no measures"
  2. decision rules   per initial action every rule of the grid (exhaustive) on the search scenarios of each seed:
                        monitor   the forecast only: expected ratio at t + lookahead of the committed portfolio
                                  ratio = max(event damage / damage target, exposed population / population target),
                                  expectation over the forecast of the 2100 climate (3-point rule), exposure held at today's
                        trigger   expected ratio > rho
                        choice    among the one-step upgrades that lower the expected ratio and fit the budget, the largest
                                  net benefit NB = PV(reduction of the annual expected damage over all return periods
                                  (治水経済調査マニュアル) from the action's completion, expected over the forecast)
                                  - PV(initial + annual cost); a long lead time delays the benefit (no penalty
                                  term); k actions per decision
                        wait      (rule option) compare NB now with the value of deciding again after wait_step years with the
                                  forecast then available (preposterior: E over the future forecast of max(0, best NB then),
                                  discounted); wait if larger and the target is expected to hold until an action decided then
                                  could be complete
                      objectives: Target Coverage (share of scenarios meeting the targets in every period from from_year,
                      maximise) and the mean cumulative cost (PV, minimise) -> Pareto front per initial action and seed
  3. evaluation       the union of the seeds' fronts on the common evaluation scenarios: the evaluated front per initial
                      action, and the agreement of the seeds (convergence: hypervolume of each seed's front on the evaluation
                      set, overlap of the policy sets).  No single rule or initial action is chosen.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import config as CF
from .. import frequency as FQ
from ..data import PROCESSED
from ..rdm import explore as EX
from ..rdm.explore_scope import nondominated
from ..rdm.uncertainty import lhs
from ..roadmap import config as RCF
from ..roadmap import run as RR
from ..roadmap import sim as SM
from ..roadmap import world as WD
from ..roadmap.social import Social

DAPP_DIR = PROCESSED / "dapp"
GH_Z = np.array([-math.sqrt(3.0), 0.0, math.sqrt(3.0)])          # 3-point Gauss-Hermite (probabilists')
GH_W = np.array([1 / 6, 2 / 3, 1 / 6])


@dataclass(frozen=True)
class Policy:
    rho: float
    lookahead: int
    k: int
    wait: int

    def describe(self):
        return {"rho": self.rho, "lookahead": self.lookahead, "k": self.k, "wait": self.wait}


def policies(P):
    return [Policy(float(r), int(L), int(k), int(w)) for r, L, k, w in
            itertools.product(P["rho_grid"], P["lookahead_grid"], P["k_grid"], P["wait_grid"])]


# ------------------------------------------------------------------ settings
def model_cfg(cfg):
    FD = cfg["forecast"]
    rc = RCF.load(FD["model"])
    rc["seed"] = cfg["seed"]
    rc["discount_rate"] = cfg["discount_rate"]
    rc["representative_event"] = cfg["representative_event"]
    rc["physics"] = {"explore_run": FD["explore_run"]}
    T = FD["targets"]
    rc["regimes"] = {"dapp": {"weights": {"damage": 1.0, "cost": 0.0}, "thresholds": {"damage_ratio": T["damage_ratio"], "pop_ratio": T["pop_ratio"]}}}
    rc["initial_regime"] = "dapp"
    rc["regime_change"] = dict(rc["regime_change"], p_none=1.0)
    return rc


def make_worlds(rc, groups, n, seed, nominal_first=False):
    F = WD.factors(rc)
    names = list(F)
    z0 = WD.unit_nominal(F)
    Z = np.tile(z0, (n, 1))
    vary = [j for j, k in enumerate(names) if F[k]["group"] in groups]
    Z[:, vary] = lhs(n, len(vary), np.random.default_rng(seed + 601))
    if nominal_first:
        Z[0] = z0
    return [WD.make_world(rc, z, seed * 1000 + s) for s, z in enumerate(Z)], Z, F


class Levels:
    """Level values per variable (State.request: option index or implementation rate) and the raw surrogate columns of a
    portfolio fully in place."""

    def __init__(self, phys, rc):
        self.phys, self.V = phys, phys.variables
        self.vals = []
        for v in self.V:
            if v["continuous"]:
                kind = v["type"]
                self.vals.append([float(x) for x in rc["measures"].get(kind, {}).get("levels", [0.0, 0.5, 1.0])])
            else:
                self.vals.append([float(i) for i in range(len(v["options"]))])
        self.cols = [[c for c, (j, _) in enumerate(phys.layout) if j == i] for i in range(len(self.V))]

    def raw(self, g):
        r = np.zeros(len(self.phys.layout))
        for i, v in enumerate(self.V):
            li = g[i]
            if not li:
                continue
            c = self.cols[i]
            x = self.vals[i][li]
            t = v["type"]
            if t == "levee":
                hs = v["heights"]
                r[c[0]] = hs[li] / max(hs)
            elif t == "relocate":
                r[c[1 if li == 2 else 0]] = 1.0
            elif t == "forest":
                r[c[0]] = r[c[1]] = x
            elif v["continuous"]:
                r[c[0]] = x
            else:
                r[c[0]] = li / max(len(v["options"]) - 1, 1)
        return r

    def snap(self, committed):
        return tuple(int(np.argmin([abs(c - x) for x in vs])) for c, vs in zip(committed, self.vals))

    def kind(self, i, li):
        return SM.kind_of(self.V[i], self.vals[i][li])

    def upgrades(self, g):
        """One-step upgrades (variable, new level index): any higher level of one variable."""
        return [(i, lj) for i in range(len(self.V)) for lj in range(g[i] + 1, len(self.vals[i]))]


# ------------------------------------------------------------------ projection (surrogate on a climate grid)
class Projector:
    """Event damage and exposed population of a portfolio fully in place on a (rain factor, sea level) grid, nominal rain
    intensity; bilinear interpolation.  Cached per portfolio."""

    def __init__(self, phys, L, PJ, gamma):
        self.phys, self.L = phys, L
        self.rg, self.sg = np.array(PJ["rain_grid"], float), np.array(PJ["slr_grid"], float)
        self.gamma = gamma
        self.cache = {}

    def _fill(self, gs):
        new = [g for g in dict.fromkeys(gs) if g not in self.cache]
        if not new:
            return
        R, S = np.meshgrid(self.rg, self.sg, indexing="ij")
        n = R.size
        raws = np.repeat(np.array([self.L.raw(g) for g in new]), n, axis=0)
        env = {"rain_factor": np.tile(R.ravel(), len(new)), "sea_level_rise_m": np.tile(S.ravel(), len(new)),
               "rain_gamma": np.full(n * len(new), self.gamma)}
        y = self.phys.predict(raws, env)
        for k, g in enumerate(new):
            sl = slice(k * n, (k + 1) * n)
            self.cache[g] = np.stack([y["damage"][sl].reshape(R.shape), y["pop_hazard"][sl].reshape(R.shape)])

    def grids(self, gs):
        self._fill(gs)
        return np.stack([self.cache[g] for g in gs])          # (n, 2, nr, ns)

    def at(self, G, rain, slr):
        """G (n, 2, nr, ns); rain, slr arrays of the same shape (m,) -> (n, 2, m) by bilinear interpolation (clipped)."""
        rg, sg = self.rg, self.sg
        r = np.clip(rain, rg[0], rg[-1]); s = np.clip(slr, sg[0], sg[-1])
        i = np.clip(np.searchsorted(rg, r) - 1, 0, len(rg) - 2)
        j = np.clip(np.searchsorted(sg, s) - 1, 0, len(sg) - 2)
        a = (r - rg[i]) / (rg[i + 1] - rg[i]); b = (s - sg[j]) / (sg[j + 1] - sg[j])
        return ((1 - a) * (1 - b) * G[..., i, j] + a * (1 - b) * G[..., i + 1, j] + (1 - a) * b * G[..., i, j + 1]
                + a * b * G[..., i + 1, j + 1])


# ------------------------------------------------------------------ the controller of one pathway
class Controller:
    needs_obs = True

    def __init__(self, M, init, policy, s):
        self.M, self.init, self.p, self.s = M, tuple(init), policy, s
        self.events = []
        self.cur = None

    def climate_points(self, fc, years, sd_scale=None):
        """Climate (rain, slr) at `years` for the 3 quadrature points of the 2100 forecast: arrays (3, len(years))."""
        M = self.M
        (mr, sr), (ms, ss) = fc["rain"], fc["slr"]
        if sd_scale is not None:
            sr, ss = sd_scale
        Xr = mr + GH_Z * sr
        Xs = np.maximum(ms + GH_Z * ss, 0.0)
        gr = np.interp(years, *M.shape_r)
        gs = np.interp(years, *M.shape_s)
        return 1.0 + (Xr[:, None] - 1.0) * gr[None], Xs[:, None] * gs[None]

    def expected(self, G, fc, years, sd=None, mean=None):
        """Expected (damage, pop) of portfolios G at years over the forecast: (n, 2, len(years))."""
        if mean is not None:
            fc = {"rain": (mean[0], fc["rain"][1]), "slr": (mean[1], fc["slr"][1])}
        R, S = self.climate_points(fc, years, sd)
        Y = self.M.proj.at(G, R.ravel(), S.ravel()).reshape(G.shape[0], 2, 3, len(years))
        return (Y * GH_W[None, None, :, None]).sum(2)

    def expected_ead(self, G, fc, years, sd=None, mean=None):
        """Expected annual damage (frequency-based, rri_sim.frequency.ead) of portfolios G at years, expected over the
        forecast: event of return period T = the representative event's rainfall x r(T) x the climate's rain factor."""
        M = self.M
        if mean is not None:
            fc = {"rain": (mean[0], fc["rain"][1]), "slr": (mean[1], fc["slr"][1])}
        R, S = self.climate_points(fc, years, sd)                                    # (3, ny)
        nT = len(M.rT)
        Rt = (R[:, :, None] * M.rT[None, None]).ravel()
        St = np.repeat(S.ravel(), nT)
        D = M.proj.at(G, Rt, St)[:, 0].reshape(G.shape[0], 3, len(years), nT)       # event damage per T
        E = FQ.ead(D, M.T)                                                          # (n, 3, ny)
        return (E * GH_W[None, :, None]).sum(1)

    def ratio(self, Y, asset, pop):
        T = self.M.targets
        return np.maximum(Y[:, 0] * asset / self.M.ref["damage"] / T["damage_ratio"],
                          Y[:, 1] * pop / self.M.ref["pop_hazard"] / T["pop_ratio"])

    def decide(self, t, st, obs):
        M, L = self.M, self.M.L
        if t == M.y0:
            for i, li in enumerate(self.init):
                if li:
                    st.request(i, L.vals[i][li], "initial")
            self.events.append({"year": t, "kind": "initial"})
            return
        if st.queue:                                             # earlier actions still wait for budget
            return
        g = L.snap(st.committed)
        fc, asset, pop = obs["forecast"], obs["asset"], obs["pop"]
        p = self.p
        tau = np.array([min(t + p.lookahead, M.y1 - 2.5)])
        G0 = M.proj.grids([g])
        r0 = float(self.ratio(self.expected(G0, fc, tau), asset, pop)[0, 0])
        if r0 <= p.rho:
            return
        busy = {a.var for a in st.actions if not a.failed and st.complete_year(a) > t}
        left = M.rc["budget"]["total_oku"] - st.spent
        cand = []
        for i, lj in L.upgrades(g):
            if i in busy:
                continue
            g2 = list(g); g2[i] = lj; g2 = tuple(g2)
            c = M.action_cost(i, g[i], lj)
            if c["initial"] > left + 1e-9:
                continue
            cand.append((i, lj, g2, c))
        if not cand:
            return
        G = M.proj.grids([c[2] for c in cand])
        r1 = self.ratio(self.expected(G, fc, tau), asset, pop)[:, 0]
        ok = [k for k in range(len(cand)) if r1[k] < r0 - 1e-6]
        if not ok:
            self.events.append({"year": t, "kind": "no_action", "ratio": round(r0, 3)})
            return
        nbv = self.net_benefit(G0, G[ok], [cand[k] for k in ok], t, fc, asset)
        nb_of = dict(zip(ok, nbv))
        if M.P.get("choice", "target_nb") == "target_nb":
            # actions that restore the target at the look-ahead year: the largest net benefit among them; when none does
            # alone, the one that lowers the expected ratio most (then the next ones, for k > 1)
            meet = [k for k in ok if r1[k] <= 1.0]
            order = sorted(meet, key=lambda k: -nb_of[k]) if meet else sorted(ok, key=lambda k: (r1[k], -nb_of[k]))
        else:                                                     # "net_benefit": the largest net benefit
            order = sorted(ok, key=lambda k: -nb_of[k])
        nb_sorted = [nb_of[k] for k in order]
        if p.wait:
            w = self.wait_value(G0, G, cand, ok, t, fc, asset, pop)
            if w is not None and w > nb_sorted[0]:
                self.events.append({"year": t, "kind": "wait", "ratio": round(r0, 3), "nb_now": round(float(nb_sorted[0]), 1),
                                    "value_wait": round(float(w), 1), "sd_rain": round(fc["rain"][1], 4)})
                return
        taken = set()
        for n_, k in enumerate(order):
            if len(taken) >= p.k:
                break
            i, lj, g2, c = cand[k]
            if i in taken:
                continue
            st.request(i, L.vals[i][lj], "forecast")
            taken.add(i)
            self.events.append({"year": t, "kind": "act", "var": int(i), "level": int(lj), "ratio": round(r0, 3),
                                "ratio_after": round(float(r1[k]), 3), "nb": round(float(nb_sorted[n_]), 1),
                                "cost": round(c["pv"](t), 1), "rain_forecast": round(fc["rain"][0], 3), "sd_rain": round(fc["rain"][1], 4)})

    def net_benefit(self, G0, G, cands, t, fc, asset, mean=None, sd=None):
        """NB = PV(reduction of the annual expected damage (all return periods, 治水経済調査マニュアル) over the years after
        the action's (gradual) completion) - PV(cost), decided at t; expected over the forecast."""
        M = self.M
        yrs = M.byears[M.byears >= t]
        Y0 = self.expected_ead(G0, fc, yrs, sd=sd, mean=mean)[0]                 # (ny,)
        Y = self.expected_ead(G, fc, yrs, sd=sd, mean=mean)                      # (n, ny)
        out = np.empty(len(cands))
        for k, (i, lj, g2, c) in enumerate(cands):
            share = c["share"](t, yrs)
            out[k] = float(((Y0 - Y[k]) * share * M.bweight[M.byears >= t]).sum()) * asset - c["pv"](t)
        return out

    def wait_value(self, G0, G, cand, ok, t, fc, asset, pop):
        # compares with the net benefit of the action that would be taken now (nb_sorted[0] in decide)
        """Value of deciding again at t + wait_step with the forecast then available: E over the future forecast mean
        (N(m_t, s_t^2 - s_{t+h}^2), 3 points) of max(0, best NB decided at t + h), with the expectation over X then
        N(m', s_{t+h}^2).  None when waiting is not allowed (target expected to be exceeded before a delayed action works)."""
        M, p = self.M, self.p
        h = M.P["wait_step"]
        t2 = t + h
        if t2 >= M.y1:
            return None
        w = M.worlds[self.s]
        lead = min(cand[k][3]["lead"] for k in ok)                  # the quickest useful action, decided at t + h
        tau = np.array([min(t2 + lead, M.y1 - 2.5)])
        if float(self.ratio(self.expected(G0, fc, tau), asset, pop)[0, 0]) > 1.0:
            return None
        sr2, ss2 = w.fc["rain"].sd(t2), w.fc["slr"].sd(t2)
        dr, ds = w.fc["rain"].spread_later(t, h), w.fc["slr"].spread_later(t, h)
        val = 0.0
        sub = [cand[k] for k in ok]
        Gs = G[ok]
        for z, wz in zip(GH_Z, GH_W):
            mean = (fc["rain"][0] + z * dr, fc["slr"][0] + z * ds)
            nb = self.net_benefit(G0, Gs, sub, t2, fc, asset, mean=mean, sd=(sr2, ss2))
            val += wz * max(0.0, float(nb.max()))
        return val


# ------------------------------------------------------------------ model
class Model:
    def __init__(self, cfg, phys, rc=None, log=print):
        FD = cfg["forecast"]
        self.cfg, self.FD, self.P = cfg, FD, FD["policy"]
        self.rc = rc or model_cfg(cfg)
        self.phys = phys
        self.ref = RR.reference(phys, self.rc)
        self.targets = FD["targets"]
        y = self.rc["years"]
        self.y0, self.y1, self.step, self.estep = y["start"], y["end"], y["decision_step"], y["eval_step"]
        self.L = Levels(phys, self.rc)
        self.proj = Projector(phys, self.L, FD["projection"], WD.uval(self.rc["climate"]["rain_gamma"]))
        self.shape_r = np.array(WD.path_shape(self.rc, "rain")).T
        self.shape_s = np.array(WD.path_shape(self.rc, "slr")).T
        # benefit years: mid-points of the evaluation periods; weight = discounted years (x the annual expected damage)
        starts = np.arange(self.y0, self.y1, self.estep)
        self.byears = starts + self.estep / 2
        self.bweight = np.array([SM.discount(self.rc, np.arange(a, a + self.estep)).sum() for a in starts])
        fq = FD["frequency"]
        self.T, self.rT = np.array(fq["T"], float), np.array(fq["ratio"], float)
        self.worlds = []
        self._cost = {}

    def action_cost(self, i, li, lj):
        """Cost of raising variable i from level li to lj, decided at t (planner's nominal phases):
        {initial, lead, pv(t), share(t, years)} - PV of the initial cost (spread over build + diffusion) and of the added
        annual cost (annual + maintenance) from completion to the end; share = expected share in place at the years."""
        key = (i, li, lj)
        if key in self._cost:
            return self._cost[key]
        rc, L, phys = self.rc, self.L, self.phys
        v = L.V[i]
        kind = L.kind(i, lj)
        m = rc["measures"][kind]
        ph = WD.nominal_phases(rc, kind)
        a0, a1 = phys.cost_parts(i, L.vals[i][li]) if li else (0.0, 0.0), phys.cost_parts(i, L.vals[i][lj])
        if v["continuous"]:
            ini, chg = max(a1[0] - a0[0], 0.0), 0.0
        else:
            chg = rc["measures"][L.kind(i, li)]["change_cost_fraction"] * a0[0] if li else 0.0
            ini = max(a1[0] - a0[0], 0.0) + chg
        ann = max(a1[1] - a0[1], 0.0) + m.get("maintenance_fraction", 0.0) * max(a1[0] - a0[0], 0.0)
        C = ph["consensus"]
        dur = ph["build"] + ph["diffuse"]
        step = kind in SM.STEP_KINDS
        lead = C + (ph["build"] if step else dur)
        y1 = self.y1
        r = rc["discount_rate"]

        def pv(t):
            ya, yb = t + C, t + C + max(dur, 1.0)
            yrs = np.arange(int(np.floor(ya)), int(np.ceil(yb)))
            cap = ini / max(yb - ya, 1e-9) * sum(max(min(yb, y + 1) - max(ya, y), 0.0) * (1 + r) ** -(y - self.y0) for y in yrs)
            done = t + lead
            mt = sum(ann * (1 + r) ** -(y - self.y0) for y in range(int(math.ceil(done)), int(y1)))
            return float(cap + mt)

        def share(t, years):
            if step:
                return (years >= t + C + ph["build"]).astype(float)
            return np.clip((years - t - C) / max(dur, 1e-9), 0.0, 1.0) if dur > 0 else (years >= t + C).astype(float)
        out = {"initial": ini, "lead": lead, "pv": pv, "share": share}
        self._cost[key] = out
        return out


# ------------------------------------------------------------------ simulation (lock step: physics batched)
def simulate(M, jobs):
    """jobs [(initial levels, Policy, scenario index)] -> [(State, Controller)]."""
    rc, phys = M.rc, M.phys
    sims = []
    for g0, pol, s in jobs:
        w = M.worlds[s]
        st = SM.State(rc, phys, w, Social(rc, w.p["social"], {}))
        sims.append((st, Controller(M, g0, pol, s)))
    for t in SM.decision_years(rc):
        for st, c in sims:
            st.settle(t)
            obs = SM.observe(st, t, M.ref, rc) if t > M.y0 else None
            c.decide(t, st, obs)
            st.process(t)
            st.add_periods(t, t + M.step)
        SM.evaluate([x[0] for x in sims], phys)
        for st, _ in sims:
            st.accrue(t + M.step)
    return sims


def record(M, st, c):
    T, ref = M.targets, M.ref
    sc = SM.score(st, M.rc, ref, [[M.y0, "dapp"]])
    viol = [p["t"] for p in st.periods if p["t"] >= T["from_year"]
            and (p["damage"] / ref["damage"] > T["damage_ratio"] or p["pop_risk"] / ref["pop_hazard"] > T["pop_ratio"])]
    df = SM.discount(M.rc, np.arange(st.y0, st.y1))
    cum = np.cumsum(df * (st.capital + st.maint + st.coord))
    return {"s": c.s, "met": not viol, "first_violation": viol[0] if viol else None, "cost_pv": sc["cost_pv"], "ead_rep_pv": sc["ead_pv"],
            "cost_path": [round(float(cum[min(t + M.step, st.y1) - st.y0 - 1]), 1) for t in SM.decision_years(M.rc)],
            "events": c.events, "n_acts": sum(e["kind"] == "act" for e in c.events), "n_wait": sum(e["kind"] == "wait" for e in c.events),
            "series": [[p["t"], round(p["damage"] / ref["damage"], 3), round(p["pop_risk"] / ref["pop_hazard"], 3)] for p in st.periods]}


def run_jobs(M, jobs, full=False):
    out = []
    ch = int(M.FD.get("chunk", 400))
    for a in range(0, len(jobs), ch):
        for st, c in simulate(M, jobs[a:a + ch]):
            r = record(M, st, c)
            if not full:
                r.pop("events"); r.pop("series")
            out.append(r)
    return out


def objectives(recs):
    return float(np.mean([r["met"] for r in recs])), float(np.mean([r["cost_pv"] for r in recs]))


def ead_pv(M, states):
    """PV of the annual expected damage (all return periods) of evaluated states: per period the surrogate at the
    period's implementation state and climate, rain factor x r(T) for every T; x the period's asset factor."""
    rows = [(k, p) for k, st in enumerate(states) for p in st.periods]
    nT = len(M.T)
    raw = np.repeat(np.array([p["raw"] for _, p in rows]), nT, axis=0)
    env = {"rain_factor": np.concatenate([p["env"]["rain_factor"] * M.rT for _, p in rows]),
           "sea_level_rise_m": np.repeat([p["env"]["sea_level_rise_m"] for _, p in rows], nT),
           "rain_gamma": np.repeat([p["env"]["rain_gamma"] for _, p in rows], nT)}
    D = M.phys.predict(raw, env)["damage"].reshape(len(rows), nT)
    E = FQ.ead(D, M.T)
    out = np.zeros(len(states))
    for (k, p), e in zip(rows, E):
        out[k] += e * p["asset"] * SM.discount(M.rc, np.arange(p["t"], p["t"] + M.estep)).sum()
    return out


# ------------------------------------------------------------------ step 1: initial actions (nominal values)
def initial_actions(M, log=print):
    """Pareto-efficient portfolios under the planner's nominal values (belief world)."""
    I = M.FD["initial"]
    L = M.L
    belief = WD.belief_world(M.rc)
    nV = len(L.V)
    memo = {}

    def static(gs):
        new = [g for g in dict.fromkeys(gs) if g not in memo]
        if new:
            sts = []
            for g in new:
                st = SM.State(M.rc, M.phys, belief, Social(M.rc, belief.p["social"], {}))
                st.cap_factor = 1e9                       # the whole portfolio starts now (its initial cost is bounded below)
                for i, li in enumerate(g):
                    if li:
                        st.request(i, L.vals[i][li], "initial")
                sts.append(st)
            for t in SM.decision_years(M.rc):
                for st in sts:
                    st.settle(t); st.process(t); st.add_periods(t, t + M.step)
                for st in sts:
                    st.accrue(t + M.step)
            SM.evaluate(sts, M.phys)
            eads = ead_pv(M, sts)
            for g, st, e in zip(new, sts, eads):
                ini = sum(M.action_cost(i, 0, li)["initial"] for i, li in enumerate(g) if li)
                pr = float(np.mean([p["pop_risk"] / M.ref["pop_hazard"] for p in st.periods]))
                memo[g] = (ini, e, pr)
        return [memo[g] for g in gs]

    space = EX.Space([SimpleNamespace(continuous=False, options=range(len(L.vals[i])), type="g") for i in range(nV)], {})
    cap = float(I["max_cost_oku"])

    def evaluate(xs):
        gs = [tuple(int(round(float(a))) for a in x) for x in xs]
        out = []
        for (ini, ead, pr) in static(gs):
            pen = max(ini - cap, 0.0) * 1e3                     # infeasible: initial cost above the cap
            out.append([ini + pen, ead + pen, pr + pen])
        return out
    seeds = [[0] * nV] + [[len(L.vals[i]) - 1 if j == i else 0 for j in range(nV)] for i in range(nV)]
    EX.nsga_mixed(space, evaluate, I["pop"], I["gens"], np.random.default_rng(M.cfg["seed"] + 5), I["p_mut"], init=seeds,
                  log=lambda m: None)
    G = [g for g in memo if memo[g][0] <= cap + 1e-9]
    Fm = np.array([memo[g] for g in G])
    nd = nondominated(Fm)
    front = [G[k] for k in np.flatnonzero(nd)]
    log(f"初期行動の候補: 評価した組合せ {len(memo)}・Pareto {len(front)}（標準値の世界）")
    none = tuple([0] * nV)
    pick = [none]
    Ff = np.array([memo[g] for g in front])
    lo, hi = Ff.min(0), Ff.max(0)
    Z = (Ff - lo) / np.where(hi > lo, hi - lo, 1.0)
    chosen = []
    if len(front):
        chosen.append(int(np.argmin(Ff[:, 1])))                 # the lowest expected damage
        while len(chosen) < min(I["n_max"] - 1, len(front)):
            d = np.min(np.linalg.norm(Z[:, None] - Z[chosen][None], axis=2), 1)
            d[chosen] = -1
            chosen.append(int(np.argmax(d)))
    pick += [front[k] for k in chosen if front[k] != none]
    return pick, {g: memo[g] for g in pick}, [{"g": list(g), "initial_cost": memo[g][0], "ead_pv": memo[g][1], "pop_ratio": memo[g][2]} for g in front]


def label(M, g):
    L = M.L
    parts = [f"{v['name']} {v['options'][li] if not v['continuous'] else f'{L.vals[i][li]:.0%}'}" for i, (v, li) in enumerate(zip(L.V, g)) if li]
    return " + ".join(parts) or "施策なし"


# ------------------------------------------------------------------ pipeline
def hypervolume(points, ref=(0.0, None)):
    """2-D hypervolume of (coverage, cost) points (maximise coverage, minimise cost) against (coverage 0, cost ref)."""
    P = sorted(points, key=lambda p: p[1])
    cmax = ref[1]
    hv, best = 0.0, 0.0
    for i, (cov, cost) in enumerate(P):
        nxt = P[i + 1][1] if i + 1 < len(P) else cmax
        best = max(best, cov)
        hv += best * max(nxt - cost, 0.0)
    return hv


def front_of(stats):
    """{policy: (coverage, cost)} -> non-dominated policies."""
    keys = list(stats)
    F = np.array([[-stats[k][0], stats[k][1]] for k in keys])
    return [keys[j] for j in np.flatnonzero(nondominated(F))]


def run(cfg=None, out_dir=None, phys=None, log=print):
    t0 = time.time()
    cfg = CF.load(cfg)
    FD = cfg["forecast"]
    rc = model_cfg(cfg)
    if phys is None:
        from ..roadmap.physics import SurrogatePhysics
        phys = SurrogatePhysics(FD["explore_run"], log)
    M = Model(cfg, phys, rc, log)
    out = Path(out_dir) if out_dir else DAPP_DIR / FD["name"]
    out.mkdir(parents=True, exist_ok=True)
    CF.save(cfg, out / "config.json")
    SC = FD["scenarios"]
    ev_worlds, Zev, F = make_worlds(rc, SC["groups"], int(SC["n_eval"]), cfg["seed"])
    search = {s: make_worlds(rc, SC["groups"], int(SC["n_search"]), int(s))[0] for s in SC["seeds"]}
    M.worlds = list(ev_worlds)
    idx = {}
    for s, ws in search.items():
        idx[s] = list(range(len(M.worlds), len(M.worlds) + len(ws)))
        M.worlds += ws
    scen_ev = list(range(len(ev_worlds)))
    inits, init_obj, front0 = initial_actions(M, log)
    pols = policies(FD["policy"])
    log(f"Adaptive DAPP（予測と学習）: 初期行動 {len(inits)}・ルール {len(pols)}（全探索）・探索シナリオ {SC['n_search']} × seed {len(SC['seeds'])}・"
        f"評価シナリオ {len(scen_ev)}")
    table = []
    for n_i, g0 in enumerate(inits):
        t1 = time.time()
        per_seed = {}
        for s in SC["seeds"]:
            recs = run_jobs(M, [(g0, p, j) for p in pols for j in idx[s]])
            n = len(idx[s])
            per_seed[s] = {p: objectives(recs[k * n:(k + 1) * n]) for k, p in enumerate(pols)}
        fronts = {s: front_of(st) for s, st in per_seed.items()}
        union = list(dict.fromkeys(p for s in SC["seeds"] for p in fronts[s]))
        recs = run_jobs(M, [(g0, p, j) for p in union for j in scen_ev], full=True)
        n = len(scen_ev)
        ev = {p: recs[k * n:(k + 1) * n] for k, p in enumerate(union)}
        evs = {p: objectives(ev[p]) for p in union}
        fe = front_of(evs)
        cost_ref = max(v[1] for v in evs.values()) * 1.05 + 1.0
        conv = {"hv_eval": {str(s): hypervolume([evs[p] for p in fronts[s]], (0.0, cost_ref)) for s in SC["seeds"]},
                "hv_union": hypervolume([evs[p] for p in fe], (0.0, cost_ref)),
                "jaccard": float(np.mean([len(set(fronts[a]) & set(fronts[b])) / max(len(set(fronts[a]) | set(fronts[b])), 1)
                                          for a, b in itertools.combinations(SC["seeds"], 2)])) if len(SC["seeds"]) > 1 else 1.0,
                "n_front": {str(s): len(fronts[s]) for s in SC["seeds"]}}
        conv["hv_ratio_min"] = min(conv["hv_eval"].values()) / max(conv["hv_union"], 1e-12)
        rows = []
        for p in sorted(fe, key=lambda p: evs[p][1]):
            R = ev[p]
            rows.append({"policy": p.describe(), "coverage": evs[p][0], "mean_cost": evs[p][1],
                         "search": {str(s): per_seed[s][p] for s in SC["seeds"]},
                         "n_acts": float(np.mean([r["n_acts"] for r in R])), "n_wait": float(np.mean([r["n_wait"] for r in R])),
                         "first_violation": [r["first_violation"] for r in R],
                         "met_so_far": np.mean([[r["first_violation"] is None or r["first_violation"] > t for t in SM.decision_years(rc)] for r in R], 0).round(4).tolist(),
                         "cost_path": np.mean([r["cost_path"] for r in R], 0).round(1).tolist(),
                         "events": [r["events"] for r in R]})
        table.append({"init": n_i, "levels": list(g0), "label": label(M, g0), "initial_cost": init_obj[g0][0], "nominal": {"ead_pv": init_obj[g0][1], "pop_ratio": init_obj[g0][2]},
                      "front": rows, "convergence": conv,
                      "all_search": {str(s): [[p.describe(), v[0], v[1]] for p, v in per_seed[s].items()] for s in SC["seeds"]}})
        log(f"  {n_i + 1}/{len(inits)} 初期行動 {label(M, g0)[:60]}: front {len(fe)}（Coverage {min(evs[p][0] for p in fe):.0%}〜{max(evs[p][0] for p in fe):.0%}）・"
            f"seed 間の HV 比の最小 {conv['hv_ratio_min']:.2f}（{time.time() - t1:.0f} s）")
    res = {"mode": "forecast", "name": FD["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg,
           "years": SM.decision_years(rc), "variables": [{k: v[k] for k in ("name", "type", "zone", "options", "continuous")} for v in M.L.V],
           "levels": M.L.vals, "ref": M.ref, "targets": FD["targets"], "initial_front": front0, "table": table,
           "worlds": [{k: w.p[k] for k in ("rain_2100", "slr_2100", "gamma")} | {"half_year": w.p["forecast"]["half_year"]} for w in ev_worlds],
           "policies": [p.describe() for p in pols]}
    res["elapsed_s"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=RR._json), encoding="utf-8")
    log(f"完了: {out / 'results.json'}（{res['elapsed_s']:.0f} s）")
    return res
