"""Adaptive DAPP: state-dependent decision rules over single-measure actions (docs/11 11.11).

    python scripts/run_dapp.py --adaptive [--quick]

Instead of a few pathways given in advance (11.9: moves between Pareto portfolios), a policy decides at every decision year,
from what is observable, whether to act and which of the actions feasible now comes next:

    monitor     observed ratio (last event's damage / exposed population ÷ target), projected ratio of the current state
                (observed climate / exposure trends, surrogate, nominal effects) at the next decade + a typical lead time,
                projected 2100 rain (trend of the observed rain; runs before 11.11.6: only from rain_from_year)
    trigger     observed > rho, projected > rho, rain >= rain threshold (each monitor on or off)
    action      every action feasible now (actors.make_actors catalog: add / strengthen / change / scale back / reconvert one
                measure; land fit and reconversion window by landuse.Transitions and sim.State, cost within the total budget)
                scored by  benefit / (1 + w_cost x initial cost / cost_ref) - w_irr x irr_penalty x irreversible
                           - w_lead x lead_penalty x lead / lead_ref,
                benefit = projected ratio of the current state - projected ratio with the action, horizon years after its
                completion; the k best with benefit > min_benefit and score > 0 start (sim.State: lead time, budget and
                deferral, change cost)
    scale back  when nothing triggers and observed and projected < lambda: the reversible measure with the largest annual cost
                (not changed in the last two decisions)
The number of paths and branches is not fixed: they arise from the observations of each scenario.  A policy is judged on the
whole 2020-2100 pathway over the scenarios (coverage, cost, utility, remaining future options; intermediate states that are
not Pareto-optimal are kept).  For each initial action of the 11.9 run the best policy is searched (NSGA-II), then the
initial actions are compared (Pareto: coverage, cost, Regret, remaining options) and with the 11.9 pathways on the same
evaluation scenarios.  Reused without changes: robust (settings, scenarios, record, simulate), multi (Ctx: observation and
batched projection; actor_tree), actors (action catalog), option_space (status of the options), sim.State, explore.nsga_mixed.

Planning climate range (adaptive.planning, 11.11.6): the policies are searched, and the initial action chosen, on scenarios
whose 2100 rain / sea-level rise lie in the range the planner considers; the evaluation stays on the common wide scenarios,
split into inside / outside (above, below) the range.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import config as CF
from . import option_space as OS
from . import robust as RB
from .actors import make_actors
from .multi import Ctx, actor_tree
from ..rdm import explore as EX
from ..rdm.explore_scope import nondominated
from ..roadmap import planner as PL
from ..roadmap import run as RR
from ..rdm.uncertainty import lhs
from ..roadmap import sim as SM
from ..roadmap import world as WD
from ..roadmap.social import Social

DAPP_DIR = RB.DAPP_DIR
MONITORS = ("observed", "projected", "rain")
LOCKIN_REASONS = ("irreversible_land_use", "land_use_conflict", "irreversible_measure")   # options lost for good (11.9.5)
CLIMATE = ("rain_2100", "slr_2100")


@dataclass(frozen=True)
class Policy:
    monitor: tuple            # (observed, projected, rain) on / off
    rho: float
    rain: float
    w_cost: float
    w_irr: float
    w_lead: float
    horizon: float
    k: int
    scale_back: float         # 0 = never

    def describe(self):
        return {"monitor": [m for m, on in zip(MONITORS, self.monitor) if on], "rho": self.rho, "rain": self.rain,
                "w_cost": self.w_cost, "w_irr": self.w_irr, "w_lead": self.w_lead, "horizon": self.horizon, "k": self.k,
                "scale_back": self.scale_back}


class Codec:
    """Integer genes (explore.nsga_mixed): 3 monitor switches, rho, rain, w_cost, w_irr, w_lead, horizon, k, scale_back."""

    def __init__(self, P):
        self.G = [[0, 1]] * 3 + [P["rho_grid"], P["rain_grid"], P["weight_grid"], P["weight_grid"], P["weight_grid"],
                                 P["horizon_grid"], P["k_grid"], P["scale_back_grid"]]
        self.sizes = [len(g) for g in self.G]

    def decode(self, x):
        v = [g[int(round(float(a)))] for g, a in zip(self.G, x)]
        mon = tuple(bool(a) for a in v[:3])
        return Policy(mon if any(mon) else (True, False, False), *[float(a) for a in v[3:9]], int(v[9]), float(v[10]))

    def encode(self, p):
        near = lambda g, a: int(np.argmin([abs(float(b) - float(a)) for b in g]))  # noqa: E731
        vals = [int(m) for m in p.monitor] + [p.rho, p.rain, p.w_cost, p.w_irr, p.w_lead, p.horizon, p.k, p.scale_back]
        return [near(g, a) for g, a in zip(self.G, vals)]


def seeds(P):
    """Hand-made policies (initial population): react to the observed ratio; anticipate; follow the rain trend; cost-aware."""
    mid = P["rho_grid"][len(P["rho_grid"]) // 2]
    return [Policy((True, False, False), mid, P["rain_grid"][-1], 0.0, 0.0, 0.0, P["horizon_grid"][1], 1, 0.0),
            Policy((True, True, False), mid, P["rain_grid"][-1], 1.0, 0.0, 0.0, P["horizon_grid"][1], 1, 0.0),
            Policy((True, True, True), P["rho_grid"][0], P["rain_grid"][2], 0.5, 0.5, 0.5, P["horizon_grid"][-1], 2, 0.0),
            Policy((False, True, False), mid, P["rain_grid"][-1], 2.0, 1.0, 0.0, P["horizon_grid"][0], 1, P["scale_back_grid"][-1])]


def _noise(world, tag, t):
    """Standard normal draw of a world (common random numbers); 0 for the planner's belief."""
    import zlib
    if world.p.get("deterministic"):
        return 0.0
    return float(np.random.default_rng([world.p["seed"], zlib.crc32(tag.encode()), int(round(t * 10))]).normal())


# ------------------------------------------------------------------ the controller of one pathway
class Controller:
    def __init__(self, A, init, policy, s):
        self.A, self.init, self.p, self.s = A, tuple(init), policy, s
        self.cur = 0                       # (robust.record / option_space hooks: no node network)
        self.events, self.hist = [], {}
        self.q0 = self.fire = self.cands = None

    def levels(self, st):
        T = self.A.T
        return [int(np.argmin([abs(st.committed[i] - lv) for lv in T.levels[i]])) for i in range(len(T.V))]

    def start(self, t, st):
        T, g0 = self.A.T, self.init
        for i, li in enumerate(g0):
            if li:
                st.request(i, T.levels[i][li], "plan")
        self.events.append({"year": t, "kind": "initial", "acts": [self.A.act.catalog.index((i, li)) for i, li in enumerate(g0) if li]})

    def _proj(self, tau):
        out = {}
        for key, v0 in (("rain", 1.0), ("slr", 0.0), ("asset", 1.0), ("pop", 1.0)):
            tl, vl, sl = PL.DAPPController._trend(self.hist[key], self.A.y0, v0, self.A.window)
            out[key] = vl + sl * (tau - tl)
        if self.A.forecast_noise:                                     # forecast error of the projected rain increase
            e = _noise(self.A.worlds[self.s], f"forecast{int(tau)}", self.t)
            out["rain"] = 1 + (out["rain"] - 1) * (1 + self.A.forecast_noise * e)
        return out

    def observe(self, t, st, obs):
        A = self.A
        tm = t - A.estep / 2
        for key in ("rain", "slr", "asset", "pop"):
            self.hist.setdefault(key, []).append((tm, float(obs[key])))
        tl, vl, sl = PL.DAPPController._trend(self.hist["rain"], A.y0, 1.0, A.window)
        self.r2100 = vl + sl * (2100 - tl)
        self.obs_ratio = A.ctx.observed_ratio(A.act, st, obs, t)
        self.g = self.levels(st)
        self.t = t
        self.q0 = (self.g, self._proj(min(t + A.step + A.typical_lead, A.y1 - A.estep / 2)))
        return self.q0

    def trigger(self, st, proj_ratio):
        """Which monitored indicator fires (None = no action); scale back when quiet."""
        A, p, t = self.A, self.p, self.t
        self.proj_ratio = proj_ratio
        if st.queue:                                                  # earlier actions still wait for budget
            return None
        if p.monitor[0] and t >= A.from_year and self.obs_ratio is not None and self.obs_ratio > p.rho:
            return "observed"
        if p.monitor[1] and proj_ratio > p.rho:
            return "projected"
        if p.monitor[2] and t >= A.rain_from and self.r2100 >= p.rain:
            return "rain"
        if p.scale_back and self.obs_ratio is not None and self.obs_ratio < p.scale_back and proj_ratio < p.scale_back:
            return "scale_back"
        return None

    def candidates(self, st):
        """Rows to project: (levels, projection) of the current state and of every feasible up action at its horizon."""
        A, T, g, t = self.A, self.A.T, self.g, self.t
        left = A.rc["budget"]["total_oku"] - st.spent
        rows, cand = [], []
        cur_rows = {}
        for k, (i, li) in enumerate(A.act.catalog):
            if li <= g[i] or (T.V[i]["type"] == "relocate" and g[i] == 2):
                continue
            if T.V[i]["type"] == "paddy_dam" and T.levels[i][li] > T.paddy_available(g, T.V[i]["zone"]) + 1e-9:
                continue                                              # afforested paddies (land)
            cost = A.space.option_cost(i, li, g[i])
            if cost > left + 1e-9 or t + A.act.lead[k] > A.y1:
                continue
            g2 = list(g); g2[i] = li
            g2 = list(T.repair(g2))
            tau = min(t + A.act.lead[k] + self.p.horizon, A.y1 - A.estep / 2)
            pj = self._proj(tau)
            if tau not in cur_rows:
                cur_rows[tau] = len(rows); rows.append((g, pj))
            cand.append((k, cost, cur_rows[tau], len(rows))); rows.append((g2, pj))
        self.cands = cand
        return rows

    def choose(self, st, ratios, why):
        """Score the candidates, request the k best (re-checking land after each), log the decision."""
        A, P, p, T = self.A, self.A.P, self.p, self.A.T
        sc = []
        for k, cost, jc, ja in self.cands:
            ben = ratios[jc] - ratios[ja]
            if ben <= P["min_benefit"]:
                continue
            kind = A.act_kind[k]
            irr = 0.0 if A.rc["measures"][kind].get("reversible") else (0.5 if kind == "forest" else 1.0)
            s = (ben / (1 + p.w_cost * cost / P["cost_ref_oku"]) - p.w_irr * P["irr_penalty"] * irr
                 - p.w_lead * P["lead_penalty"] * A.act.lead[k] / P["lead_ref_years"])
            if s > 0:                                                 # cost / irreversibility / lead time may veto
                sc.append((s, k, ben, cost))
        sc.sort(reverse=True)
        g = list(self.g)
        n = 0
        for s, k, ben, cost in sc:
            if n >= p.k:
                break
            i, li = A.act.catalog[k]
            if li <= g[i] or (T.V[i]["type"] == "paddy_dam" and T.levels[i][li] > T.paddy_available(g, T.V[i]["zone"]) + 1e-9):
                continue
            g2 = list(g); g2[i] = li
            g2 = list(T.repair(g2))
            for j in range(len(g)):
                if g2[j] != g[j]:
                    st.request(j, T.levels[j][g2[j]], "trigger")
            g = g2
            n += 1
            self._log(k, why, ben, cost, score=s)

    def scale_back(self, st):
        A, T, g, t = self.A, self.A.T, self.g, self.t
        best = None
        for k, (i, li) in enumerate(A.act.catalog):
            if li != 0 or g[i] == 0 or not st.can_decrease(i, t):
                continue
            if any(a.t_decide > t - 2 * A.step + 1e-9 for a in st.by_var[i]):
                continue                                              # changed in the last 20 years: no back and forth
            if T.V[i]["type"] == "forest":
                continue                                              # reconversion is not a saving
            save = T.costs[i][g[i]][1]
            if best is None or save > best[0]:
                best = (save, k, i)
        if best and best[0] > 0:
            st.request(best[2], 0.0, "scale_back")
            self._log(best[1], "scale_back", 0.0, 0.0)

    def _log(self, k, why, ben, cost, score=None):
        self.events.append({"year": self.t, "kind": "act", "a": int(k), "branch": 0, "reason": why,
                            "obs_ratio": None if self.obs_ratio is None else round(float(self.obs_ratio), 3),
                            "projected": round(float(self.proj_ratio), 3), "rain2100": round(float(self.r2100), 3),
                            "lead": round(float(self.A.act.lead[k]), 1), "benefit": round(float(ben), 4), "cost": round(float(cost), 1),
                            "score": None if score is None else round(float(score), 4)})


class Model:
    """Everything the policies need: the 11.9 run (settings, Transitions, surrogate, budget), the basin actor (catalog,
    observation, projection; multi.Ctx), the option space (state-based) and the worlds (evaluation + search)."""

    def __init__(self, cfg, net, worlds):
        AD = cfg["adaptive"]
        self.net, self.rc, self.T, self.phys, self.ref = net, net.rc, net.T, net.phys, net.ref
        self.P = AD["policy"]
        y = net.rc["years"]
        self.y0, self.y1, self.step, self.estep = y["start"], y["end"], y["decision_step"], y["eval_step"]
        self.window = net.rc["detect"].get("trend_window", 3)
        self.from_year = net.R["targets"]["from_year"]
        self.rain_from = self.P.get("rain_from_year") or self.y0     # abolished (11.11.6); kept for the runs that set it
        self.forecast_noise = float((AD.get("sensitivity") or {}).get("forecast_noise", 0.0))
        tg = {k: net.R["targets"][k] for k in ("damage_ratio", "pop_ratio")}
        MA = {"actors": {"流域": {"label": "流域", "measures": [v["name"] for v in net.T.V], "metric": "basin", "targets": tg,
                                "budget": dict(net.rc["budget"]), "weights": dict(net.R["weights"])}},
              "subsidy": {}, "options_year": 2060, "strategy": {}}
        self.ctx = Ctx(net.cfg, net.R, MA, net.rc, net.phys, net.ref, net.T, worlds)
        self.act = self.ctx.actors[0]
        self.act_kind = [net.T.kind(i, li) if li else net.T.kind(i, len(net.T.levels[i]) - 1) for i, li in self.act.catalog]
        self.typical_lead = float(np.median(self.act.lead))
        self.worlds = worlds
        self.space = OS.OptionSpace(net, network=False)


# ------------------------------------------------------------------ simulation (lock step, two projection rounds per decision)
def simulate(M, jobs, hook=None):
    """jobs [(initial levels, Policy, scenario)] -> [(state, controller)]."""
    rc, phys = M.rc, M.phys
    sims = []
    for g0, pol, s in jobs:
        w = M.worlds[s]
        st = SM.State(rc, phys, w, Social(rc, w.p["social"], {}))
        st.allow_decrease = True
        sims.append((st, Controller(M, g0, pol, s)))
    for t in SM.decision_years(rc):
        q, who = [], []
        for st, c in sims:
            st.settle(t)
            if hook:
                hook("pre", t, st, c)
            if t == M.y0:
                c.start(t, st)
            else:
                q.append(c.observe(t, st, SM.observe(st, t, M.ref, rc))); who.append((st, c))
        r0 = M.ctx.project(q, [M.act] * len(q))
        rows, fired = [], []
        for (st, c), r in zip(who, r0):
            why = c.trigger(st, r)
            if why == "scale_back":
                c.scale_back(st)
            elif why:
                a = len(rows)
                rows += c.candidates(st)
                fired.append((st, c, why, a, len(rows)))
        R = M.ctx.project(rows, [M.act] * len(rows))
        for st, c, why, a, b in fired:
            c.choose(st, R[a:b], why)
        for st, c in sims:
            st.process(t)
            if hook:
                hook("post", t, st, c)
            st.add_periods(t, t + M.step)
        SM.evaluate([x[0] for x in sims], phys)
        for st, _ in sims:
            st.accrue(t + M.step)
    return sims


def run_jobs(M, jobs, full=False, chunk=300):
    """Records (robust.record: met, utility, cost, series, cumulative cost) and the option status after each decision."""
    out, years = [], SM.decision_years(M.rc)
    for a in range(0, len(jobs), chunk):
        rec = OS.Recorder(M.space)
        sims = simulate(M, jobs[a:a + chunk], hook=rec)
        for st, c in sims:
            r = RB.record(M.net, st, c)
            post = np.array([rec.snap[id(c)]["post"][t] for t in years])
            r["status"] = post
            r["remaining"] = np.isin(post, (OS.FEASIBLE, OS.CONDITIONAL)).sum(1) / max(int(M.space.base.sum()), 1)
            r["n_acts"] = sum(e["kind"] == "act" for e in c.events)
            if not full:
                r.pop("events"); r.pop("series")
            out.append(r)
    return out


def summarise(recs):
    return {"met": np.array([r["met"] for r in recs]), "cost": np.array([r["cost_pv"] for r in recs]),
            "U": np.array([r["U"] for r in recs]), "remaining": np.array([r["remaining"].mean() for r in recs])}


def objectives(m):
    return [-float(m["met"].mean()), float(m["cost"].mean()), -float(m["U"].mean()), -float(m["remaining"].mean())]


def search(M, g0, scen, AD, seed, log):
    """NSGA-II over the policies for one initial action on the search scenarios; returns {Policy: arrays}, chosen."""
    codec = Codec(AD["policy"])
    S = AD["search"]
    space = EX.Space([SimpleNamespace(continuous=False, options=range(n), type="g") for n in codec.sizes], {})
    memo = {}

    def evaluate(xs):
        ps = [codec.decode(x) for x in xs]
        new = [p for p in dict.fromkeys(ps) if p not in memo]
        if new:
            recs = run_jobs(M, [(g0, p, s) for p in new for s in scen])
            for k, p in enumerate(new):
                memo[p] = summarise(recs[k * len(scen):(k + 1) * len(scen)])
        return [objectives(memo[p]) for p in ps]
    EX.nsga_mixed(space, evaluate, S["pop"], S["gens"], np.random.default_rng(seed), S["p_mut"],
                  init=[codec.encode(p) for p in seeds(AD["policy"])], log=lambda m: None)
    cov = {p: m["met"].mean() for p, m in memo.items()}
    top = max(cov.values())
    ok = [p for p in memo if cov[p] >= top - AD["choose"]["coverage_tolerance"] - 1e-12]
    chosen = min(ok, key=lambda p: (memo[p]["cost"].mean(), -memo[p]["U"].mean(), -memo[p]["remaining"].mean()))
    return memo, chosen


# ------------------------------------------------------------------ planning climate range (11.11.6)
def planning_spec(AD, F):
    """adaptive.planning -> {range, label, rain_2100: [lo, hi], slr_2100: [lo, hi]} (None: the 11.11 setting).  A preset fills
    the bounds from adaptive.planning_ranges; "custom" takes them as given, inside the factor range of docs/13."""
    pl = AD.get("planning")
    if not pl:
        return None
    key = pl.get("range", "custom")
    pre = AD["planning_ranges"].get(key) if key != "custom" else None
    if key != "custom" and not isinstance(pre, dict):
        raise ValueError(f"unknown planning range {key}")
    out = {"range": key, "label": pre["label"] if pre else "Custom"}
    for k in CLIMATE:
        lo, hi = [float(v) for v in (pre[k] if pre else pl[k])]
        if not F[k]["low"] - 1e-9 <= lo <= hi <= F[k]["high"] + 1e-9:
            raise ValueError(f"planning {k} must satisfy {F[k]['low']} <= lo <= hi <= {F[k]['high']}")
        out[k] = [lo, hi]
    return out


def planning_scenarios(rc, R, n, seed, plan):
    """Search scenarios: robust.scenarios (LHS over the same groups and seed) with the 2100 rain and sea-level rise drawn inside
    the planning range (common random numbers across ranges; the other factors are as for the 11.11 run)."""
    F = WD.factors(rc)
    names = list(F)
    Z = np.tile(WD.unit_nominal(F), (n, 1))
    vary = [j for j, k in enumerate(names) if F[k]["group"] in R["scenarios"]["groups"]]
    Z[:, vary] = lhs(n, len(vary), np.random.default_rng(seed + 601))
    for k in CLIMATE:
        j, f = names.index(k), F[k]
        lo, hi = [(v - f["low"]) / ((f["high"] - f["low"]) or 1.0) for v in plan[k]]
        Z[:, j] = lo + Z[:, j] * (hi - lo)
    return [WD.make_world(rc, z, seed * 1000 + s) for s, z in enumerate(Z)]


def in_range(clim, plan):
    """Evaluation scenarios (2100 rain, sea-level rise) -> 0 inside the planning range, +1 above (a harsher climate in either),
    -1 below.  Without a planning range all are inside."""
    if not plan:
        return np.zeros(len(clim), int)
    lo = np.array([plan[k][0] for k in CLIMATE]) - 1e-9
    hi = np.array([plan[k][1] for k in CLIMATE]) + 1e-9
    return np.where((clim > hi).any(1), 1, np.where((clim < lo).any(1), -1, 0))


def pick(table, M, tol, pareto, recommended):
    """Pareto (coverage, mean cost, max Regret, remaining options) of the initial actions and the recommended one: the
    cheapest within the coverage tolerance of the best Pareto one."""
    Fm = np.array([[-m["coverage"], m["mean_cost"], m["max_regret"], -m["remaining_mean"]] for m in M])
    nd = nondominated(Fm)
    for r, ok in zip(table, nd):
        r[pareto] = bool(ok)
    par = [k for k in range(len(table)) if nd[k]]
    top = max(M[k]["coverage"] for k in par)
    k = min((k for k in par if M[k]["coverage"] >= top - tol - 1e-12), key=lambda k: M[k]["mean_cost"])
    for j, r in enumerate(table):
        r[recommended] = j == k


# ------------------------------------------------------------------ pipeline
def option_ts(M, recs, years):
    nb = max(int(M.space.base.sum()), 1)
    P = np.array([r["status"] for r in recs])                          # (S, Y, O)
    out = {"remaining": {"mean": (np.isin(P, (OS.FEASIBLE, OS.CONDITIONAL)).sum(2) / nb).mean(0).round(4).tolist(),
                         "p10": np.quantile(np.isin(P, (OS.FEASIBLE, OS.CONDITIONAL)).sum(2) / nb, 0.1, 0).round(4).tolist(),
                         "p90": np.quantile(np.isin(P, (OS.FEASIBLE, OS.CONDITIONAL)).sum(2) / nb, 0.9, 0).round(4).tolist()},
           "used": ((P == OS.USED).sum(2) / nb).mean(0).round(4).tolist(),
           "conditional": ((P == OS.CONDITIONAL).sum(2) / nb).mean(0).round(4).tolist(),
           "loss": {q: ((P == OS.CODE[q]).sum(2) / nb).mean(0).round(4).tolist() for q in OS.REASONS}}
    out["met_so_far"] = np.mean([[r["first_violation"] is None or r["first_violation"] > t for t in years] for r in recs], 0).round(4).tolist()
    out["cost_cum"] = np.mean([r["cost_path"] for r in recs], 0).round(1).tolist()
    return out


def run(cfg=None, out_dir=None, phys=None, log=print, robust_dir=None):
    t0 = time.time()
    cfg = CF.load(cfg)
    AD = cfg["adaptive"]
    rdir = Path(robust_dir) if robust_dir else DAPP_DIR / AD["robust_run"]
    if not (rdir / "network.npz").exists():
        raise SystemExit(f"{rdir} がない。先に scripts/run_dapp.py --robust を実行する（11.9 の設定・シナリオ・初期施策を使う）。")
    rres = json.loads((rdir / "results.json").read_text(encoding="utf-8"))
    net, F, Z, vary = RB.load_net(rdir, phys, log=lambda m: None)
    sens = apply_sensitivity(net, AD)
    n_eval = min(int(AD.get("n_eval") or len(net.worlds)), len(net.worlds))
    S = AD["search"]
    plan = planning_spec(AD, F)
    if plan:
        AD["policy"].pop("rain_from_year", None)                               # abolished: the planning range replaces it
        search_worlds = planning_scenarios(net.rc, net.R, int(S["n_scenarios"]), cfg["seed"] + 313, plan)
    else:
        Rs = dict(net.R, scenarios=dict(net.R["scenarios"], n=int(S["n_scenarios"])))
        search_worlds = RB.scenarios(net.rc, Rs, cfg["seed"] + 313)[3]       # scenarios of their own, never the evaluation ones
    clim = np.array([[WD.values_from_unit(z, F)[k] for k in ("rain_2100", "slr_2100")] for z in Z[:n_eval]])
    where = in_range(clim, plan)                                               # evaluation scenarios: in / above / below the range
    worlds = list(net.worlds[:n_eval]) + list(search_worlds)
    M = Model(cfg, net, worlds)
    scen, scen_s = list(range(n_eval)), list(range(n_eval, len(worlds)))
    ev9 = rres["evaluation"]
    inits = ev9["inits"] if AD["inits"] == "all" else [r["node"] for r in ev9["table"] if r.get("promising")]
    out = Path(out_dir) if out_dir else DAPP_DIR / AD["name"]
    out.mkdir(parents=True, exist_ok=True)
    CF.save(cfg, out / "config.json")
    log(f"Adaptive DAPP: 初期行動 {len(inits)}（{AD['robust_run']} の初期施策）・行動の候補 {len(M.act.catalog)}・"
        f"評価シナリオ {n_eval}・探索シナリオ {len(scen_s)}・選択肢の基準集合 {int(M.space.base.sum())}")
    if plan:
        log(f"  計画時の気候の範囲 {plan['label']}: 2100年の降雨 ×{plan['rain_2100'][0]:.2f}〜{plan['rain_2100'][1]:.2f}・"
            f"海面 {plan['slr_2100'][0]:.2f}〜{plan['slr_2100'][1]:.2f} m（評価シナリオのうち範囲内 {int((where == 0).sum())}・"
            f"上 {int((where > 0).sum())}・下 {int((where < 0).sum())}）")
    years = SM.decision_years(M.rc)
    chosen, fronts, planned = {}, {}, {}
    fixed = fixed_policies(AD)
    if fixed:
        inits = [i for i in inits if i in fixed]
        log(f"  ルールは固定（{AD['fixed_from']} の推奨ルール）: 再評価だけ")
    for k, i in enumerate(inits):
        if fixed:
            chosen[i], fronts[i] = fixed[i], []
            continue
        g0 = tuple(int(a) for a in net.nodes[i])
        t1 = time.time()
        memo, p = search(M, g0, scen_s, AD, cfg["seed"] + 811 + k, log)
        chosen[i], planned[i] = p, memo[p]
        F_ = np.array([objectives(m) for m in memo.values()])
        keys = list(memo)
        fronts[i] = [{"policy": keys[j].describe(), "coverage": float(memo[keys[j]]["met"].mean()), "mean_cost": float(memo[keys[j]]["cost"].mean()),
                      "remaining": float(memo[keys[j]]["remaining"].mean()), "chosen": keys[j] == p} for j in np.flatnonzero(nondominated(F_))]
        log(f"  {k + 1}/{len(inits)} 初期行動 #{i}: 方策 {len(memo)}・推奨 カバー率 {memo[p]['met'].mean():.0%}（探索）（{time.time() - t1:.0f} s）")
    log("評価（評価シナリオ）: 適応方策 と 11.9 の経路")
    A = {i: run_jobs(M, [(tuple(int(a) for a in net.nodes[i]), chosen[i], s) for s in scen], full=True) for i in inits}
    # 11.9 pathways of the same initial actions on the same scenarios, with the same (state-based) option space
    rec9 = OS.Recorder(M.space)
    st9, c9 = RB.simulate(net, inits, scen, hook=rec9)
    B = {}
    for k, i in enumerate(inits):
        B[i] = []
        for st, c in zip(st9[k * n_eval:(k + 1) * n_eval], c9[k * n_eval:(k + 1) * n_eval]):
            r = RB.record(net, st, c)
            r["status"] = np.array([rec9.snap[id(c)]["post"][t] for t in years])
            r["remaining"] = np.isin(r["status"], (OS.FEASIBLE, OS.CONDITIONAL)).sum(1) / max(int(M.space.base.sum()), 1)
            r["n_acts"] = len(c.events) - 1
            B[i].append(r)
    best = np.max([[r["U"] for r in R] for R in list(A.values()) + list(B.values())], axis=0)
    lock = {r["node"]: r["lockin"].get("options") for r in ev9["table"]}
    n50 = years.index(2050) if 2050 in years else len(years) // 2
    part = {"in": where == 0, "out": where != 0, "above": where > 0, "below": where < 0}

    def row(R):
        m = summarise(R)
        reg = best - m["U"]
        O = option_ts(M, R, years)
        sub = {f"coverage_{k}": (float(m["met"][w].mean()) if w.any() else None) for k, w in part.items()}
        sub.update({f"max_regret_{k}": (float(reg[w].max()) if w.any() else None) for k, w in part.items() if k in ("in", "out")})
        sub.update({f"mean_cost_{k}": (float(m["cost"][w].mean()) if w.any() else None) for k, w in part.items() if k in ("in", "out")})
        lk = [sum(O["loss"][q][n] for q in LOCKIN_REASONS) for n in range(len(years))]
        return {"coverage": float(m["met"].mean()), "mean_cost": float(m["cost"].mean()), "max_regret": float(reg.max()),
                "mean_regret": float(reg.mean()), "mean_U": float(m["U"].mean()), **sub,
                "lockin_2050": float(lk[n50]), "lockin_end": float(lk[-1]),
                "mean_damage_ratio": float(np.mean([np.mean([p[1] for p in r["series"] if p[0] >= 2030]) for r in R])) if "series" in R[0] else None,
                "remaining_2050": O["remaining"]["mean"][n50], "remaining_mean": float(np.mean(O["remaining"]["mean"])),
                "option_loss_2050": float(sum(O["loss"][q][n50] for q in OS.REASONS)), "n_acts": float(np.mean([r["n_acts"] for r in R])),
                "options": O}
    table = []
    for i in inits:
        a, b = row(A[i]), row(B[i])
        table.append({"node": i, "label": RB.node_info(net, i)["label"], "initial_cost": float(net.mat["initial"][0, i]),
                      "lockin": lock.get(i), "adaptive": a, "robust": b, "policy": chosen[i].describe(), "front": fronts[i]})
    tol = AD["choose"]["coverage_tolerance"]
    pick(table, [r["adaptive"] for r in table], tol, "pareto", "recommended")       # on the evaluation scenarios (11.11)
    if plan and len(planned) == len(inits):
        # the planner's view: the initial actions compared on the planning scenarios (search), never on the evaluation set
        bestp = np.max([planned[i]["U"] for i in inits], axis=0)
        for r in table:
            m = planned[r["node"]]
            r["planning"] = {"coverage": float(m["met"].mean()), "mean_cost": float(m["cost"].mean()),
                             "max_regret": float((bestp - m["U"]).max()), "remaining_mean": float(m["remaining"].mean())}
        for r in table:                                                       # hindsight: the choice on the evaluation set
            r["pareto_eval"], r["recommended_eval"] = r.pop("pareto", False), r.pop("recommended", False)
        pick(table, [r["planning"] for r in table], tol, "pareto", "recommended")
    # pathway map (trie over the actions; branches arise from the observations) and the decision rules seen at the branches
    trees = {}
    for i in inits:
        recs = [{"actors": {"流域": {"events": r["events"], "met": r["met"]}}} for r in A[i]]
        trees[str(i)] = actor_tree(SimpleNamespace(y0=M.y0), M.act, recs)
    res = {"mode": "adaptive", "name": AD["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg, "robust_run": AD["robust_run"],
           "years": years, "settings": {"policy": AD["policy"], "search": S, "targets": net.R["targets"], "budget": net.rc["budget"]},
           "sensitivity": sens, "gui_inputs": AD.get("gui_inputs"), "planning": plan,
           "climate": {"in": int(part["in"].sum()), "above": int(part["above"].sum()), "below": int(part["below"].sum())},
           "catalog": M.act.labels, "catalog_lead": [round(x, 1) for x in M.act.lead], "n_eval": n_eval, "n_search": len(scen_s),
           "n_base": int(M.space.base.sum()), "reasons": list(OS.REASONS), "reason_labels": OS.LABELS,
           "worlds": [{"rain_2100": float(c[0]), "slr_2100": float(c[1]), "range": int(k)} for c, k in zip(clim, where)],
           "table": table, "trees": trees,
           "records": {str(i): [{k: r[k] for k in ("s", "met", "first_violation", "U", "cost_pv", "series", "cost_path", "events")}
                                | {"remaining": np.round(r["remaining"], 4).tolist()} for r in A[i]] for i in inits}}
    res["elapsed_s"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=RR._json), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary(res), ensure_ascii=False, default=RR._json), encoding="utf-8")
    log(f"完了: {out / 'results.json'}（{res['elapsed_s']:.0f} s）")
    return res


# ------------------------------------------------------------------ sensitivity: targets and information quality
def apply_sensitivity(net, AD):
    """Targets and information quality (adaptive.sensitivity) applied to the loaded 11.9 run, so that both methods see them:
    targets (met, triggers, 11.9 moves, the utility's penalty), observation noise (sim.observe: damage / exposed population;
    rain, sea level, exposure - the trends), trend window of the projections.  Returns the effective values."""
    s = AD.get("sensitivity") or {}
    mon, rc = net.rc["monitoring"], net.rc
    base = {k: mon.get(k, 0.0) for k in ("risk_noise", "rain_noise", "slr_noise", "exposure_noise")}   # 0 since 11.12 (no observation error)
    mon["risk_noise"] = base["risk_noise"] * float(s.get("noise_damage", 1.0))
    for k in ("rain_noise", "slr_noise", "exposure_noise"):
        mon[k] = base[k] * float(s.get("noise_climate", 1.0))
    if s.get("trend_window"):
        rc["detect"]["trend_window"] = int(s["trend_window"])
    tg = {k: float(v) for k, v in (s.get("targets") or {}).items() if v is not None}
    if tg:
        net.R["targets"] = dict(net.R["targets"], **tg)
        th = rc["regimes"]["robust"]["thresholds"]
        for k in ("damage_ratio", "pop_ratio"):
            if k in tg:
                th[k] = tg[k]
    return {"targets": {k: net.R["targets"][k] for k in ("damage_ratio", "pop_ratio", "from_year")},
            "noise_damage": float(s.get("noise_damage", 1.0)), "noise_climate": float(s.get("noise_climate", 1.0)),
            "monitoring": {k: mon[k] for k in base}, "trend_window": int(rc["detect"].get("trend_window", 3)),
            "rain_from_year": AD["policy"].get("rain_from_year"), "forecast_noise": float(s.get("forecast_noise", 0.0)),
            "mode": "fixed" if AD.get("fixed_from") else "search"}


def fixed_policies(AD):
    """The recommended rule of every initial action of an earlier run (mode "fixed": re-evaluate, no search)."""
    if not AD.get("fixed_from"):
        return None
    src = Path(AD["fixed_from_dir"]) if AD.get("fixed_from_dir") else DAPP_DIR / AD["fixed_from"]
    res = json.loads((src / "results.json").read_text(encoding="utf-8"))
    return {r["node"]: Policy(tuple(m in r["policy"]["monitor"] for m in MONITORS), r["policy"]["rho"], r["policy"]["rain"], r["policy"]["w_cost"],
                              r["policy"]["w_irr"], r["policy"]["w_lead"], r["policy"]["horizon"], int(r["policy"]["k"]), r["policy"]["scale_back"])
            for r in res["table"]}


def gui_config(cfg, inputs, mode="fixed", size="fast"):
    """Settings of a GUI recalculation from a displayed run: targets {damage_ratio, pop_ratio, from_year}, noise_damage /
    noise_climate (x the observation noise), trend_window, forecast_noise, planning {range, rain_2100, slr_2100}; mode "fixed"
    re-evaluates the displayed run's recommended rules, "search" searches again at a size (adaptive.gui.sizes) for the
    planning climate range (default Low-High).  Named <base>_gui_<hash>."""
    cfg = CF.load(copy.deepcopy(cfg))
    AD = cfg["adaptive"]
    base = (AD.get("gui_inputs") or {}).get("base", AD["name"])
    src = (AD.get("gui_inputs") or {}).get("rules_from", AD["name"]) if AD.get("fixed_from") else AD["name"]
    s = {}
    tg = {k: inputs.get("targets", {}).get(k) for k in ("damage_ratio", "pop_ratio", "from_year")}
    for k in ("damage_ratio", "pop_ratio"):
        if tg[k] is not None and not 0.3 <= float(tg[k]) <= 5:
            raise ValueError(f"{k} must be in [0.3, 5]")
    if tg["from_year"] is not None and not 2020 <= int(tg["from_year"]) <= 2090:
        raise ValueError("from_year must be in [2020, 2090]")
    t0 = cfg["robust"]["targets"]                                              # values equal to the run's own: not a change
    s["targets"] = {k: v for k, v in tg.items() if v is not None and not (k in t0 and abs(float(v) - float(t0[k])) < 1e-9)}
    for k, lo, hi in (("noise_damage", 0, 10), ("noise_climate", 0, 10), ("forecast_noise", 0, 2)):
        if inputs.get(k) is not None:
            if not lo <= float(inputs[k]) <= hi:
                raise ValueError(f"{k} must be in [{lo}, {hi}]")
            s[k] = float(inputs[k])
    if inputs.get("trend_window") is not None:
        if not 2 <= int(inputs["trend_window"]) <= 8:
            raise ValueError("trend_window must be in [2, 8]")
        s["trend_window"] = int(inputs["trend_window"])
    AD["sensitivity"] = s
    if mode == "fixed":                       # the rules (and the source's rain_from_year, if any) as they were searched
        AD["fixed_from"] = src
    elif mode == "search":
        AD.pop("fixed_from", None)
        AD["policy"].pop("rain_from_year", None)                              # abolished (11.11.6)
        if size not in AD["gui"]["sizes"]:
            raise ValueError(f"unknown size {size}")
        cfg["adaptive"] = AD = CF._merge(AD, AD["gui"]["sizes"][size])
        pl = inputs.get("planning") or {"range": "low_high"}
        AD["planning"] = {"range": pl.get("range", "custom")} | ({k: [float(v) for v in pl[k]] for k in CLIMATE} if pl.get("range") == "custom" else {})
        planning_spec(AD, WD.factors(RB.model_cfg(cfg, cfg["robust"])))                                # validate
    else:
        raise ValueError(f"unknown mode {mode}")
    key = json.dumps({"base": base, "mode": mode, "size": size if mode == "search" else None, "rules": AD.get("fixed_from"), "s": s,
                      "rain_from": AD["policy"].get("rain_from_year"), "n_eval": AD.get("n_eval"),
                      **({"planning": AD.get("planning")} if mode == "search" else {})}, sort_keys=True)
    AD["gui_inputs"] = {"base": base, "mode": mode, "size": size if mode == "search" else None, "rules_from": AD.get("fixed_from")}
    AD["name"] = f"{base}_gui_{hashlib.sha1(key.encode()).hexdigest()[:8]}"
    return cfg


def recalculate(cfg, phys=None, progress=None, out_dir=None, robust_dir=None):
    """Run with GUI settings, reporting progress(fraction, message)."""
    n = {"f": 0.0}

    def log(m):
        m = m.strip()
        if "初期行動" in m and "/" in m and "推奨" in m:
            try:
                a, b = m.split("初期行動")[0].split("/")
                n["f"] = max(n["f"], 0.05 + 0.75 * int(a) / int(b))
            except ValueError:
                pass
        elif m.startswith("評価"):
            n["f"] = max(n["f"], 0.82)
        if progress:
            progress(n["f"], m)
    run(cfg, out_dir=out_dir, phys=phys, log=log, robust_dir=robust_dir)
    return {"name": cfg["adaptive"]["name"]}


def recalculate_many(cfgs, phys=None, progress=None, root=None):
    """Several GUI runs in a row (e.g. every planning range for one target); runs already done are skipped."""
    todo = [c for c in cfgs if not (Path(root or DAPP_DIR) / c["adaptive"]["name"] / "summary.json").exists()]
    for k, c in enumerate(todo):
        def sub(f, m, k=k):
            if progress:
                progress((k + f) / len(todo), f"{k + 1}/{len(todo)} {(c['adaptive'].get('planning') or {}).get('range', '')}: {m}")
        recalculate(c, phys, sub, out_dir=Path(root) / c["adaptive"]["name"] if root else None)
    return {"name": cfgs[0]["adaptive"]["name"], "names": [c["adaptive"]["name"] for c in cfgs]}


def summary(res):
    T = res["table"]
    rec = next((r for r in T if r.get("recommended")), T[0])
    gi = res.get("gui_inputs") or {}
    st = res["settings"]
    sens = res.get("sensitivity") or {"targets": {k: st["targets"][k] for k in ("damage_ratio", "pop_ratio", "from_year")},  # runs before 11.11.5
                                      "noise_damage": 1.0, "noise_climate": 1.0, "trend_window": 3, "forecast_noise": 0.0,
                                      "rain_from_year": st["policy"].get("rain_from_year"), "mode": "search"}
    a = rec["adaptive"]
    ev = next((r for r in T if r.get("recommended_eval")), None)
    return {"name": res["name"], "created": res["created"], "base": gi.get("base", res["name"]), "mode": sens.get("mode", "search"),
            "size": gi.get("size"), "n_eval": res["n_eval"], "n_search": res["n_search"], "sensitivity": sens,
            "planning": res.get("planning"), "climate": res.get("climate"),
            "recommended": {"node": rec["node"], "label": rec["label"], "initial_cost": rec["initial_cost"], "coverage": a["coverage"],
                            "mean_cost": a["mean_cost"], "robust_coverage": rec["robust"]["coverage"], "max_regret": a["max_regret"],
                            "mean_regret": a["mean_regret"], "remaining_mean": a["remaining_mean"], "remaining_2050": a["remaining_2050"],
                            "lockin": rec.get("lockin"), "lockin_2050": a.get("lockin_2050"), "lockin_end": a.get("lockin_end"),
                            **{k: a.get(k) for k in ("coverage_in", "coverage_out", "coverage_above", "coverage_below", "max_regret_in",
                                                     "max_regret_out", "mean_cost_in", "mean_cost_out")},
                            "planning": rec.get("planning"), "policy": rec["policy"]},
            "recommended_eval": ev and {"node": ev["node"], "coverage": ev["adaptive"]["coverage"], "mean_cost": ev["adaptive"]["mean_cost"]},
            "mean_coverage": {"adaptive": float(np.mean([r["adaptive"]["coverage"] for r in T])), "robust": float(np.mean([r["robust"]["coverage"] for r in T]))},
            "mean_cost": {"adaptive": float(np.mean([r["adaptive"]["mean_cost"] for r in T])), "robust": float(np.mean([r["robust"]["mean_cost"] for r in T]))},
            "mean_remaining": {"adaptive": float(np.mean([r["adaptive"]["remaining_mean"] for r in T])), "robust": float(np.mean([r["robust"]["remaining_mean"] for r in T]))},
            "coverage": {str(r["node"]): [r["adaptive"]["coverage"], r["robust"]["coverage"]] for r in T},
            "n_better": int(sum(r["adaptive"]["coverage"] > r["robust"]["coverage"] for r in T)), "n_inits": len(T)}


def summaries(base, root=None):
    """summary() of the run `base` and of its GUI recalculations, newest first."""
    out = []
    for d in Path(root or DAPP_DIR).iterdir():
        if not (d.name == base or d.name.startswith(base + "_gui_")) or not (d / "results.json").exists():
            continue
        f = d / "summary.json"
        if not f.exists() or "climate" not in json.loads(f.read_text(encoding="utf-8")):   # written before 11.11.6
            res = json.loads((d / "results.json").read_text(encoding="utf-8"))
            if res.get("mode") != "adaptive":
                continue
            f.write_text(json.dumps(summary(res), ensure_ascii=False, default=RR._json), encoding="utf-8")
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return sorted(out, key=lambda r: r["created"], reverse=True)
