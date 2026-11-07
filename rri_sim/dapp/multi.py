"""Multi-Actor Robust DAPP: conditional adaptive strategies of single measures per actor (docs/11 11.10).

    python scripts/run_dapp.py --multi [--quick] [--verify 4]

Shared with Robust DAPP (robust.py, 11.9) and docs/12-13: the exploration surrogate and cost table, the scenarios
(roadmap.world), the implementation model (roadmap.sim.State: lead times, budgets and deferral, change costs,
scaling back / reconversion, observation), land use (landuse.Transitions), NSGA-II (explore.nsga_mixed), PRIM.

  actors      国 / 上流自治体 / 下流自治体 (multi.actors): own measures, per-decade fiscal / implementation capacity
              (sim.State.budgets; per_decade_oku binding, total_oku None = no limit; the national share of subsidised
              measures, multi.subsidy), metric and target (basin damage + exposed population, or the event damage of the
              own zone), ATP and trigger, utility weights (Regret)
  strategies  actors.Strategy: initial actions, base / severe action lists, switch on the projected 2100 rain, trigger
              rho, anticipation (lead time).  Pathways are simulated per scenario; states are never pruned - a strategy
              is judged on the whole pathway (coverage, Regret, cost, lock-in, future options)
  others      the other actors' decisions are uncertain: in the search of one actor each scenario pairs a climate /
              socio-economic world with a combination of the others' exogenous rules (積極投資 / 段階投資 / 投資抑制)
  cases       Individual: per actor, NSGA-II over its strategy against that uncertainty (a best response to a mixture
              of rules), own objectives.  Joint: NSGA-II over all actors' strategies together, objectives of the basin
              (every actor's target), the actors' budgets and measures kept.  Single: one actor with all the measures and
              the summed per-decade capacity, the same objectives and action throughput.  Searched on search.n_scenarios
              scenarios of their own (the same LHS), never on the evaluation scenarios
  compare     the three cases and every combination of the rules on the evaluation scenarios: robustness per actor and
              for the basin (every target met), Regret, cost, lock-in, options, budget use per decade;
              coordination_gap = Joint - Individual, institutional_fragmentation_gap = Single - Joint
Game-theoretic extensions (later): best_response() with the others fixed instead of a rule mixture gives a Best
Response iteration (Nash); multi.subsidy (the national share) is the leader's variable of a Stackelberg game.
Coverage is a share of a scenario sample, not a probability.  Writes data/processed/dapp/<multi.name>/results.json.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import config as CF
from . import robust as RB
from .actors import RULES, Actor, Codec, Controller, Strategy, describe, make_actors, metric_of, options, ranking, rules
from ..data import PROCESSED
from ..rdm import explore as EX
from ..rdm.explore_scope import nondominated
from ..roadmap import metrics as MT
from ..roadmap import run as RR
from ..roadmap import sim as SM
from ..roadmap import world as WD
from ..roadmap.social import Social

DAPP_DIR = RB.DAPP_DIR


def settings(cfg, single=False):
    """The Robust DAPP block with the Multi-Actor overrides, and the multi block.  single: one decision maker with one pooled
    per-decade capacity (the sum of the actors'; total: the sum, or None if any is None) paying for every measure; the actors'
    measure groups stay as its sub-strategies and the actors' targets as the criteria."""
    MA = cfg["multi"]
    R = CF._merge(cfg["robust"], MA.get("robust_overrides", {}))
    R["scenarios"]["n_eval"] = int(R["scenarios"]["n"])                 # evaluation scenarios first, then the search's own
    R["scenarios"]["n"] = R["scenarios"]["n_eval"] + int(MA["search"]["n_scenarios"])
    for a in MA["actors"].values():                                    # the basin target is the basin actor's (国)
        if a["metric"] == "basin":
            R["targets"] = dict(R["targets"], **{k: a["targets"][k] for k in ("damage_ratio", "pop_ratio") if k in a["targets"]})
    if single:
        tot = [a["budget"].get("total_oku") for a in MA["actors"].values()]
        MA = CF._merge(MA, {})
        MA["pool"] = {"name": MA.get("single", {}).get("label", "流域（単一主体）"),
                      "budget": {"total_oku": None if any(t is None for t in tot) else sum(tot),
                                 "per_decade_oku": sum(a["budget"]["per_decade_oku"] for a in MA["actors"].values())}}
    return R, MA


def shares_of(actors, subsidy, T):
    """Per variable {payer: share of its cost}: the owner, minus a subsidy paid by another actor (multi.subsidy)."""
    names = {a.name for a in actors}
    owner = {i: a.name for a in actors for i in a.vars}
    out = []
    for i, v in enumerate(T.V):
        o = owner.get(i)
        sub = subsidy.get(v["type"])
        if o is None:
            out.append({})
        elif sub and sub["payer"] in names and sub["payer"] != o:
            out.append({o: 1 - sub["share"], sub["payer"]: sub["share"]})
        else:
            out.append({o: 1.0})
    return out


class Ctx:
    """Everything a simulation needs: model, surrogate, scenarios, actors, payers."""

    def __init__(self, cfg, R, MA, rc, phys, ref, T, worlds):
        self.cfg, self.R, self.MA, self.rc, self.phys, self.ref, self.T, self.worlds = cfg, R, MA, rc, phys, ref, T, worlds
        y = rc["years"]
        self.y0, self.y1, self.step, self.estep = y["start"], y["end"], y["decision_step"], y["eval_step"]
        self.from_year = R["targets"]["from_year"]
        self.actors = make_actors(MA, T, rc)
        self.by = {a.name: a for a in self.actors}
        self.names = [a.name for a in self.actors]
        own = shares_of(self.actors, MA.get("subsidy", {}), T)               # cost attributed to the actors (criteria)
        self.Pc = np.array([[own[i].get(a, 0.0) for i in range(len(T.V))] for a in self.names])
        self.pool = MA.get("pool")
        if self.pool:                                                        # Single: one payer with the pooled capacity
            self.budgets = {self.pool["name"]: self.pool["budget"]}
            self.shares = [{self.pool["name"]: 1.0} if o else {} for o in own]
        else:
            self.budgets = {a.name: a.budget for a in self.actors}
            self.shares = own
        self.payers = list(self.budgets)
        self.n_decades = len(SM.decision_years(rc))
        self.env0 = WD.belief_world(rc).env(self.y0)
        self.option_year = int(MA["options_year"])
        self.switch_from = float(MA["strategy"].get("switch_from_year", self.y0))

    def payer(self, i):
        return self.shares[i]

    def observed_ratio(self, actor, st, obs, t):
        """The actor's own metric in the last evaluated period / target (noisy, as sim.observe)."""
        if not st.periods:
            return None
        tg = actor.targets
        if actor.metric == "basin":
            return max(obs["damage_ratio"] / tg["damage_ratio"], obs["pop_ratio"] / tg["pop_ratio"])
        z = actor.metric
        noise = self.rc["monitoring"]["risk_noise"] * st.world.noise("risk_" + z, t)
        return st.periods[-1]["damage_zone"][z] / self.ref["dz:" + z] * (1 + noise) / tg["damage_ratio"]

    def project(self, rows, actors):
        """rows [(levels, projected {rain, slr, asset, pop})] -> ratio / target of each actor's metric with that state in
        place under the projection (planning effects; climate clipped to the training range)."""
        if not rows:
            return []
        raw = np.array([self.T.raw(g) for g, _ in rows])
        env = {k: np.full(len(rows), v, float) for k, v in self.env0.items()}
        env["rain_factor"] = np.array([p["rain"] for _, p in rows])
        env["sea_level_rise_m"] = np.array([max(p["slr"], 0.0) for _, p in rows])
        y = self.phys.predict(raw, env)
        out = []
        for j, (a, (_, p)) in enumerate(zip(actors, rows)):
            yj = {k: np.atleast_1d(v[j]) for k, v in y.items()}
            yj["damage"] = yj["damage"] * max(p["asset"], 0.0)
            yj["pop_hazard"] = yj["pop_hazard"] * max(p["pop"], 0.0)
            for z in self.phys.zones:
                yj["dz:" + z] = yj["dz:" + z] * max(p["asset"], 0.0)
            out.append(float(metric_of(a, yj, self.ref)[0]))
        return out

    def metric_fn(self):
        return lambda a, y: metric_of(a, y, self.ref)

    def budget_scale(self, a):
        """Cost normaliser of a utility: the total budget, or the per-decade capacity over the horizon."""
        b = a.budget
        return b["total_oku"] if b.get("total_oku") is not None else b["per_decade_oku"] * self.n_decades


# ------------------------------------------------------------------ simulation (all jobs in lock step)
def simulate(ctx, jobs):
    """jobs [(joint {actor: Strategy}, scenario)] -> [(state, controllers, options {"mid", "end"})]."""
    rc, phys = ctx.rc, ctx.phys
    sims = []
    for joint, s in jobs:
        w = ctx.worlds[s]
        st = SM.State(rc, phys, w, Social(rc, w.p["social"], {}))
        st.allow_decrease, st.budgets, st.payer = True, ctx.budgets, ctx.payer
        sims.append((st, [Controller(ctx, ctx.by[a], S, s) for a, S in joint.items()], {}))
    for t in SM.decision_years(rc):
        rows, who = [], []
        for n, (st, cs, _) in enumerate(sims):
            st.settle(t)
            if t == ctx.y0:
                for c in cs:
                    c.start(t, st)
                continue
            obs = SM.observe(st, t, ctx.ref, rc)
            for c in cs:
                q = c.prepare(t, st, obs)
                if q is not None:
                    rows.append(q); who.append(c)
        proj = dict(zip(map(id, who), ctx.project(rows, [c.a for c in who])))
        for st, cs, opt in sims:
            if t > ctx.y0:
                for c in cs:
                    c.decide(t, st, proj.get(id(c)))
            if t == ctx.option_year:
                opt["mid"] = {c.a.name: options(ctx, c.a, st, t) for c in cs}
            st.process(t)
            st.add_periods(t, t + ctx.step)
        SM.evaluate([s[0] for s in sims], phys)
        for st, _, _ in sims:
            st.accrue(t + ctx.step)
    for st, cs, opt in sims:
        opt["end"] = {c.a.name: options(ctx, c.a, st, ctx.y1) for c in cs}
    return sims


def own_ratio(ctx, a, p):
    if a.metric == "basin":
        return p["damage"] / ctx.ref["damage"], p["pop_risk"] / ctx.ref["pop_hazard"]
    return p["damage_zone"][a.metric] / ctx.ref["dz:" + a.metric], None


def met_of(ctx, a, st):
    tg = a.targets
    viol = []
    for p in st.periods:
        if p["t"] < ctx.from_year:
            continue
        rd, rp = own_ratio(ctx, a, p)
        if rd > tg["damage_ratio"] or (rp is not None and rp > tg["pop_ratio"]):
            viol.append(p["t"])
    return not viol, (viol[0] if viol else None)


def actor_utility(ctx, a, st, cost_y):
    """Utility of actor a (weights multi.actors.<a>.weights): own damage (and exposed population for the basin metric),
    Ecology, agriculture, a penalty for periods above the target, minus the own cost relative to the own budget."""
    rc, w, tg = ctx.rc, a.weights, a.targets
    ref = ctx.ref
    num = den = 0.0
    for p in st.periods:
        d = float(SM.discount(rc, np.arange(p["t"], p["t"] + ctx.estep)).sum())
        rd, rp = own_ratio(ctx, a, p)
        viol = p["t"] >= ctx.from_year and (rd > tg["damage_ratio"] or (rp is not None and rp > tg["pop_ratio"]))
        u = (-w.get("damage", 0) * rd - w.get("pop", 0) * (rp or 0.0) + w.get("ecology", 0) * (p["ecology"] - ref["ecology"]) / 10
             + w.get("agriculture", 0) * (p["agriculture"] - 100) / 10 - rc["threshold_penalty"] * viol)
        num += d * u; den += d
    df = SM.discount(rc, np.arange(st.y0, st.y1))
    return num / max(den, 1e-12) - w.get("cost", 0) * float((df * cost_y).sum()) / ctx.budget_scale(a)


def budget_use(ctx, st):
    """Initial cost committed per decision year by each payer (the per-decade capacity applies to it) and the requests
    deferred to the next decision for lack of that capacity."""
    years = SM.decision_years(ctx.rc)
    use = {a: [0.0] * len(years) for a in ctx.payers}
    wait = {a: [0] * len(years) for a in ctx.payers}
    for e in st.log:
        if e["year"] not in years:
            continue
        k = years.index(e["year"])
        if e["event"] in ("start", "decrease") and e.get("cost"):
            for a, sh in ctx.payer(e["var"]).items():
                use[a][k] += e["cost"] * sh
        elif e["event"] == "deferred":
            for i, _ in e["actions"]:
                for a in ctx.payer(i):
                    wait[a][k] += 1
    return {"commit": use, "deferred": wait}


def record(ctx, st, cs, opt, s, full=True):
    df = SM.discount(ctx.rc, np.arange(st.y0, st.y1))
    Cc = (st.capital_var + st.maint_var) @ ctx.Pc.T                  # (years, actors): cost of each actor's measures (and subsidies)
    sc = SM.score(st, ctx.rc, ctx.ref, [[ctx.y0, "robust"]])
    out = {"s": int(s), "actors": {}}
    for k, a in enumerate(ctx.actors):                               # every actor's target, utility and cost (also in Single)
        met, fv = met_of(ctx, a, st)
        out["actors"][a.name] = {"met": met, "first_violation": fv, "U": actor_utility(ctx, a, st, Cc[:, k]),
                                 "cost_pv": float((df * Cc[:, k]).sum())}
    out["met_all"] = all(v["met"] for v in out["actors"].values())
    for c in cs:                                                     # the deciding actors: pathway, options, budget used
        a = c.a
        acts = [e for e in c.events if e["kind"] == "act"]
        d = out["actors"][a.name]
        d.update({"spent": st.spent_by.get(a.name, 0.0), "options_mid": opt.get("mid", {}).get(a.name), "options_end": opt["end"][a.name],
                  "first_act": acts[0]["year"] if acts else None, "n_acts": len(acts),
                  "switched": any(e["kind"] == "switch" for e in c.events)})
        if full:
            d["events"] = c.events
    out["budget"] = budget_use(ctx, st)
    bt = {k: ctx.R["targets"][k] for k in ("damage_ratio", "pop_ratio")}
    out["met_basin"] = met_of(ctx, SimpleNamespace(metric="basin", targets=bt), st)[0]
    out["U_basin"], out["cost_pv"] = sc["U"], sc["cost_pv"]
    if full:
        out["series"] = [[p["t"], round(p["damage"] / ctx.ref["damage"], 3), round(p["pop_risk"] / ctx.ref["pop_hazard"], 3)]
                         + [round(p["damage_zone"][z] / ctx.ref["dz:" + z], 3) for z in ctx.phys.zones] for p in st.periods]
        T = ctx.T
        out["states"] = {}                                                # committed levels at every decision year (joint network)
        for t in SM.decision_years(ctx.rc):
            g = []
            for i in range(len(T.V)):
                acts = [b for b in st.by_var[i] if not b.failed and b.t_decide <= t]
                lv = acts[-1].level if acts else 0.0
                g.append(int(np.argmin([abs(lv - x) for x in T.levels[i]])))
            out["states"][str(t)] = g
    return out


def run_jobs(ctx, jobs, full=False, chunk=400):
    out = []
    for a in range(0, len(jobs), chunk):
        sims = simulate(ctx, jobs[a:a + chunk])
        out += [record(ctx, st, cs, opt, s, full) for (st, cs, opt), (_, s) in zip(sims, jobs[a:a + chunk])]
    return out


# ------------------------------------------------------------------ summaries
def summarise(recs, name):
    """Per-actor (name) or basin (name None) arrays over the scenarios of one configuration."""
    if name is None:
        return {"met": np.array([r["met_all"] for r in recs]), "met_basin": np.array([r["met_basin"] for r in recs]),
                "U": np.array([r["U_basin"] for r in recs]), "cost": np.array([r["cost_pv"] for r in recs]),
                "lockin": np.array([np.mean([v["options_mid"]["lockin"] for v in r["actors"].values() if v.get("options_mid")] or [0])
                                    for r in recs])}
    A = [r["actors"][name] for r in recs]
    m = {"met": np.array([a["met"] for a in A]), "U": np.array([a["U"] for a in A]), "cost": np.array([a["cost_pv"] for a in A])}
    if all(a.get("options_mid") for a in A):
        m.update(lockin=np.array([a["options_mid"]["lockin"] for a in A]), options=np.array([a["options_mid"]["feasible"] for a in A]),
                 options_end=np.array([a["options_end"]["feasible"] for a in A]), first_act=[a["first_act"] for a in A])
    return m


def objectives(m):
    return [-float(m["met"].mean()), float(m["cost"].mean()), -float(m["U"].mean()), float(m["lockin"].mean())]


def choose(memo, tol):
    """Highest coverage; among those within tol of it the cheapest (then the higher mean utility)."""
    keys = list(memo)
    cov = np.array([memo[k]["met"].mean() for k in keys])
    ok = [k for k, c in zip(keys, cov) if c >= cov.max() - tol - 1e-12]
    return min(ok, key=lambda k: (memo[k]["cost"].mean(), -memo[k]["U"].mean()))


def front(memo):
    keys = list(memo)
    F = np.array([objectives(memo[k]) for k in keys])
    return [keys[i] for i in np.flatnonzero(nondominated(F))]


def table_row(m, best_U=None):
    r = {"coverage": float(m["met"].mean()), "mean_cost": float(m["cost"].mean()), "mean_U": float(m["U"].mean()),
         "lockin": float(m["lockin"].mean()) if "lockin" in m else None}
    if "options" in m:
        r["options"] = float(m["options"].mean()); r["options_end"] = float(m["options_end"].mean())
        fa = [y for y in m["first_act"] if y is not None]
        r["first_act_median"] = float(np.median(fa)) if fa else None
        r["share_acting"] = len(fa) / max(len(m["first_act"]), 1)
    if "met_basin" in m:
        r["coverage_basin_metric"] = float(m["met_basin"].mean())
    if best_U is not None:
        reg = best_U - m["U"]
        r["max_regret"], r["mean_regret"] = float(reg.max()), float(reg.mean())
    return r


# ------------------------------------------------------------------ search: one actor against the others' rules / the basin
def rule_mix(ctx, actor, RL, scen):
    """Per optimisation scenario a combination of the other actors' rules (cycled so that every combination appears)."""
    others = [a for a in ctx.names if a != actor]
    combos = list(itertools.product(RULES, repeat=len(others)))
    return [{o: RL[o][r] for o, r in zip(others, combos[j % len(combos)])} for j in range(len(scen))], \
        [dict(zip(others, combos[j % len(combos)])) for j in range(len(scen))]


def best_response(ctx, actor, scen, others, seeds, S, seed, log):
    """NSGA-II over the strategies of `actor` (others[j]: the other actors' strategies in scenario j - rules for the
    robust search, fixed strategies for a Best Response iteration).  Returns {Strategy: per-scenario arrays}."""
    a = ctx.by[actor]
    codec = Codec(a, ctx.T, ctx.MA["strategy"])
    space = EX.Space([SimpleNamespace(continuous=False, options=range(n), type="g") for n in codec.sizes], {})
    memo = {}

    def evaluate(xs):
        st = [codec.decode(x) for x in xs]
        new = [s for s in dict.fromkeys(st) if s not in memo]
        if new:
            recs = run_jobs(ctx, [({**others[j], actor: s}, scen[j]) for s in new for j in range(len(scen))])
            n = len(scen)
            for k, s in enumerate(new):
                memo[s] = summarise(recs[k * n:(k + 1) * n], actor)
        return [objectives(memo[s]) for s in st]
    init = [codec.encode(s) for s in seeds]
    EX.nsga_mixed(space, evaluate, S["pop"], S["gens"], np.random.default_rng(seed), S["p_mut"], init=init, log=log)
    return memo, codec


def basin_search(ctx, scen, seeds, S, seed, log):
    """NSGA-II over all actors' strategies together (concatenated genomes), objectives of the whole basin."""
    codecs = [Codec(a, ctx.T, ctx.MA["strategy"]) for a in ctx.actors]
    cut = np.cumsum([0] + [len(c.sizes) for c in codecs])
    sizes = [n for c in codecs for n in c.sizes]
    space = EX.Space([SimpleNamespace(continuous=False, options=range(n), type="g") for n in sizes], {})
    memo = {}

    def joint(x):
        return tuple(c.decode(x[cut[k]:cut[k + 1]]) for k, c in enumerate(codecs))

    def evaluate(xs):
        J = [joint(x) for x in xs]
        new = [j for j in dict.fromkeys(J) if j not in memo]
        if new:
            recs = run_jobs(ctx, [({s.actor: s for s in j}, sc) for j in new for sc in scen])
            n = len(scen)
            for k, j in enumerate(new):
                memo[j] = summarise(recs[k * n:(k + 1) * n], None)
        return [objectives(memo[j]) for j in J]
    init = [[g for c, s in zip(codecs, js) for g in c.encode(s)] for js in seeds]
    EX.nsga_mixed(space, evaluate, S["basin_pop"], S["basin_gens"], np.random.default_rng(seed), S["p_mut"], init=init, log=log)
    return memo


def pareto_seeds(ctx, a, rank, R, n=4):
    """Seeds from the exploration's Pareto portfolios (docs/12): their levels of the actor's measures, started as far as the
    first decade's budget allows, the rest (and the actor's ranking up to the most protective Pareto levels) as list A,
    the remaining actions as the severe list B.  Pareto portfolios only seed the search - they are not pathway states."""
    T = ctx.T
    src = PROCESSED / "explore" / R["explore_run"] / "results.json"
    if not src.exists():
        return []
    ex = json.loads(src.read_text(encoding="utf-8"))
    if [v["name"] for v in ex["variables"]] != [v["name"] for v in T.V]:
        return []
    subs = {tuple(RB.snap(T, ex["archive"][i]["x"])[v] for v in a.vars) for i in ex["pareto"]}
    cost = lambda g: sum(T.costs[v][li][0] for v, li in zip(a.vars, g))  # noqa: E731
    subs = sorted(subs, key=cost)
    top = subs[-1] if subs else ()
    out = []
    K = int(ctx.MA["strategy"]["seq_len"])
    for g in [subs[i] for i in np.unique(np.linspace(0, len(subs) - 1, min(n, len(subs))).round().astype(int))]:
        g = list(g)
        while cost(g) > a.budget["per_decade_oku"] + 1e-9 and any(g):
            j = max(range(len(g)), key=lambda j: T.costs[a.vars[j]][g[j]][0] if g[j] else -1)
            g[j] -= 1
        lev = dict(zip(a.vars, g))
        A = [k for k in rank if a.catalog[k][1] > lev[a.catalog[k][0]] and a.catalog[k][1] <= dict(zip(a.vars, top))[a.catalog[k][0]]]
        B = [k for k in rank if a.catalog[k][1] > lev[a.catalog[k][0]] and k not in A[:K]]
        out.append(Strategy(a.name, tuple(g), (tuple(A[:K]), tuple(B[:K])), switch=1.14))
    return out


# ------------------------------------------------------------------ pathways: per actor (map) and joint (network)
def integrated(ctx, recs):
    """Single as one actor for the map: one catalog of all the groups' actions, the groups' events merged in time order
    (a switch keeps the name of its group); its target = every actor's target."""
    off, cat, lab, lead = {}, [], [], []
    for a in ctx.actors:
        off[a.name] = len(cat)
        cat += a.catalog; lab += a.labels; lead += a.lead
    sa = Actor(ctx.pool["name"], ctx.pool["name"], [i for a in ctx.actors for i in a.vars], ctx.pool["budget"], "all", {}, {}, cat, lead, lab)
    out = []
    for r in recs:
        ev0 = {"year": ctx.y0, "kind": "initial", "acts": []}
        ev = []
        for g, a in enumerate(ctx.actors):
            for e in r["actors"][a.name]["events"]:
                if e["kind"] == "initial":
                    ev0["acts"] += [x + off[a.name] for x in e["acts"]]
                else:
                    ev.append((e["year"], g, {**e, "a": e["a"] + off[a.name]} if e["kind"] == "act" else {**e, "group": a.name}))
        ev.sort(key=lambda x: (x[0], x[1]))
        out.append({"actors": {sa.name: {"events": [ev0] + [e for _, _, e in ev], "met": r["met_all"]}}})
    return sa, out


def actor_tree(ctx, a, recs):
    """Representative pathways of one actor: a trie over its decisions (the switch to list B and each action; the years
    vary by scenario: median / range); per branch the scenarios, observed signals, share meeting the actor's target."""
    root = {"id": 0, "parent": None, "kind": "initial", "label": None, "years": [], "n": 0, "met": 0, "obs_ratio": [], "rain2100": [],
            "reasons": {}, "branch": 0, "acts": None}
    tree, child = [root], {}
    for r in recs:
        A = r["actors"][a.name]
        k = 0
        ev = A["events"]
        root["n"] += 1; root["met"] += A["met"]; root["years"].append(ctx.y0)
        if root["acts"] is None:
            root["acts"] = ev[0]["acts"]
        for e in ev[1:]:
            key = (k, e["kind"], e.get("a"), e.get("group"))
            if key not in child:
                child[key] = len(tree)
                tree.append({"id": len(tree), "parent": k, "kind": e["kind"], "a": e.get("a"), "group": e.get("group"),
                             "label": a.labels[e["a"]] if e["kind"] == "act" else
                             (f"{e['group']}の施策: 経路 B へ切替" if e.get("group") else "厳しい経路 B へ切替"),
                             "branch": e.get("branch", 1), "lead": e.get("lead"), "years": [], "n": 0, "met": 0,
                             "obs_ratio": [], "rain2100": [], "reasons": {}})
            k = child[key]
            b = tree[k]
            b["years"].append(e["year"]); b["n"] += 1; b["met"] += A["met"]
            if e.get("obs_ratio") is not None:
                b["obs_ratio"].append(e["obs_ratio"])
            b["rain2100"].append(e["rain2100"])
            if e["kind"] == "act":
                b["reasons"][e["reason"]] = b["reasons"].get(e["reason"], 0) + 1
        tree[k]["end"] = tree[k].get("end", 0) + 1
        tree[k]["end_met"] = tree[k].get("end_met", 0) + A["met"]
    return _finish_tree(tree, ctx, len(recs))


def _finish_tree(tree, ctx, n, min_share=None):
    for b in tree:
        b.setdefault("end", 0); b.setdefault("end_met", 0)
        ys = sorted(b.pop("years"))
        b["year"] = int(ys[(len(ys) - 1) // 2]) if ys else ctx.y0
        b["year_range"] = [int(ys[0]), int(ys[-1])] if ys else [ctx.y0, ctx.y0]
        for k in ("obs_ratio", "rain2100"):
            v = b.get(k)
            b[k] = [round(min(v), 3), round(max(v), 3)] if v else None
        b["met_share"] = b["met"] / max(b["n"], 1)
    return tree


def joint_network(ctx, recs, max_states):
    """Pathways network of the joint state (committed level of every measure) at the decision years: the states visited by
    the scenarios (at most max_states per year, the rest merged into "other"), edges = transitions taken (which actor
    changed which measure), with the scenario counts and the share meeting all targets."""
    T = ctx.T
    owner = {i: a.name for a in ctx.actors for i in a.vars}              # the authority of each measure (its group in Single)
    years = SM.decision_years(ctx.rc)
    keep = {}
    for t in years:
        cnt = {}
        for r in recs:
            g = tuple(r["states"][str(t)])
            cnt[g] = cnt.get(g, 0) + 1
        keep[t] = [g for g, _ in sorted(cnt.items(), key=lambda x: -x[1])[:max_states]]
    nodes, idx, edges = [], {}, {}

    def nid(t, g):
        key = (t, g if g in keep[t] else None)
        if key not in idx:
            idx[key] = len(nodes)
            nodes.append({"id": len(nodes), "year": t, "levels": list(g) if key[1] is not None else None, "n": 0, "met": 0,
                          "label": "その他の状態" if key[1] is None else None})
        return idx[key]
    for r in recs:
        prev = None
        for t in years:
            g = tuple(r["states"][str(t)])
            k = nid(t, g)
            nodes[k]["n"] += 1; nodes[k]["met"] += r["met_all"]
            if prev is not None:
                e = edges.setdefault((prev[0], k), {"from": prev[0], "to": k, "n": 0, "met": 0, "changes": None})
                e["n"] += 1; e["met"] += r["met_all"]
                if e["changes"] is None:
                    e["changes"] = [{"actor": owner.get(i, "—"), "var": T.V[i]["name"], "from": T.label(i, prev[1][i]), "to": T.label(i, g[i])}
                                    for i in range(len(g)) if g[i] != prev[1][i]]
            prev = (k, g)
    for nd in nodes:
        if nd["levels"] is not None:
            nd["label"] = " + ".join(f"{T.V[i]['name']} {T.label(i, li)}" for i, li in enumerate(nd["levels"]) if li) or "施策なし"
            nd["by_actor"] = {a.name: [f"{T.V[i]['name']} {T.label(i, nd['levels'][i])}" for i in a.vars if nd["levels"][i]] for a in ctx.actors}
        nd["met_share"] = nd["met"] / max(nd["n"], 1)
    E = list(edges.values())
    for e in E:
        e["met_share"] = e["met"] / max(e["n"], 1)
    return {"years": years, "nodes": nodes, "edges": E}


# ------------------------------------------------------------------ interdependence
def spillover(ctx, RL):
    """Effect of each actor's measures on every actor's metric (ratio / target), each actor's 積極投資 levels alone and all
    together, with no measures as the reference; planned 2095 climate and a severe one (rain x1.2, sea level 0.7 m)."""
    T = ctx.T
    G = {"施策なし": [0] * len(T.V)}
    for a in ctx.actors:
        g = [0] * len(T.V)
        for i, li in zip(a.vars, RL[a.name]["積極投資"].init):
            g[i] = li
        G[a.name] = list(T.repair(g))
    G["全主体"] = list(T.repair(np.max([G[a.name] for a in ctx.actors], axis=0)))
    out = {}
    for lab, env in (("計画（2℃, 2095年）", WD.belief_world(ctx.rc).env(2095)),
                     ("厳しい気候（降雨 ×1.2・海面 0.7 m）", dict(WD.belief_world(ctx.rc).env(2095), rain_factor=1.2, sea_level_rise_m=0.7))):
        y = ctx.phys.predict(np.array([T.raw(g) for g in G.values()]), {k: np.full(len(G), v) for k, v in env.items()})
        out[lab] = {g: {a.name: float(metric_of(a, {k: np.atleast_1d(v[j]) for k, v in y.items()}, ctx.ref)[0]) for a in ctx.actors}
                    for j, g in enumerate(G)}
    return {"portfolios": {k: [f"{T.V[i]['name']} {T.label(i, li)}" for i, li in enumerate(g) if li] for k, g in G.items()},
            "ratio_to_target": out}


def signal_quality(ctx):
    """How much the controllers' switch signal (projected 2100 rain from the observed trend) says about the scenario's true
    2100 rain at each decision year: quantiles and correlation over the scenarios."""
    from ..roadmap import planner as PL
    mon, win = ctx.rc["monitoring"], ctx.rc["detect"].get("trend_window", 3)
    W = ctx.worlds[:int(ctx.R["scenarios"].get("n_eval", len(ctx.worlds)))]
    true = np.array([w.p["rain_pts"][-1][1] for w in W])
    P = {}
    for w in W:
        h = []
        for t in SM.decision_years(ctx.rc)[1:]:
            tm = t - ctx.estep / 2
            h.append((tm, w.rain(tm) + mon["rain_noise"] * w.noise("rain", t)))
            tl, vl, sl = PL.DAPPController._trend(h, ctx.y0, 1.0, win)
            P.setdefault(t, []).append(vl + sl * (2100 - tl))
    return [{"year": t, "q10": float(np.quantile(v, 0.1)), "median": float(np.median(v)), "q90": float(np.quantile(v, 0.9)),
             "corr": float(np.corrcoef(v, true)[0, 1]) if len(v) > 2 else None} for t, v in P.items()]


# ------------------------------------------------------------------ pipeline
def setup(cfg, single=False, phys=None, log=print):
    R, MA = settings(cfg, single)
    rc, phys, ref, T, F, Z, vary, worlds = RB.setup(cfg, phys, log, R=R)
    return Ctx(cfg, R, MA, rc, phys, ref, T, worlds), F, Z, vary


CASES = {"individual": "Individual", "joint": "Joint", "single": "Single"}


def gap_of(a, b, ma, mb, seed=0, n_boot=2000):
    """b - a for the basin row of two cases (coverage of every target, cost, Regret, lock-in); ma, mb: every target met per
    evaluation scenario, for a paired bootstrap 90% interval of the coverage gap (the evaluation sample is finite)."""
    out = {k: b["basin"][k] - a["basin"][k] for k in ("coverage", "mean_cost", "max_regret", "mean_regret")}
    out["lockin"] = (b["basin"]["lockin"] or 0.0) - (a["basin"]["lockin"] or 0.0)
    out["actors"] = {n: b["actors"][n]["coverage"] - a["actors"][n]["coverage"] for n in a["actors"]}
    d = np.asarray(mb, float) - np.asarray(ma, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    out["coverage_ci90"] = np.quantile(d[idx].mean(1), [0.05, 0.95]).tolist()
    out["scenarios_gained"], out["scenarios_lost"] = int((d > 0).sum()), int((d < 0).sum())
    return out


def frontier(c, memo, chosen_key, scen):
    """The search's coverage-cost front (non-dominated on the search scenarios) re-evaluated on the evaluation scenarios."""
    keys = list(memo)
    F = np.array([[-memo[k]["met"].mean(), memo[k]["cost"].mean()] for k in keys])
    pts = [keys[i] for i in np.flatnonzero(nondominated(F))]
    if chosen_key not in pts:
        pts.append(chosen_key)
    recs = run_jobs(c, [({s.actor: s for s in j}, sc) for j in pts for sc in scen])
    n = len(scen)
    out = []
    for k, j in enumerate(pts):
        m = summarise(recs[k * n:(k + 1) * n], None)
        out.append({"coverage": float(m["met"].mean()), "mean_cost": float(m["cost"].mean()), "chosen": j == chosen_key,
                    "search_coverage": float(memo[j]["met"].mean()), "search_cost": float(memo[j]["cost"].mean())})
    return sorted(out, key=lambda r: r["mean_cost"])


def budget_summary(ctx, recs):
    """Per payer and decision year: committed initial cost (mean / 90% / max over the scenarios), the per-decade capacity,
    and the share of scenarios in which requests had to wait for the next decade."""
    out = {}
    for a, b in ctx.budgets.items():
        U = np.array([r["budget"]["commit"][a] for r in recs])
        W = np.array([r["budget"]["deferred"][a] for r in recs])
        out[a] = {"cap": b["per_decade_oku"], "total": b.get("total_oku"), "mean": U.mean(0).tolist(),
                  "p90": np.quantile(U, 0.9, axis=0).tolist(), "max": U.max(0).tolist(), "deferred_share": (W > 0).mean(0).tolist()}
    return out


def run(cfg=None, out_dir=None, phys=None, log=print):
    """Individual (each actor robust against the others' rules), Joint (the actors' strategies searched together, own budgets
    and measures kept) and Single (one actor with all measures and the summed per-decade capacity), on the same evaluation
    scenarios; coordination_gap = Joint - Individual, institutional_fragmentation_gap = Single - Joint."""
    t0 = time.time()
    cfg = CF.load(cfg)
    ctx, F, Z, vary = setup(cfg, False, phys, log)
    sctx = setup(cfg, True, ctx.phys, lambda m: None)[0]
    R, MA, S = ctx.R, ctx.MA, ctx.MA["search"]
    out = Path(out_dir) if out_dir else DAPP_DIR / MA["name"]
    out.mkdir(parents=True, exist_ok=True)
    CF.save(cfg, out / "config.json")
    nS = int(R["scenarios"]["n_eval"])                 # the search never sees the evaluation scenarios (out-of-sample comparison)
    scen_opt = list(range(nS, len(ctx.worlds)))
    log(f"Multi-Actor Robust DAPP: 主体 {len(ctx.actors)}（{'・'.join(ctx.names)}）・シナリオ 評価 {nS}・探索 {len(scen_opt)}")
    fmt_b = lambda b: f"{b['per_decade_oku']:.0f} 億円/10年" + ("" if b.get("total_oku") is None else f"（総額 {b['total_oku']:.0f}）")  # noqa: E731
    for a in ctx.actors:
        log(f"  {a.name}: 施策 {len(a.vars)}・行動 {len(a.catalog)}・容量 {fmt_b(a.budget)}・指標 {a.metric}")
    log(f"  Single（{sctx.pool['name']}）: 全 {sum(len(a.vars) for a in ctx.actors)} 施策・容量 {fmt_b(sctx.pool['budget'])}")
    rank = {a.name: ranking(a, ctx.T, ctx.phys, ctx.rc, ctx.metric_fn()) for a in ctx.actors}
    RL = {a.name: rules(a, rank[a.name], MA["rules"]) for a in ctx.actors}
    tol = MA["choose"]["coverage_tolerance"]

    # 1. Individual: each actor's robust strategy against the uncertain decisions of the others (rule mixture)
    memos, chosen, mixes = {}, {}, {}
    for k, a in enumerate(ctx.actors):
        log(f"1. Individual: {a.name} の戦略の探索（他主体の行動は外生ルールの組合せ: 不確実性）")
        others, mixes[a.name] = rule_mix(ctx, a.name, RL, scen_opt)
        seeds = list(RL[a.name].values()) + pareto_seeds(ctx, a, rank[a.name], R)
        memos[a.name], _ = best_response(ctx, a.name, scen_opt, others, seeds, S, cfg["seed"] + 701 + k, log)
        chosen[a.name] = choose(memos[a.name], tol)
        m = memos[a.name][chosen[a.name]]
        log(f"   {len(memos[a.name])} 戦略・推奨: カバー率 {m['met'].mean():.0%}・費用 {m['cost'].mean():.0f} 億円")

    # 2. Joint: all the actors' strategies together, objectives of the basin (every target), own budgets and measures
    log("2. Joint: 全主体の戦略を同時に探索（主体別の予算・施策の権限は維持）")
    seeds = [tuple(chosen[n] for n in ctx.names)] + [tuple(RL[n][r] for n in ctx.names) for r in RULES]
    bmemo = basin_search(ctx, scen_opt, seeds, S, cfg["seed"] + 709, log)
    joint = choose(bmemo, tol)
    log(f"   {len(bmemo)} 組合せ・推奨: 全主体の目標 {bmemo[joint]['met'].mean():.0%}・総費用 {bmemo[joint]['cost'].mean():.0f} 億円")

    # 3. Single: one decision maker for all the measures with the pooled per-decade capacity, the same objectives; every
    #    Joint strategy is also a Single strategy (seeded with the Joint and Individual choices)
    log("3. Single: 全施策と期別の財政容量を1主体に統合して探索")
    seeds = [joint, tuple(chosen[n] for n in ctx.names)] + [tuple(RL[n][r] for n in ctx.names) for r in RULES]
    smemo = basin_search(sctx, scen_opt, seeds, S, cfg["seed"] + 719, log)
    single = choose(smemo, tol)
    log(f"   {len(smemo)} 組合せ・推奨: 全主体の目標 {smemo[single]['met'].mean():.0%}・総費用 {smemo[single]['cost'].mean():.0f} 億円")

    # 4. evaluation on the evaluation scenarios: the three cases and every combination of the actors' rules
    log("4. 評価: Individual・Joint・Single・外生ルールの全組合せ")
    scen = list(range(nS))
    configs = {"individual": (ctx, {n: chosen[n] for n in ctx.names}), "joint": (ctx, {s.actor: s for s in joint}),
               "single": (sctx, {s.actor: s for s in single})}
    for combo in itertools.product(RULES, repeat=len(ctx.names)):
        configs["rules:" + "/".join(combo)] = (ctx, {n: RL[n][r] for n, r in zip(ctx.names, combo)})
    recs = {name: run_jobs(c, [(J, s) for s in scen], full=name in CASES) for name, (c, J) in configs.items()}
    best = {n: np.max([summarise(r, n)["U"] for r in recs.values()], axis=0) for n in ctx.names}
    best[None] = np.max([summarise(r, None)["U"] for r in recs.values()], axis=0)
    comp = {}
    for name, rr in recs.items():
        c, J = configs[name]
        comp[name] = {"actors": {n: table_row(summarise(rr, n), best[n]) for n in ctx.names},
                      "basin": table_row(summarise(rr, None), best[None]),
                      "strategies": {n: describe(c.by[n], s) for n, s in J.items()}}
    met = {k: summarise(recs[k], None)["met"] for k in CASES}
    gap = {"coordination_gap": gap_of(comp["individual"], comp["joint"], met["individual"], met["joint"], cfg["seed"]),
           "institutional_fragmentation_gap": gap_of(comp["joint"], comp["single"], met["joint"], met["single"], cfg["seed"] + 1)}
    log("   coverage–cost の前線（探索の非劣解を評価シナリオで再評価）")
    front_eval = {"individual": [{"coverage": comp["individual"]["basin"]["coverage"], "mean_cost": comp["individual"]["basin"]["mean_cost"],
                                  "chosen": True}],
                  "joint": frontier(ctx, bmemo, joint, scen), "single": frontier(sctx, smemo, single, scen)}
    front_eval["any_case_coverage"] = float(np.mean(np.any([met[k] for k in CASES], axis=0)))
    log("   全主体の目標: " + " → ".join(f"{CASES[k]} {comp[k]['basin']['coverage']:.0%}" for k in CASES)
        + f"（coordination gap {gap['coordination_gap']['coverage']:+.0%}・institutional fragmentation gap "
        f"{gap['institutional_fragmentation_gap']['coverage']:+.0%}）")

    # 5. dependence on the others: each actor's choice under every combination of the others' rules
    log("5. 他主体の行動への依存（各主体の推奨戦略 × 他主体のルールの組合せ）")
    sd = scen[:: max(1, nS // int(MA["dependence_scenarios"]))]
    dep = {}
    for n in ctx.names:
        others = [o for o in ctx.names if o != n]
        rows = []
        for combo in itertools.product(RULES, repeat=len(others)):
            J = {n: chosen[n], **{o: RL[o][r] for o, r in zip(others, combo)}}
            rows.append({"others": dict(zip(others, combo)), **table_row(summarise(run_jobs(ctx, [(J, s) for s in sd]), n))})
        rows.append({"others": {o: "個別最適" for o in others}, **table_row(summarise([recs["individual"][s] for s in sd], n))})
        dep[n] = rows

    # 6. pathways (per deciding actor; joint network), budget use, unmet futures, search fronts
    cx = {k: configs[k][0] for k in CASES}
    trees = {k: {a.name: actor_tree(cx[k], a, recs[k]) for a in cx[k].actors} for k in CASES}
    sa, srecs = integrated(sctx, recs["single"])
    trees["single"][sa.name] = actor_tree(sctx, sa, srecs)              # one map of all the measures (Single)
    nets = {k: joint_network(cx[k], recs[k], MA["network"]["max_states_per_year"]) for k in CASES}
    budget = {k: budget_summary(cx[k], recs[k]) for k in CASES}
    Fv = {k: F[k] for j, k in enumerate(F) if j in vary}
    X = Z[np.ix_(scen, vary)]
    pr = R["prim"]
    unmet = {}
    for k in CASES:
        u = {"all": ~summarise(recs[k], None)["met"]}
        u.update({n: ~summarise(recs[k], n)["met"] for n in ctx.names})
        unmet[k] = {q: {"scenarios": np.flatnonzero(v).tolist(), **MT.prim_box(X, v, Fv, pr["alpha"], pr["min_mass"], pr["density"])}
                    for q, v in u.items()}
    search = {}
    for n in ctx.names:
        mm = memos[n]
        bU = np.max([v["U"] for v in mm.values()], axis=0)
        search[n] = {"n_evaluated": len(mm), "mix": mixes[n],
                     "front": [{"strategy": describe(ctx.by[n], s), "chosen": s == chosen[n], **table_row(mm[s], bU)} for s in front(mm)]}
    for key, mm, pick, c in (("joint", bmemo, joint, ctx), ("single", smemo, single, sctx)):
        bU = np.max([v["U"] for v in mm.values()], axis=0)
        search[key] = {"n_evaluated": len(mm), "front": [{"strategies": {s.actor: describe(c.by[s.actor], s) for s in j},
                                                           "chosen": j == pick, **table_row(mm[j], bU)} for j in front(mm)]}

    def actor_info(c, a, rk, rl):
        return {"name": a.name, "label": a.label, "metric": a.metric, "budget": a.budget, "targets": a.targets, "weights": a.weights,
                "measures": [c.T.V[i]["name"] for i in a.vars], "catalog": a.labels, "lead": [round(x, 1) for x in a.lead],
                "catalog_vars": [c.T.V[i]["name"] for i, _ in a.catalog], "catalog_up": [li > 0 for _, li in a.catalog],
                "ranking": rk, "rules": {r: describe(a, s) for r, s in rl.items()}, "payer": {c.T.V[i]["name"]: c.shares[i] for i in a.vars}}
    E, _, _ = RB.env_table(ctx.worlds, [float(y) + ctx.estep / 2 for y in SM.eval_starts(ctx.rc)])
    res = {"mode": "multi", "name": MA["name"], "created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg,
           "settings": {"robust": R, "multi": MA, "single": sctx.pool},
           "variables": ctx.phys.variables, "zones": ctx.phys.zones, "reference": ctx.ref, "years": SM.decision_years(ctx.rc),
           "cases": CASES, "actors": [actor_info(ctx, a, rank[a.name], RL[a.name]) for a in ctx.actors],
           "single_actor": {"name": sa.name, "label": sa.label, "budget": sa.budget, "catalog": sa.labels, "lead": [round(x, 1) for x in sa.lead],
                            "groups": {a.name: [sctx.T.V[i]["name"] for i in a.vars] for a in sctx.actors}},
           "factors": F, "varied": [list(F)[j] for j in vary], "Z": Z[:nS].tolist(), "n_search_scenarios": len(scen_opt),
           "worlds": [{"rain_2100": w.p["rain_pts"][-1][1], "slr_2100": w.p["slr_pts"][-1][1]} for w in ctx.worlds[:nS]],
           "physics": {"explore_run": R["explore_run"], "ranges": ctx.phys.ranges, "outside_range": RB.outside_range(ctx.phys, E)},
           "search": search, "comparison": comp, "gap": gap, "frontier": front_eval, "budget": budget, "dependence": dep, "spillover": spillover(ctx, RL),
           "signal_quality": signal_quality(ctx), "trees": trees, "networks": nets, "unmet": unmet,
           "records": {k: [{q: r[q] for q in ("s", "met_all", "met_basin", "U_basin", "cost_pv", "series")}
                           | {"actors": {n: {q: v for q, v in a.items() if q != "events"} for n, a in r["actors"].items()}} for r in recs[k]]
                       for k in CASES}}
    res["elapsed_s"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=_json), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary(res), ensure_ascii=False, default=_json), encoding="utf-8")
    log(f"完了: {out / 'results.json'}（{res['elapsed_s']:.0f} s）")
    return res


# ------------------------------------------------------------------ GUI: recalculation with other capacities / targets / weights
def gui_config(cfg, inputs, size="standard"):
    """Settings of a GUI recalculation: the base run's settings with, per actor, the per-decade capacity, the target
    (damage ratio; exposed-population ratio for the basin metric) and the utility weights, the basin weights, at a computing
    size (multi.gui.sizes).  The run is named <base>_gui_<hash of the inputs> (the same inputs reuse the result)."""
    cfg = CF.load(copy.deepcopy(cfg))                  # settings saved by an older version get the current defaults
    MA = cfg["multi"]
    sizes = MA["gui"]["sizes"]
    if size not in sizes:
        raise ValueError(f"unknown size {size}")
    base = MA.get("gui_inputs", {}).get("base", MA["name"])
    for n, x in (inputs.get("actors") or {}).items():
        if n not in MA["actors"]:
            raise ValueError(f"unknown actor {n}")
        a = MA["actors"][n]
        if x.get("per_decade_oku") is not None:
            v = float(x["per_decade_oku"])
            if not 0 < v <= 10000:
                raise ValueError(f"{n}: per_decade_oku must be in (0, 10000]")
            a["budget"]["per_decade_oku"] = v
        for k, v in (x.get("targets") or {}).items():
            if v is None:
                continue
            if k not in ("damage_ratio", "pop_ratio") or (k == "pop_ratio" and a["metric"] != "basin"):
                raise ValueError(f"{n}: no target {k} (exposed population only for the basin metric)")
            if not 0.3 <= float(v) <= 5.0:
                raise ValueError(f"{n}: {k} must be in [0.3, 5]")
            a["targets"][k] = float(v)
        if x.get("weights"):
            a["weights"] = {k: float(v) for k, v in x["weights"].items() if float(v) >= 0}
    if inputs.get("basin_weights"):
        MA.setdefault("robust_overrides", {})["weights"] = {k: float(v) for k, v in inputs["basin_weights"].items() if float(v) >= 0}
    cfg["multi"] = MA = CF._merge(MA, sizes[size])
    key = json.dumps({"base": base, "size": size, "actors": {n: {"b": a["budget"], "t": a["targets"], "w": a["weights"]} for n, a in MA["actors"].items()},
                      "w": MA.get("robust_overrides", {}).get("weights")}, sort_keys=True, ensure_ascii=False)
    MA["gui_inputs"] = {"base": base, "size": size}
    MA["name"] = f"{base}_gui_{hashlib.sha1(key.encode()).hexdigest()[:8]}"
    return cfg


PHASES = (("1. Individual", 0.03, 0.45), ("2. Joint", 0.45, 0.62), ("3. Single", 0.62, 0.8), ("4. 評価", 0.8, 0.92), ("5. ", 0.92, 0.99))


def recalculate(cfg, phys=None, progress=None, out_dir=None):
    """Run the three cases with GUI settings (gui_config), reporting progress(fraction, message); returns the run name."""
    state = {"lo": 0.0, "hi": 0.03, "k": 0, "f": 0.0}
    n_act = len(cfg["multi"]["actors"])

    def log(m):
        m = m.strip()
        for tag, lo, hi in PHASES:
            if m.startswith(tag):
                if tag.startswith("1."):                                   # one actor after the other
                    w = (hi - lo) / n_act
                    lo, hi = lo + w * state["k"], lo + w * (state["k"] + 1)
                    state["k"] += 1
                state["lo"], state["hi"] = lo, hi
        f = state["lo"]
        if "世代" in m and "/" in m:                                       # NSGA-II generation g/G within the phase
            try:
                g, G = m.split("世代")[1].split(":")[0].strip().split("/")
                f = state["lo"] + (state["hi"] - state["lo"]) * int(g) / int(G)
            except ValueError:
                pass
        state["f"] = max(state["f"], f)                                     # never backwards
        if progress:
            progress(state["f"], m)
    run(cfg, out_dir=out_dir, phys=phys, log=log)
    return {"name": cfg["multi"]["name"]}


def summary(res):
    """Conditions and the headline results of a run (GUI: the table of conditions and results)."""
    MA = res["settings"]["multi"]
    g = res["gap"]
    return {"name": res["name"], "created": res["created"], "elapsed_s": res.get("elapsed_s"),
            "base": MA.get("gui_inputs", {}).get("base", res["name"]), "size": MA.get("gui_inputs", {}).get("size", "full"),
            "n_eval": len(res["worlds"]), "n_search": res.get("n_search_scenarios"),
            "actors": {a["name"]: {"per_decade_oku": a["budget"]["per_decade_oku"], "total_oku": a["budget"].get("total_oku"),
                                   "targets": a["targets"], "weights": a["weights"]} for a in res["actors"]},
            "basin_weights": res["settings"]["robust"]["weights"], "single_per_decade_oku": res["single_actor"]["budget"]["per_decade_oku"],
            "coverage": {k: res["comparison"][k]["basin"]["coverage"] for k in CASES},
            "cost": {k: res["comparison"][k]["basin"]["mean_cost"] for k in CASES},
            "max_regret": {k: res["comparison"][k]["basin"]["max_regret"] for k in CASES},
            "gap": {k: {"coverage": v["coverage"], "ci90": v["coverage_ci90"], "mean_cost": v["mean_cost"]} for k, v in g.items()},
            "deferred": {k: bool(any(x > 0 for b in res["budget"][k].values() for x in b["deferred_share"])) for k in CASES}}


def summaries(base, root=None):
    """summary() of the run `base` and of its GUI recalculations (in root, default DAPP_DIR), newest first."""
    out = []
    for d in Path(root or DAPP_DIR).iterdir():
        if not (d.name == base or d.name.startswith(base + "_gui_")) or not (d / "results.json").exists():
            continue
        f = d / "summary.json"
        if not f.exists():                                                 # runs made before summary.json
            res = json.loads((d / "results.json").read_text(encoding="utf-8"))
            if res.get("mode") != "multi" or "gap" not in res or "coordination_gap" not in res["gap"]:
                continue
            f.write_text(json.dumps(summary(res), ensure_ascii=False, default=_json), encoding="utf-8")
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return sorted(out, key=lambda r: r["created"], reverse=True)


def _json(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return RR._json(o)


# ------------------------------------------------------------------ RRI check of representative joint states
def verify_rri(out_dir, n_states=4, log=print):
    """Re-run RRI for joint states of the individual-choice pathways (scenarios with the lowest / median / highest 2100 rain,
    the last evaluated period) and compare the event damage with the surrogate."""
    from ..roadmap.physics import rri_evaluator
    from ..rdm.sensitivity import error_stats
    out = Path(out_dir)
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    ctx, *_ = setup(cfg, False, log=log)
    J = {}
    for n, d in res["comparison"]["individual"]["strategies"].items():
        g = d["genes"]
        a = ctx.by[n]
        J[n] = Strategy(n, tuple(g["init"]), (tuple(g["A"]), tuple(g["B"])), float("inf") if d["switch"] is None else d["switch"],
                        d["rho"], d["anticipate"])
    rain = np.array([w["rain_2100"] for w in res["worlds"]])
    order = list(np.argsort(rain))
    picks = list(dict.fromkeys([order[0], order[len(order) // 2], order[-1]]))
    rri = rri_evaluator(ctx.R["explore_run"], log)
    rows = []
    for s in picks:
        st = simulate(ctx, [(J, int(s))])[0][0]
        for p in (st.periods[len(st.periods) // 2], st.periods[-1]):
            if len(rows) >= n_states:
                break
            x, sim = ctx.phys.to_rri(p["raw"], p["env"])
            y = rri.simulate(x, sim)
            pred = ctx.phys.predict(p["raw"][None], {k: np.atleast_1d(v) for k, v in p["env"].items()})["damage"][0]
            rows.append({"s": int(s), "t": p["t"], "rri": y["damage"], "surrogate": float(pred)})
            log(f"  シナリオ {s} {p['t']}年: RRI {y['damage']:.0f} / 代理 {pred:.0f} 億円")
    o = np.array([r["rri"] for r in rows]); q = np.array([r["surrogate"] for r in rows])
    res["verification_rri"] = {"states": rows, **(error_stats(o, q) if len(rows) > 1 else {})}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=_json), encoding="utf-8")
    return res["verification_rri"]
