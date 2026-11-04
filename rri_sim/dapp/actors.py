"""Actors and their conditional adaptive strategies (Multi-Actor Robust DAPP, docs/11 11.10).

The unit of a pathway is a single measure action, not a portfolio:

    action      (variable, level index) of one of the actor's measures: add / strengthen (a higher level),
                change (raise -> relocate), scale back (level 0 of a reversible measure: paddy dams, private
                storage) or reconvert (forest back to farmland while young, measures.forest.reconvert)
    state       what the implementation model (rri_sim.roadmap.sim.State) carries: committed levels, actions under
                way (consensus / build / diffuse / mature), each payer's budget used, land use (landuse.Transitions)

A strategy is conditional - what to do next depends on what is observed:

    initial     level per measure started in the first decision year
    pathways    two ordered action lists: base (A) and severe (B); the actor follows A until the projected 2100 rain
                (trend of the observed rain) reaches `switch` (not before strategy.switch_from_year), then B for good
    trigger     the next action of the active list starts when the observed ratio of the actor's own metric exceeds
                rho x target, when (anticipate) the current state is projected (observed climate / exposure trends, the
                surrogate, nominal effects) to exceed it by the time that action could be complete (lead time) plus one
                decision step, or at the switch to B
Infeasible actions (land, irreversibility) are skipped; actions above the payer's budget wait (sim.State.process).

Rules for the other actors' unknown decisions (積極投資 / 段階投資 / 投資抑制) are strategies of the same form.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..roadmap import planner as PL
from ..roadmap import sim as SM
from ..roadmap import world as WD

RULES = ("積極投資", "段階投資", "投資抑制")


@dataclass(frozen=True)
class Strategy:
    actor: str
    init: tuple                     # level index per actor variable
    paths: tuple                    # (A, B): catalog indices
    switch: float = float("inf")    # projected 2100 rain at which the actor moves to B (inf = never)
    rho: float = 0.9
    anticipate: bool = True
    rule: str | None = None         # name of an exogenous decision rule (None = searched strategy)

    def key(self):
        return (self.actor, self.init, self.paths, self.switch, self.rho, self.anticipate, self.rule)


@dataclass
class Actor:
    name: str
    label: str
    vars: list
    budget: dict
    metric: str                     # "basin" (damage + exposed population) or a zone (its event damage)
    targets: dict
    weights: dict
    catalog: list = field(default_factory=list)    # [(var, level index)]
    lead: list = field(default_factory=list)       # nominal lead time [years] of each catalog action
    labels: list = field(default_factory=list)


def make_actors(MA, T, rc):
    """Actors of the settings (multi.actors; variables by name) with their action catalogs."""
    names = [v["name"] for v in T.V]
    bw = WD.belief_world(rc)
    out = []
    for name, a in MA["actors"].items():
        vs = [names.index(n) for n in a["measures"]]
        cat, lead, lab = [], [], []
        for i in vs:
            v = T.V[i]
            for li in range(1, len(T.levels[i])):
                cat.append((i, li)); lead.append(PL.lead_years(bw, T.kind(i, li))); lab.append(f"{v['name']} {T.label(i, li)}")
            k = T.kind(i, len(T.levels[i]) - 1)
            m = rc["measures"][k]
            if m.get("reversible") or (m.get("reconvert") or {}).get("allowed"):
                cat.append((i, 0)); lead.append(1.0)
                lab.append(f"{v['name']} {'農地へ再転換' if v['type'] == 'forest' else '縮小（0%）'}")
        out.append(Actor(name, a.get("label", name), vs, dict(a["budget"]), a["metric"], dict(a["targets"]), dict(a["weights"]),
                         cat, lead, lab))
    return out


# ------------------------------------------------------------------ genome (integer genes for explore.nsga_mixed)
class Codec:
    """genes = [initial level per variable] + [A_1..A_K] + [B_1..B_K] (0 = none, k = catalog k-1) + [switch] + [rho] + [anticipate]."""

    def __init__(self, actor, T, S):
        self.a, self.T = actor, T
        self.K = int(S["seq_len"])
        self.rho = [float(x) for x in S["rho_grid"]]
        self.sw = [float(x) for x in S["switch_grid"]] + [float("inf")]
        n = len(actor.catalog) + 1
        self.sizes = [len(T.levels[i]) for i in actor.vars] + [n] * (2 * self.K) + [len(self.sw), len(self.rho), 2]

    def decode(self, x):
        x = [int(round(float(v))) for v in x]
        nv, K = len(self.a.vars), self.K
        A = tuple(dict.fromkeys(c - 1 for c in x[nv:nv + K] if c > 0))          # repeated actions would be skipped anyway
        B = tuple(dict.fromkeys(c - 1 for c in x[nv + K:nv + 2 * K] if c > 0))
        return Strategy(self.a.name, tuple(x[:nv]), (A, B), self.sw[x[-3]], self.rho[x[-2]], bool(x[-1]))

    def encode(self, s):
        K = self.K
        seq = lambda p: [c + 1 for c in p[:K]] + [0] * (K - len(p[:K]))  # noqa: E731
        near = lambda g, v: int(np.argmin([abs(a - v) if np.isfinite(a) else (0 if not np.isfinite(v) else 9e9) for a in g]))  # noqa: E731
        return list(s.init) + seq(s.paths[0]) + seq(s.paths[1]) + [near(self.sw, s.switch), near(self.rho, s.rho), int(s.anticipate)]


def describe(actor, s):
    """Readable strategy (GUI / results)."""
    lab = lambda i, li: actor.labels[actor.catalog.index((i, li))]  # noqa: E731
    return {"actor": actor.name, "rule": s.rule, "init": [lab(i, li) for i, li in zip(actor.vars, s.init) if li],
            "A": [actor.labels[c] for c in s.paths[0]], "B": [actor.labels[c] for c in s.paths[1]],
            "switch": None if not np.isfinite(s.switch) else s.switch, "rho": s.rho, "anticipate": s.anticipate,
            "genes": {"init": list(s.init), "A": list(s.paths[0]), "B": list(s.paths[1])}}


# ------------------------------------------------------------------ exogenous decision rules (the other actors)
def ranking(actor, T, phys, rc, metric):
    """Up actions of the actor ordered by the reduction of its own metric (metric(actor, outputs) -> ratio) per initial
    cost, each action alone under the planned 2095 climate; levels of one variable keep their order (strengthen later)."""
    env = WD.belief_world(rc).env(2095)
    ups = [k for k, (i, li) in enumerate(actor.catalog) if li > 0]
    G = [tuple([0] * len(T.V))]
    for k in ups:
        i, li = actor.catalog[k]
        g = [0] * len(T.V); g[i] = li
        G.append(T.repair(g))
    m = metric(actor, phys.predict(np.array([T.raw(g) for g in G]), {k: np.full(len(G), v) for k, v in env.items()}))
    cost = np.array([T.costs[actor.catalog[k][0]][actor.catalog[k][1]][0] for k in ups])
    order = [ups[j] for j in np.argsort(-(m[0] - m[1:]) / np.maximum(cost, 1.0), kind="stable")]
    by_var = {}
    for k in sorted(order, key=lambda k: actor.catalog[k][1]):
        by_var.setdefault(actor.catalog[k][0], []).append(k)
    return [by_var[actor.catalog[k][0]].pop(0) for k in order]


def rules(actor, rank, R):
    """{rule: Strategy}: 積極投資 every measure at its top level at the start (the budget defers the rest), 段階投資 the ranking,
    one action per trigger (rho = staged_rho), 投資抑制 nothing until the observed ratio exceeds restrained_rho, then the
    actions with the shortest lead times first, observed signals only."""
    top = tuple(max(li for (i, li) in actor.catalog if i == v) for v in actor.vars)
    first = actor.catalog[rank[0]] if rank else None
    init = tuple(first[1] if first and v == first[0] else 0 for v in actor.vars)
    quick = tuple(sorted(rank, key=lambda k: actor.lead[k]))
    return {"積極投資": Strategy(actor.name, top, ((), ()), rule="積極投資"),
            "段階投資": Strategy(actor.name, init, (tuple(rank[1:]), ()), rho=float(R["staged_rho"]), rule="段階投資"),
            "投資抑制": Strategy(actor.name, tuple([0] * len(actor.vars)), (quick, ()), rho=float(R["restrained_rho"]),
                             anticipate=False, rule="投資抑制")}


# ------------------------------------------------------------------ controller of one actor in one scenario
class Controller:
    def __init__(self, ctx, actor, strat, s):
        self.ctx, self.a, self.S, self.s = ctx, actor, strat, s
        self.ptr = [0, 0]
        self.branch = 0
        self.hist = {}
        self.events = []
        self.r2100 = None
        self._next = None

    def level_index(self, st, i):
        L = self.ctx.T.levels[i]
        return int(np.argmin([abs(st.committed[i] - lv) for lv in L]))

    def levels(self, st):
        return [self.level_index(st, i) for i in range(len(self.ctx.T.V))]

    def own_pending(self, st):
        mine = set(self.a.vars)
        return any(q[0] in mine for q in st.queue)

    # ---- which action is next on the active list (skipping done / infeasible ones)
    def _peek(self, st, t, branch=None):
        b = self.branch if branch is None else branch
        seq = self.S.paths[b]
        T = self.ctx.T
        g = self.levels(st)
        k = self.ptr[b]
        while k < len(seq):
            i, li = self.a.catalog[seq[k]]
            if li > g[i]:
                if T.V[i]["type"] == "paddy_dam" and T.levels[i][li] > T.paddy_available(g, T.V[i]["zone"]) + 1e-9:
                    k += 1; continue                                   # the land no longer allows it
                return k
            if li < g[i] and st.can_decrease(i, t):
                return k
            k += 1
        return None

    def start(self, t, st):
        T = self.ctx.T
        g = self.levels(st)
        for i, li in zip(self.a.vars, self.S.init):
            g[i] = li
        g = T.repair(g)
        for i in self.a.vars:
            if g[i]:
                st.request(i, T.levels[i][g[i]], "plan")
        self.events.append({"year": t, "kind": "initial", "acts": [self.a.catalog.index((i, g[i])) for i in self.a.vars if g[i]]})

    def prepare(self, t, st, obs):
        """Observations of this decision; returns the state to project (levels, horizon) or None."""
        c = self.ctx
        tm = t - c.estep / 2
        for k in ("rain", "slr", "asset", "pop"):
            self.hist.setdefault(k, []).append((tm, float(obs[k])))
        tl, vl, sl = PL.DAPPController._trend(self.hist["rain"], c.y0, 1.0, c.rc["detect"].get("trend_window", 3))
        self.r2100 = vl + sl * (2100 - tl)
        self.obs_ratio = c.observed_ratio(self.a, st, obs, t)
        k = self._peek(st, t)
        self._next = k
        if not self.S.anticipate or k is None:
            return None
        lead = self.a.lead[self.S.paths[self.branch][k]]
        tau = min(t + lead + c.step, c.y1 - c.estep / 2)
        proj = {}
        for key, v0 in (("rain", 1.0), ("slr", 0.0), ("asset", 1.0), ("pop", 1.0)):
            tl, vl, sl = PL.DAPPController._trend(self.hist[key], c.y0, v0, c.rc["detect"].get("trend_window", 3))
            proj[key] = vl + sl * (tau - tl)
        return self.levels(st), proj

    def decide(self, t, st, proj_ratio):
        c, tg = self.ctx, self.a.targets
        switched = False
        if (self.branch == 0 and np.isfinite(self.S.switch) and t >= c.switch_from and self.r2100 is not None
                and self.r2100 >= self.S.switch):
            self.branch, switched = 1, True
            self.events.append({"year": t, "kind": "switch", "rain2100": round(float(self.r2100), 3)})
            self._next = self._peek(st, t)
        if self.own_pending(st) or self._next is None:
            return
        fire_o = t >= tg.get("from_year", c.from_year) and self.obs_ratio is not None and self.obs_ratio > self.S.rho
        fire_p = proj_ratio is not None and proj_ratio > self.S.rho
        if not (fire_o or fire_p or switched):
            return
        T = c.T
        k = self._next
        seq = self.S.paths[self.branch]
        i, li = self.a.catalog[seq[k]]
        g = self.levels(st)
        g2 = list(g); g2[i] = li
        g2 = T.repair(g2)
        for j in range(len(g)):
            if g2[j] != g[j]:
                st.request(j, T.levels[j][g2[j]], "trigger")
        self.ptr[self.branch] = k + 1
        self.events.append({"year": t, "kind": "act", "a": int(seq[k]), "branch": self.branch,
                            "reason": "switch" if switched and not (fire_o or fire_p) else ("observed" if fire_o else "projected"),
                            "obs_ratio": None if self.obs_ratio is None else round(float(self.obs_ratio), 3),
                            "projected": None if proj_ratio is None else round(float(proj_ratio), 3),
                            "rain2100": round(float(self.r2100), 3), "lead": round(float(self.a.lead[seq[k]]), 1)})


def metric_of(actor, y, ref, targets=None):
    """Ratio / target of the actor's own metric from surrogate outputs y (arrays), exposure factors applied by the caller."""
    tg = targets or actor.targets
    if actor.metric == "basin":
        return np.maximum(y["damage"] / ref["damage"] / tg["damage_ratio"], y["pop_hazard"] / ref["pop_hazard"] / tg["pop_ratio"])
    return y["dz:" + actor.metric] / ref["dz:" + actor.metric] / tg["damage_ratio"]


def options(ctx, actor, st, t):
    """Future options of the actor at t: share of its catalog still possible (feasible), done, or lost through the land
    (afforested paddies), irreversibility (no scaling back / forest too old) or its budget."""
    T = ctx.T
    g = [int(np.argmin([abs(st.committed[i] - lv) for lv in T.levels[i]])) for i in range(len(T.V))]
    out = dict.fromkeys(("feasible", "done", "land", "irreversible", "budget"), 0)
    left = {a: b["total_oku"] - st.spent_by.get(a, 0.0) for a, b in ctx.budgets.items()}
    for i, li in actor.catalog:
        if li == 0:
            if g[i] == 0:
                out["done"] += 1
            elif st.can_decrease(i, t):
                out["feasible"] += 1
            else:
                out["irreversible"] += 1
            continue
        if g[i] >= li:
            out["done"] += 1
            continue
        if T.V[i]["type"] == "paddy_dam" and T.levels[i][li] > T.paddy_available(g, T.V[i]["zone"]) + 1e-9:
            out["land"] += 1
            continue
        cost = max(T.costs[i][li][0] - T.costs[i][g[i]][0], 0.0)
        if not T.V[i]["continuous"] and g[i]:
            cost += ctx.rc["measures"][T.kind(i, g[i])]["change_cost_fraction"] * T.costs[i][g[i]][0]
        if any(cost * sh > left[a] + 1e-9 for a, sh in ctx.payer(i).items()):
            out["budget"] += 1
        else:
            out["feasible"] += 1
    n = max(len(actor.catalog), 1)
    return {k: v / n for k, v in out.items()} | {"lockin": (out["land"] + out["irreversible"] + out["budget"]) / n}
