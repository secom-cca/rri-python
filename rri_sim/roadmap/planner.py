"""Roadmaps, their (re-)optimisation from an inherited state, and the policies P0-P3.

Roadmap: actions [{var, level, year}] - level = option index (levee, retention, housing) or target rate
(paddy dams, forest).  Genome per variable: (level, decision index, stage); stage s > 0 for a staged
variable (levee +2, housing relocation) builds the intermediate option first and the final one s
decisions later (e.g. levee +1 -> +2, raise -> relocate).

optimise(): GA over genomes; each genome is simulated (open loop, sim.run with FixedController) from
the inherited state (measures done / under way, budget spent, engagement) in every world of the
planner's belief ensemble; fitness = mean utility (sim.score) of the remaining horizon under the
criteria the planner holds.  Physics is evaluated in one batch per generation.

Policies (cfg["policies"]):
  P0 fixed       the roadmap optimised at the start, executed as scheduled (no retries)
  P1 dapp        the same roadmap as a pathway: bundles in planned order; the first now, each next one when
                 the observed risk (damage / exposed population of the representative event, noisy)
                 reaches trigger.ratio_of_threshold x the tolerated ratio, or would reach the tolerated
                 ratio within the bundle's lead time (anticipate); failed actions are retried
  P2 + model     P1 + monitoring of the model assumptions (climate, land use, measure effects, delays):
                 a deviation beyond detect.*_tol updates the belief and re-optimises the remaining roadmap
  P3 + criteria  P2 + the criteria regime (weights, tolerated ratios) observed with regime_lag_years
"""
from __future__ import annotations

import copy

import numpy as np

from . import sim as SM
from . import world as WD


# ------------------------------------------------------------------ genome
def levels_of(cfg, v):
    if v["type"] == "levee":
        return list(range(len(v["heights"])))
    if v["continuous"]:
        return [float(x) for x in cfg["measures"][v["type"]]["levels"]]
    return list(range(len(v["options"])))


def staged(cfg, v):
    return (v["type"] == "levee" and cfg["measures"]["levee"].get("staged")) or (v["type"] == "relocate" and cfg.get("housing_staged"))


class Space:
    def __init__(self, cfg, phys):
        self.cfg, self.V = cfg, phys.variables
        self.levels = [levels_of(cfg, v) for v in self.V]
        self.staged = [bool(staged(cfg, v)) for v in self.V]
        self.years = SM.decision_years(cfg)
        self.nd = len(self.years)

    def random(self, rng, now_idx, p_active=0.35):
        g = np.zeros((len(self.V), 3), int)
        for i, L in enumerate(self.levels):
            if rng.random() < p_active:
                g[i, 0] = rng.integers(1, len(L))
            g[i, 1] = rng.integers(now_idx, self.nd)
            g[i, 2] = rng.integers(0, 4) if self.staged[i] and rng.random() < 0.3 else 0
        return g

    def decode(self, g, now_idx=0, committed=None):
        acts = []
        for i, (li, di, s) in enumerate(g):
            if li <= 0:
                continue
            L = self.levels[i]
            d = max(int(di), now_idx)
            cur = committed[i] if committed is not None else 0.0
            if self.staged[i] and s > 0 and li == len(L) - 1 and len(L) > 2:
                if L[li - 1] > cur + 1e-9:
                    acts.append({"var": i, "level": L[li - 1], "year": self.years[d]})
                if d + s < self.nd:
                    acts.append({"var": i, "level": L[li], "year": self.years[d + s]})
            elif L[li] > cur + 1e-9:
                acts.append({"var": i, "level": L[li], "year": self.years[d]})
        return sorted(acts, key=lambda a: (a["year"], a["var"]))

    def encode(self, actions, now_idx=0):
        g = np.zeros((len(self.V), 3), int)
        g[:, 1] = now_idx
        by = {}
        for a in actions:
            if a["year"] >= self.years[now_idx]:
                by.setdefault(a["var"], []).append(a)
        for i, acts in by.items():
            acts = sorted(acts, key=lambda a: a["year"])
            L = self.levels[i]
            top = max(acts, key=lambda a: a["level"])
            g[i, 0] = int(np.argmin([abs(x - top["level"]) for x in L]))
            g[i, 1] = self.years.index(acts[0]["year"])
            if len(acts) > 1 and self.staged[i]:
                g[i, 2] = min(3, self.years.index(acts[-1]["year"]) - g[i, 1])
        return g

    def mutate(self, g, rng, p, now_idx):
        g = g.copy()
        for i, L in enumerate(self.levels):
            if rng.random() < p:
                g[i, 0] = rng.integers(0, len(L))
            if rng.random() < p:
                g[i, 1] = int(np.clip(g[i, 1] + rng.choice([-1, 1]), now_idx, self.nd - 1))
            if self.staged[i] and rng.random() < p:
                g[i, 2] = rng.integers(0, 4)
        return g


class FixedController:
    needs_obs = False

    def __init__(self, actions, modes=None):
        self.actions = actions
        self.modes = modes or {}

    def decide(self, t, st, obs):
        for a in self.actions:
            if a["year"] == t:
                st.request(a["var"], a["level"], a.get("source", "plan"))


def rollouts(cfg, phys, ref, genomes_actions, start_state, worlds, t_now, regimes):
    """Simulate each action list in each world from start_state (or nothing); returns mean U per list."""
    states = []
    for acts in genomes_actions:
        for w in worlds:
            states.append(SM.run(w, FixedController(acts, start_state.social.modes), phys, cfg, ref,
                                 state=start_state.with_world(w), t_from=t_now, batch=True))
    SM.evaluate(states, phys)
    U = np.array([SM.score(s, cfg, ref, regimes, t_from=t_now)["U"] for s in states]).reshape(len(genomes_actions), len(worlds))
    return U.mean(1), states


def allowed_vars(cfg, phys, tiers):
    """Variables whose measures can be implemented by the given tiers (measures.<kind>.tier)."""
    M = cfg["measures"]
    out = []
    for i, v in enumerate(phys.variables):
        kinds = {SM.kind_of(v, lv) for lv in levels_of(cfg, v)[1:]}
        if all(M[k].get("tier", "国") in tiers for k in kinds):
            out.append(i)
    return out


def optimise(cfg, phys, ref, worlds, regimes, t_now=None, state=None, init=(), seed=0, modes=None, log=None, earliest=None,
             allowed=None):
    """Best roadmap (actions from t_now on, none before `earliest`) for the planner's belief ensemble
    `worlds` and criteria `regimes`, continuing from `state` (None = nothing done, start year);
    only the variables in `allowed` (None = all) may be used."""
    sp = Space(cfg, phys)
    off = np.ones(len(sp.V), bool)
    if allowed is not None:
        off[list(allowed)] = False
    else:
        off[:] = False
    t_now = cfg["years"]["start"] if t_now is None else t_now
    now_idx = sp.years.index(max(t_now, earliest or t_now))
    if state is None:
        from .social import Social
        state = SM.State(cfg, phys, worlds[0], Social(cfg, worlds[0].p["social"], modes))
    committed = list(state.committed)
    for (i, lv, _) in state.queue:                         # deferred requests count as planned
        committed[i] = max(committed[i], lv)
    op = cfg["optimiser"]
    rng = np.random.default_rng(seed)
    pop = [np.zeros((len(sp.V), 3), int)]
    pop[0][:, 1] = now_idx
    for acts in init:
        pop.append(sp.encode(acts, now_idx))
    while len(pop) < op["pop"]:
        pop.append(sp.random(rng, now_idx))
    cache = {}

    def fitness(G):
        for g in G:
            g[off, 0] = 0                                  # measures outside the allowed set
        todo = []
        for g in G:
            k = g.tobytes()
            if k not in cache:
                todo.append(k); cache[k] = None
        if todo:
            gs = [np.frombuffer(k, dtype=G[0].dtype).reshape(G[0].shape) for k in todo]
            acts = [sp.decode(g, now_idx, committed) for g in gs]
            U, _ = rollouts(cfg, phys, ref, acts, state, worlds, t_now, regimes)
            U = U + 1e-6 * np.array([-len(a) for a in acts])       # ties: fewer actions
            for k, u in zip(todo, U):
                cache[k] = float(u)
        return np.array([cache[g.tobytes()] for g in G])

    F = fitness(pop)
    for gen in range(op["gens"]):
        kids = []
        order = np.argsort(-F)
        elite = [pop[i] for i in order[:max(2, op["pop"] // 6)]]
        while len(kids) + len(elite) < op["pop"]:
            a, b = rng.integers(0, len(pop), 2), rng.integers(0, len(pop), 2)
            pa = pop[a[0]] if F[a[0]] > F[a[1]] else pop[a[1]]
            pb = pop[b[0]] if F[b[0]] > F[b[1]] else pop[b[1]]
            mask = rng.random(len(sp.V)) < 0.5
            c = np.where(mask[:, None], pa, pb)
            kids.append(sp.mutate(c, rng, op["p_mut"], now_idx))
        pop = elite + kids
        F = fitness(pop)
        if log:
            log(f"    GA 世代 {gen + 1}/{op['gens']}: 最良 U {F.max():.4f}（評価 {len(cache)} 案）")
    best = int(np.argmax(F))
    acts = sp.decode(pop[best], now_idx, committed)
    init_U = [cache.get(sp.encode(a, now_idx).tobytes()) for a in init] if init else []
    return acts, {"U": float(F[best]), "n_eval": len(cache), "init_U": init_U}


# ------------------------------------------------------------------ policies
def lead_years(world, kind):
    ph = world.phases(kind)
    return ph["consensus"] + ph["build"] + ph["diffuse"] + (ph["mature"] / 2 if kind == "forest" else 0.0)


def bundles_of(actions, t_now):
    """Actions grouped by planned year (the pathway order)."""
    out = {}
    for a in actions:
        if a["year"] >= t_now:
            out.setdefault(a["year"], []).append(a)
    return [out[y] for y in sorted(out)]


class DAPPController:
    needs_obs = True

    def __init__(self, cfg, phys, ref, roadmap, spec, modes=None, belief=None, seed=0, allowed=None):
        self.cfg, self.phys, self.ref, self.spec = cfg, phys, ref, spec
        self.allowed = allowed
        self.modes = modes or {}
        self.belief = copy.deepcopy(belief or WD.belief_params(cfg))
        self.regimes = [[cfg["years"]["start"], cfg["initial_regime"]]]
        self.roadmap = [dict(a) for a in roadmap]
        self.bundles = bundles_of(self.roadmap, cfg["years"]["start"])
        self.started = []
        self.first_pending = True             # the first bundle starts at its planned year, later ones on a trigger
        self.replans = []
        self.triggers = []
        self.hist = {}
        self.t_prev = None
        self.seed = seed

    # ---- detection of changed assumptions / criteria
    def _series(self, key, tm, v):
        h = self.hist.setdefault(key, [])
        h.append((tm, v))
        return h

    @staticmethod
    def _trend(h, y0, v0, window):
        """Level (mean of the last two observations, at their mean time) and slope (least squares over
        the last `window` observations, anchored at (y0, v0) when fewer than three)."""
        last = h[-2:]
        tl, vl = float(np.mean([q[0] for q in last])), float(np.mean([q[1] for q in last]))
        pts = h[-window:]
        if len(pts) >= 3:
            T = np.array([q[0] for q in pts]); V = np.array([q[1] for q in pts])
            slope = float(np.polyfit(T, V, 1)[0])
        else:
            slope = (vl - v0) / max(tl - y0, 1.0)
        return tl, vl, slope

    def _detect(self, t, obs):
        """A deviation is detected when the mean of the last two observations differs from the belief
        at their mean time by more than the tolerance (persistence over two decisions); the belief path
        is then replaced by the observed level and trend."""
        cfg, b, tol = self.cfg, self.belief, self.cfg["detect"]
        bw = WD.World(b, cfg)
        y0 = cfg["years"]["start"]
        tm = t - cfg["years"]["eval_step"] / 2 if t > y0 else t
        out = []
        if self.spec.get("update_model") and t > y0:
            series = (("rain", "rain_pts", "降雨倍率", "climate", tol["rain_tol"], 1.0, (1.0, 1.3), False),
                      ("slr", "slr_pts", "海面上昇", "climate", tol["slr_tol"], 0.0, (0.0, 0.7), False),
                      ("asset", "asset_pts", "資産（浸水域）", "landuse", tol["exposure_tol"], 1.0, (0.4, 1.6), True),
                      ("pop", "pop_pts", "人口（浸水域）", "landuse", tol["exposure_tol"], 1.0, (0.4, 1.6), True))
            for key, pk, what, kind, tl_, v0, lim, rel in series:
                h = self._series(key, tm, obs[key])
                if len(h) < 2:
                    continue
                tl, vl, slope = self._trend(h, y0, v0, cfg["detect"].get("trend_window", 3))
                bv = WD._interp(b[pk], tl)
                dev = (vl / max(bv, 1e-9) - 1) if rel else (vl - bv)
                if abs(dev) > tl_:
                    v2100 = float(np.clip(vl + slope * (2100 - tl), *lim))
                    out.append({"kind": kind, "what": what, "observed": vl, "believed": bv, "new_2100": v2100, "old_2100": b[pk][-1][1]})
                    b[pk] = [q for q in b[pk] if q[0] < tl] + [[tl, vl], [2100, v2100]]
            for k, e in obs["effects"].items():
                kk = "raise" if k == "relocate" else k
                h = self._series("eff_" + kk, tm, e)
                if len(h) < 2:
                    continue
                ev = float(np.mean([q[1] for q in h[-2:]]))
                be = bw.effect(kk, tm)
                if abs(ev - be) > tol["effect_tol"]:
                    out.append({"kind": "effect", "what": f"{cfg['measures'][kk]['label']}の効果", "observed": ev, "believed": be})
                    b["eff0"][kk] = float(np.clip(ev, *bw.ranges[kk]))
            seen = {}
            for o in obs["phases"]:
                seen.setdefault((o["kind"], o["phase"]), []).append(o["years"])
            for (k, ph), ys in seen.items():
                bv = b["phases"][k][ph]
                ov = float(np.mean(ys))
                if bv > 0 and abs(ov / bv - 1) > tol["delay_tol"]:
                    out.append({"kind": "delay", "what": f"{cfg['measures'][k]['label']}の{ {'build': '施工', 'diffuse': '普及', 'mature': '成熟'}[ph]}期間",
                                "observed": ov, "believed": bv})
                    b["phases"][k][ph] = ov
        if self.spec.get("update_criteria") and obs["regime"] != self.regimes[-1][1]:
            out.append({"kind": "criteria", "what": "評価基準（重み・許容閾値）", "observed": obs["regime"], "believed": self.regimes[-1][1]})
            self.regimes.append([t, obs["regime"]])
        return out

    def _thresholds(self):
        return self.cfg["regimes"][self.regimes[-1][1]]["thresholds"]

    def _trigger(self, t, obs, bundle):
        th = self._thresholds()
        r = max(obs["damage_ratio"] / th["damage_ratio"], obs["pop_ratio"] / th["pop_ratio"])
        tr = self.cfg["trigger"]
        if r > tr["ratio_of_threshold"]:
            return {"reason": "observed", "ratio": r}
        if tr.get("anticipate") and "damage_ratio_prev" in obs:
            r0 = max(obs["damage_ratio_prev"] / th["damage_ratio"], obs["pop_ratio_prev"] / th["pop_ratio"])
            bw = WD.World(self.belief, self.cfg)
            lead = max(lead_years(bw, SM.kind_of(self.phys.variables[a["var"]], a["level"])) for a in bundle)
            proj = r + (r - r0) * lead / self.cfg["years"]["decision_step"]
            if proj > 1.0:
                return {"reason": "anticipated", "ratio": r, "projected": proj, "lead": lead}
        return None

    def decide(self, t, st, obs):
        cfg = self.cfg
        y0 = cfg["years"]["start"]
        reasons = self._detect(t, obs) if t > y0 else []
        if reasons and len(self.replans) < cfg["replan"]["max"]:
            from .world import belief_ensemble
            worlds = belief_ensemble(cfg, self.belief, cfg["optimiser"]["n_belief"], self.seed + t)
            remaining = [a for b in self.bundles for a in b]
            old = [dict(a, year=max(a["year"], t)) for a in remaining]
            acts, info = optimise(cfg, self.phys, self.ref, worlds, self.regimes, t, st, init=[old], seed=self.seed + t,
                                  modes=self.modes, allowed=self.allowed)
            self.replans.append({"year": t, "reasons": reasons, "old": old, "new": acts, "U_new": info["U"],
                                 "U_old": (info["init_U"] or [None])[0], "diff": diff_roadmaps(old, acts),
                                 "regime": self.regimes[-1][1]})
            self.bundles = bundles_of(acts, t)
            self.first_pending = True
        elif reasons:
            self.replans.append({"year": t, "reasons": reasons, "skipped": True, "regime": self.regimes[-1][1]})
        # retries of failed consensus
        for e in st.failures_since(self.t_prev if self.t_prev is not None else y0):
            st.request(e["var"], e["level"], "retry")
        # pathway: the first bundle at its planned year, each next one on a trigger
        if self.bundles and self.first_pending and self.bundles[0][0]["year"] <= t:
            self._start(t, st, self.bundles.pop(0), {"reason": "replan" if self.replans and not self.replans[-1].get("skipped") else "plan"})
            self.first_pending = False
        elif self.bundles and not self.first_pending and obs.get("damage_ratio") is not None:
            trig = self._trigger(t, obs, self.bundles[0])
            if trig:
                self._start(t, st, self.bundles.pop(0), trig)
        self.t_prev = t

    def _start(self, t, st, bundle, why):
        src = {"plan": "plan", "replan": "replan"}.get(why["reason"], "trigger")
        for a in bundle:
            st.request(a["var"], a["level"], src)
        self.started.append({"year": t, "actions": bundle, **why})
        if src == "trigger":
            self.triggers.append({"year": t, **why, "actions": bundle})


def diff_roadmaps(old, new):
    """Changes between two action lists: added / dropped / advanced / delayed / changed level."""
    def key(a):
        return a["var"]
    o, n = {}, {}
    for a in old:
        o.setdefault(key(a), []).append(a)
    for a in new:
        n.setdefault(key(a), []).append(a)
    out = []
    for v in sorted(set(o) | set(n)):
        A, B = o.get(v, []), n.get(v, [])
        if not A:
            out += [{"change": "added", "var": v, "level": b["level"], "year": b["year"]} for b in B]
        elif not B:
            out += [{"change": "dropped", "var": v, "level": a["level"], "year": a["year"]} for a in A]
        else:
            ta, tb = max(a["level"] for a in A), max(b["level"] for b in B)
            ya, yb = min(a["year"] for a in A), min(b["year"] for b in B)
            if abs(ta - tb) > 1e-9:
                out.append({"change": "level", "var": v, "from": ta, "to": tb, "year": yb})
            if ya != yb:
                out.append({"change": "advanced" if yb < ya else "delayed", "var": v, "from": ya, "to": yb})
    return out


def controller(cfg, phys, ref, name, roadmap, modes=None, seed=0, allowed=None):
    spec = cfg["policies"][name]
    if spec["kind"] == "fixed":
        return FixedController([dict(a) for a in roadmap], modes)
    return DAPPController(cfg, phys, ref, roadmap, spec, modes, seed=seed, allowed=allowed)


def initial_roadmap(cfg, phys, ref, modes=None, seed=0, log=None, delay_scale=1.0, earliest=None, allowed=None):
    """The roadmap all policies start from: optimised at the start year under the planner's belief
    ensemble and the initial criteria (no action before `earliest`)."""
    bel = WD.belief_params(cfg, delay_scale)
    worlds = WD.belief_ensemble(cfg, bel, cfg["optimiser"]["n_belief"], seed)
    acts, info = optimise(cfg, phys, ref, worlds, [[cfg["years"]["start"], cfg["initial_regime"]]], seed=seed, modes=modes, log=log,
                          earliest=earliest, allowed=allowed)
    return acts, info


class Postponed:
    """A controller that does nothing before `year` (the value of an early decision)."""

    def __init__(self, inner, year):
        self.inner, self.year = inner, year
        self.needs_obs, self.modes = inner.needs_obs, inner.modes

    def decide(self, t, st, obs):
        if t >= self.year:
            self.inner.decide(t, st, obs)

    def __getattr__(self, k):
        return getattr(self.inner, k)
