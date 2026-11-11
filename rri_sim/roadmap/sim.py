"""Implementation state of a roadmap and its simulation in a world.

Time: decisions every `decision_step` years; the flood outcome is evaluated at
the middle of every `eval_step` years (the representative event, docs/11, under
the climate and measures in place at that time).

A request (variable, level) at decision year t becomes an Action:
  1. budget   committed initial cost = cost(new) - cost(current) (+ change_cost_fraction x cost(current)
              when an option is replaced); it must fit the decade cap and the total budget, otherwise the
              request waits for the next decision (deferred).  A single project above the decade cap may
              start when nothing else has started in that decision (it takes the decade's budget)
  2. consensus the needed stakeholders are engaged (social.py); consensus time C, failure (drawn
              once per attempt from the world), adoption ceiling A
  3. build    levees / retention: in place at t + C + build (step).  Pumps, public storage: effect grows linearly from 0
              to 100 % over the build years
  4. diffuse  paddy dams, forest management, housing (raise / relocate), private storage, institutions: implementation
              share grows linearly over diffuse years to A
  (5. mature  afforestation of earlier runs only: effect of a planted cohort grows over mature years)
  A failed action is known at t + C; its committed budget is released and the variable goes back to
  its previous level (the controller may try again).
Payments: the committed cost (x adoption for paddy dams / forest / housing) is spread over the
build / diffusion years; maintenance (costs.py annual cost + maintenance_fraction x initial) and
coordination cost (social.py) accrue every year.  Levels never decrease (no removal) unless allow_decrease
(Robust DAPP).  budgets / payer (Multi-Actor DAPP): each payer's share of a cost must fit its own caps.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .social import Social

ADOPTION_KINDS = ("paddy_dam", "forest", "forest_mgmt", "raise", "relocate", "storage_private", "floodproof", "evacuation")
STEP_KINDS = ("levee", "retention")    # in place when built; every other kind grows linearly over its build + diffuse years


def decision_years(cfg):
    y = cfg["years"]
    return list(range(y["start"], y["end"], y["decision_step"]))


def eval_starts(cfg):
    y = cfg["years"]
    return list(range(y["start"], y["end"], y["eval_step"]))


@dataclass
class Action:
    var: int
    kind: str
    level: float
    t_decide: float
    t_cons: float
    failed: bool
    build: float
    diffuse: float
    mature: float
    adopt: float
    prev_level: float
    final: tuple
    cost: float
    change_cost: float
    attempt: int
    source: str = "plan"
    settled: bool = False


def kind_of(v, level):
    """Measure kind of variable v at `level` (housing: option 1 raise, 2 relocate)."""
    if v["type"] == "relocate":
        return "raise" if int(round(level)) == 1 else "relocate"
    return v["type"]


class State:
    def __init__(self, cfg, phys, world, social):
        self.cfg, self.phys, self.world, self.social = cfg, phys, world, social
        V = phys.variables
        self.nV = len(V)
        self.y0, self.y1 = cfg["years"]["start"], cfg["years"]["end"]
        n = self.y1 - self.y0
        self.capital = np.zeros(n); self.maint = np.zeros(n); self.coord = np.zeros(n); self.change = np.zeros(n)
        self.capital_var = np.zeros((n, self.nV)); self.maint_var = np.zeros((n, self.nV))     # per variable (cost by actor)
        self.budgets = None                  # {payer: {total_oku (None = no limit), per_decade_oku}} (Multi-Actor DAPP); None = cfg budget
        self.payer = None                    # var -> {payer: share of its cost}
        self.spent_by = {}
        self.actions: list[Action] = []
        self.by_var = [[] for _ in range(self.nV)]
        self.committed = [0.0] * self.nV
        self.queue = []                      # [(var, level, source)]
        self.spent = 0.0
        self.attempts = {}
        self.log = []
        self.periods = []                    # evaluated periods (dicts)
        self.pending = []                    # periods waiting for the physics
        self.accrued_to = self.y0
        self.cols = [[c for c, (j, _) in enumerate(phys.layout) if j == i] for i in range(self.nV)]

    def copy(self):
        o = copy.copy(self)
        o.capital, o.maint, o.coord, o.change = self.capital.copy(), self.maint.copy(), self.coord.copy(), self.change.copy()
        o.capital_var, o.maint_var, o.spent_by = self.capital_var.copy(), self.maint_var.copy(), dict(self.spent_by)
        o.actions = [copy.copy(a) for a in self.actions]
        o.by_var = [[] for _ in range(self.nV)]
        for a in o.actions:
            o.by_var[a.var].append(a)
        o.committed = list(self.committed); o.queue = list(self.queue); o.attempts = dict(self.attempts)
        o.log = list(self.log); o.periods = list(self.periods); o.pending = list(self.pending)
        o.social = self.social.copy()
        return o

    def with_world(self, world):
        """Copy of this state continuing in another world (planning in a belief world)."""
        o = self.copy()
        o.world = world
        o.social = self.social.copy(); o.social.v = world.p["social"]
        return o

    # ---------------------------------------------------------------- requests and actions
    def request(self, var, level, source="plan"):
        self.queue.append((int(var), float(level), source))

    def settle(self, t):
        """Failures known by t: release the budget, restore the level, log."""
        for a in self.actions:
            if a.failed and not a.settled and a.t_cons <= t:
                a.settled = True
                self.spent -= a.cost
                self._charge(a.var, -a.cost)
                later = [b for b in self.by_var[a.var] if b.t_decide > a.t_decide and not b.failed]
                if not later:
                    self.committed[a.var] = a.prev_level
                self.log.append({"year": round(a.t_cons, 1), "event": "failed", "var": a.var, "kind": a.kind, "level": a.level,
                                 "attempt": a.attempt})

    def failures_since(self, t_prev):
        return [e for e in self.log if e["event"] == "failed" and e["year"] > t_prev - 1e-9]

    def process(self, t):
        """Start the queued requests that the budget allows (in order); the rest wait."""
        cfg, V, P = self.cfg, self.phys.variables, self.phys
        B = cfg["budget"]
        cap = B["per_decade_oku"] * getattr(self, "cap_factor", 1.0)
        dec = 0.0
        dec_by = {}
        keep = []
        for (i, level, source) in self.queue:
            v = V[i]
            cur = self.committed[i]
            if level <= cur + 1e-9:
                if level < cur - 1e-9 and self.can_decrease(i, t):
                    self._decrease(i, level, t, source)
                continue
            kind = kind_of(v, level)
            m = cfg["measures"][kind]
            ini_new = P.cost_parts(i, level)[0]
            ini_old = P.cost_parts(i, cur)[0] if cur > 0 else 0.0
            if v["continuous"]:
                c, chg = max(ini_new - ini_old, 0.0), 0.0
            else:
                chg = cfg["measures"][kind_of(v, cur)]["change_cost_fraction"] * ini_old if cur > 0 else 0.0
                c = max(ini_new - ini_old, 0.0) + chg
            # the decade cap; a single project above the cap may start when nothing else started this decade
            if self.budgets is None:
                if (dec + c > cap + 1e-9 and dec > 0) or self.spent + c > B["total_oku"] + 1e-9:
                    keep.append((i, level, source))
                    continue
            elif not self._fits(i, c, dec_by):                   # each payer's own caps (shares of the cost)
                keep.append((i, level, source))
                continue
            dec += c; self.spent += c
            self._charge(i, c, dec_by)
            S = self.social
            S.engage(m["stakeholders"], t)
            ph = self.world.phases(kind)
            C = S.consensus_years(kind, t, ph["consensus"])
            att = self.attempts.get(kind, 0); self.attempts[kind] = att + 1
            failed = self.world.fail_u(kind, att) < S.fail_prob(kind, t)
            adopt = S.adoption(kind, t + C + ph["build"] + ph["diffuse"]) if kind in ADOPTION_KINDS else 1.0
            final = self._final(i, kind, level, cur, adopt)
            a = Action(i, kind, level, float(t), float(t + C), bool(failed), ph["build"], ph["diffuse"], ph["mature"], adopt,
                       cur, final, c, chg, att, source)
            self.actions.append(a)
            self.by_var[i].append(a)
            self.committed[i] = level
            if not failed:
                pay = c * (adopt if kind in ADOPTION_KINDS else 1.0)
                y_a, y_b = t + C, t + C + max(a.build + a.diffuse, 1.0)
                self._spread(self.capital, pay, y_a, y_b)
                self._spread(self.capital_var[:, i], pay, y_a, y_b)
                if chg:
                    self._spread(self.change, chg, y_a, y_b)
            self.log.append({"year": t, "event": "start", "var": i, "kind": kind, "level": level, "cost": c, "change_cost": chg,
                             "consensus_end": round(t + C, 2), "complete": round(self.complete_year(a), 2),
                             "adopt": round(adopt, 3), "source": source, "failed_later": bool(failed)})
        if keep:
            self.log.append({"year": t, "event": "deferred", "actions": [[i, lv] for i, lv, _ in keep]})
        self.queue = keep
        return dec

    def can_decrease(self, i, t):
        """A lower level may be requested only when the State allows it (allow_decrease, Robust DAPP) and the
        measure is reversible (measures.<kind>.reversible: paddy dams, private storage) or can be reconverted
        (measures.<kind>.reconvert: young forest back to farmland, max_age_years after planting)."""
        if not getattr(self, "allow_decrease", False):
            return False
        v = self.phys.variables[i]
        m = self.cfg["measures"][kind_of(v, self.committed[i])]
        if m.get("reversible"):
            return True
        rc = m.get("reconvert")
        acts = [a for a in self.by_var[i] if not a.failed and a.final[0] > 0]
        return bool(rc and rc.get("allowed") and acts and t - max(a.t_cons for a in acts) <= rc["max_age_years"] + 1e-9)

    def _decrease(self, i, level, t, source):
        """Scale a reversible measure back (or reconvert forest land); no refund of the committed cost,
        reconversion costs reconvert.cost_fraction x the planting cost of the cleared share."""
        v, P = self.phys.variables[i], self.phys
        cur = self.committed[i]
        kind = kind_of(v, cur)
        m = self.cfg["measures"][kind]
        c = 0.0
        if not m.get("reversible"):
            c = m["reconvert"].get("cost_fraction", 0.0) * max(P.cost_parts(i, cur)[0] - P.cost_parts(i, level)[0], 0.0)
        final = (level - cur,) if v["type"] == "forest" else (level,)
        a = Action(i, kind, level, float(t), float(t), False, 1.0 if v["type"] == "forest" else 0.0,
                   0.0 if v["type"] == "forest" else 1.0, 0.0, 1.0, cur, final, c, c, 0, source)     # one year to stop / clear
        self.actions.append(a)
        self.by_var[i].append(a)
        self.committed[i] = level
        self.spent += c
        self._charge(i, c)
        if c:
            self._spread(self.capital, c, t, t + 1)
            self._spread(self.capital_var[:, i], c, t, t + 1)
            self._spread(self.change, c, t, t + 1)
        self.log.append({"year": t, "event": "decrease", "var": i, "kind": kind, "level": level, "from": cur, "cost": c, "source": source})

    def shares(self, i):
        return self.payer(i) if self.payer else {}

    def _fits(self, i, c, dec_by):
        for a, sh in self.shares(i).items():
            b, x = self.budgets[a], c * sh
            d = dec_by.get(a, 0.0)
            tot = b.get("total_oku")                             # None: no limit on the total (per-decade capacity only)
            if (d + x > b["per_decade_oku"] + 1e-9 and d > 0) or (tot is not None and self.spent_by.get(a, 0.0) + x > tot + 1e-9):
                return False
        return True

    def _charge(self, i, c, dec_by=None):
        for a, sh in self.shares(i).items():
            self.spent_by[a] = self.spent_by.get(a, 0.0) + c * sh
            if dec_by is not None:
                dec_by[a] = dec_by.get(a, 0.0) + c * sh

    def complete_year(self, a):
        if a.kind in STEP_KINDS:
            return a.t_cons + a.build
        if a.kind == "forest":
            return a.t_cons + a.build + a.mature
        return a.t_cons + a.build + a.diffuse

    def _final(self, i, kind, level, cur, adopt):
        v = self.phys.variables[i]
        if kind == "levee":
            hs = v["heights"]
            return (hs[int(round(level))] / max(hs),)
        if kind == "retention":
            return (1.0,)
        if kind == "paddy_dam":
            return (max(cur, min(level, adopt)),)
        if kind == "forest":
            return (max(min(level, adopt) - cur, 0.0),)            # planted share of this cohort
        if kind in ("raise", "relocate"):
            prev = self._final_vec(i)
            if kind == "raise":
                return (max(prev[0], adopt), prev[1])
            return (max(prev[0] - adopt, 0.0), max(prev[1], adopt))     # relocate (raised houses may move)
        if v["continuous"] or kind in self.cfg["measures"] and len(self.cfg["measures"][kind].get("levels", [])) > 2:
            return (max(cur, min(level, adopt)),)                       # implementation rate (storage, floodproofing)
        frac = level / max(len(v["options"]) - 1, 1)
        return (frac * (adopt if kind in ADOPTION_KINDS else 1.0),)    # pumps, storage, zoning, evacuation ...

    def _final_vec(self, i):
        acts = [a for a in self.by_var[i] if not a.failed]
        return acts[-1].final if acts else (0.0, 0.0)

    def _spread(self, arr, amount, ya, yb):
        spread_years(arr, amount, ya, yb, self.y0)

    # ---------------------------------------------------------------- state at time t
    def vec(self, i, t):
        v = self.phys.variables[i]
        acts = [a for a in self.by_var[i] if not a.failed and a.t_decide <= t]
        if v["type"] == "forest":
            land = eff = 0.0
            for a in acts:
                pl = a.final[0] * (np.clip((t - a.t_cons) / a.build, 0, 1) if a.build > 0 else float(t >= a.t_cons))
                if pl < 0:                                   # reconversion to farmland: the forest shrinks proportionally
                    keep = max(land + pl, 0.0) / land if land > 1e-12 else 0.0
                    land *= keep; eff *= keep
                    continue
                mat = np.clip((t - a.t_cons - a.build / 2) / a.mature, 0, 1) if a.mature > 0 else float(t >= a.t_cons)
                land += pl; eff += pl * mat
            return (float(eff), float(land))
        out = (0.0, 0.0) if v["type"] == "relocate" else (0.0,)
        for a in acts:
            if a.kind in STEP_KINDS:
                p = float(t >= a.t_cons + a.build)
            else:                                     # linear 0 -> 100 % over the build and diffusion years
                dur = a.build + a.diffuse
                p = float(np.clip((t - a.t_cons) / dur, 0, 1)) if dur > 0 else float(t >= a.t_cons)
            out = tuple(o + p * (f - o) for o, f in zip(out, a.final))
        return out

    def raw(self, t):
        r = np.zeros(len(self.phys.layout))
        for i in range(self.nV):
            if not self.by_var[i]:
                continue
            for c, x in zip(self.cols[i], self.vec(i, t)):
                r[c] = x
        return r

    def in_place(self, i, t):
        """Level fully in place at t (option index or rate), for the GUI and the maintenance cost."""
        v = self.phys.variables[i]
        x = self.vec(i, t)
        if v["type"] == "levee":
            hs = v["heights"]
            return x[0] * max(hs)
        return x

    # ---------------------------------------------------------------- yearly accounting
    def accrue(self, t1):
        cfg, P, V = self.cfg, self.phys, self.phys.variables
        M = cfg["measures"]
        for y in range(int(self.accrued_to), int(min(t1, self.y1))):
            k = y - self.y0
            tm = y + 0.5
            m = 0.0
            for i, v in enumerate(V):
                if not self.by_var[i]:
                    continue
                m0 = m
                x = self.vec(i, tm)
                if v["type"] == "levee" and x[0] > 0:
                    lv = x[0] * (len(v["heights"]) - 1)
                    lo = int(np.floor(lv)); hi = min(lo + 1, len(v["heights"]) - 1); w = lv - lo
                    ini = (1 - w) * P.cost_parts(i, lo)[0] + w * P.cost_parts(i, hi)[0]
                    ann = (1 - w) * P.cost_parts(i, lo)[1] + w * P.cost_parts(i, hi)[1]
                    m += ann + M["levee"]["maintenance_fraction"] * ini
                elif v["type"] == "retention" and x[0] > 0:
                    ini, ann = P.cost_parts(i, 1)
                    m += x[0] * (ann + M["retention"]["maintenance_fraction"] * ini)
                elif v["type"] == "relocate":
                    for o, share in ((1, x[0]), (2, x[1])):
                        if share > 0:
                            ini, ann = P.cost_parts(i, o)
                            m += share * (ann + M[kind_of(v, o)]["maintenance_fraction"] * ini)
                elif v["type"] == "paddy_dam" and x[0] > 0:
                    m += P.cost_parts(i, x[0])[1]
                elif v["type"] == "forest" and x[1] > 0:
                    m += P.cost_parts(i, x[1])[1]
                elif x[0] > 0:                               # pumps, storage, institutional measures
                    if v["continuous"] or len(M.get(v["type"], {}).get("levels", [])) > 2:
                        m += P.cost_parts(i, x[0])[1]
                    else:
                        ini, ann = P.cost_parts(i, len(v["options"]) - 1)
                        m += x[0] * (ann + M.get(v["type"], {}).get("maintenance_fraction", 0.0) * ini)
                self.maint_var[k, i] = m - m0
            self.maint[k] = m
            kinds = [a.kind for a in self.actions if a.t_decide <= tm < a.t_cons]
            self.coord[k] = self.social.coordination_cost(y, kinds)
        self.accrued_to = max(self.accrued_to, min(t1, self.y1))

    # ---------------------------------------------------------------- periods (physics deferred or immediate)
    def add_periods(self, t0, t1):
        step = self.cfg["years"]["eval_step"]
        for a in range(int(t0), int(min(t1, self.y1)), step):
            m = a + step / 2
            self.pending.append({"t": a, "mid": m, "raw": self.raw(m), "env": self.world.env(m),
                                 "asset": self.world.asset(m), "pop": self.world.pop(m), "inst": self.institutions(m)})

    def institutions(self, t):
        """Share in place of the institutional measures (no physics), by kind."""
        out = {}
        for i, v in enumerate(self.phys.variables):
            if v.get("institutional") and self.by_var[i]:
                out[v["type"]] = max(out.get(v["type"], 0.0), self.vec(i, t)[0])
        return out


def spread_years(arr, amount, ya, yb, y0):
    """Spread `amount` uniformly over the years [ya, yb) into arr (index 0 = y0)."""
    n = len(arr)
    span = max(yb - ya, 1e-9)
    for y in range(int(np.floor(ya)), int(np.ceil(yb))):
        k = y - y0
        if 0 <= k < n:
            ov = min(yb, y + 1) - max(ya, y)
            if ov > 0:
                arr[k] += amount * ov / span


def institutional_effects(cfg, asset, pop, inst):
    """(asset factor, population factor, damage multiplier, human-risk multiplier) with the
    institutional measures in place (shares s in [0, 1]; effects measures.<kind>.effect, assumptions):
        zoning      growth of assets / people in flood-prone areas x (1 - growth_block s)
        induce      people x (1 - pop s), assets x (1 - asset s)
        floodproof  damage x (1 - damage s)
        evacuation  people at risk x (1 - risk s)   (damage unchanged)"""
    M = cfg["measures"]
    s = inst.get("zoning", 0.0)
    if s:
        g = M["zoning"]["effect"]["growth_block"] * s
        asset = 1 + (asset - 1) * (1 - g) if asset > 1 else asset
        pop = 1 + (pop - 1) * (1 - g) if pop > 1 else pop
    s = inst.get("induce", 0.0)
    if s:
        pop *= 1 - M["induce"]["effect"]["pop"] * s
        asset *= 1 - M["induce"]["effect"]["asset"] * s
    dm = 1 - M["floodproof"]["effect"]["damage"] * inst.get("floodproof", 0.0) if "floodproof" in M else 1.0
    rk = 1 - M["evacuation"]["effect"]["risk"] * inst.get("evacuation", 0.0) if "evacuation" in M else 1.0
    return asset, pop, dm, rk


def evaluate(states, phys):
    """Run the physics for the pending periods of all states in one batch."""
    rows = [(s, p) for s in states for p in s.pending]
    if not rows:
        return
    raw = np.array([p["raw"] for _, p in rows])
    env = {k: np.array([p["env"][k] for _, p in rows]) for k in rows[0][1]["env"]}
    out = phys.predict(raw, env)
    agri = phys.agriculture(raw)
    zones = [k[3:] for k in out if k.startswith("dz:")]
    for j, (s, p) in enumerate(rows):
        a, q, dm, rk = institutional_effects(s.cfg, p["asset"], p["pop"], p.get("inst", {}))
        p.update(damage=float(out["damage"][j]) * a * dm, pop_hazard=float(out["pop_hazard"][j]) * q,
                 pop_risk=float(out["pop_hazard"][j]) * q * rk,
                 households=float(out["households"][j]) * q, livability=float(out["livability"][j]),
                 ecology=float(out["ecology"][j]), agriculture=float(agri[j]),
                 damage_zone={z: float(out["dz:" + z][j]) * a * dm for z in zones})
    for s in states:
        s.periods = s.periods + s.pending
        s.pending = []


def observe(state, t, ref, cfg):
    """What a controller knows at decision year t (no observation error): the outcome of the last evaluated period, the
    current climate and exposure, and the forecast of the 2100 climate (world.forecast)."""
    w, mon = state.world, cfg["monitoring"]
    last = state.periods[-1] if state.periods else None
    obs = {"t": t}
    if last is not None:
        obs["damage_ratio"] = last["damage"] / max(ref["damage"], 1e-9)
        obs["pop_ratio"] = last["pop_risk"] / max(ref["pop_hazard"], 1e-9)
        prev = state.periods[-3] if len(state.periods) >= 3 else None
        if prev is not None:
            obs["damage_ratio_prev"] = prev["damage"] / max(ref["damage"], 1e-9)
            obs["pop_ratio_prev"] = prev["pop_risk"] / max(ref["pop_hazard"], 1e-9)
    tm = t - cfg["years"]["eval_step"] / 2 if t > cfg["years"]["start"] else t
    obs["rain"], obs["slr"], obs["asset"], obs["pop"] = w.rain(tm), w.slr(tm), w.asset(tm), w.pop(tm)
    obs["forecast"] = w.forecast(t)
    obs["effects"] = {}
    obs["phases"] = [{"kind": a.kind, "phase": ph, "years": getattr(a, ph)} for a in state.actions
                     if not a.failed and state.complete_year(a) <= t
                     for ph in ("build", "diffuse", "mature") if getattr(a, ph) > 0]
    obs["regime"] = w.regime_at(t - mon.get("regime_lag_years", 0))
    obs["failures"] = [e for e in state.log if e["event"] == "failed" and e["year"] <= t]
    return obs


def run(world, controller, phys, cfg, ref, state=None, t_from=None, batch=False, snapshots=()):
    """Simulate `controller` in `world` from `state` (default: nothing done) and decision year t_from.
    batch=True (open-loop controllers only): the periods stay pending, evaluate() many states at once."""
    physics_in_loop = controller.needs_obs
    st = state if state is not None else State(cfg, phys, world, Social(cfg, world.p["social"], controller.modes))
    st.cap_factor = getattr(controller, "cap_factor", 1.0)
    step = cfg["years"]["decision_step"]
    snaps = {}
    for t in decision_years(cfg):
        if t_from is not None and t < t_from:
            continue
        st.settle(t)
        if t in snapshots:
            snaps[t] = st.copy()
        obs = observe(st, t, ref, cfg) if controller.needs_obs else None
        controller.decide(t, st, obs)
        st.process(t)
        st.add_periods(t, t + step)
        if physics_in_loop:
            evaluate([st], phys)
        st.accrue(t + step)
    if not batch:
        evaluate([st], phys)
    st.snapshots = snaps
    return st


# ---------------------------------------------------------------- scoring
def discount(cfg, years):
    return (1 + cfg["discount_rate"]) ** -(np.asarray(years, float) - cfg["years"]["start"])


def criteria_schedule(cfg, regimes):
    """[[year, regime name], ...] -> function year -> regime dict."""
    R = cfg["regimes"]

    def at(y):
        name = regimes[0][1]
        for yy, n in regimes:
            if y >= yy:
                name = n
        return R[name]
    return at


def score(st, cfg, ref, regimes, t_from=None):
    """Utility and metrics of an evaluated state.  regimes = [[year, name], ...] (criteria in force).

    U = sum_p d_p [-w_d D_p/D0 - w_pop P_p/P0 + w_e (E_p - E0)/10 + w_l (L_p - L0)/10 + w_a (A_p - 100)/10
                   - penalty x 1(D_p/D0 > theta_d or P_p/P0 > theta_p)] / sum_p d_p
        - sum_y w_c(y) df(y) cost(y) / total budget
    (d_p = discount factors summed over the period's years; ref = no measures, current climate)."""
    at = criteria_schedule(cfg, regimes)
    step = cfg["years"]["eval_step"]
    p_ann = cfg["representative_event"]["annual_probability"]
    P = [p for p in st.periods if t_from is None or p["t"] >= t_from]
    lam = cfg["threshold_penalty"]
    num = den = 0.0
    ead = 0.0
    viol_years, atp = 0, None
    rows = []
    for p in P:
        d = float(discount(cfg, np.arange(p["t"], p["t"] + step)).sum())
        R = at(p["t"]); w = R["weights"]; th = R["thresholds"]
        rd, rp = p["damage"] / max(ref["damage"], 1e-9), p["pop_risk"] / max(ref["pop_hazard"], 1e-9)
        viol = rd > th["damage_ratio"] or rp > th["pop_ratio"]
        u = (-w.get("damage", 0) * rd - w.get("pop", 0) * rp + w.get("ecology", 0) * (p["ecology"] - ref["ecology"]) / 10
             + w.get("livability", 0) * (p["livability"] - ref["livability"]) / 10 + w.get("agriculture", 0) * (p["agriculture"] - 100) / 10
             - lam * viol)
        num += d * u; den += d
        ead += p_ann * p["damage"] * d
        if viol:
            viol_years += step
            atp = p["t"] if atp is None else atp
        rows.append({"t": p["t"], "damage_ratio": rd, "pop_ratio": rp, "violation": viol})
    y0 = cfg["years"]["start"]
    yrs = np.arange(st.y0, st.y1)
    sel = yrs >= (t_from if t_from is not None else y0)
    df = discount(cfg, yrs)
    cost_y = st.capital + st.maint + st.coord
    wc = np.array([at(y)["weights"].get("cost", 0) for y in yrs])
    cost_pv = float((df * cost_y)[sel].sum())
    U = (num / max(den, 1e-12)) - float((wc * df * cost_y)[sel].sum()) / cfg["budget"]["total_oku"]
    n_per = max(len(P), 1)
    return {"U": float(U), "ead_pv": float(ead), "cost_pv": cost_pv,
            "capital_pv": float((df * st.capital)[sel].sum()), "maint_pv": float((df * st.maint)[sel].sum()),
            "coord_pv": float((df * st.coord)[sel].sum()), "change_pv": float((df * st.change)[sel].sum()),
            "violation_share": viol_years / (n_per * step), "atp": atp,
            "mean_ecology": float(np.mean([p["ecology"] for p in P])) if P else None,
            "mean_livability": float(np.mean([p["livability"] for p in P])) if P else None,
            "mean_agriculture": float(np.mean([p["agriculture"] for p in P])) if P else None,
            "mean_pop": float(np.mean([p["pop_hazard"] for p in P])) if P else None,
            "mean_pop_risk": float(np.mean([p["pop_risk"] for p in P])) if P else None,
            "failures": sum(1 for e in st.log if e["event"] == "failed"),
            "deferred": sum(1 for e in st.log if e["event"] == "deferred"),
            "spent": st.spent}
