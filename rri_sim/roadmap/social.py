"""SOCIAL / INSTITUTIONAL layer: stakeholder participation -> feasibility, adoption, consensus time.

Each stakeholder s has an engagement e_s(t) in [0, 1] (trust and shared
understanding built in a consultation process):

    e_s(t) = trust_s (1 - exp(-(t - t0_s) / tau))      for t >= t0_s, else 0

t0_s depends on the participation mode:
    early   (早期参加)  t0 = start year (pays the coordination cost from then on)
    staged  (段階参加)  t0 = decision year of the first measure that needs s
    late    (事後参加)  t0 = as staged, but s is informed after the decision:
                        trust_s = late.trust, consensus x late.consensus_factor,
                        failure x late.fail_factor, coordination cost x late.conflict_cost_factor

A measure of type m decided at t needs the stakeholders S_m (measures.<m>.stakeholders);
with e = mean engagement of S_m at t:
    consensus time   C = consensus_m (1 - a e) [x late factor]
    failure prob.    p = fail_prob_m x fail_scale x (1 - b e) [x late factor]   (drawn once per attempt)
    adoption ceiling A = lo_m + (hi_m - lo_m) e(end of planting / diffusion) + shift   (paddy dams, forest, housing)
Coordination cost per year = sum over engaged s of coord_cost_s x scale (x conflict factor
while a late stakeholder is in a consensus process) + consensus_cost per measure in consensus.
"""
from __future__ import annotations

import math

import numpy as np

MODES = ("early", "staged", "late")


class Social:
    def __init__(self, cfg, values, modes=None):
        """values: tau, alpha, fail_scale, adoption_shift, coord_scale (world.social_values)."""
        self.cfg = cfg
        self.P = cfg["participation"]
        self.M = cfg["measures"]
        self.v = values
        self.modes = dict(self.P["mode"], **(modes or {}))
        self.since = {s: (float(cfg["years"]["start"]) if self.modes.get(s) == "early" else None) for s in self.P["stakeholders"]}

    def copy(self):
        o = Social.__new__(Social)
        o.cfg, o.P, o.M, o.v, o.modes, o.since = self.cfg, self.P, self.M, self.v, self.modes, dict(self.since)
        return o

    def late(self, s):
        return self.modes.get(s) == "late"

    def engage(self, stakeholders, t):
        for s in stakeholders:
            if self.since.get(s) is None:
                self.since[s] = float(t)

    def engagement(self, s, t):
        t0 = self.since.get(s)
        if t0 is None or t < t0:
            return 0.0
        e = 1.0 - math.exp(-(t - t0) / max(self.v["tau"], 1e-6))
        return e * (self.P["late"]["trust"] if self.late(s) else 1.0)

    def mean_engagement(self, kind, t):
        S = self.M[kind]["stakeholders"]
        return float(np.mean([self.engagement(s, t) for s in S])) if S else 1.0

    def any_late(self, kind):
        return any(self.late(s) for s in self.M[kind]["stakeholders"])

    def consensus_years(self, kind, t, base=None):
        e = self.mean_engagement(kind, t)
        c = (self.M[kind]["consensus"] if base is None else base) * (1.0 - self.v["alpha"] * e)
        return c * (self.P["late"]["consensus_factor"] if self.any_late(kind) else 1.0)

    def fail_prob(self, kind, t):
        e = self.mean_engagement(kind, t)
        p = self.M[kind]["fail_prob"] * self.v["fail_scale"] * (1.0 - self.P["fail_engagement_reduction"] * e)
        return float(min(0.95, p * (self.P["late"]["fail_factor"] if self.any_late(kind) else 1.0)))

    def adoption(self, kind, t):
        lim = self.P["adoption"].get(kind)
        if lim is None:
            return 1.0
        e = self.mean_engagement(kind, t)
        return float(np.clip(lim[0] + (lim[1] - lim[0]) * e + self.v["adoption_shift"], 0.05, 1.0))

    def coordination_cost(self, year, in_consensus):
        """Cost [億円] of one year; in_consensus = measure kinds in a consensus process that year."""
        c = 0.0
        late_conflict = {s for k in in_consensus for s in self.M[k]["stakeholders"] if self.late(s)}
        for s, st in self.P["stakeholders"].items():
            t0 = self.since.get(s)
            if t0 is not None and year >= t0:
                f = self.P["late"]["conflict_cost_factor"] if s in late_conflict else 1.0
                c += st["coord_cost_oku_yr"] * f
        c += self.P["consensus_cost_oku_yr"] * len(in_consensus)
        return c * self.v["coord_scale"]

    def status(self, t):
        """{stakeholder: {mode, since, engagement}} (GUI)."""
        return {s: {"mode": self.modes.get(s), "since": self.since.get(s), "engagement": round(self.engagement(s, t), 3)}
                for s in self.P["stakeholders"]}
