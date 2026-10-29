"""Stakeholder utilities, regret and the minimax-regret recommendation.

Utility of stakeholder j for metrics m (one policy in one scenario):

    U_j = - w_damage  D_j / D_ref - w_cost C_j / budget - w_households H / H_ref
          + w_livability (L - L_ref) / 10 + w_ecology (E - E_ref) / 10

D_j = damage in the stakeholder's zones, C_j = project cost in its cost
zones (where the measures are; [] = the stakeholder pays nothing), refs =
the no-measure policy in the nominal scenario (10 = ten score points).

Regret of policy p in scenario s: R = max_{p' feasible in s} U(p', s) - U(p, s),
feasible = total cost (with the scenario's cost factor) within the budget.
A policy satisfices in s when it is feasible and R <= regret_threshold.
"""
from __future__ import annotations

import numpy as np


def _scope(d, zones):
    if zones == "all":
        return sum(d.values())
    return sum(d.get(z, 0.0) for z in zones)


def utility(m, st, ref, budget):
    w = st["weights"]
    D = _scope(m["damage_zone"], st["damage_zones"])
    C = _scope(m["cost_zone"], st["cost_zones"]) if st["cost_zones"] else 0.0
    Dr = max(_scope(ref["damage_zone"], st["damage_zones"]), 1e-9)
    Hr = max(ref["households"], 1e-9)
    return (-w.get("damage", 0) * D / Dr - w.get("cost", 0) * C / budget - w.get("households", 0) * m["households"] / Hr
            + w.get("livability", 0) * (m["livability"] - ref["livability"]) / 10.0
            + w.get("ecology", 0) * (m["ecology"] - ref["ecology"]) / 10.0)


def regret(U, feasible):
    """U, feasible: (n_policy, n_scenario).  Returns regret (n_policy, n_scenario)."""
    U = np.asarray(U, float); F = np.asarray(feasible, bool)
    best = np.where(F.any(0), np.where(F, U, -np.inf).max(0), U.max(0))
    return best[None] - U


def summary(R, feasible, threshold):
    F = np.asarray(feasible, bool)
    return {"max": R.max(1).tolist(), "mean": R.mean(1).tolist(), "p90": np.quantile(R, 0.9, axis=1).tolist(),
            "satisficing": ((R <= threshold) & F).mean(1).tolist(), "feasible": F.mean(1).tolist()}


def minimax(R, eligible=None):
    mx = R.max(1).astype(float)
    if eligible is not None:
        mx = np.where(eligible, mx, np.inf)
    return int(np.argmin(mx))


def evaluate(matrix, stakeholders, ref, budget, eligible=None):
    """matrix[p][s] = post-applied metrics.  Returns per-stakeholder U, R,
    summaries, the minimax-regret policy for each, and an overall
    recommendation minimising the largest max-regret relative to each
    stakeholder's threshold."""
    n_p, n_s = len(matrix), len(matrix[0])
    feas = np.array([[matrix[p][s]["cost"] <= budget * (1 + 1e-9) for s in range(n_s)] for p in range(n_p)])
    out = {"feasible": feas.tolist(), "stakeholders": {}}
    rel = np.zeros((len(stakeholders), n_p))
    for j, (name, st) in enumerate(stakeholders.items()):
        U = np.array([[utility(matrix[p][s], st, ref, budget) for s in range(n_s)] for p in range(n_p)])
        R = regret(U, feas)
        rec = minimax(R, eligible)
        out["stakeholders"][name] = {"utility": U.tolist(), "regret": R.tolist(), "summary": summary(R, feas, st["regret_threshold"]),
                                     "minimax": rec}
        rel[j] = R.max(1) / max(st["regret_threshold"], 1e-9)
    worst = rel.max(0)
    if eligible is not None:
        worst = np.where(eligible, worst, np.inf)
    out["overall"] = {"minimax": int(np.argmin(worst)), "max_relative_regret": rel.max(0).tolist()}
    return out
