"""Future option space of Robust DAPP pathways (docs/11 11.9.5): how many of the initial options are still possible
after the choices of a pathway.

    option          (variable, level index >= 1) of the decision space (landuse.Transitions levels: levee +1 / +2 m,
                    paddy dams 50 / 100 %, raise / relocate, ...); options are levels to reach - scaling back and
                    reconversion are transitions that can restore an option (forest -> paddy dams)
    base set        the options feasible at the start (no measures, 2020); levels outside the pathways network or above
                    the total budget from the start are not in it (never "lost")
    status at t     used (the level or a higher one is in place / committed; a raise is used only if it was done), feasible,
                    conditional (feasible only by reconverting young forest, before its reconversion deadline), or lost:
        irreversible_land_use   relocation sites (移転跡地: no new housing) - a raise is no longer possible
        land_use_conflict       paddy dams on paddies afforested longer ago than measures.forest.reconvert.max_age_years
        irreversible_measure    the same conflict where forest may never be reconverted (reconvert not allowed)
        transition_unavailable  no node with that level is reachable in the pathways network from the current node
        lead_time               it could not be complete before the end (t + nominal lead time > end year)
        budget                  its own initial cost (increment + change cost) exceeds the total budget left
    remaining_option_ratio = feasible (incl. conditional) / base; option loss = lost / base (used options are not lost).
Retention basins convert farmland irreversibly (landuse.Transitions.landuse: 遊水地) but no option of the decision space
needs that land (paddy-dam rates are shares of the zone's paddies; forest candidates are high-HAND farmland, retention sites
low-HAND), so they show as converted land, not as option loss.
"""
from __future__ import annotations

import numpy as np

from ..roadmap import sim as SM

REASONS = ("irreversible_land_use", "land_use_conflict", "irreversible_measure", "transition_unavailable", "lead_time", "budget")
LABELS = {"irreversible_land_use": "不可逆な土地利用（移転跡地）", "land_use_conflict": "土地利用の競合（森林化と田んぼダム, 再転換期限後）",
          "irreversible_measure": "不可逆な施策（再転換できない森林）", "transition_unavailable": "移行できない（経路網に移行先がない）",
          "lead_time": "リードタイム（2100年までに完成しない）", "budget": "予算（総額の残りで足りない）"}
FEASIBLE, CONDITIONAL, USED = 0, 1, 2
CODE = {r: 3 + k for k, r in enumerate(REASONS)}
OUTSIDE = 9                                                       # not in the base set


def closure(A):
    """Reachability (incl. itself) over the directed graph A (N, N) bool (Floyd-Warshall on booleans)."""
    R = A.copy() | np.eye(len(A), dtype=bool)
    for k in range(len(A)):
        R |= R[:, k:k + 1] & R[k:k + 1, :]
    return R


class OptionSpace:
    """Option set of a Robust DAPP network (robust.Net) and the status of each option for a pathway state.
    network=False: pathways that are not bound to the nodes (adaptive.py): no transition_unavailable, every level is an option."""

    def __init__(self, net, network=True):
        self.net, T = net, net.T
        self.T = T
        self.opts = [(i, li) for i in range(len(T.V)) for li in range(1, len(T.levels[i]))]
        self.labels = [f"{T.V[i]['name']} {T.label(i, li)}" for i, li in self.opts]
        self.lead = np.array([T.lead_nominal[T.kind(i, li)] for i, li in self.opts])
        self.reach = self.top = None
        if network:
            G = np.array(net.nodes)
            M = net.mat
            self.reach = {True: closure(M["ok"]), False: closure(M["ok"] & ~M["conditional"])}    # with / without reconversion
            self.top = {k: np.array([[G[R[c], i].max() for i in range(len(T.V))] for c in range(len(G))]) for k, R in self.reach.items()}
        self.forest_of = {z: T.forest[z] for z in T.paddy}
        self.reconvert = (net.rc["measures"].get("forest", {}).get("reconvert") or {}).get("allowed", False)
        st0 = SM.State(net.rc, net.phys, net.worlds[0], None)
        st0.allow_decrease = True
        self.base = np.isin(self.status(st0, 0, net.y0, base=False), (FEASIBLE, CONDITIONAL))

    def levels(self, st):
        return [int(np.argmin([abs(st.committed[i] - lv) for lv in self.T.levels[i]])) for i in range(len(self.T.V))]

    def status(self, st, node, t, base=True):
        """(n_options,) status codes of the options for the state st, the pathway's current node and decision year t."""
        T, net = self.T, self.net
        g = self.levels(st)
        left = net.rc["budget"]["total_oku"] - st.spent
        out = np.empty(len(self.opts), np.int8)
        young = {i: st.can_decrease(i, t) for i in set(self.forest_of.values()) if i is not None and g[i] > 0}
        rec = all(young.values()) if young else False
        for o, (i, li) in enumerate(self.opts):
            v = T.V[i]
            if base is not False and not self.base[o]:
                out[o] = OUTSIDE
                continue
            if v["type"] == "relocate" and li == 1 and g[i] == 2:          # relocated: raised houses only if raised before
                done = any(not a.failed and a.kind == "raise" for a in st.by_var[i])
                out[o] = USED if done else CODE["irreversible_land_use"]
                continue
            if g[i] >= li:
                out[o] = USED
                continue
            cond = False
            if v["type"] == "paddy_dam" and T.levels[i][li] > T.paddy_available(g, v["zone"]) + 1e-9:
                jf = self.forest_of.get(v["zone"])
                if not self.reconvert:
                    out[o] = CODE["irreversible_measure"]; continue
                if jf is None or not young.get(jf, False):
                    out[o] = CODE["land_use_conflict"]; continue
                cond = True                                               # only by reconverting the young forest
            k = cond or rec
            if self.top is not None and self.top[k][node, i] < li:
                out[o] = CODE["transition_unavailable"]; continue
            if t + self.lead[o] > net.y1 + 1e-9:
                out[o] = CODE["lead_time"]; continue
            if self.option_cost(i, li, g[i]) > left + 1e-9:
                out[o] = CODE["budget"]; continue
            out[o] = CONDITIONAL if cond else FEASIBLE
        return out

    def option_cost(self, i, li, cur):
        """Initial cost of taking variable i from level cur to li (increment + change cost of a replaced option, as sim.State)."""
        T = self.T
        c = max(T.costs[i][li][0] - T.costs[i][cur][0], 0.0)
        if not T.V[i]["continuous"] and cur:
            c += self.net.rc["measures"][T.kind(i, cur)]["change_cost_fraction"] * T.costs[i][cur][0]
        return c

    def deadline(self, st, zone):
        """Last year the forest of `zone` can still be reconverted (planting year + max_age_years), or None."""
        jf = self.forest_of.get(zone)
        rc = self.net.rc["measures"].get("forest", {}).get("reconvert") or {}
        acts = [a for a in st.by_var[jf] if not a.failed and a.final[0] > 0] if jf is not None else []
        return max(a.t_cons for a in acts) + rc.get("max_age_years", 0) if acts and rc.get("allowed") else None


class Recorder:
    """robust.simulate hook: option status before ("pre") and after ("post") each decision of every pathway."""

    def __init__(self, space):
        self.space = space
        self.snap = {}

    def __call__(self, when, t, st, ctrl):
        self.snap.setdefault(id(ctrl), {}).setdefault(when, {})[t] = self.space.status(st, ctrl.cur, t)


def summarise(space, rec, ctrls, recs, years, trees, inits, nS, checkpoints=(2050, 2090)):
    """Per initial node: time series (remaining ratio, used, conditional, loss by reason; mean / p10 / p90), checkpoint values,
    move-driven option loss per branch of the pathway tree (pre -> post of a decision) and time-driven loss (post of the
    previous decision -> pre of this one: reconversion deadlines, lead times)."""
    nb = max(int(space.base.sum()), 1)
    out = {"options": [{"label": l, "lead": float(d), "base": bool(b)} for l, d, b in zip(space.labels, space.lead, space.base)],
           "n_base": int(space.base.sum()), "reasons": list(REASONS), "labels": LABELS, "years": list(years), "inits": {}}
    lost_codes = np.array([CODE[r] for r in REASONS])
    for k, i in enumerate(inits):
        idx = range(k * nS, (k + 1) * nS)
        post = np.array([[rec.snap[id(ctrls[j])]["post"][t] for t in years] for j in idx])       # (S, Y, O)
        pre = np.array([[rec.snap[id(ctrls[j])]["pre"][t] for t in years] for j in idx])
        rem = np.isin(post, (FEASIBLE, CONDITIONAL)).sum(2) / nb
        loss = {r: (post == CODE[r]).sum(2) / nb for r in REASONS}
        q = lambda a: {"mean": a.mean(0).round(4).tolist(), "p10": np.quantile(a, 0.1, 0).round(4).tolist(),  # noqa: E731
                       "p90": np.quantile(a, 0.9, 0).round(4).tolist()}
        ts = {"remaining": q(rem), "used": q((post == USED).sum(2) / nb), "conditional": q((post == CONDITIONAL).sum(2) / nb),
              "loss": {r: v.mean(0).round(4).tolist() for r, v in loss.items()}}
        R = [recs[j] for j in idx]
        ts["cost_cum"] = np.mean([r["cost_path"] for r in R], 0).round(1).tolist()
        dr = np.array([[next((p[1] for p in r["series"] if p[0] == t), np.nan) for t in years] for r in R])
        ts["damage_ratio"] = np.nanmean(dr, 0).round(4).tolist()
        ts["met_so_far"] = np.mean([[r["first_violation"] is None or r["first_violation"] > t for t in years] for r in R], 0).round(4).tolist()
        cp = {}
        for y in checkpoints:
            if y in years:
                n = years.index(y)
                cp[str(y)] = {"remaining": float(rem[:, n].mean()), "loss": {r: float(v[:, n].mean()) for r, v in loss.items()},
                              "loss_total": float(sum(v[:, n].mean() for v in loss.values()))}
        struct = sum(loss[r] for r in ("irreversible_land_use", "land_use_conflict", "irreversible_measure", "transition_unavailable"))
        # move-driven loss per branch (the trie of pathway_tree) and time-driven loss per decade
        tree = trees[str(i)]
        child = {(b["parent"], b["node"]): b["id"] for b in tree if b["parent"] is not None}
        br = {}

        def add(bid, a, b):
            new = np.isin(b, lost_codes) & ~np.isin(a, lost_codes)
            d = br.setdefault(bid, {"n": 0, "lost": dict.fromkeys(REASONS, 0), "options": {}, "d_remaining": 0.0})
            d["n"] += 1
            d["d_remaining"] += (np.isin(b, (FEASIBLE, CONDITIONAL)).sum() - np.isin(a, (FEASIBLE, CONDITIONAL)).sum()) / nb
            for o in np.flatnonzero(new):
                rs = REASONS[int(b[o]) - 3]
                d["lost"][rs] += 1
                d["options"][f"{space.labels[o]}|{rs}"] = d["options"].get(f"{space.labels[o]}|{rs}", 0) + 1
        timed = np.zeros((len(years), len(REASONS)))
        for s_, j in enumerate(idx):
            ev = recs[j]["events"]
            add(0, pre[s_, 0], post[s_, 0])
            node = 0
            for e in ev[1:]:
                node = child.get((node, e["to"]), node)
                n = years.index(e["year"])
                add(node, pre[s_, n], post[s_, n])
            for n in range(1, len(years)):
                new = np.isin(pre[s_, n], lost_codes) & ~np.isin(post[s_, n - 1], lost_codes)
                for o in np.flatnonzero(new):
                    timed[n, int(pre[s_, n, o]) - 3] += 1
        for d in br.values():
            n = max(d["n"], 1)
            d["d_remaining"] = round(d["d_remaining"] / n, 4)
            d["lost"] = {r: round(v / n, 3) for r, v in d["lost"].items()}
            d["options"] = sorted(([*k.split("|"), round(v / n, 3)] for k, v in d["options"].items()), key=lambda x: -x[2])[:8]
        out["inits"][str(i)] = {"timeseries": ts, "checkpoints": cp, "remaining_mean": float(rem.mean()),
                                "transition_feasibility": float((1 - struct).mean()), "branches": {str(b): d for b, d in br.items()},
                                "time_driven": {r: (timed[:, q_] / nS).round(3).tolist() for q_, r in enumerate(REASONS)}}    # options / pathway
    return out


def export(out_dir, ev, names):
    """CSV / JSON of the extended scorecard, the time series, the loss by reason and the move-driven loss per branch."""
    import csv
    import json
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    O = ev["option_space"]
    (out / "option_space.json").write_text(json.dumps(O, ensure_ascii=False), encoding="utf-8")
    cols = ["node", "coverage", "coverage_perfect", "mean_damage_ratio", "mean_cost", "max_regret", "mean_regret",
            "remaining_options_2050", "remaining_options_mean", "option_loss_2050", "transition_feasibility"]
    with open(out / "scorecard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial"] + cols + ["lockin_options", "promising"])
        for r in ev["table"]:
            w.writerow([names.get(str(r["node"]), r["node"])] + [r.get(c) for c in cols] + [r["lockin"].get("options"), bool(r.get("promising"))])
    with open(out / "timeseries.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", "year", "remaining_mean", "remaining_p10", "remaining_p90", "used", "conditional", "damage_ratio",
                    "met_so_far", "cost_cum_pv"] + [f"loss_{r}" for r in REASONS])
        for i, d in O["inits"].items():
            ts = d["timeseries"]
            for n, y in enumerate(O["years"]):
                w.writerow([i, y, ts["remaining"]["mean"][n], ts["remaining"]["p10"][n], ts["remaining"]["p90"][n], ts["used"]["mean"][n],
                            ts["conditional"]["mean"][n], ts["damage_ratio"][n], ts["met_so_far"][n], ts["cost_cum"][n]]
                           + [ts["loss"][r][n] for r in REASONS])
    with open(out / "branches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", "branch", "n", "d_remaining"] + [f"lost_{r}" for r in REASONS] + ["options_lost"])
        for i, d in O["inits"].items():
            for b, x in d["branches"].items():
                w.writerow([i, b, x["n"], x["d_remaining"]] + [x["lost"][r] for r in REASONS]
                           + ["; ".join(f"{o} ({LABELS[r]}) {v}" for o, r, v in x["options"])])
