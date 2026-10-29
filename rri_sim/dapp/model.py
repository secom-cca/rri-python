"""Simulate one strategy under one climate path, decade by decade.

For decade y .. y+9:
  1. measures whose completion year <= y + 5 (mid-decade) are in place;
     forest effect = (mid-year - completion) / ramp_years, clipped to [0, 1]
     (mean over the forested zones, quantised to 0.25)
  2. decision at y: static strategies start the actions scheduled for y (and
     any deferred ones); the adaptive strategy starts its next pathway bundle
     when the indicator observed `monitor_lag` decades earlier (damage or
     exposed population of the representative event, relative to no measures
     under the current climate) exceeds trigger_ratio
  3. an action costs its initial cost increment (new option - current option)
     plus change_cost_fraction x the initial cost of the replaced option; it
     starts only if the decade cap (per_decade_oku x budget_cap_factor) and the
     total budget allow, otherwise it waits for the next decade
  4. the representative event is simulated with the measures in place and the
     decade's climate (rain factor, sea-level rise at mid-decade)
  5. expected damage of each year = annual_probability x event damage;
     maintenance from completion; everything discounted to the start year
  6. an adaptation tipping point (ATP) is the first decade whose event damage
     or exposed population exceeds atp ratio x the no-measure current-climate value
"""
from __future__ import annotations

import numpy as np


def climate_at(path, year):
    pts = np.array(path["points"], float)
    return float(np.interp(year, pts[:, 0], pts[:, 1])), float(np.interp(year, pts[:, 0], pts[:, 2]))


def parse_action(spec, variables):
    """'type[@zone]:option' -> [(variable index, option)]."""
    head, opt = spec.rsplit(":", 1)
    typ, _, zone = head.partition("@")
    out = [(i, int(opt)) for i, v in enumerate(variables)
           if v.type == typ and (not zone or v.zone == zone) and int(opt) < len(v.options)]
    return out


def action_kind(v, opt):
    p = v.params[opt]
    return (p or {}).get("type", v.type)


def years(cfg):
    y = cfg["years"]
    return list(range(y["start"], y["end"], y["step"]))


def simulate(name, spec, clim, ev, cfg, ref):
    """Returns the pathway record of strategy `name` under climate path `clim`."""
    V = ev.variables
    path = cfg["climate_paths"][clim]
    step, r = cfg["years"]["step"], cfg["discount_rate"]
    y0 = cfg["years"]["start"]
    p_ann = cfg["representative_event"]["annual_probability"]
    meas = cfg["measures"]
    B = cfg["budget"]
    cap = B["per_decade_oku"] * spec.get("budget_cap_factor", 1.0)
    df = lambda yr: (1 + r) ** -(yr - y0)  # noqa: E731
    planned = [0] * len(V)        # option committed (started or done)
    done_at = [None] * len(V)     # completion year of the committed option
    built = [0] * len(V)          # option in place
    built_done = [None] * len(V)
    pending = []                  # [(i, opt, completion year)]
    queue = []                    # actions waiting for budget
    spent = 0.0
    log, periods = [], []
    ini0, ann0 = ev.cost_parts([0] * len(V))
    next_bundle = 0
    if spec["kind"] == "static":
        sched = {int(y): acts for y, acts in spec["schedule"]}
    for k, y in enumerate(years(cfg)):
        mid = y + step / 2
        # completions up to the start of this decade's decision
        for (i, o, c) in list(pending):
            if c <= y:
                built[i], built_done[i] = o, c
                pending.remove((i, o, c))
        # ---- decision
        new = []
        if spec["kind"] == "static":
            new = sched.get(y, [])
        else:
            if k == 0:
                for _ in range(spec.get("initial", 1)):
                    if next_bundle < len(spec["pathway"]):
                        new += spec["pathway"][next_bundle]; next_bundle += 1
            else:
                lag = spec.get("monitor_lag", 1)
                obs = periods[max(0, k - lag)]
                if max(obs["damage_ratio"], obs["pop_ratio"]) > spec["trigger_ratio"] and next_bundle < len(spec["pathway"]):
                    new = spec["pathway"][next_bundle]; next_bundle += 1
                    log.append({"year": y, "event": "trigger", "observed_year": obs["year"],
                                "ratio": max(obs["damage_ratio"], obs["pop_ratio"]), "bundle": new})
        for a in new:
            queue += parse_action(a, V)
        decade_spend = 0.0
        for (i, o) in list(queue):
            if o <= planned[i]:
                queue.remove((i, o)); continue
            x_old = list(planned); x_new = list(planned); x_new[i] = o
            ini_old = ev.cost_parts(x_old)[0][i]
            ini_new = ev.cost_parts(x_new)[0][i]
            kind = action_kind(V[i], o)
            change = meas[action_kind(V[i], planned[i])]["change_cost_fraction"] * ini_old if planned[i] else 0.0
            c = max(ini_new - ini_old, 0.0) + change
            if decade_spend + c > cap + 1e-9 or spent + c > B["total_oku"] + 1e-9:
                continue                                   # deferred
            queue.remove((i, o))
            decade_spend += c; spent += c
            lead = meas[kind]["lead_years"]
            pending = [(pi, po, pc) for (pi, po, pc) in pending if pi != i]   # a newer decision replaces a pending one
            pending.append((i, o, y + lead))
            planned[i], done_at[i] = o, y + lead
            log.append({"year": y, "event": "start", "var": V[i].name, "option": V[i].options[o], "kind": kind,
                        "cost": c, "change_cost": change, "complete": y + lead})
        if queue:
            log.append({"year": y, "event": "deferred", "actions": [f"{V[i].name}: {V[i].options[o]}" for i, o in queue]})
        # ---- measures in place at mid-decade
        x = list(built); comp = list(built_done)
        for (i, o, c) in pending:
            if c <= mid:
                x[i], comp[i] = o, c
        fk = [min(max((mid - comp[i]) / meas["forest"]["ramp_years"], 0.0), 1.0) if meas["forest"]["ramp_years"] else 1.0
              for i, v in enumerate(V) if v.type == "forest" and x[i]]
        forest_k = round(float(np.mean(fk)) * 4) / 4 if fk else 1.0
        if fk and forest_k == 0.0:          # planted but no effect yet: simulate without the forest measures
            x = [0 if V[i].type == "forest" else xi for i, xi in enumerate(x)]
            forest_k = 1.0
        rf, slr = climate_at(path, mid)
        sim = {"rain_factor": round(rf, 3), "sea_level_rise_m": round(slr, 2), "forest_effect": forest_k}
        m = ev.simulate(x, sim)
        # ---- accounting (PV to the start year)
        yrs = np.arange(y, y + step)
        dfs = np.array([df(t) for t in yrs])
        dmg = p_ann * m["damage"] * dfs.sum()
        dmg_zone = {z: p_ann * v * dfs.sum() for z, v in m["damage_zone"].items()}
        cap_pv = decade_spend * df(y)
        ini, ann = ev.cost_parts(x)
        maint = sum(ann[i] * sum(df(t) for t in yrs if comp[i] is not None and t >= comp[i]) for i in range(len(V)) if x[i])
        cost_zone = {z: 0.0 for z in m["damage_zone"]}
        for e in log:
            if e.get("event") == "start" and e["year"] == y:
                zi = next(v.zone for v in V if v.name == e["var"])
                cost_zone[zi] = cost_zone.get(zi, 0.0) + e["cost"] * df(y)
        for i in range(len(V)):
            if x[i]:
                cost_zone[V[i].zone] = cost_zone.get(V[i].zone, 0.0) + ann[i] * sum(df(t) for t in yrs if comp[i] is not None and t >= comp[i])
        periods.append({"year": y, "mid": mid, "rain_factor": rf, "sea_level_rise_m": slr, "x": x, "forest_effect": forest_k,
                        "event_damage": m["damage"], "event_damage_zone": m["damage_zone"], "pop_hazard": m["pop_hazard"],
                        "households": m["households"], "livability": m["livability"], "ecology": m["ecology"],
                        "damage_ratio": m["damage"] / max(ref["damage"], 1e-9), "pop_ratio": m["pop_hazard"] / max(ref["pop_hazard"], 1e-9),
                        "exp_damage_pv": dmg, "exp_damage_zone_pv": dmg_zone, "capital": decade_spend, "capital_pv": cap_pv,
                        "maintenance_pv": maint, "cost_zone_pv": cost_zone, "spent_cum": spent})
    # ---- adaptation tipping points
    at = cfg["atp"]
    atp = {"damage": next((p["year"] for p in periods if p["damage_ratio"] > at["damage_ratio"]), None),
           "pop": next((p["year"] for p in periods if p["pop_ratio"] > at["pop_ratio"]), None),
           "budget": next((e["year"] for e in log if e["event"] == "deferred"), None)}
    return {"strategy": name, "climate": clim, "periods": periods, "log": log, "atp": atp,
            "cum_damage_pv": sum(p["exp_damage_pv"] for p in periods),
            "cum_cost_pv": sum(p["capital_pv"] + p["maintenance_pv"] for p in periods),
            "cum_capital": spent,
            "mean_livability": float(np.mean([p["livability"] for p in periods])),
            "mean_ecology": float(np.mean([p["ecology"] for p in periods])),
            "mean_households": float(np.mean([p["households"] for p in periods])),
            "mean_pop_hazard": float(np.mean([p["pop_hazard"] for p in periods]))}
