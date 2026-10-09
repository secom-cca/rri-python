"""Flood-management policy scenarios as pure transformations of a Domain.

Every function returns a modified *copy*, so a baseline and any number of
policies can be simulated with identical forcing and compared with
`compare_scenarios`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .domain import Domain, enforce_monotone_bed


def dredge(d: Domain, river_idx, depth_m: float, widen_m: float = 0.0) -> Domain:
    """浚渫・河道掘削: lower the bed by depth_m (and optionally widen) on the
    given river cells."""
    d2 = d.copy()
    r = d2.river
    idx = np.atleast_1d(river_idx)
    r.zb[idx] -= depth_m
    r.depth[idx] += depth_m
    r.width[idx] = np.minimum(r.width[idx] + widen_m, 0.95 * min(d.dx, d.dy))
    enforce_monotone_bed(r)
    return d2


def raise_levee(d: Domain, river_idx, height_m: float) -> Domain:
    """堤防嵩上げ / 築堤."""
    d2 = d.copy()
    d2.river.embankment[np.atleast_1d(river_idx)] += height_m
    return d2


def retention_basin(d: Domain, cells_mask: np.ndarray, excavate_m: float,
                    river_idx=None, inlet_crest_m: float | None = None) -> Domain:
    """遊水地: excavate the ground in `cells_mask` by excavate_m.  If river_idx
    and inlet_crest_m are given, the embankment on those river cells is set
    to inlet_crest_m above the (new) ground, acting as an overflow weir
    (越流堤) that lets flood peaks spill into the basin."""
    d2 = d.copy()
    d2.z = np.where(cells_mask & d2.mask, d2.z - excavate_m, d2.z)
    if river_idx is not None and inlet_crest_m is not None:
        idx = np.atleast_1d(river_idx)
        r = d2.river
        # river bed must stay below the new ground; crest relative to new ground
        zs_new = d2.z[r.ri[idx], r.rj[idx]]
        r.embankment[idx] = inlet_crest_m
        r.depth[idx] = zs_new - r.zb[idx]
    return d2


def change_landuse(d: Domain, cells_mask: np.ndarray, new_class: int) -> Domain:
    """土地利用変更 (e.g. paddy -> urban, or 田んぼダム via a dedicated class)."""
    d2 = d.copy()
    d2.landclass = np.where(cells_mask & d2.mask, new_class, d2.landclass)
    return d2


@dataclass
class Scenario:
    name: str
    transform: Callable[[Domain], Domain]
    description: str = ""
    cost_yen: float | None = None
    meta: dict = field(default_factory=dict)


def compare_scenarios(base: Domain, scenarios: list[Scenario], run: Callable[[Domain], object],
                      damage: Callable[[object], float] | None = None) -> list[dict]:
    """Run baseline + scenarios with the same `run(domain) -> Results` and
    optional `damage(results) -> yen`.  Returns a table of key indicators."""
    rows = []
    for sc in [Scenario("baseline", lambda x: x)] + list(scenarios):
        dom = sc.transform(base)
        res = run(dom)
        wet = np.nan_to_num(res.max_depth) > 0.5
        row = {"scenario": sc.name, "peak_outlet_q": float(np.max(res.outlet_q)),
               "inundated_area_km2_gt0.5m": float(wet.sum() * dom.cell_area / 1e6),
               "max_depth_m": float(np.nanmax(res.max_depth)),
               "mass_rel_err": res.mass.relative_error}
        if damage is not None:
            row["damage_yen"] = float(damage(res))
        if sc.cost_yen is not None:
            row["cost_yen"] = sc.cost_yen
        rows.append(row)
    if damage is not None:
        base_dmg = rows[0]["damage_yen"]
        for r in rows:
            r["damage_reduction_yen"] = base_dmg - r["damage_yen"]
    return rows
