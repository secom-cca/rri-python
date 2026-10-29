"""Policy x scenario -> metrics through the GUI engine (RRI unchanged), cached on disk.

An evaluator provides
    cost(x)          -> {"total", "by_zone", "by_type"} [億円, present value, nominal unit costs]
    simulate(x, sim) -> {"damage", "damage_zone", "households", "livability",
                         "ecology", "inundated_km2", "peak_q"}
for a decision vector x (see candidates) and a simulated scenario
sim = {"climate", "rain_gamma", "paddy_effect", "forest_effect", "retention_effect"}.
Post-hoc factors (cost_factor, asset_factor) are applied later (uncertainty.apply_post).
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from . import candidates as CA

FOREST_N_FARM, FOREST_N_FULL = 0.25, 0.4        # farmland-average and full new-forest roughness


REQUIRED = ("damage", "damage_zone", "households", "livability", "ecology", "pop_hazard")


def gparams_for(sim: dict) -> dict:
    """Engine parameters of a simulated scenario (effect coefficients scale the measure parameters).
    Climate: a preset (sim["climate"]) or continuous rain_factor / sea_level_rise_m (custom)."""
    kf = sim.get("forest_effect", 1.0)
    if "rain_factor" in sim or "sea_level_rise_m" in sim:
        clim = {"preset": "custom", "rain_factor": sim.get("rain_factor", 1.0),
                "sea_level_rise_m": sim.get("sea_level_rise_m", 0.0), "intensity_gamma": sim.get("rain_gamma", 1.0)}
    else:
        clim = {"preset": sim.get("climate", "current"), "intensity_gamma": sim.get("rain_gamma", 1.0)}
    return {"paddy_store_m": 0.15 * sim.get("paddy_effect", 1.0), "paddy_release_mmh": 5.0,
            "forest_n": FOREST_N_FARM + kf * (FOREST_N_FULL - FOREST_N_FARM), "forest_soil_m": 1.0 * kf,
            "climate": clim}


def xkey(x):
    """Decision vector for cache keys / JSON: integers stay integers (keys of integer
    policies unchanged), implementation rates are rounded to 1e-4."""
    return [int(round(float(v))) if abs(float(v) - round(float(v))) < 1e-9 else round(float(v), 4) for v in x]


class RRIEvaluator:
    """`gextra` (dict) is merged into the engine parameters of every run, e.g.
    {"baseline": "current"}: measures are set against the current-climate baseline
    instead of running a no-measures baseline for each scenario climate."""

    def __init__(self, session, cfg, variables, zone_masks, cache_path=None, log=print, gextra=None):
        self.s, self.cfg, self.variables, self.zm = session, cfg, variables, zone_masks
        self.event = cfg["event"]
        self.log = log
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec["m"]
        self._cost = {}
        self.n_sim = 0
        self.gextra = dict(gextra or {})
        self.pop_total = float(np.asarray(session.pop)[session.d.mask].sum())
        from ..gui import engine as E
        self.version = f"sg{E.SUBGRID_VERSION}_as{E.ASSET_FILE_VERSION}"

    def _key(self, x, sim):
        k = json.dumps({"b": self.cfg["basin"], "r": self.cfg["res"], "e": self.event, "x": xkey(x),
                        "sim": {k: sim[k] for k in sorted(sim)}, "v": self.version,
                        **({"g": self.gextra} if self.gextra else {}),
                        "vars": hashlib.sha1(json.dumps([v.to_dict() for v in self.variables]).encode()).hexdigest()[:8]},
                       sort_keys=True)
        return hashlib.sha1(k.encode()).hexdigest()

    def cost(self, x):
        t = tuple(xkey(x))
        if t not in self._cost:
            eds = CA.edits(self.variables, t)
            per = self.s.estimate(self.event, eds, {})["edit_cost_yen"] if eds else []
            by_zone = {z: 0.0 for z in self.zm}
            by_type = {}
            active = [v for v, xi in zip(self.variables, t) if v.active(xi)]
            for v, c in zip(active, per):
                by_zone[v.zone] = by_zone.get(v.zone, 0.0) + c / 1e8
                by_type[v.type] = by_type.get(v.type, 0.0) + c / 1e8
            self._cost[t] = {"total": float(sum(per)) / 1e8, "by_zone": by_zone, "by_type": by_type}
        return self._cost[t]

    def cost_parts(self, x):
        """Per variable (initial [億円], annual maintenance [億円/年]) of decision vector x
        (0 for variables not implemented).  Used by the DAPP analysis."""
        t = tuple(xkey(x))
        key = ("parts",) + t
        if key not in self._cost:
            eds = CA.edits(self.variables, t)
            est = self.s.estimate(self.event, eds, {}) if eds else {"edit_initial_yen": [], "edit_annual_yen": []}
            ini, ann = [0.0] * len(t), [0.0] * len(t)
            active = [i for i, (v, xi) in enumerate(zip(self.variables, t)) if v.active(xi)]
            for i, a, b in zip(active, est["edit_initial_yen"], est["edit_annual_yen"]):
                ini[i], ann[i] = a / 1e8, b / 1e8
            self._cost[key] = (ini, ann)
        return self._cost[key]

    def simulate(self, x, sim, strict=True):
        """strict=False accepts cached results of older versions lacking newer metrics (e.g. pop_hazard)."""
        key = self._key(x, sim)
        if key in self.cache and (not strict or all(k in self.cache[key] for k in REQUIRED)):
            return self.cache[key]
        eds = CA.edits(self.variables, x, sim)
        res = self.s.run(self.event, eds, {**gparams_for(sim), **self.gextra})
        self.n_sim += 1
        dmg = np.frombuffer(base64.b64decode(res["damage"]), "<f4").reshape(self.s.d.shape)
        g = max(self.s.meta["gauge_index"], key=lambda n: self.s.d.river.area_km2[self.s.meta["gauge_index"][n]])
        m = {"damage": res["damage_total_oku"],
             "damage_zone": {z: float(dmg[mk].sum() / 1e8) for z, mk in self.zm.items()},
             "households": res["households_above_floor"],
             "livability": res["livability"]["score"], "ecology": res["ecology"]["score"],
             # 浸水人口: residents of cells flooded >= 0.5 m (indicators.livability, 安全・安心)
             "pop_hazard": float(res["livability"].get("pop_in_hazard_share", 0.0) * self.pop_total),
             "inundated_km2": res["inundated_km2"], "peak_q": float(np.nanmax(res["gauges"][g]["sim"]))}
        self.cache[key] = m
        if self.cache_path:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "x": xkey(x), "sim": sim, "m": m}, ensure_ascii=False) + "\n")
        self.log(f"  sim #{self.n_sim}: {CA.describe(self.variables, x) or ['施策なし']} | {sim} -> 被害 {m['damage']:.0f} 億円")
        return m
