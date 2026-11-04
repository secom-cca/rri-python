"""One possible future ("world") and the planner's belief about it, in the same structure.

A world is given by factors in the unit cube (FACTORS; ranges from the config):

    climate   rain_2100, slr_2100, climate_shift_year, rain_gamma
              (the planning path = 2C until the shift year, then linear to the 2100 values)
    land use  asset_2100, pop_2100, landuse_shift_year    (exposure: assets / people in flood-prone areas)
    effects   eff_<measure> (effect coefficient at the start), tech_gain, tech_year
              (technology changes the effects of tech_types by tech_gain over 10 years)
    delays    delay_<measure> (the uncertain phase: build / diffuse / mature)
    social    engagement_tau, consensus_reduction, fail_scale, adoption_shift, coord_cost_scale
    regime    regime_year, regime_kind (none with probability p_none, else another criteria regime)

Random draws that are not factors (consensus failures, observation noise) come
from the world seed, identical for every policy (common random numbers).
"""
from __future__ import annotations

import copy
import zlib

import numpy as np

from .config import uval

DELAY_FACTOR = {"levee": "delay_levee", "retention": "delay_retention", "paddy_dam": "delay_paddy_dam",
                "forest": "delay_forest", "raise": "delay_housing", "relocate": "delay_housing",
                "pump_urban": "delay_drainage", "pump_agri": "delay_drainage", "storage_public": "delay_drainage",
                "storage_private": "delay_private", "floodproof": "delay_private",
                "zoning": "delay_policy", "induce": "delay_policy", "evacuation": "delay_policy"}
# factors given in years of one measure's phase; the others are positions (0-1) within each measure's own range
YEAR_FACTORS = ("delay_levee", "delay_retention", "delay_paddy_dam", "delay_forest")
UNIT_FACTOR_LABEL = {"delay_housing": "住宅対策の実施期間", "delay_drainage": "排水ポンプ・公共貯留の施工期間",
                     "delay_private": "各戸貯留・耐水化の普及期間", "delay_policy": "土地利用規制・居住誘導・避難の浸透期間"}


def delay_phase(m):
    """The uncertain phase ({nominal, low, high}) of a measure."""
    return next(ph for ph in ("build", "diffuse", "mature") if isinstance(m.get(ph), dict))
EFFECT_KINDS = ("levee", "retention", "paddy_dam", "forest", "raise")
GROUPS = {"climate": "気候", "landuse": "土地利用", "effects": "施策効果", "delays": "時間遅れ", "social": "社会・参加", "regime": "評価基準"}


def factors(cfg):
    """{name: {low, high, nominal, label, group}} in a fixed order."""
    C, L, E, P = cfg["climate"], cfg["landuse"], cfg["effects"], cfg["participation"]
    M = cfg["measures"]

    def f(spec, label, group, **kw):
        return {"low": float(spec["low"]), "high": float(spec["high"]), "nominal": float(spec["nominal"]),
                "label": label, "group": group, **kw}
    out = {
        "rain_2100": f(C["rain_2100"], "2100年の降雨倍率", "climate"),
        "slr_2100": f(C["slr_2100"], "2100年の海面上昇 [m]", "climate"),
        "climate_shift_year": f(C["shift_year"], "気候が想定から外れ始める年", "climate"),
        "rain_gamma": f(C["rain_gamma"], "降雨強度の強調 γ", "climate"),
        "asset_2100": f(L["asset_2100"], "2100年の資産係数（浸水域）", "landuse"),
        "pop_2100": f(L["pop_2100"], "2100年の人口係数（浸水域）", "landuse"),
        "landuse_shift_year": f(L["shift_year"], "資産変化の開始年", "landuse"),
    }
    for k in EFFECT_KINDS:
        out[f"eff_{k}"] = f(E[k], f"{M[k]['label']}の効果係数（開始時）", "effects")
    out["tech_gain"] = f(E["tech_gain"], "技術変化による効果係数の増減", "effects")
    out["tech_year"] = f(E["tech_year"], "技術変化の年", "effects")
    for k in ("levee", "retention", "paddy_dam", "forest"):
        if k in M:
            ph = delay_phase(M[k])
            out[DELAY_FACTOR[k]] = f(M[k][ph], f"{M[k]['label']}の{ {'build': '施工', 'diffuse': '普及', 'mature': '成熟'}[ph]}期間 [年]", "delays")
    for fac, lab in UNIT_FACTOR_LABEL.items():
        ks = [k for k in M if DELAY_FACTOR.get(k) == fac]
        if not ks:
            continue
        m0 = M[ks[0]][delay_phase(M[ks[0]])]
        rng = "・".join(f"{M[k]['label']} {M[k][delay_phase(M[k])]['low']:.0f}〜{M[k][delay_phase(M[k])]['high']:.0f}" for k in ks)
        out[fac] = {"low": 0.0, "high": 1.0, "nominal": (m0["nominal"] - m0["low"]) / ((m0["high"] - m0["low"]) or 1.0),
                    "label": f"{lab}（範囲内の位置: {rng} 年）", "group": "delays"}
    out["engagement_tau"] = f(P["engagement_tau_years"], "関係の構築にかかる年数 τ", "social")
    out["consensus_reduction"] = f(P["consensus_reduction"], "参加による合意期間の短縮率", "social")
    out["fail_scale"] = f(P["fail_scale"], "合意失敗確率の倍率", "social")
    out["adoption_shift"] = f(P["adoption_shift"], "参加率（実施率の上限）の増減", "social")
    out["coord_cost_scale"] = f(P["coord_cost_scale"], "調整費用の倍率", "social")
    out["regime_year"] = f(cfg["regime_change"]["year"], "評価基準が変わる年", "regime")
    others = [r for r in cfg["regimes"] if r != cfg["initial_regime"]]
    out["regime_kind"] = {"low": 0.0, "high": 1.0, "nominal": 0.0, "label": "評価基準の変化（" + " / ".join(["なし"] + others) + "）",
                          "group": "regime", "categories": ["なし"] + others}
    return out


def unit_nominal(F):
    return np.array([(v["nominal"] - v["low"]) / ((v["high"] - v["low"]) or 1.0) for v in F.values()])


def values_from_unit(z, F):
    return {n: float(v["low"] + zi * (v["high"] - v["low"])) for (n, v), zi in zip(F.items(), z)}


def planning_paths(cfg):
    """Climate paths assumed at the start (2C shape of docs/11: 60 % of the rain change, 39 % of the
    sea-level rise by 2050)."""
    b = cfg["climate"]["belief"]
    y0 = cfg["years"]["start"]
    rain = [[y0, 1.0], [2050, 1.0 + 0.6 * (b["rain_2100"] - 1.0)], [2100, b["rain_2100"]]]
    slr = [[y0, 0.0], [2050, 0.39 * b["slr_2100"]], [2100, b["slr_2100"]]]
    return rain, slr


def _interp(pts, t):
    p = np.asarray(pts, float)
    return float(np.interp(t, p[:, 0], p[:, 1]))


def _shift_path(plan, year, v2100):
    """The planning path until `year`, then a straight line to v2100."""
    y = float(np.clip(year, plan[0][0], 2100))
    pts = [q for q in plan if q[0] < y] + [[y, _interp(plan, y)]]
    if y < 2100:
        pts.append([2100.0, v2100])
    return pts


class World:
    """params: rain_pts, slr_pts, gamma, eff0 {kind}, tech_gain, tech_year, tech_types, asset_pts, pop_pts,
    phases {kind: {consensus, build, diffuse, mature}}, social {tau, alpha, fail_scale, adoption_shift, coord_scale},
    regimes [[year, name], ...], seed, deterministic (belief worlds: no noise, failure iff p > 0.5)."""

    def __init__(self, params, cfg):
        self.p = params
        self.cfg = cfg
        self.ranges = {k: (cfg["effects"][k]["low"], cfg["effects"][k]["high"]) for k in EFFECT_KINDS}

    def copy(self):
        return World(copy.deepcopy(self.p), self.cfg)

    # ---- environment
    def rain(self, t):
        return _interp(self.p["rain_pts"], t)

    def slr(self, t):
        return _interp(self.p["slr_pts"], t)

    def effect(self, kind, t):
        k = "raise" if kind == "relocate" else kind
        e = self.p["eff0"][k]
        if k in self.p["tech_types"] and self.p["tech_gain"]:
            e += self.p["tech_gain"] * float(np.clip((t - self.p["tech_year"]) / 10.0, 0.0, 1.0))
        return float(np.clip(e, *self.ranges[k]))

    def asset(self, t):
        return _interp(self.p["asset_pts"], t)

    def pop(self, t):
        return _interp(self.p["pop_pts"], t)

    def env(self, t):
        return {"rain_factor": self.rain(t), "sea_level_rise_m": self.slr(t), "rain_gamma": self.p["gamma"],
                **{f"{'paddy' if k == 'paddy_dam' else k}_effect": self.effect(k, t) for k in EFFECT_KINDS}}

    def phases(self, kind):
        return self.p["phases"][kind]

    def regime_at(self, t):
        name = self.p["regimes"][0][1]
        for y, n in self.p["regimes"]:
            if t >= y:
                name = n
        return name

    # ---- random draws (common random numbers across policies)
    def fail_u(self, kind, attempt):
        if self.p.get("deterministic"):
            return 0.5
        return float(np.random.default_rng([self.p["seed"], zlib.crc32(kind.encode()), attempt]).random())

    def noise(self, tag, t):
        if self.p.get("deterministic"):
            return 0.0
        return float(np.random.default_rng([self.p["seed"], zlib.crc32(tag.encode()), int(round(t * 10))]).normal())


def nominal_phases(cfg, kind, scale=1.0):
    m = cfg["measures"][kind]
    return {ph: uval(m[ph]) * (scale if ph != "consensus" else 1.0) for ph in ("consensus", "build", "diffuse", "mature")}


def belief_params(cfg, delay_scale=1.0):
    """The planner's assumptions at the start year (nominal values, no surprises)."""
    rain, slr = planning_paths(cfg)
    y0 = cfg["years"]["start"]
    P = cfg["participation"]
    return {"rain_pts": rain, "slr_pts": slr, "gamma": uval(cfg["climate"]["rain_gamma"]),
            "eff0": {k: uval(cfg["effects"][k]) for k in EFFECT_KINDS}, "tech_gain": 0.0,
            "tech_year": uval(cfg["effects"]["tech_year"]), "tech_types": list(cfg["effects"]["tech_types"]),
            "asset_pts": [[y0, 1.0], [2100, 1.0]], "pop_pts": [[y0, 1.0], [2100, uval(cfg["landuse"]["pop_2100"])]],
            "phases": {k: nominal_phases(cfg, k, delay_scale) for k in cfg["measures"]},
            "social": {"tau": uval(P["engagement_tau_years"]), "alpha": uval(P["consensus_reduction"]),
                       "fail_scale": uval(P["fail_scale"]), "adoption_shift": uval(P["adoption_shift"]),
                       "coord_scale": uval(P["coord_cost_scale"])},
            "regimes": [[y0, cfg["initial_regime"]]], "seed": 0, "deterministic": True}


def world_params(cfg, v, seed, delay_scale=1.0):
    """True world from factor values v (values_from_unit)."""
    rain, slr = planning_paths(cfg)
    y0 = cfg["years"]["start"]
    M = cfg["measures"]
    F = factors(cfg)
    phases = {}
    for k in M:
        ph = nominal_phases(cfg, k, delay_scale)
        main = delay_phase(M[k])
        fac = DELAY_FACTOR[k]
        if fac in YEAR_FACTORS:
            ph[main] = v[fac] * delay_scale
        else:
            s = M[k][main]
            ph[main] = (s["low"] + v[fac] * (s["high"] - s["low"])) * delay_scale
        phases[k] = ph
    ly = v["landuse_shift_year"]
    cats = F["regime_kind"]["categories"]
    pn = cfg["regime_change"]["p_none"]
    rk = v["regime_kind"]
    regimes = [[y0, cfg["initial_regime"]]]
    if rk >= pn and len(cats) > 1:
        idx = min(int((rk - pn) / max(1 - pn, 1e-9) * (len(cats) - 1)), len(cats) - 2)
        regimes.append([round(v["regime_year"]), cats[1 + idx]])
    return {"rain_pts": _shift_path(rain, v["climate_shift_year"], v["rain_2100"]),
            "slr_pts": _shift_path(slr, v["climate_shift_year"], v["slr_2100"]),
            "gamma": v["rain_gamma"], "eff0": {k: v[f"eff_{k}"] for k in EFFECT_KINDS},
            "tech_gain": v["tech_gain"], "tech_year": v["tech_year"], "tech_types": list(cfg["effects"]["tech_types"]),
            "asset_pts": [[y0, 1.0], [max(y0, min(ly, 2099)), 1.0], [2100, v["asset_2100"]]],
            "pop_pts": [[y0, 1.0], [2100, v["pop_2100"]]],
            "phases": phases,
            "social": {"tau": v["engagement_tau"], "alpha": v["consensus_reduction"], "fail_scale": v["fail_scale"],
                       "adoption_shift": v["adoption_shift"], "coord_scale": v["coord_cost_scale"]},
            "regimes": regimes, "seed": int(seed), "deterministic": False}


def make_world(cfg, z, seed, delay_scale=1.0):
    F = factors(cfg)
    return World(world_params(cfg, values_from_unit(z, F), seed, delay_scale), cfg)


def belief_world(cfg, belief=None, delay_scale=1.0):
    return World(copy.deepcopy(belief) if belief else belief_params(cfg, delay_scale), cfg)


def belief_ensemble(cfg, belief, n, rng_seed):
    """The planner's uncertainty around its belief: the central world + n-1 perturbed ones (LHS)."""
    from ..rdm.uncertainty import lhs
    sp = cfg["optimiser"]["belief_spread"]
    out = [World(copy.deepcopy(belief), cfg)]
    if n <= 1:
        return out
    U = lhs(n - 1, 3, np.random.default_rng(rng_seed)) * 2 - 1
    for u in U:
        p = copy.deepcopy(belief)
        d = u[0] * sp["rain_2100"]
        p["rain_pts"] = [[y, r + d * (y - p["rain_pts"][0][0]) / max(2100 - p["rain_pts"][0][0], 1)] for y, r in p["rain_pts"]]
        p["eff0"] = {k: e + u[1] * sp["effect"] for k, e in p["eff0"].items()}
        p["phases"] = {k: {ph: (x * (1 + u[2] * sp["delay"]) if ph != "consensus" else x) for ph, x in v.items()}
                       for k, v in p["phases"].items()}
        out.append(World(p, cfg))
    return out
