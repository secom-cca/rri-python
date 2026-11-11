"""One possible future ("world") and the planner's belief about it, in the same structure.

A world is given by factors in the unit cube (FACTORS; ranges from the config):

    climate   rain_2100, slr_2100 (the true 2100 values X; true path 1 + (X - 1) g(t) / X g(t) with a fixed shape g),
              rain_gamma, forecast_half_year (learning speed of the forecast)
    land use  asset_2100, pop_2100   (exposure in flood-prone areas, linear from the start year)
    delays    delay_<measure> (the uncertain phase: build / diffuse)
    social    engagement_tau, consensus_reduction, fail_scale, adoption_shift, coord_cost_scale
    regime    regime_year, regime_kind (none with probability p_none, else another criteria regime)

The decisions see the forecast (Forecast: posterior mean and s.d. of the 2100 rain factor and sea-level rise at each time,
improving as information accumulates), the physics sees the true path.  There is no observation error and no effect
coefficient (measures act with their physical parameters).  Random draws that are not factors (consensus failures, the
forecast's information path) come from the world seed, identical for every policy (common random numbers).
"""
from __future__ import annotations

import copy
import math
import zlib

import numpy as np

from .config import uval

DELAY_FACTOR = {"levee": "delay_levee", "retention": "delay_retention", "paddy_dam": "delay_paddy_dam",
                "forest_mgmt": "delay_forest_mgmt", "raise": "delay_housing", "relocate": "delay_housing",
                "pump_urban": "delay_drainage", "pump_agri": "delay_drainage", "storage_public": "delay_drainage",
                "storage_private": "delay_private", "floodproof": "delay_private",
                "zoning": "delay_policy", "induce": "delay_policy", "evacuation": "delay_policy"}
# factors given in years of one measure's phase; the others are positions (0-1) within each measure's own range
YEAR_FACTORS = ("delay_levee", "delay_retention", "delay_paddy_dam", "delay_forest_mgmt")
UNIT_FACTOR_LABEL = {"delay_housing": "住宅対策の実施期間", "delay_drainage": "排水ポンプ・公共貯留の施工期間",
                     "delay_private": "各戸貯留・耐水化の普及期間", "delay_policy": "土地利用規制・居住誘導・避難の浸透期間"}


def delay_phase(m):
    """The uncertain phase ({nominal, low, high}) of a measure."""
    return next(ph for ph in ("build", "diffuse", "mature") if isinstance(m.get(ph), dict))
GROUPS = {"climate": "気候", "landuse": "土地利用", "delays": "時間遅れ", "social": "社会・参加", "regime": "評価基準"}


def factors(cfg):
    """{name: {low, high, nominal, label, group}} in a fixed order."""
    C, L, P = cfg["climate"], cfg["landuse"], cfg["participation"]
    M = cfg["measures"]

    def f(spec, label, group, **kw):
        return {"low": float(spec["low"]), "high": float(spec["high"]), "nominal": float(spec["nominal"]),
                "label": label, "group": group, **kw}
    out = {
        "rain_2100": f(C["rain_2100"], "2100年の降雨倍率（真値）", "climate"),
        "slr_2100": f(C["slr_2100"], "2100年の海面上昇 [m]（真値）", "climate"),
        "rain_gamma": f(C["rain_gamma"], "降雨強度の強調 γ", "climate"),
        "forecast_half_year": f(C["forecast"]["half_year"], "予測の標準偏差が半分になる年（学習の速さ）", "climate"),
        "asset_2100": f(L["asset_2100"], "2100年の資産係数（浸水域）", "landuse"),
        "pop_2100": f(L["pop_2100"], "2100年の人口係数（浸水域）", "landuse"),
    }
    for k in ("levee", "retention", "paddy_dam", "forest_mgmt"):
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


def path_shape(cfg, var):
    """[[year, g]]: share of the change to 2100 reached by each year (var "rain" / "slr")."""
    y0 = cfg["years"]["start"]
    return [[y0, 0.0], [2050, float(cfg["climate"]["path_2050"][var])], [2100, 1.0]]


def true_paths(cfg, rain_2100, slr_2100):
    """(rain_pts, slr_pts) of a world with these 2100 values (the same shape in every world)."""
    gr, gs = path_shape(cfg, "rain"), path_shape(cfg, "slr")
    return [[y, 1.0 + (rain_2100 - 1.0) * g] for y, g in gr], [[y, slr_2100 * g] for y, g in gs]


def planning_paths(cfg):
    """Climate paths assumed at the start (the belief's 2100 values on the common shape)."""
    b = cfg["climate"]["belief"]
    return true_paths(cfg, b["rain_2100"], b["slr_2100"])


def _interp(pts, t):
    p = np.asarray(pts, float)
    return float(np.interp(t, p[:, 0], p[:, 1]))


def forecast_prior(cfg):
    """{var: (mu0, sigma0)} of the 2100 values: the belief and (high - low) / sqrt(12) of the factor range."""
    C = cfg["climate"]
    return {"rain": (float(C["belief"]["rain_2100"]), (C["rain_2100"]["high"] - C["rain_2100"]["low"]) / math.sqrt(12)),
            "slr": (float(C["belief"]["slr_2100"]), (C["slr_2100"]["high"] - C["slr_2100"]["low"]) / math.sqrt(12))}


def kappa_of(sigma0, half_year, y0):
    """Information rate so that the forecast s.d. halves by half_year: 1 + kappa s0^2 (half_year - y0) = 4."""
    return 3.0 / (sigma0 ** 2 * max(half_year - y0, 1e-6))


class Forecast:
    """Gaussian learning about the 2100 value X of one variable.  Prior N(mu0, s0^2); information dY = X dt + kappa^-1/2 dW
    from y0 (the world's Brownian path W, yearly increments from its seed).  At time t:
        precision  p_t = 1/s0^2 + kappa (t - y0),   s.d. s_t = p_t^-1/2
        forecast   m_t = (mu0/s0^2 + kappa Y_t) / p_t,   Y_t = X (t - y0) + kappa^-1/2 W(t - y0)
    m_t is a martingale (under the prior) and tends to X.  Seen from t, the forecast at t + h is
    N(m_t, s_t^2 - s_{t+h}^2) (preposterior), and X is N(m_t, s_t^2).  truth None: the planner's view (no information path,
    m_t = mu0)."""

    def __init__(self, mu0, s0, kappa, y0, truth=None, seed=None, tag="rain"):
        self.mu0, self.s0, self.kappa, self.y0, self.X = float(mu0), float(s0), float(kappa), float(y0), truth
        if truth is not None:
            rng = np.random.default_rng([int(seed), zlib.crc32(("forecast_" + tag).encode())])
            self.W = np.concatenate([[0.0], np.cumsum(rng.normal(size=200))])     # yearly Brownian path

    def sd(self, t):
        return (1.0 / self.s0 ** 2 + self.kappa * max(t - self.y0, 0.0)) ** -0.5

    def mean(self, t):
        if self.X is None:
            return self.mu0
        tau = max(t - self.y0, 0.0)
        k = int(np.floor(tau))
        w = self.W[k] + (tau - k) * (self.W[min(k + 1, len(self.W) - 1)] - self.W[k])
        Y = self.X * tau + w / math.sqrt(self.kappa)
        return (self.mu0 / self.s0 ** 2 + self.kappa * Y) / (1.0 / self.s0 ** 2 + self.kappa * tau)

    def spread_later(self, t, h):
        """s.d. of the forecast that will be available at t + h, seen from t (preposterior)."""
        return math.sqrt(max(self.sd(t) ** 2 - self.sd(t + h) ** 2, 0.0))


class World:
    """params: rain_pts, slr_pts (true paths), rain_2100, slr_2100 (true values; None in the planner's belief), gamma,
    forecast {half_year}, asset_pts, pop_pts, phases {kind: {consensus, build, diffuse, mature}}, social {tau, alpha,
    fail_scale, adoption_shift, coord_scale}, regimes [[year, name], ...], seed, deterministic (belief worlds: failure iff
    p > 0.5, no information path)."""

    def __init__(self, params, cfg):
        self.p = params
        self.cfg = cfg
        y0 = cfg["years"]["start"]
        pri = forecast_prior(cfg)
        hy = float(params.get("forecast", {}).get("half_year", uval(cfg["climate"]["forecast"]["half_year"])))
        det = params.get("deterministic")
        self.fc = {v: Forecast(mu, s0, kappa_of(s0, hy, y0), y0, None if det else params.get(f"{v}_2100"),
                               params.get("seed", 0), v) for v, (mu, s0) in pri.items()}

    def copy(self):
        return World(copy.deepcopy(self.p), self.cfg)

    # ---- environment (true)
    def rain(self, t):
        return _interp(self.p["rain_pts"], t)

    def slr(self, t):
        return _interp(self.p["slr_pts"], t)

    def asset(self, t):
        return _interp(self.p["asset_pts"], t)

    def pop(self, t):
        return _interp(self.p["pop_pts"], t)

    def env(self, t):
        return {"rain_factor": self.rain(t), "sea_level_rise_m": self.slr(t), "rain_gamma": self.p["gamma"]}

    # ---- forecast (what the decisions see)
    def forecast(self, t):
        """{rain: (mean, sd), slr: (mean, sd)} of the 2100 values available at time t."""
        return {v: (f.mean(t), f.sd(t)) for v, f in self.fc.items()}

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


def nominal_phases(cfg, kind, scale=1.0):
    m = cfg["measures"][kind]
    return {ph: uval(m[ph]) * (scale if ph != "consensus" else 1.0) for ph in ("consensus", "build", "diffuse", "mature")}


def belief_params(cfg, delay_scale=1.0):
    """The planner's assumptions at the start year (nominal values, no surprises)."""
    rain, slr = planning_paths(cfg)
    y0 = cfg["years"]["start"]
    P, L = cfg["participation"], cfg["landuse"]
    return {"rain_pts": rain, "slr_pts": slr, "rain_2100": None, "slr_2100": None, "gamma": uval(cfg["climate"]["rain_gamma"]),
            "forecast": {"half_year": uval(cfg["climate"]["forecast"]["half_year"])},
            "asset_pts": [[y0, 1.0], [2100, uval(L["asset_2100"])]], "pop_pts": [[y0, 1.0], [2100, uval(L["pop_2100"])]],
            "phases": {k: nominal_phases(cfg, k, delay_scale) for k in cfg["measures"]},
            "social": {"tau": uval(P["engagement_tau_years"]), "alpha": uval(P["consensus_reduction"]),
                       "fail_scale": uval(P["fail_scale"]), "adoption_shift": uval(P["adoption_shift"]),
                       "coord_scale": uval(P["coord_cost_scale"])},
            "regimes": [[y0, cfg["initial_regime"]]], "seed": 0, "deterministic": True}


def world_params(cfg, v, seed, delay_scale=1.0):
    """True world from factor values v (values_from_unit)."""
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
    cats = F["regime_kind"]["categories"]
    pn = cfg["regime_change"]["p_none"]
    rk = v["regime_kind"]
    regimes = [[y0, cfg["initial_regime"]]]
    if rk >= pn and len(cats) > 1:
        idx = min(int((rk - pn) / max(1 - pn, 1e-9) * (len(cats) - 1)), len(cats) - 2)
        regimes.append([round(v["regime_year"]), cats[1 + idx]])
    rain, slr = true_paths(cfg, v["rain_2100"], v["slr_2100"])
    return {"rain_pts": rain, "slr_pts": slr, "rain_2100": v["rain_2100"], "slr_2100": v["slr_2100"],
            "gamma": v["rain_gamma"], "forecast": {"half_year": v["forecast_half_year"]},
            "asset_pts": [[y0, 1.0], [2100, v["asset_2100"]]], "pop_pts": [[y0, 1.0], [2100, v["pop_2100"]]],
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
    U = lhs(n - 1, 2, np.random.default_rng(rng_seed)) * 2 - 1
    for u in U:
        p = copy.deepcopy(belief)
        d = u[0] * sp["rain_2100"]
        p["rain_pts"] = [[y, r + d * (y - p["rain_pts"][0][0]) / max(2100 - p["rain_pts"][0][0], 1)] for y, r in p["rain_pts"]]
        p["phases"] = {k: {ph: (x * (1 + u[1] * sp["delay"]) if ph != "consensus" else x) for ph, x in v.items()}
                       for k, v in p["phases"].items()}
        out.append(World(p, cfg))
    return out
