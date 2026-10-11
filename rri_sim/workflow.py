"""High-level helpers: build a registered basin with parameter presets (cached)."""
from __future__ import annotations

import pickle

from .basin import build_basin, default_land_params
from .cases import BASINS
from .data import ROOT
from .params import LandClassParams, ModelParams

CACHE = ROOT / "data" / "cache"

# Calibrated / recommended parameter sets per basin (see docs/04_progress_report.md)
PRESETS = {
    # 玖珠川: calibrated on 2018/7, validated on 2012/7, 2021/8, 2020/7 (NSE 0.95-0.96)
    "kusu_ogase": {"sets": ["forest.n=0.08025", "forest.ka=0.05008", "forest.soil_depth=0.518",
                            "river.cw=10.19", "river.n_river=0.04037"], "threshold": 5.0},
    # 阿武隈川上流: a-priori values (須賀川 NSE 0.98 for 2019); regional calibration pending
    "abukuma_akutsu": {"sets": [], "threshold": 5.0},
    # 筑後川全域: 玖珠川 calibration transferred (same region / volcanic uplands)
    "chikugo_whole": {"sets": ["forest.n=0.08025", "forest.ka=0.05008", "forest.soil_depth=0.518",
                               "river.cw=10.19", "river.n_river=0.04037"], "threshold": 5.0},
    "abukuma_whole": {"sets": [], "threshold": 5.0},
}

# extra land classes used only by scenarios
PADDY_DAM = "paddy_dam"
FOREST_NEW = "forest_new"


def model_land_params() -> list[LandClassParams]:
    """Default classes + 田んぼダム (paddy with outlet plate).

    paddy_dam: same soil as paddy, but 0.15 m of surface water is held by the
    plate and released slowly (≈5 mm/h when full: leak = 5e-3/3600/0.15)."""
    lp = default_land_params()
    paddy = next(c for c in lp if c.name == "paddy")
    lp.append(LandClassParams(name=PADDY_DAM, n=paddy.n, kv=paddy.kv,
                              phi_minus_theta=paddy.phi_minus_theta, sf=paddy.sf,
                              infilt_limit=paddy.infilt_limit, store_depth=0.15,
                              leak_rate=5e-3 / 3600 / 0.15))
    # 森林化: newly forested land (afforestation of fields / built-up land).
    # Not the basin-calibrated mountain forest (whose n is an effective value for
    # steep slopes) but a-priori forest-floor roughness and forest soil.
    lp.append(LandClassParams(name=FOREST_NEW, n=0.4, ka=0.05, soil_depth=1.0, gamma_a=0.4,
                              gamma_m=0.1, beta=8.0))
    # 移転跡地の緑地化 (grassland / open space) and 遊水地 wetland
    field = next(c for c in lp if c.name == "field")
    lp.append(LandClassParams(name="green", n=0.3, kv=field.kv, phi_minus_theta=field.phi_minus_theta,
                              sf=field.sf, infilt_limit=field.infilt_limit))
    lp.append(LandClassParams(name="wetland", n=0.1, kv=paddy.kv, phi_minus_theta=paddy.phi_minus_theta,
                              sf=paddy.sf, infilt_limit=paddy.infilt_limit))
    return lp


def apply_overrides(params, sets):
    """'class.attr=value', 'river.attr=value', 'num.attr=value', 'all.attr=value'."""
    names = {c.name: c for c in params.land_classes}
    for s in sets or []:
        key, val = s.split("=")
        obj, attr = key.split(".")
        if obj == "all":
            for c in params.land_classes:
                setattr(c, attr, float(val))
            continue
        target = {"river": params.river, "num": params.numerics}.get(obj) or names[obj]
        cur = getattr(target, attr)
        setattr(target, attr, type(cur)(val) if not isinstance(cur, bool) else val == "True")


def build(basin_key, sets=None, threshold=None, use_cache=False):
    """Build a registered basin. Returns (domain, meta, basin_entry)."""
    b = BASINS[basin_key]
    cfg = b["cfg"]
    if threshold is not None:
        cfg.river_threshold_km2 = threshold
    import hashlib
    sig = hashlib.sha1(("|".join(sets or [])).encode()).hexdigest()[:10]   # stable across runs
    key = f"{basin_key}_{cfg.res:.0f}_{cfg.river_threshold_km2}_{sig}"
    path = CACHE / f"domain_{key}.pkl"
    if use_cache and path.exists():
        with open(path, "rb") as f:
            d, meta = pickle.load(f)
        # land-class list may have grown since caching (classes are only appended)
        want = model_land_params()
        if [c.name for c in d.params.land_classes] != [c.name for c in want]:
            fresh = ModelParams(land_classes=want, river=d.params.river, numerics=d.params.numerics)
            apply_overrides(fresh, [s_ for s_ in (sets or []) if not s_.startswith(("river.", "num."))])
            d.params = fresh
        return d, meta, b
    params = ModelParams(land_classes=model_land_params())
    apply_overrides(params, sets)
    d, meta = build_basin(cfg, params, verbose=False)
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((d, meta), f)
    return d, meta, b


def build_preset(basin_key, use_cache=True):
    p = PRESETS.get(basin_key, {"sets": [], "threshold": None})
    return build(basin_key, p["sets"], p["threshold"], use_cache=use_cache)


def boundary_conditions(basin_key, domain, meta, t0, t_end):
    """Dams and tidal stage boundary for a registered basin.

    Returns (dams: list[Dam], stage_bc: {outlet idx: (t_s, H_TP)}, info dict)."""
    import numpy as np
    import pandas as pd

    from .basin import lonlat_to_rc, snap_to_river
    from .data import mlit_water
    from .solver import Dam
    b = BASINS[basin_key]
    r = domain.river
    dams, info = [], {"dams": {}}
    for name, spec in b.get("dams", {}).items():
        row, col, _, _ = lonlat_to_rc(*spec["lonlat"], domain.transform, domain.crs)
        try:
            k = snap_to_river(r, row, col, meta["acc_km2"], 4)
        except ValueError:
            continue
        dams.append(Dam(k=k, capacity=spec["capacity"], q_cut=spec["q_cut"], name=name))
        info["dams"][name] = {"river_index": int(k), "area_km2": float(r.area_km2[k]), **spec}
    stage_bc = {}
    tide = b.get("tide_gauge")
    if tide:
        T0, T1 = pd.Timestamp(t0), pd.Timestamp(t_end)
        lvl = mlit_water.hourly(tide, mlit_water.KIND_LEVEL, str(T0), str(T1))
        lvl = lvl.interpolate(limit_direction="both")
        zero = mlit_water.gauge_zero_tp(tide) or 0.0
        ts = ((lvl.index - T0) / pd.Timedelta(seconds=1)).values.astype(float)
        H = lvl.values.astype(float) + zero
        for k in np.nonzero(r.dn < 0)[0]:
            if r.area_km2[k] > 0.5 * r.area_km2.max():      # main outlet only
                stage_bc[int(k)] = (ts, H)
        info["tide"] = {"station": tide, "zero_TP": zero, "H_range": [float(np.nanmin(H)), float(np.nanmax(H))]}
    return dams, stage_bc, info
