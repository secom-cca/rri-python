"""High-level helpers: build a registered basin with parameter presets (cached)."""
from __future__ import annotations

import pickle

import numpy as np

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


def default_res(basin_key) -> float:
    return BASINS[basin_key]["cfg"].res


def cache_key(basin_key, res=None) -> str:
    """Cache-file stem for a basin at a grid resolution (the registered default
    resolution keeps the bare basin key, so existing caches stay valid)."""
    if res is None or float(res) == default_res(basin_key):
        return basin_key
    return f"{basin_key}_{float(res):.0f}m"


def basin_config(basin_key, res=None, threshold=None):
    """Registered BasinConfig with grid resolution / river threshold overridden
    (a copy: the registry is never mutated)."""
    from dataclasses import replace
    cfg = BASINS[basin_key]["cfg"]
    kw = {}
    if res is not None:
        kw["res"] = float(res)
        # source DEM: zoom 13 (~20 m) is ample for >= 150 m cells; zoom 14 (DEM10B, ~10 m) below
        kw["dem_zoom"] = 14 if float(res) < 150 else cfg.dem_zoom
    if threshold is not None:
        kw["river_threshold_km2"] = threshold
    return replace(cfg, **kw) if kw else cfg


def _domain_path(basin_key, sets, cfg):
    import hashlib
    b = BASINS[basin_key]
    sig = hashlib.sha1(("|".join(sets or []) + str(b.get("existing_levees")) + str(b.get("lowland_river_n")) + "w3coast").encode()).hexdigest()[:10]
    return CACHE / f"domain_{basin_key}_{cfg.res:.0f}_{cfg.river_threshold_km2}_{sig}.pkl"


def domain_cached(basin_key, res=None) -> bool:
    """Whether the preset model of a basin at `res` is already built (cache file exists)."""
    p = PRESETS.get(basin_key, {"sets": [], "threshold": None})
    return _domain_path(basin_key, p["sets"], basin_config(basin_key, res, p["threshold"])).exists()


def build(basin_key, sets=None, threshold=None, use_cache=False, res=None):
    """Build a registered basin. Returns (domain, meta, basin_entry).
    `res` [m] overrides the registered grid resolution (default 250 m)."""
    b = BASINS[basin_key]
    cfg = basin_config(basin_key, res, threshold)
    path = _domain_path(basin_key, sets, cfg)
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
    meta["res"] = cfg.res
    nl = b.get("lowland_river_n")
    if nl:
        # Manning n of large lowland reaches is smaller than that of steep gravel
        # rivers (the calibrated value): n by local bed slope
        r = d.river
        slope = np.full(r.n, 1e-2)
        has = r.dn >= 0
        slope[has] = (r.zb[has] - r.zb[r.dn[has]]) / r.length[has]
        slope[~has] = r.outlet_slope[~has]
        for s_max, n_val in sorted(nl.items()):
            r.n_river[slope < s_max] = np.minimum(r.n_river[slope < s_max], n_val)
        meta["lowland_n_cells"] = int((slope < max(nl)).sum())
    lev = b.get("existing_levees")
    if lev:
        from .levees import estimate_levees
        emb, stats, width = estimate_levees(d, min_area_km2=lev.get("min_area_km2", 200.0))
        d.river.embankment[:] = np.maximum(d.river.embankment, emb)
        ok = np.isfinite(width)
        cap = lev.get("max_width_cells", 2.5) * d.dx
        d.river.width[ok] = np.clip(np.maximum(d.river.width[ok], width[ok]), None, cap)
        meta["levee_stats"] = stats
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((d, meta), f)
    return d, meta, b


def build_preset(basin_key, use_cache=True, res=None):
    p = PRESETS.get(basin_key, {"sets": [], "threshold": None})
    return build(basin_key, p["sets"], p["threshold"], use_cache=use_cache, res=res)


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
            k = snap_to_river(r, row, col, meta["acc_km2"], max(2, round(1000.0 / domain.dx)))   # ~1 km
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


def drainage_arrays(basin_key, domain, meta, landclass=None):
    """Lowland drainage (pumps / sewers / creeks) for a registered basin:
    (rate [m/s] per cell, target river index per cell) or None.
    Applies to cells with HAND <= hand_max_m; urban and farmland rates differ."""
    spec = BASINS[basin_key].get("lowland_drainage")
    if not spec:
        return None
    from .hydro import drain_target
    d = domain
    r = d.river
    if "drain_target_k" not in meta:
        is_riv = np.zeros(d.shape, bool)
        is_riv[r.ri, r.rj] = True
        tgt = drain_target(meta["fdir"], is_riv, d.mask)
        lut = -np.ones(d.shape[0] * d.shape[1], np.int64)
        lut[r.ri * d.shape[1] + r.rj] = np.arange(r.n)
        meta["drain_target_k"] = np.where(tgt >= 0, lut[np.maximum(tgt, 0)], -1)
    lc = d.landclass if landclass is None else landclass
    names = np.asarray([c.name for c in d.params.land_classes])[lc]
    low = d.mask & (np.nan_to_num(meta["hand"], nan=1e9) <= spec.get("hand_max_m", 5.0))
    rate = np.zeros(d.shape)
    rate[low & (names == "urban")] = spec.get("urban_mmh", 10.0) / 1000 / 3600
    rate[low & np.isin(names, ["paddy", "field", "paddy_dam", "green"])] = spec.get("agri_mmh", 2.0) / 1000 / 3600
    return rate, meta["drain_target_k"]
