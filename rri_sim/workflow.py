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
