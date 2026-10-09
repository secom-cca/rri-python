"""Build a real-basin model from public data (DEM, land use, gauges)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .domain import Domain, build_river
from .hydro import D8_DI, D8_DJ, d8_directions, fill_depressions, flow_accumulation
from .params import LandClassParams, ModelParams, RiverGeometryParams, NumericsParams


@dataclass
class BasinConfig:
    name: str
    outlet_lonlat: tuple[float, float]
    bbox: tuple[float, float, float, float]       # lon_min, lat_min, lon_max, lat_max
    crs: str                                      # projected CRS (e.g. EPSG:6690 UTM52N)
    res: float = 250.0
    dem_zoom: int = 13
    river_threshold_km2: float = 5.0
    snap_radius_m: float = 1500.0
    expected_area_km2: float | None = None
    gauges: dict = field(default_factory=dict)    # name -> (lon, lat)


def upstream_mask(fdir, i0, j0):
    """Cells draining to (i0, j0) following D8 directions."""
    ny, nx = fdir.shape
    # reverse graph via downstream index
    di = np.where(fdir >= 0, D8_DI[np.clip(fdir, 0, 7)], 0)
    dj = np.where(fdir >= 0, D8_DJ[np.clip(fdir, 0, 7)], 0)
    mask = np.zeros((ny, nx), bool)
    mask[i0, j0] = True
    changed = True
    # iterative propagation (vectorised); converges in <= longest flow path steps
    while changed:
        ii = np.arange(ny)[:, None] + di
        jj = np.arange(nx)[None, :] + dj
        valid = (fdir >= 0) & (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
        down = np.zeros((ny, nx), bool)
        down[valid] = mask[ii[valid], jj[valid]]
        new = mask | down
        changed = bool((new != mask).any())
        mask = new
    return mask


def lonlat_to_rc(lon, lat, transform, crs):
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    col = int(np.floor((x - transform.c) / transform.a))
    row = int(np.floor((y - transform.f) / transform.e))
    return row, col, x, y


def default_land_params() -> list[LandClassParams]:
    """A-priori RRI-type parameters per class (order = landuse.CLASSES).

    forest : lateral subsurface flow (RRI mountain setting)
    paddy/field/urban : Green-Ampt vertical infiltration, no lateral subsurface
    water  : impervious, smooth
    """
    return [
        LandClassParams(name="forest", n=0.4, ka=0.1, soil_depth=1.0, gamma_a=0.4,
                        gamma_m=0.1, beta=8.0),
        LandClassParams(name="paddy", n=0.3, kv=2e-7, phi_minus_theta=0.2, sf=0.1,
                        infilt_limit=0.05),
        LandClassParams(name="field", n=0.2, kv=5e-6, phi_minus_theta=0.3, sf=0.2,
                        infilt_limit=0.3),
        LandClassParams(name="urban", n=0.05, kv=1e-7, phi_minus_theta=0.1, sf=0.1,
                        infilt_limit=0.01),
        LandClassParams(name="water", n=0.03),
    ]


def build_basin(cfg: BasinConfig, params: ModelParams | None = None, cache_dir: Path | None = None,
                verbose=True):
    """DEM -> watershed -> river network -> land classes.  Returns (Domain, meta)."""
    from .data import gsi_dem, landuse
    lon0, lat0, lon1, lat1 = cfg.bbox
    z_full, tf, crs = gsi_dem.dem_on_grid(lon0, lat0, lon1, lat1, cfg.crs, cfg.res, cfg.dem_zoom)
    valid = np.isfinite(z_full)
    zf = fill_depressions(np.where(valid, z_full, 0.0), valid)
    fdir = d8_directions(zf, valid, cfg.res, cfg.res)
    acc = flow_accumulation(fdir, valid, cfg.res ** 2) / 1e6
    # snap outlet to max accumulation within radius
    r0, c0, _, _ = lonlat_to_rc(*cfg.outlet_lonlat, tf, crs)
    rad = int(np.ceil(cfg.snap_radius_m / cfg.res))
    sub = acc[max(r0 - rad, 0):r0 + rad + 1, max(c0 - rad, 0):c0 + rad + 1]
    a, b = np.unravel_index(np.argmax(sub), sub.shape)
    ro, co = max(r0 - rad, 0) + a, max(c0 - rad, 0) + b
    ws = upstream_mask(fdir, ro, co)
    area = ws.sum() * cfg.res ** 2 / 1e6
    if verbose:
        print(f"[{cfg.name}] outlet snapped ({r0},{c0})->({ro},{co}); "
              f"watershed {area:.1f} km2 (expected {cfg.expected_area_km2})")
    # crop
    rows = np.nonzero(ws.any(1))[0]
    cols = np.nonzero(ws.any(0))[0]
    r_a, r_b = max(rows[0] - 1, 0), min(rows[-1] + 2, ws.shape[0])
    c_a, c_b = max(cols[0] - 1, 0), min(cols[-1] + 2, ws.shape[1])
    from rasterio.transform import from_origin
    tf_c = from_origin(tf.c + c_a * cfg.res, tf.f - r_a * cfg.res, cfg.res, cfg.res)
    z = z_full[r_a:r_b, c_a:c_b]
    mask = ws[r_a:r_b, c_a:c_b]
    fd = fdir[r_a:r_b, c_a:c_b].copy()
    zf_c = zf[r_a:r_b, c_a:c_b]
    acc_c = acc[r_a:r_b, c_a:c_b]
    # land use
    frac = landuse.class_fractions(tf_c, z.shape, crs, lon0, lat0, lon1, lat1)
    lc = np.argmax(frac, axis=0)
    lc[frac.sum(0) == 0] = 0
    params = params or ModelParams(land_classes=default_land_params(),
                                   river=RiverGeometryParams(), numerics=NumericsParams())
    # the outlet cell must drain out of the domain
    fd[ro - r_a, co - c_a] = -1
    acc_ws = flow_accumulation(fd, mask, cfg.res ** 2) / 1e6
    river = build_river(np.where(mask, z, np.nan), zf_c, mask, fd, acc_ws, cfg.res, cfg.res,
                        params.river, cfg.river_threshold_km2)
    d = Domain(z=np.where(mask, z, np.nan), mask=mask, dx=cfg.res, dy=cfg.res,
               landclass=lc.astype(np.int64), params=params, river=river,
               transform=tf_c, crs=crs)
    meta = {"outlet_rc": (int(ro - r_a), int(co - c_a)), "area_km2": float(area),
            "landuse_fraction": frac, "acc_km2": acc_ws, "fdir": fd}
    # gauge river indices
    g = {}
    for name, (lon, lat) in cfg.gauges.items():
        r, c, _, _ = lonlat_to_rc(lon, lat, tf_c, crs)
        g[name] = snap_to_river(river, r, c, acc_ws, rad)
    meta["gauge_index"] = g
    return d, meta


def snap_to_river(river, r, c, acc_km2, rad):
    """River cell with the largest upstream area within rad cells of (r, c)."""
    best, bk = -1.0, -1
    for k in range(river.n):
        if abs(river.ri[k] - r) <= rad and abs(river.rj[k] - c) <= rad:
            if river.area_km2[k] > best:
                best, bk = river.area_km2[k], k
    if bk < 0:
        raise ValueError("no river cell near gauge")
    return bk


def save_domain(d: Domain, meta: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = d.river
    np.savez_compressed(
        path, z=d.z, mask=d.mask, landclass=d.landclass, dx=d.dx, dy=d.dy,
        transform=np.array(d.transform)[:6], crs=d.crs,
        ri=r.ri, rj=r.rj, dn=r.dn, length=r.length, width=r.width, depth=r.depth, zb=r.zb,
        embankment=r.embankment, n_river=r.n_river, area_km2=r.area_km2,
        outlet_slope=r.outlet_slope, landuse_fraction=meta["landuse_fraction"],
        meta_json=json.dumps({k: v for k, v in meta.items()
                              if k in ("outlet_rc", "area_km2", "gauge_index")}))
