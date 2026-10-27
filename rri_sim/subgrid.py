"""Sub-grid ground elevations of assets for damage estimation.

A model cell has one ground elevation z (cell mean) and one water depth h, so
the damage code used to flood every house of a cell with h.  On coarse grids
(1 km) this floods whole towns that in reality sit on natural levees, terraces
or fill a few metres above the lowest part of the cell.

Here each cell gets the distribution of the ground elevation *where the assets
are*, from

    国土数値情報 土地利用細分メッシュ (L03-b, 100 m)  -> where buildings / farmland are
    国土地理院 DEM5A (5 m, z=15 tiles; DEM10B where missing) -> their ground elevation

For every 100 m mesh in a damage-eligible cell (HAND <= hand_max) the median of
2 x 2 DEM samples is its ground elevation e.  Per cell and asset group
(建物用地 = 0700; 農地 = 0100 田 + 0200 その他の農用地; all meshes) the K
quantiles of the offset  o = e - z  are stored.

Water level: the model stores the slope water of a cell as a depth h over a
flat cell (volume h A).  The water level H in the real (sub-grid) terrain is
the one holding the same volume over the cell's hypsometry (all meshes):

    mean_q max(H - o_q^all, 0) = h

so in a valley cell the water gathers in the valley floor (deeper than h, over
a smaller area) and in a flat cell H ~ h.  A mesh of offset o is flooded to

    d = max(H - o, 0)            (only where the cell is wet, h > 0)

and damage is the mean over the K quantiles (each holds 1/K of the cell's
assets of that group).  Cells without meshes of a group use o = 0 (the old
cell-uniform depth).
"""
from __future__ import annotations

import numpy as np

from .data import landuse

GROUPS = {"urban": ["0700"], "farm": ["0100", "0200"], "all": None}
K = 20                                   # quantiles per cell and group


def _sample(lon, lat, z, layer, say=None):
    """DEM values at points, reading each web-mercator tile once."""
    import math
    from .data import gsi_dem
    n = 2 ** z
    xt = (lon + 180.0) / 360.0 * n
    yt = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * n
    tx, ty = np.floor(xt).astype(np.int64), np.floor(yt).astype(np.int64)
    px = np.clip(((xt - tx) * 256).astype(int), 0, 255)
    py = np.clip(((yt - ty) * 256).astype(int), 0, 255)
    out = np.full(lon.shape, np.nan)
    key = tx * n + ty
    uk, inv = np.unique(key, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uk) + 1))
    for t, k in enumerate(uk):
        idx = order[bounds[t]:bounds[t + 1]]
        tile = gsi_dem.fetch_tile(z, int(k // n), int(k % n), layer)
        out[idx] = tile[py[idx], px[idx]]
        if say and t % 200 == 0:
            say(0.1 + 0.8 * t / len(uk), f"5 m標高タイルを取得中 {t}/{len(uk)}")
    return out


def _dem(lon, lat, say=None):
    z = _sample(lon, lat, 15, "dem5a_png", say)
    miss = ~np.isfinite(z)
    if miss.any():
        z[miss] = _sample(lon[miss], lat[miss], 14, "dem_png")
    return z


def elevation_offsets(domain, bbox, hand, hand_max=10.0, k=K, say=lambda f, msg: None):
    """{group: (k, ny, nx) float32 offsets [m] (NaN = no mesh of that group)}."""
    from pyproj import Transformer
    tf, (ny, nx) = domain.transform, domain.shape
    tr = Transformer.from_crs("EPSG:6668", domain.crs, always_xy=True)
    eligible = domain.mask & (np.nan_to_num(hand, nan=1e9) <= hand_max)
    lon0, lat0, lon1, lat1 = bbox
    cells, lus, lons, lats = [], [], [], []
    for mesh in landuse.first_meshes(lon0, lat0, lon1, lat1):
        codes, lu = landuse._load_mesh(mesh)
        if not codes:
            continue
        ll = np.array([landuse.mesh_center(c) for c in codes])
        keep = (ll[:, 0] >= lat0) & (ll[:, 0] <= lat1) & (ll[:, 1] >= lon0) & (ll[:, 1] <= lon1)
        if not keep.any():
            continue
        x, y = tr.transform(ll[keep, 1], ll[keep, 0])
        col = np.floor((np.asarray(x) - tf.c) / tf.a).astype(int)
        row = np.floor((np.asarray(y) - tf.f) / tf.e).astype(int)
        ok = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
        ok[ok] = eligible[row[ok], col[ok]]
        cells.append(row[ok] * nx + col[ok]); lus.append(np.asarray(lu)[keep][ok])
        lats.append(ll[keep, 0][ok]); lons.append(ll[keep, 1][ok])
    if not cells:
        return {g: np.full((k, ny, nx), np.nan, np.float32) for g in GROUPS}
    cells, lus = np.concatenate(cells), np.concatenate(lus)
    lats, lons = np.concatenate(lats), np.concatenate(lons)
    # 2 x 2 samples inside each 100 m mesh (3" x 4.5"), median
    say(0.1, f"建物・農地の標高を取得中（100 mメッシュ {len(cells):,} 個、5 m標高）")
    dlat, dlon = 3 / 3600 / 4, 4.5 / 3600 / 4
    m = len(cells)
    pts_lon = np.concatenate([lons + b * dlon for a, b in [(-1, -1), (-1, 1), (1, -1), (1, 1)]])
    pts_lat = np.concatenate([lats + a * dlat for a, b in [(-1, -1), (-1, 1), (1, -1), (1, 1)]])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            e = np.nanmedian(_dem(pts_lon, pts_lat, say).reshape(4, m), axis=0)
    z = domain.z.ravel()
    off = e - z[cells]
    good = np.isfinite(off)
    qs = (np.arange(k) + 0.5) / k
    out = {}
    for g, codes in GROUPS.items():
        sel = good & (np.isin(lus, codes) if codes else True)
        arr = np.full((k, ny * nx), np.nan, np.float32)
        order = np.argsort(cells[sel], kind="stable")
        cs, os_ = cells[sel][order], off[sel][order]
        starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        ends = np.r_[starts[1:], len(cs)]
        for s0, s1 in zip(starts, ends):
            arr[:, cs[s0]] = np.quantile(os_[s0:s1], qs)
        out[g] = arr.reshape(k, ny, nx)
    return out


def water_level(h, off_all, iters=40):
    """Water level H (relative to the model ground z) holding the cell's water
    volume h A over the sub-grid hypsometry; H = h where there is no data."""
    h = np.nan_to_num(np.asarray(h, float))
    o = np.asarray(off_all, float)
    have = np.isfinite(o).all(0)
    o = np.where(have[None], o, 0.0)
    lo, hi = o.min(0), o.max(0) + h          # f(lo) = 0 <= h <= f(hi)
    for _ in range(iters):                   # bisection; f(H) = mean max(H - o, 0) is monotone
        mid = 0.5 * (lo + hi)
        f = np.maximum(mid[None] - o, 0.0).mean(0)
        up = f < h
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return np.where(h > 0, 0.5 * (lo + hi), 0.0)


def depths(h, offsets, level):
    """(k, ny, nx) depths of the meshes of one asset group for water level
    `level` (from water_level); 0 where the cell is dry.  Cells without meshes
    of the group keep the cell depth h."""
    h = np.nan_to_num(np.asarray(h, float))
    have = np.isfinite(offsets).all(0)
    d = np.maximum(level[None] - np.nan_to_num(offsets, nan=0.0), 0.0)
    d = np.where(have[None], d, h[None])
    return np.where(h[None] > 0, d, 0.0)
