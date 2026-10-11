"""Existing levee (embankment) heights from the GSI 5 m DEM (DEM5A, aerial lidar).

At 250 m the averaged DEM erases levees, so a leveed river spills over its
whole floodplain.  For every river cell with a large upstream area, the 5 m
DEM is sampled inside the cell; the levee crest appears as the upper tail of
the elevation distribution.

    crest     = P99.5 of 5 m elevations sampled at ~5 m spacing in a 2x2-cell
                window (levee crests are only 5-10 m wide)
    ground    = model (250 m) ground elevation of the cell
    embankment = clip(crest - ground, 0, max_height)

Cells whose 5 m relief (crest - P10) exceeds `max_relief` are valley reaches
where the upper tail is hillside, not a levee; they are left unchanged.
Where DEM5A tiles are missing, DEM10B (z=14) is used.

Between-levee width (堤間幅): a leveed river conveys floods over the whole
space between its levees (low-water channel + 高水敷), which is much wider than
the area-width regression of the low-water channel.  It is estimated as the
area of the window lower than  P10 + 0.4 (crest - P10)  divided by the window
length, smoothed along the river, and capped at `max_width_cells` x dx.
"""
from __future__ import annotations

import math

import numpy as np

from .data import gsi_dem


def _sample(lon, lat, z, layer):
    n = 2 ** z
    xt = (lon + 180.0) / 360.0 * n
    yt = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * n
    tx, ty = np.floor(xt).astype(int), np.floor(yt).astype(int)
    px = np.clip(((xt - tx) * 256).astype(int), 0, 255)
    py = np.clip(((yt - ty) * 256).astype(int), 0, 255)
    out = np.full(lon.shape, np.nan)
    for (a, b) in set(zip(tx.tolist(), ty.tolist())):
        sel = (tx == a) & (ty == b)
        tile = gsi_dem.fetch_tile(z, a, b, layer)
        out[sel] = tile[py[sel], px[sel]]
    return out


def estimate_levees(domain, min_area_km2=200.0, max_height=12.0, max_relief=20.0,
                    samples=100, window=2.0, smooth=3, crest_pct=99.5, verbose=False):
    """Return embankment heights [m] for every river cell (0 where not estimated)."""
    from pyproj import Transformer
    r = domain.river
    tf = domain.transform
    tr = Transformer.from_crs(domain.crs, "EPSG:6668", always_xy=True)
    emb = np.zeros(r.n)
    width = np.full(r.n, np.nan)
    sel = np.nonzero(r.area_km2 >= min_area_km2)[0]
    # sample a window of `window` x `window` cells centred on the river cell so
    # that levee crests of wide (tidal) channels are captured
    u = (np.arange(samples) + 0.5) / samples * window - (window - 1) / 2
    U, V = np.meshgrid(u, u)
    stats = {"cells": len(sel), "leveed": 0, "valley": 0, "no_data": 0}
    for k in sel:
        x = tf.c + (r.rj[k] + U.ravel()) * tf.a
        y = tf.f + (r.ri[k] + V.ravel()) * tf.e
        lon, lat = tr.transform(x, y)
        lon, lat = np.asarray(lon), np.asarray(lat)
        zz = _sample(lon, lat, 15, "dem5a_png")
        if np.isfinite(zz).mean() < 0.5:
            zz = _sample(lon, lat, 14, "dem_png")
        zz = zz[np.isfinite(zz)]
        if zz.size < 20:
            stats["no_data"] += 1
            continue
        crest, p10 = np.percentile(zz, [crest_pct, 10])
        if crest - p10 > max_relief:
            stats["valley"] += 1
            continue
        ground = domain.z[r.ri[k], r.rj[k]]
        h = float(np.clip(crest - ground, 0.0, max_height))
        emb[k] = h
        if crest - p10 >= 3.0:          # a clear levee/bank profile
            low = (zz < p10 + 0.4 * (crest - p10)).mean()
            width[k] = low * window * abs(tf.a)
        stats["leveed"] += int(h > 0.5)
    # levees are continuous: running median along the downstream path (±smooth cells)
    if smooth:
        est = emb.copy()
        west = width.copy()
        width_s = np.full(r.n, np.nan)
        for k in sel:
            nb = [k]
            up = k
            for _ in range(smooth):
                ups = np.nonzero(r.dn == up)[0]
                if not len(ups):
                    break
                up = ups[np.argmax(r.area_km2[ups])]
                nb.append(up)
            dn = k
            for _ in range(smooth):
                dn = r.dn[dn]
                if dn < 0:
                    break
                nb.append(dn)
            emb[k] = float(np.median(est[nb]))
            w = west[nb]
            if np.isfinite(w).sum() >= 2:
                width_s[k] = float(np.nanmedian(w))
        width = width_s
    stats["leveed_after_smoothing"] = int((emb[sel] > 0.5).sum())
    stats["width_estimated"] = int(np.isfinite(width).sum())
    if verbose:
        print("levee estimation:", stats)
    return emb, stats, width
