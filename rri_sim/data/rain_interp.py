"""Spatial interpolation of gauge rainfall onto the model grid."""
from __future__ import annotations

import numpy as np


def idw_weights(gx, gy, X, Y, power=2.0, max_neighbours=6):
    """Inverse-distance weights of gauges (gx, gy) for grid points (X, Y).
    Returns (idx[npts, k], w[npts, k])."""
    P = np.column_stack([X.ravel(), Y.ravel()])
    G = np.column_stack([gx, gy])
    d = np.sqrt(((P[:, None, :] - G[None, :, :]) ** 2).sum(-1))
    k = min(max_neighbours, len(gx))
    idx = np.argsort(d, axis=1)[:, :k]
    dk = np.take_along_axis(d, idx, axis=1)
    w = 1.0 / np.maximum(dk, 1.0) ** power
    return idx, w


def idw_fields(values, gx, gy, X, Y, power=2.0, max_neighbours=6):
    """values: (nt, ngauge) with NaN for missing.  Missing gauges are skipped
    per time step (weights renormalised).  Returns (nt, ny, nx)."""
    values = np.asarray(values, float)
    idx, w = idw_weights(gx, gy, X, Y, power, max_neighbours=len(gx))
    nt = values.shape[0]
    out = np.empty((nt,) + X.shape)
    for t in range(nt):
        v = values[t][idx]                       # (npts, ng) sorted by distance
        ok = np.isfinite(v)
        # keep only the nearest `max_neighbours` valid gauges
        rank = np.cumsum(ok, axis=1)
        use = ok & (rank <= max_neighbours)
        ww = np.where(use, w, 0.0)
        s = ww.sum(1)
        f = np.where(s > 0, (np.where(use, v, 0.0) * ww).sum(1) / np.where(s > 0, s, 1), np.nan)
        out[t] = f.reshape(X.shape)
    return out
