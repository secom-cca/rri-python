"""Goodness-of-fit metrics for hydrographs and inundation maps."""
from __future__ import annotations

import numpy as np


def _pair(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def nse(obs, sim):
    o, s = _pair(obs, sim)
    return 1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def kge(obs, sim):
    """Kling-Gupta efficiency (Gupta et al., 2009)."""
    o, s = _pair(obs, sim)
    r = np.corrcoef(o, s)[0, 1]
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def peak_error(obs, sim):
    o, s = _pair(obs, sim)
    return (s.max() - o.max()) / o.max()


def peak_timing_hours(times, obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    to = times[np.nanargmax(obs)]
    ts = times[np.nanargmax(sim)]
    return (ts - to) / np.timedelta64(1, "h") if hasattr(ts - to, "dtype") else (ts - to)


def volume_error(obs, sim):
    o, s = _pair(obs, sim)
    return (s.sum() - o.sum()) / o.sum()


def hydrograph_scores(times, obs, sim) -> dict:
    return {"NSE": nse(obs, sim), "KGE": kge(obs, sim), "peak_err": peak_error(obs, sim),
            "peak_obs": float(np.nanmax(obs)), "peak_sim": float(np.nanmax(sim)),
            "peak_timing_h": float(peak_timing_hours(np.asarray(times), obs, sim)),
            "volume_err": volume_error(obs, sim)}


def flood_extent_scores(sim_wet, obs_wet, valid=None) -> dict:
    """Binary inundation comparison.  CSI = TP/(TP+FP+FN) (Bates & De Roo 2000),
    hit rate H = TP/(TP+FN), false alarm ratio FAR = FP/(TP+FP)."""
    s = np.asarray(sim_wet, bool)
    o = np.asarray(obs_wet, bool)
    if valid is None:
        valid = np.ones_like(s, bool)
    tp = np.sum(s & o & valid)
    fp = np.sum(s & ~o & valid)
    fn = np.sum(~s & o & valid)
    return {"CSI": tp / max(tp + fp + fn, 1), "H": tp / max(tp + fn, 1),
            "FAR": fp / max(tp + fp, 1), "TP": int(tp), "FP": int(fp), "FN": int(fn)}


def depth_scores(sim_depth, obs_depth, valid) -> dict:
    d = np.asarray(sim_depth)[valid] - np.asarray(obs_depth)[valid]
    return {"bias_m": float(np.mean(d)), "rmse_m": float(np.sqrt(np.mean(d ** 2))),
            "mae_m": float(np.mean(np.abs(d))), "n": int(valid.sum())}
