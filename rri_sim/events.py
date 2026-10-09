"""Event forcing assembly: gauge rainfall (MLIT + JMA) -> gridded hourly fields."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import PROCESSED, jma, mlit_water
from .data.rain_interp import idw_fields
from .forcing import Rainfall


def collect_gauges(bbox, start, end, mlit_suikei=None, jma_prec_nos=(), buffer_deg=0.15,
                   cache_tag=None, verbose=True):
    """Hourly rainfall [mm] at all gauges inside bbox+buffer.

    Returns DataFrame (index = hour end JST, columns = gauge id) and a station
    table with lon/lat/source.
    """
    cache = PROCESSED / f"rain_gauges_{cache_tag}.pkl" if cache_tag else None
    if cache is not None and cache.exists():
        obj = pd.read_pickle(cache)
        return obj["data"], obj["stations"]
    lon0, lat0, lon1, lat1 = bbox
    series, rows = {}, []
    if mlit_suikei:
        for sid, name in mlit_water.search_stations(mlit_suikei, "01"):
            info = mlit_water.site_info(sid)
            if info["lat"] is None:
                continue
            if not (lon0 - buffer_deg <= info["lon"] <= lon1 + buffer_deg
                    and lat0 - buffer_deg <= info["lat"] <= lat1 + buffer_deg):
                continue
            try:
                s = mlit_water.hourly(sid, mlit_water.KIND_RAIN, start, end)
            except RuntimeError:
                continue
            if np.isfinite(s.values).sum() == 0:
                continue
            series[f"mlit_{sid}"] = s
            rows.append({"id": f"mlit_{sid}", "name": name, "lon": info["lon"],
                         "lat": info["lat"], "source": "MLIT"})
    for st in jma.stations_in_bbox(jma_prec_nos, lon0, lat0, lon1, lat1, buffer_deg):
        s = jma.hourly_precip(st, start, end)
        if np.isfinite(s.values).sum() == 0:
            continue
        series[f"jma_{st.block}"] = s
        rows.append({"id": f"jma_{st.block}", "name": st.name, "lon": st.lon, "lat": st.lat,
                     "source": "JMA"})
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h")
    data = pd.DataFrame({k: v.reindex(idx) for k, v in series.items()})
    stations = pd.DataFrame(rows).set_index("id")
    if verbose:
        print(f"{len(stations)} gauges ({(stations.source == 'MLIT').sum()} MLIT, "
              f"{(stations.source == 'JMA').sum()} JMA)")
    if cache is not None:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"data": data, "stations": stations}, cache)
    return data, stations


def gridded_rainfall(domain, data: pd.DataFrame, stations: pd.DataFrame, t0: pd.Timestamp,
                     power=2.0, max_neighbours=6):
    """IDW fields on the model grid -> Rainfall object with t=0 at t0.

    Hour label T (accumulation over (T-1h, T]) becomes the interval
    [T-1h-t0, T-t0] in seconds.
    """
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", domain.crs, always_xy=True)
    gx, gy = tr.transform(stations.lon.values, stations.lat.values)
    ny, nx = domain.shape
    tf = domain.transform
    X = tf.c + (np.arange(nx) + 0.5) * tf.a
    Y = tf.f + (np.arange(ny) + 0.5) * tf.e
    XX, YY = np.meshgrid(X, Y)
    vals = data[stations.index].values
    fields = idw_fields(vals, np.asarray(gx), np.asarray(gy), XX, YY, power, max_neighbours)
    fields = np.nan_to_num(fields, nan=0.0)
    ends = (data.index - t0) / pd.Timedelta(seconds=1)
    edges = np.concatenate([[ends[0] - 3600.0], ends.values.astype(float)])
    return Rainfall(edges, fields), fields


def basin_mean_series(fields, mask):
    return np.array([f[mask].mean() for f in fields])
