"""e-Stat 地域メッシュ統計 (統計GIS) -> asset base quantities on the model grid.

  T001142 : 令和2年国勢調査 4分の1地域メッシュ (≈250 m)  -> 一般世帯数 (column '一般世帯数')
  T001147 : 経済センサス 2分の1地域メッシュ (≈500 m)      -> 従業者数 by 産業大分類 C..S
            (columns T001147022..042 = 従業者数; 001..021 = 事業所数)

Download: https://www.e-stat.go.jp/gis/statmap-search/data?statsId=...&code=<1st mesh>&downloadType=2
Masked cells ('*', 秘匿) are treated as 0; their totals are merged into the
mesh named in HTKSAKI by e-Stat, so totals are approximately preserved.
"""
from __future__ import annotations

import csv
import io
import zipfile

import numpy as np

from . import RAW, http_get

URL = "https://www.e-stat.go.jp/gis/statmap-search/data?statsId={sid}&code={mesh}&downloadType=2"
INDUSTRY_CODES = list("CDEFGHIJKLMNOPQRS")
# employee columns of T001147: 022 A-S total, 023 A-R, 024 C-E, 025 C, 026 D, 027 E,
# 028 F-S, 029 F ... 042 S
EMPLOYEE_COLUMN = {"C": 25, "D": 26, "E": 27, **{c: 29 + i for i, c in enumerate("FGHIJKLMNOPQRS")}}


def _table(sid: str, mesh: str):
    zp = RAW / "estat" / f"{sid}_{mesh}.zip"
    if not zp.exists():
        zp.parent.mkdir(parents=True, exist_ok=True)
        r = http_get(URL.format(sid=sid, mesh=mesh), timeout=120)
        zp.write_bytes(r.content)
    try:
        with zipfile.ZipFile(zp) as z:
            name = [n for n in z.namelist() if n.endswith(".txt")][0]
            txt = z.read(name).decode("cp932", "replace")
    except zipfile.BadZipFile:
        return [], [], []
    rows = list(csv.reader(io.StringIO(txt)))
    return rows[0], [c.strip().replace("　", "") for c in rows[1]], rows[2:]


def mesh_bounds(code: str):
    """(lat_s, lat_n, lon_w, lon_e) of a 3rd / half / quarter mesh code."""
    pq, rs = int(code[0:2]), int(code[2:4])
    t, u, v, w = (int(c) for c in code[4:8])
    lat = pq / 1.5 + t * 5 / 60 + v * 30 / 3600
    lon = 100 + rs + u * 7.5 / 60 + w * 45 / 3600
    dlat, dlon = 30 / 3600, 45 / 3600
    for c in code[8:]:
        k = int(c) - 1                     # 1 SW, 2 SE, 3 NW, 4 NE
        dlat, dlon = dlat / 2, dlon / 2
        lat += dlat * (k // 2)
        lon += dlon * (k % 2)
    return lat, lat + dlat, lon, lon + dlon


def _num(s):
    try:
        return float(s)
    except ValueError:
        return 0.0


def distribute_to_grid(values: dict, transform, shape, crs, weight=None):
    """Distribute mesh totals {code: value} onto model cells whose centres
    fall inside each mesh, proportionally to `weight` (e.g. built-up fraction,
    manual 4.2(3)); uniform if all weights are zero.  A mesh that contains no
    cell centre (cells coarser than the mesh) goes to the cell containing the
    mesh centre, so totals are conserved on any grid."""
    from pyproj import Transformer
    tr = Transformer.from_crs(crs, "EPSG:6668", always_xy=True)
    inv = Transformer.from_crs("EPSG:6668", crs, always_xy=True)
    ny, nx = shape
    X = transform.c + (np.arange(nx) + 0.5) * transform.a
    Y = transform.f + (np.arange(ny) + 0.5) * transform.e
    XX, YY = np.meshgrid(X, Y)
    lon, lat = tr.transform(XX, YY)
    lon, lat = np.asarray(lon), np.asarray(lat)
    out = np.zeros(shape)
    w = np.ones(shape) if weight is None else np.asarray(weight, float)
    for code, val in values.items():
        if val <= 0:
            continue
        s, n, wl, e = mesh_bounds(code)
        m = (lat >= s) & (lat < n) & (lon >= wl) & (lon < e)
        if not m.any():
            x, y = inv.transform(0.5 * (wl + e), 0.5 * (s + n))
            i = int(np.floor((y - transform.f) / transform.e))
            j = int(np.floor((x - transform.c) / transform.a))
            if 0 <= i < ny and 0 <= j < nx:
                out[i, j] += val
            continue
        ww = w[m]
        if ww.sum() <= 0:
            ww = np.ones_like(ww)
        out[m] += val * ww / ww.sum()
    return out


def households(first_meshes, transform, shape, crs, weight=None, column="一般世帯数"):
    """一般世帯数 (default) or another T001142 column, e.g. '人口（総数）'."""
    vals = {}
    for mesh in first_meshes:
        h0, h1, rows = _table("T001142", mesh)
        if not rows:
            continue
        col = h1.index(column)
        for r in rows:
            vals[r[0]] = vals.get(r[0], 0.0) + _num(r[col])
    return distribute_to_grid(vals, transform, shape, crs, weight)


def employees(first_meshes, transform, shape, crs, weight=None):
    """{industry code: employees array}."""
    out = {c: {} for c in INDUSTRY_CODES}
    for mesh in first_meshes:
        h0, h1, rows = _table("T001147", mesh)
        if not rows:
            continue
        for code in INDUSTRY_CODES:
            name = h0.index(f"T001147{EMPLOYEE_COLUMN[code]:03d}")
            for r in rows:
                out[code][r[0]] = out[code].get(r[0], 0.0) + _num(r[name])
    return {c: distribute_to_grid(v, transform, shape, crs, weight) for c, v in out.items()}


def population(first_meshes, transform, shape, crs, weight=None):
    return households(first_meshes, transform, shape, crs, weight, column="人口（総数）")
