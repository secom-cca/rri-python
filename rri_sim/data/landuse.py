"""国土数値情報 土地利用細分メッシュ (L03-b, 100 m) -> class fractions on the model grid.

10-digit mesh code  pq rs t u v w x y :
  lat_SW = pq/1.5 + t*5'/60 + v*30"/3600 + x*3"/3600
  lon_SW = 100 + rs + u*7.5'/60 + w*45"/3600 + y*4.5"/3600
Land-use codes (L03-b-21): 0100 田, 0200 その他の農用地, 0500 森林, 0600 荒地,
0700 建物用地, 0901 道路, 0902 鉄道, 1000 その他の用地, 1100 河川地及び湖沼,
1400 海浜, 1500 海水域, 1600 ゴルフ場.
"""
from __future__ import annotations

import zipfile

import numpy as np
import requests

from . import RAW

URL = "https://nlftp.mlit.go.jp/ksj/gml/data/L03-b/L03-b-{yy}/L03-b-{yy}_{mesh}-jgd2011_GML.zip"

# model land classes
CLASSES = ["forest", "paddy", "field", "urban", "water"]
CODE_TO_CLASS = {"0100": "paddy", "0200": "field", "0500": "forest", "0600": "forest",
                 "1600": "forest", "0700": "urban", "0901": "urban", "0902": "urban",
                 "1000": "urban", "1100": "water", "1400": "water", "1500": "water"}


def first_meshes(lon_min, lat_min, lon_max, lat_max):
    out = []
    for p in range(int(lat_min * 1.5), int(lat_max * 1.5) + 1):
        for r in range(int(lon_min) - 100, int(lon_max) - 100 + 1):
            out.append(f"{p:02d}{r:02d}")
    return out


def _load_mesh(mesh: str, yy: str = "21"):
    import shapefile
    d = RAW / "ksj" / f"L03-b-{yy}"
    zp = d / f"L03-b-{yy}_{mesh}-jgd2011_GML.zip"
    if not zp.exists():
        d.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL.format(yy=yy, mesh=mesh), timeout=300)
        if r.status_code == 404:
            return [], []
        r.raise_for_status()
        zp.write_bytes(r.content)
    with zipfile.ZipFile(zp) as z:
        import io
        dbf = [n for n in z.namelist() if n.lower().endswith(".dbf")][0]
        rd = shapefile.Reader(dbf=io.BytesIO(z.read(dbf)), encoding="cp932")
        codes, lus = [], []
        for rec in rd.iterRecords():
            codes.append(rec[0])
            lus.append(rec[1])
    return codes, lus


def mesh_center(code: str):
    pq, rs = int(code[0:2]), int(code[2:4])
    t, u, v, w, x, y = (int(c) for c in code[4:10])
    lat = pq / 1.5 + t * 5 / 60 + v * 30 / 3600 + x * 3 / 3600 + 1.5 / 3600
    lon = 100 + rs + u * 7.5 / 60 + w * 45 / 3600 + y * 4.5 / 3600 + 2.25 / 3600
    return lat, lon


def class_fractions(transform, shape, crs, lon_min, lat_min, lon_max, lat_max, yy="21"):
    """Fraction of each model class per grid cell, (nclass, ny, nx)."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:6668", crs, always_xy=True)  # JGD2011 geographic
    ny, nx = shape
    counts = np.zeros((len(CLASSES), ny, nx))
    for mesh in first_meshes(lon_min, lat_min, lon_max, lat_max):
        codes, lus = _load_mesh(mesh, yy)
        if not codes:
            continue
        ll = np.array([mesh_center(c) for c in codes])
        keep = ((ll[:, 0] >= lat_min - 0.02) & (ll[:, 0] <= lat_max + 0.02)
                & (ll[:, 1] >= lon_min - 0.02) & (ll[:, 1] <= lon_max + 0.02))
        if not keep.any():
            continue
        x, y = tr.transform(ll[keep, 1], ll[keep, 0])
        col = np.floor((np.asarray(x) - transform.c) / transform.a).astype(int)
        row = np.floor((np.asarray(y) - transform.f) / transform.e).astype(int)
        lu = np.asarray(lus)[keep]
        ok = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
        for k, name in enumerate(CLASSES):
            sel = ok & np.isin(lu, [c for c, n in CODE_TO_CLASS.items() if n == name])
            np.add.at(counts[k], (row[sel], col[sel]), 1)
    tot = counts.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(tot > 0, counts / tot, 0.0)
    return frac
