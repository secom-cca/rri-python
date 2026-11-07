"""国土数値情報 国有林野 (A45, 2019) -> mask of the model cells inside national forest.

Used to split 森林整備 by implementing actor: national forest (国有林, 林野庁 = 国) and the
other forest (民有林: 都道府県・市町村の森林整備計画, 森林所有者).  A cell is national forest
when its centre lies inside an A45 sub-compartment polygon.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np

from . import RAW, http_get

URL = "https://nlftp.mlit.go.jp/ksj/gml/data/A45/A45-19/A45-19_{pref:02d}_GML.zip"


def _reader(pref: int):
    import shapefile
    zp = RAW / "ksj" / "A45" / f"A45-19_{pref:02d}_GML.zip"
    if not zp.exists():
        zp.parent.mkdir(parents=True, exist_ok=True)
        r = http_get(URL.format(pref=pref), timeout=600)
        r.raise_for_status()
        zp.write_bytes(r.content)
    z = zipfile.ZipFile(zp)
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    return shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")),
                            dbf=io.BytesIO(z.read(stem + ".dbf")), encoding="cp932")


def national_forest_grid(transform, shape, crs, prefs) -> np.ndarray:
    """(ny, nx) bool: cell centre inside a 国有林野 polygon of the prefectures `prefs`."""
    from matplotlib.path import Path
    from pyproj import Transformer
    ny, nx = shape
    to_grid = Transformer.from_crs("EPSG:6668", crs, always_xy=True)
    X = transform.c + (np.arange(nx) + 0.5) * transform.a
    Y = transform.f + (np.arange(ny) + 0.5) * transform.e
    to_ll = Transformer.from_crs(crs, "EPSG:6668", always_xy=True)
    lon, lat = to_ll.transform(*np.meshgrid(X, Y))
    out = np.zeros((ny, nx), bool)
    for pref in prefs:
        for shp in _reader(pref).iterShapes():
            if not shp.points:
                continue
            bx = shp.bbox
            # candidate cells: the polygon's bounding box mapped to grid rows / columns (+1 cell margin)
            gx, gy = to_grid.transform([bx[0], bx[2], bx[0], bx[2]], [bx[1], bx[1], bx[3], bx[3]])
            j0 = max(int(np.floor((min(gx) - transform.c) / transform.a)) - 1, 0)
            j1 = min(int(np.ceil((max(gx) - transform.c) / transform.a)) + 1, nx)
            i0 = max(int(np.floor((max(gy) - transform.f) / transform.e)) - 1, 0)
            i1 = min(int(np.ceil((min(gy) - transform.f) / transform.e)) + 1, ny)
            if i0 >= i1 or j0 >= j1:
                continue
            parts = list(shp.parts) + [len(shp.points)]
            verts, codes = [], []
            for a, b in zip(parts[:-1], parts[1:]):
                ring = shp.points[a:b]
                verts += ring
                codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 1)
            pts = np.c_[lon[i0:i1, j0:j1].ravel(), lat[i0:i1, j0:j1].ravel()]
            inside = Path(np.asarray(verts), codes).contains_points(pts).reshape(i1 - i0, j1 - j0)
            out[i0:i1, j0:j1] |= inside
    return out
