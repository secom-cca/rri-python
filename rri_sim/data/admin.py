"""国土数値情報 行政区域 (N03, 2020-01-01) -> municipality of every model cell."""
from __future__ import annotations

import io
import zipfile

import numpy as np

from . import RAW, http_get

URL = "https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2020/N03-20200101_{pref:02d}_GML.zip"


def _reader(pref: int):
    import shapefile
    zp = RAW / "ksj" / "N03" / f"N03-20200101_{pref:02d}_GML.zip"
    if not zp.exists():
        zp.parent.mkdir(parents=True, exist_ok=True)
        r = http_get(URL.format(pref=pref), timeout=300)
        r.raise_for_status()
        zp.write_bytes(r.content)
    z = zipfile.ZipFile(zp)
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    return shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), dbf=io.BytesIO(z.read(stem + ".dbf")),
                            shx=io.BytesIO(z.read(stem + ".shx")), encoding="cp932")


def municipality_grid(transform, shape, crs, prefs):
    """(codes array (ny, nx) of 5-digit 市区町村 codes as int, 0 = none;
    {code: (prefecture, municipality name)}, {(prefecture, name): area km2}) from
    cell centres (政令市: wards)."""
    from matplotlib.path import Path
    from pyproj import Transformer
    ny, nx = shape
    tr = Transformer.from_crs(crs, "EPSG:6668", always_xy=True)
    X = transform.c + (np.arange(nx) + 0.5) * transform.a
    Y = transform.f + (np.arange(ny) + 0.5) * transform.e
    lon, lat = tr.transform(*np.meshgrid(X, Y))
    pts = np.c_[np.ravel(lon), np.ravel(lat)]
    lo0, la0, lo1, la1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
    out = np.zeros(ny * nx, np.int64)
    names, areas = {}, {}
    for pref in prefs:
        rd = _reader(pref)
        for sr in rd.iterShapeRecords():
            rec, shp = sr.record, sr.shape
            if not rec[4] or not shp.points:
                continue
            bx = shp.bbox
            if bx[2] < lo0 or bx[0] > lo1 or bx[3] < la0 or bx[1] > la1:
                continue
            code = int(rec[4])
            parts = list(shp.parts) + [len(shp.points)]
            area = 0.0                              # signed shoelace (holes have the opposite sign), km2
            for a, b in zip(parts[:-1], parts[1:]):
                p = np.asarray(shp.points[a:b])
                kx = 111.32 * np.cos(np.radians(p[:, 1].mean()))
                area += 0.5 * np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]) * kx * 110.57
            nm = (rec[0], rec[3] or rec[2])
            names[code] = nm
            areas[nm] = areas.get(nm, 0.0) + abs(area)
            sel = (pts[:, 0] >= bx[0]) & (pts[:, 0] <= bx[2]) & (pts[:, 1] >= bx[1]) & (pts[:, 1] <= bx[3])
            if not sel.any():
                continue
            parts = list(shp.parts) + [len(shp.points)]
            verts, codes = [], []
            for a, b in zip(parts[:-1], parts[1:]):
                ring = shp.points[a:b]
                verts += ring
                codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 1)
            inside = Path(np.asarray(verts), codes).contains_points(pts[sel])
            idx = np.flatnonzero(sel)[inside]
            out[idx] = code
    return out.reshape(ny, nx), names, areas
