"""国土数値情報 河川 (W05, 2007-08 edition) -> rasterised stream network.

Used for *stream burning*: on flat deltas and alluvial plains a 250 m averaged
DEM does not determine the drainage path (filled flats route water to the
nearest sea edge), so the mapped channels of the target water system are
lowered before computing D8 flow directions.  The hydraulic model itself
keeps the un-burned DEM.

Stream attributes: W05_001 水系域コード (筑後川 890906, 阿武隈川 830207?) - use
`system_codes_by_name` to look codes up from the river name.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np

from . import RAW, http_get

URL = "https://nlftp.mlit.go.jp/ksj/gml/data/W05/W05-07/W05-07_{pref:02d}_GML.zip"


def _reader(pref: int):
    import shapefile
    zp = RAW / "ksj" / "W05" / f"W05-07_{pref:02d}_GML.zip"
    if not zp.exists():
        zp.parent.mkdir(parents=True, exist_ok=True)
        zp.write_bytes(http_get(URL.format(pref=pref), timeout=300).content)
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    base = [n for n in names if n.endswith("_Stream.shp")][0][:-4]
    return shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                            shx=io.BytesIO(z.read(base + ".shx")), encoding="cp932")


def system_codes_by_name(prefs, river_name: str) -> set[str]:
    codes = set()
    for p in prefs:
        for rec in _reader(p).iterRecords():
            if rec[3] == river_name:
                codes.add(rec[0])
    return codes


def stream_lines(prefs, system_codes=None):
    """List of (lon, lat) arrays for streams of the given water systems."""
    out = []
    for p in prefs:
        r = _reader(p)
        for sr in r.iterShapeRecords():
            if system_codes and sr.record[0] not in system_codes:
                continue
            pts = np.asarray(sr.shape.points, float)
            if len(pts) >= 2:
                out.append(pts)
    return out


def rasterize_lines(lines, transform, shape, crs, step_frac=0.25):
    """Boolean raster of cells touched by the polylines (dense sampling)."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:6668", crs, always_xy=True)
    ny, nx = shape
    res = abs(transform.a)
    hit = np.zeros(shape, bool)
    for pts in lines:
        x, y = tr.transform(pts[:, 0], pts[:, 1])
        x, y = np.asarray(x), np.asarray(y)
        seg = np.hypot(np.diff(x), np.diff(y))
        n = np.maximum(1, np.ceil(seg / (res * step_frac)).astype(int))
        xs = np.concatenate([np.linspace(x[i], x[i + 1], n[i], endpoint=False) for i in range(len(seg))] + [x[-1:]])
        ys = np.concatenate([np.linspace(y[i], y[i + 1], n[i], endpoint=False) for i in range(len(seg))] + [y[-1:]])
        col = np.floor((xs - transform.c) / transform.a).astype(int)
        row = np.floor((ys - transform.f) / transform.e).astype(int)
        ok = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
        hit[row[ok], col[ok]] = True
    return hit
