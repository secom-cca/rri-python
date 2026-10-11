"""Daily-life service facilities (国土数値情報) for accessibility indicators.

  医療 : P04-20 医療機関 - 病院(1)・診療所(2) whose 診療科目 include 内科 or 外科
         (都市構造の評価に関するハンドブック: 医療施設 = 病院(内科又は外科)及び診療所)
  教育 : P29-23 学校 - 小学校 (16001), 中学校 (16002)
  公共 : P05-22 市町村役場等 - 本庁(1), 支所・出張所・連絡所(2), (3)
  商業 : not in 国土数値情報; proxied in `indicators` by 経済センサス 卸売・小売業 (I)
         employment (handbook definition: 商業施設 = 百貨店, スーパー, コンビニ等).
"""
from __future__ import annotations

import io
import zipfile

import numpy as np

from . import RAW, http_get

SRC = {
    "medical": ("P04/P04-20/P04-20_{p:02d}_GML.zip", "cp932"),
    "school": ("P29/P29-23/P29-23_{p:02d}_GML.zip", "utf-8"),
    "public": ("P05/P05-22/P05-22_{p:02d}_GML.zip", "cp932"),
}


def _reader(kind, pref):
    import shapefile
    rel, enc = SRC[kind]
    rel = rel.format(p=pref)
    zp = RAW / "ksj" / "fac" / rel.split("/")[-1]
    if not zp.exists():
        zp.parent.mkdir(parents=True, exist_ok=True)
        zp.write_bytes(http_get("https://nlftp.mlit.go.jp/ksj/gml/data/" + rel, timeout=300).content)
    z = zipfile.ZipFile(zp)
    b = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    return shapefile.Reader(shp=io.BytesIO(z.read(b + ".shp")), dbf=io.BytesIO(z.read(b + ".dbf")),
                            shx=io.BytesIO(z.read(b + ".shx")), encoding=enc)


def points(kind: str, prefs) -> np.ndarray:
    """(n, 2) lon/lat of facilities of a kind for the given prefectures."""
    out = []
    for p in prefs:
        for sr in _reader(kind, p).iterShapeRecords():
            rec = sr.record
            if kind == "medical":
                if int(rec[0]) not in (1, 2):
                    continue
                subj = str(rec[3])
                if "内科" not in subj and "外科" not in subj:
                    continue
            elif kind == "school":
                if rec[2] not in ("16001", "16002"):
                    continue
            elif kind == "public":
                if str(rec[1]) not in ("1", "2", "3"):
                    continue
            out.append(sr.shape.points[0])
    return np.asarray(out, float).reshape(-1, 2)


def to_cells(lonlat, transform, shape, crs):
    """Flat indices of grid cells containing the points (inside the grid)."""
    from pyproj import Transformer
    if len(lonlat) == 0:
        return np.array([], np.int64)
    x, y = Transformer.from_crs("EPSG:6668", crs, always_xy=True).transform(lonlat[:, 0], lonlat[:, 1])
    col = np.floor((np.asarray(x) - transform.c) / transform.a).astype(int)
    row = np.floor((np.asarray(y) - transform.f) / transform.e).astype(int)
    ny, nx = shape
    ok = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
    return np.unique(row[ok] * nx + col[ok])
