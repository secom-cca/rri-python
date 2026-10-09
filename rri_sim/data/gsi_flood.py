"""GSI 浸水推定段彩図 / 浸水推定図 (PNG tiles) -> observed inundation extent on the model grid.

Tiles: https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.png
  e.g. 20191012typhoon19_abukuma_1014dansaizu (阿武隈川, 2019-10-14 16時時点)
       20200703oame_hita_0707dansaizu          (花月川 日田市, 2020-07-07 14時作成)
Any opaque pixel = estimated inundation.  Depth is encoded by a continuous
blue colour ramp whose range is given only in the map legend, so only the
extent is used here (documented limitation).  Tiles that do not exist (404)
are outside the mapped area and excluded from the evaluation.
"""
from __future__ import annotations

import io
import math
import time

import numpy as np
import requests
from PIL import Image

from . import RAW
from .gsi_dem import R_EARTH, lonlat_to_tile

URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.png"


def _tile(layer, z, x, y):
    cache = RAW / "gsi_flood" / layer / str(z) / str(x) / f"{y}.png"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL.format(layer=layer, z=z, x=x, y=y), timeout=60)
        cache.write_bytes(r.content if r.status_code == 200 else b"")
        time.sleep(0.05)
    data = cache.read_bytes()
    if not data:
        return None
    return np.array(Image.open(io.BytesIO(data)).convert("RGBA"))[..., 3] > 0


def extent_on_grid(layer, bbox, transform, shape, crs, z=15, frac_threshold=0.5):
    """Returns (observed_wet[ny,nx] bool, mapped[ny,nx] bool, wet_fraction)."""
    import rasterio.warp as rw
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    lon0, lat0, lon1, lat1 = bbox
    x0, y1 = lonlat_to_tile(lon0, lat0, z)
    x1, y0 = lonlat_to_tile(lon1, lat1, z)
    tx = range(int(x0), int(x1) + 1)
    ty = range(int(y0), int(y1) + 1)
    wet = np.zeros((256 * len(ty), 256 * len(tx)), np.float32)
    have = np.zeros_like(wet)
    for a, yy in enumerate(ty):
        for b, xx in enumerate(tx):
            t = _tile(layer, z, xx, yy)
            if t is None:
                continue
            wet[a * 256:(a + 1) * 256, b * 256:(b + 1) * 256] = t
            have[a * 256:(a + 1) * 256, b * 256:(b + 1) * 256] = 1.0
    size = 2 * math.pi * R_EARTH / 2 ** z
    src_tf = from_origin(-math.pi * R_EARTH + tx[0] * size, math.pi * R_EARTH - ty[0] * size,
                         size / 256, size / 256)
    out_w = np.zeros(shape, np.float32)
    out_h = np.zeros(shape, np.float32)
    for src, dst in [(wet, out_w), (have, out_h)]:
        rw.reproject(src, dst, src_transform=src_tf, src_crs="EPSG:3857", dst_transform=transform,
                     dst_crs=crs, resampling=Resampling.average)
    mapped = out_h > 0.99
    return (out_w >= frac_threshold) & mapped, mapped, out_w
