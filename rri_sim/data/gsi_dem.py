"""GSI (国土地理院) elevation tiles -> model grid.

Tiles: https://cyberjapandata.gsi.go.jp/xyz/dem_png/{z}/{x}/{y}.png  (DEM10B, z<=14)
       https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/{z}/{x}/{y}.png (DEM5A, z=15, partial)
PNG encoding (地理院タイル標高PNG): x = R*2^16 + G*2^8 + B;
  x < 2^23 -> h = 0.01 x ; x > 2^23 -> h = 0.01 (x - 2^24) ; x == 2^23 -> NA.
"""
from __future__ import annotations

import io
import math
import time

import numpy as np
import requests
from PIL import Image

from . import RAW

URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.png"
R_EARTH = 6378137.0


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def decode_png(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.int64)
    g = arr[..., 1].astype(np.int64)
    b = arr[..., 2].astype(np.int64)
    x = (r << 16) + (g << 8) + b
    h = np.where(x < 2 ** 23, x * 0.01, (x - 2 ** 24) * 0.01).astype(float)
    h[x == 2 ** 23] = np.nan
    return h


def fetch_tile(z, x, y, layer="dem_png"):
    cache = RAW / "gsi" / layer / str(z) / str(x) / f"{y}.png"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL.format(layer=layer, z=z, x=x, y=y), timeout=60)
        if r.status_code == 404:
            cache.write_bytes(b"")          # remember missing tile (sea / outside Japan)
        else:
            r.raise_for_status()
            cache.write_bytes(r.content)
        time.sleep(0.05)
    data = cache.read_bytes()
    if not data:
        return np.full((256, 256), np.nan)
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return decode_png(img)


def mosaic(lon_min, lat_min, lon_max, lat_max, z=13, layer="dem_png"):
    """Mosaic tiles covering the bbox. Returns (elev, transform) in EPSG:3857."""
    from rasterio.transform import from_origin
    x0, y1 = lonlat_to_tile(lon_min, lat_min, z)
    x1, y0 = lonlat_to_tile(lon_max, lat_max, z)
    tx = range(int(x0), int(x1) + 1)
    ty = range(int(y0), int(y1) + 1)
    out = np.full((256 * len(ty), 256 * len(tx)), np.nan)
    for a, ty_ in enumerate(ty):
        for b, tx_ in enumerate(tx):
            out[a * 256:(a + 1) * 256, b * 256:(b + 1) * 256] = fetch_tile(z, tx_, ty_, layer)
    size = 2 * math.pi * R_EARTH / 2 ** z
    res = size / 256
    west = -math.pi * R_EARTH + tx[0] * size
    north = math.pi * R_EARTH - ty[0] * size
    return out, from_origin(west, north, res, res)


def dem_on_grid(lon_min, lat_min, lon_max, lat_max, dst_crs: str, res: float, z=13,
                resampling="average"):
    """GSI DEM averaged onto a regular grid in a projected CRS.

    Returns (elev[ny, nx], transform, crs).  Cells with no valid source
    pixels (sea) are NaN.
    """
    import rasterio.warp as rw
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    src, src_tf = mosaic(lon_min, lat_min, lon_max, lat_max, z)
    xs, ys = rw.transform("EPSG:4326", dst_crs, [lon_min, lon_max, lon_min, lon_max],
                          [lat_min, lat_min, lat_max, lat_max])
    xmin = math.floor(min(xs) / res) * res
    xmax = math.ceil(max(xs) / res) * res
    ymin = math.floor(min(ys) / res) * res
    ymax = math.ceil(max(ys) / res) * res
    nx = int(round((xmax - xmin) / res))
    ny = int(round((ymax - ymin) / res))
    dst_tf = from_origin(xmin, ymax, res, res)
    dst = np.full((ny, nx), np.nan)
    rw.reproject(src, dst, src_transform=src_tf, src_crs="EPSG:3857", src_nodata=np.nan,
                 dst_transform=dst_tf, dst_crs=dst_crs, dst_nodata=np.nan,
                 resampling=getattr(Resampling, resampling))
    return dst, dst_tf, dst_crs
