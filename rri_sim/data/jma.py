"""JMA 過去の気象データ (etrn) hourly precipitation at AMeDAS / observatories.

Station list : etrn/select/prefecture.php?prec_no=NN (viewPoint(...) entries,
               lat/lon given in degrees + decimal minutes)
Hourly data  : etrn/view/hourly_a1.php (AMeDAS, type 'a'; precip = column 1)
               etrn/view/hourly_s1.php (observatory, type 's'; precip = column 3)
Hour label h is the accumulation over (h-1, h] JST.  '--' = no precipitation
(0), '×'/'///'/'#' = missing (NaN).  Quality marks ')' ']' are stripped and
the value used (flagged in the docs as quasi-normal/incomplete).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from . import RAW, http_get

BASE = "https://www.data.jma.go.jp/stats/etrn"
VP = re.compile(r"viewPoint\('([as])','(\d+)','([^']*)','[^']*','(\d+)','([\d.]+)',"
                r"'(\d+)','([\d.]+)','([\d.\-]+)','(\d)'")


@dataclass
class Station:
    kind: str       # 'a' AMeDAS, 's' observatory
    block: str
    name: str
    lat: float
    lon: float
    elev: float
    prec_no: int
    has_rain: bool


def _get(url, cache):
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    txt = http_get(url).content.decode("utf-8", "replace")
    cache.write_text(txt, encoding="utf-8")
    time.sleep(0.5)
    return txt


def stations(prec_no: int) -> list[Station]:
    url = f"{BASE}/select/prefecture.php?prec_no={prec_no}&block_no=&year=&month=&day=&view="
    html = _get(url, RAW / "jma" / f"pref_{prec_no}.html")
    out = {}
    for m in VP.finditer(html):
        kind, blk, name, latd, latm, lond, lonm, elev, f_pre = m.groups()
        out[blk] = Station(kind, blk, name, int(latd) + float(latm) / 60,
                           int(lond) + float(lonm) / 60, float(elev), prec_no, f_pre == "1")
    return list(out.values())


def _parse_value(s: str) -> float:
    s = s.strip()
    if s in ("--",):
        return 0.0
    s2 = re.sub(r"[)\]\s]", "", s)
    try:
        return float(s2)
    except ValueError:
        return np.nan


def hourly_precip_day(st: Station, day: pd.Timestamp) -> pd.Series:
    page = "hourly_a1.php" if st.kind == "a" else "hourly_s1.php"
    col = 1 if st.kind == "a" else 3
    url = (f"{BASE}/view/{page}?prec_no={st.prec_no}&block_no={st.block}"
           f"&year={day.year}&month={day.month}&day={day.day}&view=")
    cache = RAW / "jma" / "hourly" / st.block / f"{day:%Y%m%d}.html"
    html = _get(url, cache)
    vals = {}
    for row in re.findall(r'<tr class="mtx"[^>]*>(.*?)</tr>', html, re.S):
        cells = [re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if not cells or not cells[0].strip().isdigit():
            continue
        h = int(cells[0])
        vals[day + pd.Timedelta(hours=h)] = _parse_value(cells[col]) if len(cells) > col else np.nan
    return pd.Series(vals, dtype=float)


def hourly_precip(st: Station, start: str, end: str) -> pd.Series:
    """Hourly precipitation [mm] for hours ending in [start, end] (JST)."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range((t0 - pd.Timedelta(hours=1)).normalize(), t1.normalize(), freq="D")
    s = pd.concat([hourly_precip_day(st, d) for d in days]).sort_index()
    return s[(s.index >= t0) & (s.index <= t1)]


def stations_in_bbox(prec_nos, lon_min, lat_min, lon_max, lat_max, buffer_deg=0.15):
    out = []
    for p in prec_nos:
        for s in stations(p):
            if (s.has_rain and lon_min - buffer_deg <= s.lon <= lon_max + buffer_deg
                    and lat_min - buffer_deg <= s.lat <= lat_max + buffer_deg):
                out.append(s)
    return out
