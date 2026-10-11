"""MLIT 水文水質データベース (www1.river.go.jp) hourly stage / discharge.

KIND codes (時刻データ): 2 = hourly water level [m], 6 = hourly discharge [m3/s].
Hour labels 1..24 are the END of each hour (JST).  Flags: '*' provisional,
'$' missing, '#' closed, '-' not registered -> NaN.
"""
from __future__ import annotations

import re
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from . import RAW, http_get

BASE = "http://www1.river.go.jp"
KIND_LEVEL = 2
KIND_DISCHARGE = 6
KIND_RAIN = "rain"   # hourly rainfall via DspRainData.exe (KIND=2) [mm]


def _month_iter(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _fetch_month(station_id: str, kind: int, y: int, m: int) -> str:
    cache = RAW / "mlit_water" / f"{station_id}_k{kind}_{y}{m:02d}.dat"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    exe, k = ("DspRainData.exe", 2) if kind == KIND_RAIN else ("DspWaterData.exe", kind)
    url = (f"{BASE}/cgi-bin/{exe}?KIND={k}&ID={station_id}"
           f"&BGNDATE={y}{m:02d}01&ENDDATE={y}1231&KAWABOU=NO")
    hdr = {"User-Agent": "Mozilla/5.0"}
    html = http_get(url).content.decode("euc-jp", "replace")
    mm = re.search(r'(/dat/dload/download/[^"\']+\.dat)', html)
    if not mm:
        raise RuntimeError(f"no data file for {station_id} kind={kind} {y}-{m}")
    txt = http_get(BASE + mm.group(1)).content.decode(
        "shift_jis", "replace")
    cache.write_text(txt, encoding="utf-8")
    time.sleep(1.0)  # be polite to the server
    return txt


def parse_hourly(txt: str) -> pd.Series:
    rows = []
    for line in txt.splitlines():
        if not re.match(r"^\d{4}/\d{2}/\d{2},", line):
            continue
        parts = line.split(",")
        day = pd.Timestamp(parts[0])
        for h in range(24):
            v = parts[1 + 2 * h].strip()
            flag = parts[2 + 2 * h].strip() if 2 + 2 * h < len(parts) else ""
            try:
                val = float(v)
            except ValueError:
                val = np.nan
            if flag in ("$", "#", "-") or val <= -9999:
                val = np.nan
            rows.append((day + pd.Timedelta(hours=h + 1), val))
    s = pd.Series(dict(rows)).sort_index()
    s.index.name = "time_jst"
    return s


def hourly(station_id: str, kind: int, start: str, end: str) -> pd.Series:
    """Hourly series [start, end] (JST, inclusive)."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    parts = [parse_hourly(_fetch_month(station_id, kind, y, m))
             for y, m in _month_iter(t0.date(), t1.date())]
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    return s[(s.index >= t0) & (s.index <= t1)]


def search_stations(suikei: str, komoku: str = "01", max_pages: int = 10) -> list[tuple]:
    """(id, name) of stations in a water system.  komoku: '01' rain, '02' stage/discharge.
    suikei codes: 筑後川 '89094000', 阿武隈川 '82020000'."""
    out = {}
    for page in range(max_pages):
        url = (f"{BASE}/cgi-bin/SrchSite.exe?KOMOKU={komoku}&NAME=&SUIKEI={suikei}"
               f"&KASEN=&KEN=-00001&CITY=&PAGE={page}")
        html = http_get(url).content.decode(
            "euc-jp", "replace")
        found = re.findall(r'siteDetail1_(\d+)" HREF="#">([^<]+)<', html)
        new = {k: v for k, v in found if k not in out}
        if not new:
            break
        out.update(new)
        time.sleep(0.5)
    return sorted(out.items())


def site_info(station_id: str) -> dict:
    """Name, river and coordinates (decimal degrees, from DMS) of a station (cached)."""
    import json
    cache = RAW / "mlit_water" / f"site_{station_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"{BASE}/cgi-bin/SiteInfo.exe?ID={station_id}"
    html = http_get(url).content.decode(
        "euc-jp", "replace")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    m = re.search(r"北緯 (\d+)度(\d+)分(\d+)秒 東経 (\d+)度(\d+)分(\d+)秒", text)
    lat = lon = None
    if m:
        a = list(map(float, m.groups()))
        lat = a[0] + a[1] / 60 + a[2] / 3600
        lon = a[3] + a[4] / 60 + a[5] / 3600
    name = re.search(r"観測所名 (\S+)", text)
    river = re.search(r"河川名 (\S+)", text)
    info = {"id": station_id, "name": name.group(1) if name else None,
            "river": river.group(1) if river else None, "lat": lat, "lon": lon}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return info


def gauge_zero_tp(station_id: str) -> float | None:
    """Latest gauge datum (零点高) [m, T.P.] from the station page (cached)."""
    import json
    cache = RAW / "mlit_water" / f"zero_{station_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())["zero"]
    html = http_get(f"{BASE}/cgi-bin/SiteInfo.exe?ID={station_id}").content.decode("euc-jp", "replace")
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    m = re.search(r"最新の零点高 T\.P\. ?(-?[\d.]+)m", text)
    z = float(m.group(1)) if m else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"zero": z}))
    return z
