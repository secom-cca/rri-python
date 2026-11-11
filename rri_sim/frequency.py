"""Rainfall frequency of a basin for the annual expected damage (治水経済調査マニュアル: 年平均被害（軽減）期待額).

Annual maximum daily precipitation of the JMA AMeDAS stations inside the basin (etrn 年ごとの値, 日最大降水量), Gumbel
distribution per station fitted by L-moments (Hosking 1990):
    x_T = xi + alpha y_T,  y_T = -ln(-ln(1 - 1/T)),  alpha = l2 / ln 2,  xi = l1 - 0.5772 alpha
The event scale of return period T relative to the representative event (annual exceedance probability p_rep, 1/100) is
the median over the stations of x_T / x_{1/p_rep}.

Assumptions (stated where used, docs/14):
    - the representative event (2019年台風19号) is the 1/p_rep-year event of the basin (as before)
    - the event of return period T is the representative event's rainfall scaled by r(T) = x_T / x_rep (引き伸ばし; the
      spatial / temporal pattern is that of the representative event)
    - point daily maxima give the ratio r(T) (areal reduction and the duration of the basin's design rainfall ignored)
    - climate change multiplies the rainfall of every return period by the same rain factor
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from .data import RAW, http_get
from .data import jma

EULER = 0.5772156649
ROW = re.compile(r"<tr class=\"mtx\" style=\"text-align:right;\"><td[^>]*>(?:<div[^>]*>)?(?:<a[^>]*>)?(\d{4})(?:</a>)?(?:</div>)?</td>(.*?)</tr>", re.S)
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def annual_max_daily(st):
    """{year: annual maximum daily precipitation [mm]} of a station (quality marks ')' kept, ']' and missing dropped)."""
    cache = RAW / "jma" / f"annual_{st.kind}_{st.prec_no}_{st.block}.html"
    if cache.exists():
        txt = cache.read_text(encoding="utf-8")
    else:
        url = f"{jma.BASE}/view/annually_{st.kind}.php?prec_no={st.prec_no}&block_no={st.block}&year=&month=&day=&view="
        txt = http_get(url, timeout=60).text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(txt, encoding="utf-8")
    out = {}
    for y, body in ROW.findall(txt):
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in CELL.findall(body)]
        if len(cells) < 2:
            continue
        v = cells[1]
        if not v or "]" in v or v in ("×", "///", "#", "--"):
            continue
        try:
            out[int(y)] = float(v.replace(")", "").replace(" ", ""))
        except ValueError:
            continue
    return out


def gumbel_lmom(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    b0 = x.mean()
    b1 = np.sum((np.arange(n) / (n - 1)) * x) / n
    l1, l2 = b0, 2 * b1 - b0
    alpha = l2 / math.log(2)
    return l1 - EULER * alpha, alpha


def quantile(xi, alpha, T):
    return xi + alpha * -np.log(-np.log(1 - 1 / np.asarray(T, float)))


def basin_ratios(mask, transform, crs, prec_nos, bbox, T, T_rep=100.0, min_years=30, log=print):
    """Median over the basin's stations of x_T / x_{T_rep} (Gumbel, L-moments) and the station table."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    rows = []
    for st in jma.stations_in_bbox(prec_nos, *bbox):
        if not st.has_rain:
            continue
        x, y = tr.transform(st.lon, st.lat)
        c, r = int((x - transform.c) / transform.a), int((y - transform.f) / transform.e)
        if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]):
            continue
        am = annual_max_daily(st)
        if len(am) < min_years:
            continue
        xi, al = gumbel_lmom(list(am.values()))
        rows.append({"name": st.name, "n_years": len(am), "years": [min(am), max(am)], "xi": xi, "alpha": al,
                     "ratio": (quantile(xi, al, T) / quantile(xi, al, T_rep)).tolist(), "x_rep": float(quantile(xi, al, T_rep))})
    if not rows:
        raise RuntimeError("no station with enough annual maxima inside the basin")
    ratio = np.median(np.array([r["ratio"] for r in rows]), 0)
    log(f"確率雨量の比（{len(rows)} 地点の中央値, 1/{T_rep:.0f} 年 = 1）: " + ", ".join(f"1/{t:g} {v:.2f}" for t, v in zip(T, ratio)))
    return ratio, rows


def ead(damage_T, T):
    """Annual expected damage from damages at return periods T (ascending; damage 0 assumed at the smallest T and
    nothing beyond the largest T counted): sum over intervals of the mean damage x the interval probability."""
    p = 1.0 / np.asarray(T, float)
    D = np.asarray(damage_T, float)
    return ((D[..., :-1] + D[..., 1:]) / 2 * (p[:-1] - p[1:])).sum(-1)
