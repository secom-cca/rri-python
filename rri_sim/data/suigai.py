"""国土交通省 水害統計調査 (e-Stat) — municipal flood damage for validation.

表-3 市区町村別水害被害 (one year, all floods of the year; e-Stat file download):
  2020: statInfId 000032217685, 2021: 000040171170
Columns used (row 8 on): A 都道府県, B 市区町村, C/D/E 水害区域面積 [m2] 宅地・その他/農地/計,
F 床下浸水 [棟], J 床上浸水 計 [棟], K 半壊, L 全壊流失, N 床下世帯, O 床上世帯,
T 一般資産等被害額 [千円] (営業停止損失を含む), U 農作物, V 合計.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile

from . import RAW, http_get

URL = "https://www.e-stat.go.jp/stat-search/file-download?statInfId={sid}&fileKind=0"
TABLE3 = {2020: "000032217685", 2021: "000040171170"}
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _rows(path):
    """Rows of the first sheet of an .xlsx as {column letter: value} (no openpyxl)."""
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            ss.append("".join(t.text or "" for t in si.iter("{%s}t" % _NS["m"])))
    sheets = sorted((n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n)),
                    key=lambda n: int(re.findall(r"\d+", n)[0]))
    for r in ET.fromstring(z.read(sheets[0])).iter("{%s}row" % _NS["m"]):
        row = {}
        for c in r.findall("m:c", _NS):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", _NS)
            if v is not None:
                row[col] = ss[int(v.text)] if c.get("t") == "s" else v.text
        yield int(r.get("r")), row


def municipal_damage(year: int = 2020):
    """{(都道府県, 市区町村): dict} from 表-3 of the year."""
    path = RAW / "suigai" / f"table3_{year}.xlsx"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = http_get(URL.format(sid=TABLE3[year]), timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    f = lambda row, c: float(row.get(c) or 0)  # noqa: E731
    out = {}
    for i, row in _rows(path):
        if i < 8 or not row.get("A") or not row.get("B"):
            continue
        out[(row["A"].strip(), row["B"].strip())] = {
            "area_m2": f(row, "E"), "area_urban_m2": f(row, "C"), "area_farm_m2": f(row, "D"),
            "bldg_below": f(row, "F"), "bldg_above": f(row, "J"), "bldg_half": f(row, "K"), "bldg_full": f(row, "L"),
            "hh_below": f(row, "N"), "hh_above": f(row, "O"),
            "general_kyen": f(row, "T"), "crop_kyen": f(row, "U"), "total_kyen": f(row, "V")}
    return out
