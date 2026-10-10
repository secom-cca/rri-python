"""Reproduce the intervention example table of docs/05_gui.md through the GUI API.

Requires the server:  python -m rri_sim.gui --no-browser
"""
import time

import numpy as np
import requests

B = "http://127.0.0.1:8765"
BASIN, EVENT = "kusu_ogase", "kusu_2020"


def b64(s, dt):
    import base64
    return np.frombuffer(base64.b64decode(s), dt)


def run(edits, params=None):
    j = requests.post(f"{B}/api/run", json={"basin": BASIN, "event": EVENT, "edits": edits,
                                            "params": params or {}}).json()["job"]
    while True:
        r = requests.get(f"{B}/api/job/{j}").json()
        if r["status"] == "done":
            return r["result"]
        if r["status"] == "error":
            raise RuntimeError(r["message"])
        time.sleep(1)


st = requests.get(f"{B}/api/basin/{BASIN}").json()
nx, cn = st["nx"], st["class_names"]
lc, hand = b64(st["landclass"], "u1"), b64(st["hand"], "<f4")
ri, rj, area = (np.array(st["river"][k]) for k in ("ri", "rj", "area"))
paddy = np.flatnonzero(lc == cn.index("paddy"))
farm = np.flatnonzero(np.isin(lc, [cn.index("paddy"), cn.index("field")]))
low_urban = np.flatnonzero((lc == cn.index("urban")) & (hand >= 0) & (hand < 5))
main_riv = (ri * nx + rj)[area > 200]
near = set((ri * nx + rj)[area > 300].tolist())
basin_cells = [int(c) for c in np.flatnonzero((lc == cn.index("paddy")) & (hand >= 0) & (hand < 3))
               if any((int(c) + o) in near for o in (1, -1, nx, -nx))][:25]
cases = {
    "現況": [],
    "森林化（水田・畑）": [{"type": "forest", "cells": farm.tolist()}],
    "田んぼダム（全水田, 15 cm）": [{"type": "paddy_dam", "cells": paddy.tolist()}],
    "移転（低地の建物用地）": [{"type": "relocate", "cells": low_urban.tolist()}],
    "嵩上げ +1 m（同, 床高）": [{"type": "raise", "cells": low_urban.tolist(), "height": 1.0}],
    "堤防 3 m（上流域200 km²以上）": [{"type": "levee", "cells": main_riv.tolist(), "height": 3.0}],
    "遊水地（掘削5 m, 越流堤2 m）": [{"type": "retention", "cells": basin_cells, "depth": 5.0, "inlet": 2.0}],
}
print("| 施策 | 適用 | 被害額 [億円] | 浸水面積 [km²] | 床上世帯 | ピーク流量 小ヶ瀬 [m³/s] | 計算時間 [s] |")
print("|---|---|---|---|---|---|---|")
for name, ed in cases.items():
    r = run(ed)
    print(f"| {name} | {'; '.join(r['edits_log']) or '—'} | {r['damage_total_oku']:.1f} | {r['inundated_km2']:.1f} | "
          f"{r['households_above_floor']:.0f} | {max(r['gauges']['ogase']['sim']):.0f} | {r['wall_s']:.0f} |", flush=True)
