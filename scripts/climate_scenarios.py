"""Run an event under several climate scenarios (no measures) and tabulate impacts.

python scripts/climate_scenarios.py kusu_ogase kusu_2020 [gauge]
Outputs outputs/climate/<event>/table.md, scenarios.json, hydrograph.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from rri_sim import viz  # noqa: F401  (sets matplotlib backend / fonts)
from rri_sim.gui.engine import BasinSession

import matplotlib.pyplot as plt  # noqa: E402

basin, event = sys.argv[1], sys.argv[2]
OUT = Path("outputs/climate") / event
OUT.mkdir(parents=True, exist_ok=True)
SCEN = [("現在気候", {"preset": "current"}),
        ("×1.1", {"preset": "custom", "rain_factor": 1.1}),
        ("×1.2", {"preset": "custom", "rain_factor": 1.2}),
        ("2℃上昇", {"preset": "2C"}),
        ("4℃上昇", {"preset": "4C"}),
        ("4℃上昇・強雨強調γ=1.2", {"preset": "4C", "intensity_gamma": 1.2})]

s = BasinSession(basin)
gauge = sys.argv[3] if len(sys.argv) > 3 else list(s.meta["gauge_index"])[-1]
rows, series = [], {}
for name, cp in SCEN:
    sc = s.climate({"climate": cp})
    if name.startswith("4℃") and "intensity_gamma" in cp:
        cp = {**cp, "rain_factor": sc.rain_factor, "sea_level_rise_m": sc.sea_level_rise_m}
    t = time.time()
    r = s.run(event, [], {"climate": cp})
    peaks = {g: float(np.nanmax(v["sim"])) for g, v in r["gauges"].items()}
    rows.append({"scenario": name, "rain_factor": r["climate"]["rain_factor"],
                 "gamma": r["climate"]["intensity_gamma"], "slr_m": r["climate"]["sea_level_rise_m"],
                 "rain_mm": float(np.sum(r["basin_rain"])), "damage_oku": r["damage_total_oku"],
                 "inundated_km2": r["inundated_km2"], "hh_above_floor": r["households_above_floor"],
                 "peaks": peaks, "livability": r["livability"]["score"], "ecology": r["ecology"]["score"],
                 "wall_s": time.time() - t})
    series[name] = (r["times"], r["gauges"][gauge]["sim"], r["gauges"][gauge]["obs"])
    print(f"{name:20s} rain {rows[-1]['rain_mm']:6.0f} mm  damage {r['damage_total_oku']:8.1f} 億円  "
          f"inund {r['inundated_km2']:6.1f} km2  peak[{gauge}] {peaks[gauge]:7.0f}  ({time.time() - t:.0f}s)", flush=True)

base = rows[0]
gs = list(base["peaks"])
lines = ["| シナリオ | 降雨倍率 | γ | 海面上昇 [m] | 評価期間雨量 [mm] | 被害額 [億円] | 浸水面積 [km²] | 床上世帯 | "
         + " | ".join(f"ピーク {g} [m³/s]" for g in gs) + " |",
         "|" + "---|" * (8 + len(gs))]
for r in rows:
    d = lambda k: f"{r[k]:.1f}" + (f" ({(r[k] / base[k] - 1) * 100:+.0f}%)" if r is not base and base[k] else "")  # noqa: E731
    pk = [f"{r['peaks'][g]:.0f}" + (f" ({(r['peaks'][g] / base['peaks'][g] - 1) * 100:+.0f}%)" if r is not base else "")
          for g in gs]
    lines.append(f"| {r['scenario']} | {r['rain_factor']:.2f} | {r['gamma']:.2f} | {r['slr_m']:.2f} | {r['rain_mm']:.0f} | "
                 f"{d('damage_oku')} | {d('inundated_km2')} | {r['hh_above_floor']:.0f} | " + " | ".join(pk) + " |")
(OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
json.dump(rows, open(OUT / "scenarios.json", "w"), ensure_ascii=False, indent=1)

fig, ax = plt.subplots(figsize=(9, 4.2))
times, _, obs = series["現在気候"]
x = np.arange(len(times))
ax.plot(x, [np.nan if v is None else v for v in obs], "k.", ms=3, label="観測")
for (name, (_, q, _)), c in zip(series.items(), ["#2563eb", "#93c5fd", "#60a5fa", "#f59e0b", "#dc2626", "#7f1d1d"]):
    ax.plot(x, q, color=c, lw=1.4, label=name)
ax.set_xticks(x[::24]); ax.set_xticklabels([t[5:13].replace("T", " ") for t in times[::24]], rotation=30)
ax.set_ylabel("Q [m³/s]"); ax.set_title(f"{event} {gauge}: 気候変動シナリオ（施策なし）"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "hydrograph.png", dpi=150)
print("\n".join(lines))
