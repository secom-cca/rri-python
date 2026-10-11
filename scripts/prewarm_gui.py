"""Pre-compute GUI caches (domain, assets, indicator data, spin-up, baseline) for a basin/event.

python scripts/prewarm_gui.py chikugo_whole chikugo_2020
"""
import sys
import time

from rri_sim.gui.engine import BasinSession

basin, event = sys.argv[1], sys.argv[2]
t = time.time()
s = BasinSession(basin)
print(f"session {time.time() - t:.0f}s: cells {int(s.d.mask.sum())}, households {s.assets.households.sum():.0f}, "
      f"population {s.pop[s.d.mask].sum():.0f}, facilities { {k: len(v) for k, v in s.fac.items()} }", flush=True)
t = time.time()
r = s.run(event, progress=lambda f, m: None)
print(f"baseline {time.time() - t:.0f}s (sim {r['wall_s']:.0f}s): damage {r['damage_total_oku']:.0f} 億円, "
      f"inundated {r['inundated_km2']:.1f} km2, livability {r['livability']['score']:.1f}, "
      f"ecology {r['ecology']['score']:.1f}, dams { {k: round(v['max_storage_mcm'], 1) for k, v in r['dams'].items()} }, "
      f"mass err {r['mass_rel_err']:.1e}", flush=True)
