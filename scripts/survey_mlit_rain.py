"""Survey MLIT rain gauges: coordinates and data completeness for an event."""
import json, sys
import numpy as np
from rri_sim.data import mlit_water as M, PROCESSED

suikei, start, end, tag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
out = []
for sid, name in M.search_stations(suikei, "01"):
    try:
        s = M.hourly(sid, M.KIND_RAIN, start, end)
        info = M.site_info(sid)
    except Exception as e:  # noqa
        print(sid, name, "ERR", e); continue
    n_ok = int(np.isfinite(s.values).sum())
    print(f"{sid} {name:8s} lat={info['lat']} lon={info['lon']} valid={n_ok}/{len(s)} total={np.nansum(s.values):.0f}")
    out.append({**info, "valid": n_ok, "n": len(s), "total_mm": float(np.nansum(s.values))})
PROCESSED.mkdir(parents=True, exist_ok=True)
json.dump(out, open(PROCESSED / f"mlit_rain_stations_{tag}.json", "w"), ensure_ascii=False, indent=1)
