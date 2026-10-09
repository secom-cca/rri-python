"""Direct flood damage (治水経済調査マニュアル) from a simulation's max depth.

python scripts/damage_from_result.py <result.pkl> <basin_key> <prefecture>
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_event import build  # noqa: E402

from rri_sim import damage as D  # noqa: E402
from rri_sim.assets import build_assets  # noqa: E402
from rri_sim.basin import floodplain_inundation  # noqa: E402


def main(pkl, basin_key, pref):
    r = pickle.load(open(pkl, "rb"))
    d, meta, b = build(basin_key)
    assets = build_assets(d, meta["landuse_fraction"], b["cfg"].bbox, pref)
    gs = D.ground_slope_from_dem(d.z, d.dx, d.dy)
    depth = floodplain_inundation(r["max_depth"], meta["hand"], depth_min=0.1, hand_max=10.0)
    res = D.estimate_damage(depth, assets, ground_slope=gs)
    hh_flooded = float(assets.households[depth >= 0.45].sum())
    hh_below = float(assets.households[(depth > 0) & (depth < 0.45)].sum())
    out = {"total_oku_yen": res.total / 1e8,
           "by_category_oku_yen": {k: v / 1e8 for k, v in res.totals.items()},
           "households_above_floor": hh_flooded, "households_below_floor": hh_below,
           "inundated_area_km2": float((depth > 0).sum() * d.cell_area / 1e6),
           "definition": "max depth >= 0.1 m on cells with HAND <= 10 m"}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open(Path(pkl).parent / "damage.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
