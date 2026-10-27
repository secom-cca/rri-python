"""Compare modelled flood damage with 水害統計 (表-3 市区町村別水害被害) by municipality.

    python scripts/validate_damage.py <basin> <res> <event> [year]
    e.g.  python scripts/validate_damage.py kusu_ogase 1000 kusu_2020

Model (baseline, no measures) aggregated over the basin cells of each
municipality, two ways: cell-uniform depth (old) and sub-grid building /
farmland elevations (rri_sim.subgrid, the GUI default).  水害統計 counts the
whole municipality and every flood of the year, so compare municipalities
mostly inside the basin (column 'in basin' = share of its area in the model).
Writes data/processed/validation_<basin>_<res>_<event>.csv.
"""
import csv
import sys

import numpy as np

from rri_sim import damage as D
from rri_sim import subgrid as SG
from rri_sim.cases import BASINS
from rri_sim.data import PROCESSED, admin, suigai
from rri_sim.gui.engine import DEPTH_MIN, FLOOR_HEIGHT, Engine


def main(key, res, ev, year=2020):
    s = Engine().session(key, float(res))
    b = s.run(ev)
    d, a = s.d, s.assets
    depth = b["_depth"]
    fr = np.frombuffer(__import__("base64").b64decode(b["frames_depth_cm"]), "<u2").reshape(b["nt"], -1) / 100
    dur = np.zeros(d.shape)
    dur.ravel()[s.flood_cells] = (fr >= DEPTH_MIN).sum(0) / 24.0
    floor = np.full(d.shape, FLOOR_HEIGHT)
    level = SG.water_level(depth, s.subgrid["all"])
    m_old = D.estimate_damage(depth, a, s.gs, dur, floor)
    m_new = D.estimate_damage(depth, a, s.gs, dur, floor, subgrid=(s.subgrid, level))
    hh = a.households
    du = SG.depths(depth, s.subgrid["urban"], level)
    k = du.shape[0]
    maps = {
        "old": dict(above=hh * (depth >= floor), below=hh * ((depth > 0) & (depth < floor)), area=(depth > 0) * d.cell_area,
                    general=m_old.total_map - sum(m_old.by_category.get(c, 0) for c in ("paddy", "field")),
                    crop=sum(m_old.by_category.get(c, 0) for c in ("paddy", "field"))),
        "new": dict(above=(hh[None] / k * (du >= floor[None])).sum(0), below=(hh[None] / k * ((du > 0) & (du < floor[None]))).sum(0),
                    area=(SG.depths(depth, s.subgrid["all"], level) >= DEPTH_MIN).mean(0) * d.cell_area,
                    general=m_new.total_map - sum(m_new.by_category.get(c, 0) for c in ("paddy", "field")),
                    crop=sum(m_new.by_category.get(c, 0) for c in ("paddy", "field"))),
    }
    codes, names, areas = admin.municipality_grid(d.transform, d.shape, d.crs, BASINS[key]["prefs"])
    stat = suigai.municipal_damage(year)
    rows = []
    for code in np.unique(codes[d.mask]):
        if code == 0:
            continue
        nm = names[code]
        sel = d.mask & (codes == code)
        r = {"pref": nm[0], "muni": nm[1], "in_basin": sel.sum() * d.cell_area / 1e6 / max(areas.get(nm, 1e-9), 1e-9),
             "hh_model": hh[sel].sum()}
        for meth, mp in maps.items():
            r[f"above_{meth}"] = mp["above"][sel].sum(); r[f"below_{meth}"] = mp["below"][sel].sum()
            r[f"area_{meth}_km2"] = mp["area"][sel].sum() / 1e6
            r[f"general_{meth}_oku"] = mp["general"][sel].sum() / 1e8; r[f"crop_{meth}_oku"] = mp["crop"][sel].sum() / 1e8
        st = stat.get(nm)
        if st:
            r.update(above_obs=st["hh_above"], below_obs=st["hh_below"], area_obs_km2=st["area_m2"] / 1e6,
                     general_obs_oku=st["general_kyen"] / 1e5, crop_obs_oku=st["crop_kyen"] / 1e5)
        rows.append(r)
    rows.sort(key=lambda r: -r["hh_model"])
    out = PROCESSED / f"validation_{key}_{int(float(res))}_{ev}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda x: list(rows[0]).index(x) if x in rows[0] else 99))
        w.writeheader(); w.writerows(rows)
    g = lambda r, k: r.get(k, float("nan"))  # noqa: E731
    print(f"## {key} {res} m {ev}（モデル: 一律 → セル内標高 / 水害統計 {year}年）")
    print("| 市町村 | 流域内 | 床上世帯 | 床下世帯 | 浸水面積 km² | 一般資産 億円 | 農作物 億円 |")
    print("|---|---|---|---|---|---|---|")
    tot = {}
    for r in rows:
        if r["hh_model"] < 50:
            continue
        cells = [f"{g(r, 'above_old'):,.0f} → {g(r, 'above_new'):,.0f} / {g(r, 'above_obs'):,.0f}",
                 f"{g(r, 'below_old'):,.0f} → {g(r, 'below_new'):,.0f} / {g(r, 'below_obs'):,.0f}",
                 f"{g(r, 'area_old_km2'):.1f} → {g(r, 'area_new_km2'):.1f} / {g(r, 'area_obs_km2'):.1f}",
                 f"{g(r, 'general_old_oku'):,.0f} → {g(r, 'general_new_oku'):,.0f} / {g(r, 'general_obs_oku'):,.0f}",
                 f"{g(r, 'crop_old_oku'):.1f} → {g(r, 'crop_new_oku'):.1f} / {g(r, 'crop_obs_oku'):.1f}"]
        print(f"| {r['pref']}{r['muni']} | {r['in_basin'] * 100:.0f}% | " + " | ".join(cells) + " |")
        if r["in_basin"] >= 0.5:
            for kk in ("above_old", "above_new", "above_obs", "below_old", "below_new", "below_obs", "area_old_km2", "area_new_km2", "area_obs_km2",
                       "general_old_oku", "general_new_oku", "general_obs_oku", "crop_old_oku", "crop_new_oku", "crop_obs_oku"):
                tot[kk] = tot.get(kk, 0.0) + np.nan_to_num(g(r, kk))
    if tot:
        print(f"| **流域内50%以上の市町村 計** | | {tot['above_old']:,.0f} → {tot['above_new']:,.0f} / {tot['above_obs']:,.0f} | "
              f"{tot['below_old']:,.0f} → {tot['below_new']:,.0f} / {tot['below_obs']:,.0f} | "
              f"{tot['area_old_km2']:.1f} → {tot['area_new_km2']:.1f} / {tot['area_obs_km2']:.1f} | "
              f"{tot['general_old_oku']:,.0f} → {tot['general_new_oku']:,.0f} / {tot['general_obs_oku']:,.0f} | "
              f"{tot['crop_old_oku']:.1f} → {tot['crop_new_oku']:.1f} / {tot['crop_obs_oku']:.1f} |")
    print(f"\n(csv: {out})")


if __name__ == "__main__":
    main(*sys.argv[1:4], *(int(x) for x in sys.argv[4:5]))
