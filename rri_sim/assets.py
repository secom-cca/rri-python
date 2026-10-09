"""Asset base quantities on the model grid for damage estimation.

Sources: e-Stat mesh statistics (households 250 m, employees 500 m by
industry), 国土数値情報 land-use fractions (paddy / field area).

Floor area (延床面積): the manual uses JACIC 100 m data (paid).  As a
documented substitute we estimate
    floor area = households x A_dwelling + employees x A_workplace
with A_dwelling from 住宅・土地統計調査 (2018, 1住宅当たり延べ面積, 都道府県)
and A_workplace = 30 m2/employee (assumption; sensitivity to be reported).
"""
from __future__ import annotations

import numpy as np

from .damage import AssetGrid
from .data import estat, landuse

# 平成30年住宅・土地統計調査: 1住宅当たり延べ面積 [m2] (prefectural values; check before use)
DWELLING_FLOOR_M2 = {"福岡": 84.4, "佐賀": 109.4, "大分": 98.5, "熊本": 102.6,
                     "福島": 111.3, "宮城": 94.4}
WORKPLACE_FLOOR_M2 = 30.0


def build_assets(domain, landuse_fraction, bbox, prefecture: str) -> AssetGrid:
    tf, shape, crs = domain.transform, domain.shape, domain.crs
    lon0, lat0, lon1, lat1 = bbox
    meshes = landuse.first_meshes(lon0, lat0, lon1, lat1)
    urban = landuse_fraction[landuse.CLASSES.index("urban")]
    weight = urban + 1e-3            # 宅地面積比率で按分 (manual 4.2(3)); small floor avoids 0/0
    hh = estat.households(meshes, tf, shape, crs, weight)
    emp = estat.employees(meshes, tf, shape, crs, weight)
    tot_emp = sum(emp.values())
    floor = hh * DWELLING_FLOOR_M2.get(prefecture, 95.0) + tot_emp * WORKPLACE_FLOOR_M2
    cell = domain.cell_area
    paddy = landuse_fraction[landuse.CLASSES.index("paddy")] * cell
    field = landuse_fraction[landuse.CLASSES.index("field")] * cell
    m = domain.mask
    z = lambda a: np.where(m, a, 0.0)  # noqa: E731
    return AssetGrid(floor_area_m2=z(floor), households=z(hh),
                     employees={k: z(v) for k, v in emp.items()},
                     paddy_area_m2=z(paddy), field_area_m2=z(field), prefecture=prefecture)
