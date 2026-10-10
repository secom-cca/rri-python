"""Direct flood damage after MLIT 治水経済調査マニュアル（案）.

Damage rates: 治水経済調査マニュアル（案）令和7年7月, 表-4.2 ～ 表-4.6
Unit asset values: 各種資産評価単価及びデフレーター 令和7年6月改正 (令和5年評価額)

    damage = Σ_asset  asset value(cell) × rate(depth[, ground slope, duration])

Depth convention (documented interpretation, see docs/damage.md):
  The manual classifies 床下 / 床上 and states that, for residential assets,
  a mesh depth >= 45 cm is 床上 (floor height 0.45 m).  The 床上 bins
  (<50, 50-99, 100-199, 200-299, >=300 cm) are applied to the depth above the
  floor, d_f = h - floor_height.  Cars use depth above ground (表-4.3.2).
  Crops use depth above ground and inundation duration (表-4.6).
  Sediment (土砂堆積) rates are tabulated but not applied (no sediment model).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------ rate tables
# columns: 床下, 床上 <0.5, 0.5-0.99, 1.0-1.99, 2.0-2.99, >=3.0 m (above floor)
HOUSE_RATES = {  # 表-4.2, by ground slope group
    "A": [0.047, 0.189, 0.253, 0.406, 0.592, 0.800],   # < 1/1000
    "B": [0.058, 0.219, 0.301, 0.468, 0.657, 0.843],   # 1/1000 - 1/500
    "C": [0.064, 0.235, 0.325, 0.499, 0.690, 0.865],   # >= 1/500
}
HOUSEHOLD_GOODS_RATES = [0.037, 0.308, 0.533, 0.701, 0.948, 0.977]   # 表-4.3.1
BUSINESS_DEPRECIABLE_RATES = [0.064, 0.296, 0.573, 0.801, 0.920, 0.940]  # 表-4.4 償却
BUSINESS_INVENTORY_RATES = [0.053, 0.282, 0.440, 0.814, 0.946, 0.975]    # 表-4.4 在庫
FARM_DEPRECIABLE_RATES = [0.000, 0.113, 0.327, 0.483, 0.828, 1.000]      # 表-4.5 償却
FARM_INVENTORY_RATES = [0.000, 0.223, 0.584, 0.618, 0.792, 0.942]        # 表-4.5 在庫
SEDIMENT_RATES = {  # 土砂堆積（床上） <50cm, >=50cm — not applied (no sediment model)
    "house": (0.43, 0.785), "household_goods": (0.50, 0.845),
    "business_depreciable": (0.54, 0.815), "business_inventory": (0.48, 0.780),
    "farm_depreciable": (0.370, 0.725), "farm_inventory": (0.580, 0.845),
}
# 表-4.3.2 car, depth above ground: <0.3, 0.3-0.49, 0.5-0.69, >=0.7 m
CAR_EDGES = np.array([0.3, 0.5, 0.7])
CAR_RATES = np.array([0.0, 0.1, 0.5, 1.0])
# 表-4.6 crops [%]: rows depth (<0.5, 0.5-0.99, >=1.0 m), cols days (1-2,3-4,5-6,>=7)
CROP_RATES = {
    "paddy": np.array([[21, 30, 36, 50], [24, 44, 50, 71], [37, 54, 64, 74]]) / 100.0,
    "field": np.array([[27, 42, 54, 67], [35, 48, 67, 74], [51, 67, 81, 91]]) / 100.0,  # 畑平均
}

# ------------------------------------------------------------- unit values R5
# 第1表 家屋1m2当たり評価額 [千円/m2] (令和5年評価額, 令和7年6月改正)
HOUSE_VALUE_PER_M2 = {
    "北海道": 300.2, "青森": 241.8, "岩手": 230.9, "宮城": 242.7, "秋田": 236.0,
    "山形": 240.1, "福島": 247.1, "茨城": 218.0, "栃木": 236.1, "群馬": 243.9,
    "埼玉": 243.1, "千葉": 244.5, "東京": 369.9, "神奈川": 273.6, "新潟": 240.6,
    "富山": 240.8, "石川": 276.6, "福井": 260.0, "山梨": 286.0, "長野": 270.1,
    "岐阜": 232.6, "静岡": 261.4, "愛知": 240.4, "三重": 242.5, "滋賀": 218.3,
    "京都": 249.3, "大阪": 275.0, "兵庫": 250.5, "奈良": 252.6, "和歌山": 231.6,
    "鳥取": 249.2, "島根": 250.8, "岡山": 258.8, "広島": 256.2, "山口": 262.4,
    "徳島": 265.8, "香川": 248.4, "愛媛": 234.0, "高知": 268.0, "福岡": 230.8,
    "佐賀": 220.9, "長崎": 242.6, "熊本": 245.5, "大分": 240.1, "宮崎": 231.9,
    "鹿児島": 267.8, "沖縄": 283.0,
}
# 第2表 [千円/世帯]
HOUSEHOLD_GOODS_VALUE = 9368.0
CAR_VALUE = 3502.0
# 第3表 [千円/人] (償却, 在庫) by 大分類
BUSINESS_VALUE_PER_EMPLOYEE = {
    "C": (23438, 4319), "D": (1956, 3175), "E": (5943, 6157), "F": (136370, 6550),
    "G": (4674, 1085), "H": (6920, 1690), "I": (2585, 2307), "J": (1208, 287),
    "K": (21446, 12111), "L": (2283, 593), "M": (1625, 120), "N": (3297, 236),
    "O": (979, 152), "P": (1281, 71), "Q": (1208, 287), "R": (1208, 287), "S": (1208, 287),
}
# 第4表 [千円/戸]
FARM_VALUE = (3110.0, 981.0)
# 第5表 水稲平年収量 [kg/10a] (selected prefectures; extend as needed)
RICE_YIELD_KG_PER_10A = {"福岡": 456, "佐賀": 487, "大分": 475, "熊本": 479,
                         "福島": 532, "宮城": 511}
# 第6表 [千円/t]
RICE_PRICE = 187.0

DEPTH_BIN_EDGES_ABOVE_FLOOR = np.array([0.5, 1.0, 2.0, 3.0])


def general_rate_index(depth, floor_height=0.45):
    """Index into 6-column general-asset tables; -1 = not inundated.
    floor_height may be a scalar or a per-cell array (嵩上げ)."""
    depth = np.asarray(depth, float)
    fh = np.broadcast_to(np.asarray(floor_height, float), depth.shape)
    idx = np.full(depth.shape, -1, dtype=np.int64)
    wet = np.isfinite(depth) & (depth > 0.0)
    below = wet & (depth < fh)
    above = wet & ~below
    idx[below] = 0
    df = np.round(depth[above] - fh[above], 6)
    idx[above] = 1 + np.searchsorted(DEPTH_BIN_EDGES_ABOVE_FLOOR, df, side="right")
    return idx


def lookup(rates, idx):
    rates = np.asarray(rates, float)
    out = np.zeros(idx.shape)
    m = idx >= 0
    out[m] = rates[idx[m]]
    return out


def slope_group(ground_slope):
    """Ground slope -> 'A' (<1/1000), 'B' (1/1000-1/500), 'C' (>=1/500)."""
    s = np.asarray(ground_slope, float)
    g = np.full(s.shape, "A", dtype="<U1")
    g[s >= 1 / 1000] = "B"
    g[s >= 1 / 500] = "C"
    return g


def house_rate(depth, ground_slope, floor_height=0.45):
    idx = general_rate_index(depth, floor_height)
    grp = slope_group(ground_slope)
    out = np.zeros(idx.shape)
    for k, rates in HOUSE_RATES.items():
        m = grp == k
        out[m] = lookup(rates, idx[m])
    return out


def car_rate(depth):
    d = np.nan_to_num(np.asarray(depth, float), nan=0.0)
    return CAR_RATES[np.searchsorted(CAR_EDGES, d, side="right")] * (d > 0)


def crop_rate(depth, duration_days, crop="paddy"):
    d = np.nan_to_num(np.asarray(depth, float), nan=0.0)
    days = np.asarray(duration_days, float)
    di = np.searchsorted([0.5, 1.0], d, side="right")
    ti = np.clip(np.searchsorted([2.0, 4.0, 6.0], np.ceil(days), side="left"), 0, 3)
    r = CROP_RATES[crop][di, ti]
    return np.where(d > 0, r, 0.0)


def floor_correction(mean_floors):
    """γ = 1 (f < 3), 2/f (f >= 3)  (manual 4.3.1)."""
    f = np.asarray(mean_floors, float)
    return np.where(f < 3, 1.0, 2.0 / np.maximum(f, 1e-9))


@dataclass
class AssetGrid:
    """Basic quantities per computational cell (same shape as the model grid)."""
    floor_area_m2: np.ndarray | None = None        # 延床面積
    households: np.ndarray | None = None           # 一般世帯数
    employees: dict = field(default_factory=dict)  # {大分類 code: array}
    farm_households: np.ndarray | None = None
    paddy_area_m2: np.ndarray | None = None
    field_area_m2: np.ndarray | None = None
    mean_floors: np.ndarray | None = None
    prefecture: str = "福岡"
    field_value_kyen_per_a: float | None = None    # c [千円/a]; regional, user-supplied


@dataclass
class DamageResult:
    by_category: dict          # name -> array [yen]
    total_map: np.ndarray      # [yen]

    @property
    def totals(self) -> dict:
        return {k: float(np.nansum(v)) for k, v in self.by_category.items()}

    @property
    def total(self) -> float:
        return float(np.nansum(self.total_map))


def estimate_damage(max_depth, assets: AssetGrid, ground_slope=None,
                    duration_days=None, floor_height=0.45) -> DamageResult:
    """Direct damage [yen] per cell from the maximum inundation depth."""
    h = np.nan_to_num(np.asarray(max_depth, float), nan=0.0)
    shape = h.shape
    zeros = np.zeros(shape)
    gam = floor_correction(assets.mean_floors) if assets.mean_floors is not None else 1.0
    gs = zeros if ground_slope is None else ground_slope
    idx = general_rate_index(h, floor_height)
    cat = {}
    if assets.floor_area_m2 is not None:
        v = assets.floor_area_m2 * HOUSE_VALUE_PER_M2[assets.prefecture] * 1e3 * gam
        cat["house"] = v * house_rate(h, gs, floor_height)
    if assets.households is not None:
        v = assets.households * HOUSEHOLD_GOODS_VALUE * 1e3 * gam
        cat["household_goods"] = v * lookup(HOUSEHOLD_GOODS_RATES, idx)
        cat["car"] = assets.households * CAR_VALUE * 1e3 * car_rate(h)
    if assets.employees:
        dep = np.zeros(shape)
        inv = np.zeros(shape)
        for code, n in assets.employees.items():
            vd, vi = BUSINESS_VALUE_PER_EMPLOYEE[code]
            dep += n * vd * 1e3
            inv += n * vi * 1e3
        cat["business_depreciable"] = dep * gam * lookup(BUSINESS_DEPRECIABLE_RATES, idx)
        cat["business_inventory"] = inv * gam * lookup(BUSINESS_INVENTORY_RATES, idx)
    if assets.farm_households is not None:
        cat["farm_depreciable"] = (assets.farm_households * FARM_VALUE[0] * 1e3
                                   * lookup(FARM_DEPRECIABLE_RATES, idx))
        cat["farm_inventory"] = (assets.farm_households * FARM_VALUE[1] * 1e3
                                 * lookup(FARM_INVENTORY_RATES, idx))
    dur = np.ones(shape) if duration_days is None else duration_days
    if assets.paddy_area_m2 is not None:
        y = RICE_YIELD_KG_PER_10A.get(assets.prefecture)
        if y is not None:
            # area[m2]/1000 -> 10a units; kg -> t; 千円/t -> yen
            v = assets.paddy_area_m2 / 1000.0 * y / 1000.0 * RICE_PRICE * 1e3
            cat["paddy"] = v * crop_rate(h, dur, "paddy")
    if assets.field_area_m2 is not None and assets.field_value_kyen_per_a is not None:
        v = assets.field_area_m2 / 100.0 * assets.field_value_kyen_per_a * 1e3
        cat["field"] = v * crop_rate(h, dur, "field")
    total = sum(cat.values()) if cat else zeros
    return DamageResult(by_category=cat, total_map=np.asarray(total, float))


def ground_slope_from_dem(z, dx, dy, smooth_cells=3):
    """Mean ground slope magnitude, smoothed over a window (manual 4.3.1(1)2:
    mechanically computed mesh slopes should be averaged over a wider area)."""
    from scipy.ndimage import uniform_filter
    zf = np.where(np.isfinite(z), z, np.nanmean(z))
    gy, gx = np.gradient(zf, dy, dx)
    s = np.hypot(gx, gy)
    return uniform_filter(s, size=smooth_cells, mode="nearest")


def inundation_duration_days(snapshots, threshold=0.05):
    """Duration [days] with surface depth > threshold from equally spaced
    snapshots [(t, depth, hr), ...]."""
    if len(snapshots) < 2:
        return None
    dt = snapshots[1][0] - snapshots[0][0]
    wet = sum((np.nan_to_num(s[1]) > threshold).astype(float) for s in snapshots)
    return wet * dt / 86400.0


def asset_value(assets: AssetGrid) -> np.ndarray:
    """Total exposed asset value per cell [yen] (house + household goods + car +
    business depreciable/inventory + farm), i.e. damage at rate 1."""
    gam = floor_correction(assets.mean_floors) if assets.mean_floors is not None else 1.0
    v = 0.0
    if assets.floor_area_m2 is not None:
        v = v + assets.floor_area_m2 * HOUSE_VALUE_PER_M2[assets.prefecture] * 1e3 * gam
    if assets.households is not None:
        v = v + assets.households * (HOUSEHOLD_GOODS_VALUE * gam + CAR_VALUE) * 1e3
    for code, n in (assets.employees or {}).items():
        vd, vi = BUSINESS_VALUE_PER_EMPLOYEE[code]
        v = v + n * (vd + vi) * 1e3 * gam
    if assets.farm_households is not None:
        v = v + assets.farm_households * sum(FARM_VALUE) * 1e3
    return np.asarray(v, float)
