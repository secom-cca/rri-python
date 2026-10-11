"""Livability and Ecology scores (0-100) for flood-management scenarios.

Livability (生活利便性) follows the indicator set of 国土交通省「都市構造の評価に
関するハンドブック」(2014):
  - 生活サービス施設の徒歩圏人口カバー率 (徒歩圏 = 800 m)          -> walk coverage
  - travel time by car to the nearest facility (rural basins are car-based)
  - 居住を誘導する区域の人口密度 / 利用圏平均人口密度                -> 居住集約度
for four service types: 医療, 教育, 商業, 公共サービス.
Additionally (安全・安心 分野):
  - 防災上危険性が懸念される地域に居住する人口の割合 (max depth >= 0.5 m)
  - 洪水時の医療アクセス維持率 (car access within 30 min with cells >= 0.3 m
    impassable and flooded facilities unusable).

Ecology (生態系):
  E1 生息地の質   : InVEST Habitat Quality-type model (suitability x
                    (1 - degradation^z / (degradation^z + k^z)); threats =
                    built-up land, roads, farmland with exponential decay)
  E2 生態的ネットワークの連結性 : terrestrial - kernel approximation of the
                    probability-of-connectivity index (equivalent connected
                    area / total area, dispersal scale 1 km); riverine - lateral
                    connectivity (share of channel length without levee) and
                    longitudinal dendritic connectivity index (dams = barriers)
  E3 水文・生態条件 : rainfall retained in the basin during the event (1 -
                    outflow / rain) and the natural flood pulse (share of natural
                    floodplain cells that are inundated and connected)

Every sub-indicator is in [0, 1]; scores are 100 x the mean.  All parameters
(speeds, suitabilities, threat weights, distances) are in the dataclasses
below and are reported with the results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import sparse
from scipy.signal import fftconvolve
from scipy.sparse.csgraph import dijkstra

# ---------------------------------------------------------------- parameters


@dataclass
class AccessParams:
    walk_speed_m_min: float = 80.0          # handbook: 徒歩圏 800 m ≈ 10 min
    walk_radius_m: float = 800.0
    # car speeds by cell type [km/h] (approximation of the road network from land use)
    v_road: float = 40.0                    # cells with >= 10 % road land (L03-b 0901)
    v_urban: float = 30.0
    v_farm: float = 25.0                    # paddy / field (farm roads)
    v_forest: float = 10.0
    v_water: float = 0.0                    # impassable unless road (bridge)
    car_full_min: float = 10.0              # full score if nearest facility within 10 min
    car_zero_min: float = 30.0              # zero score beyond 30 min
    density_ref_per_ha: float = 40.0        # DID criterion (人口集中地区: 40 人/ha)
    commercial_min_employees: float = 10.0  # 卸売・小売 employees in a cell = commercial service
    flood_car_depth_m: float = 0.3          # vehicles cannot pass (冠水 30 cm)
    flood_walk_depth_m: float = 0.5
    hazard_depth_m: float = 0.5             # 防災上危険性が懸念される地域


@dataclass
class EcoParams:
    # habitat suitability by land class
    suitability: dict = field(default_factory=lambda: {
        "forest": 1.0, "forest_new": 0.8, "water": 0.9, "wetland": 0.85, "paddy": 0.5,
        "paddy_dam": 0.5, "green": 0.6, "field": 0.3, "urban": 0.0})
    # threats: (weight, decay distance [m])
    threats: dict = field(default_factory=lambda: {
        "urban": (1.0, 2000.0), "road": (0.7, 1000.0), "farm": (0.3, 500.0)})
    z: float = 2.5
    k: float = 0.25                          # half-saturation of degradation
    habitat_min: float = 0.5                 # cells counted as habitat for connectivity
    dispersal_m: float = 1000.0
    levee_threshold_m: float = 0.5           # channel cells with levees above this are disconnected
    min_river_area_km2: float = 10.0
    floodplain_hand_m: float = 5.0


# ---------------------------------------------------------------- accessibility


def _grid_graph(speed_mps, dx, dy):
    """Directed 8-neighbour graph with travel time [s] = distance / mean speed."""
    ny, nx = speed_mps.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, w = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[max(0, -di):ny - max(0, di), max(0, -dj):nx - max(0, dj)]
        b = idx[max(0, di):ny - max(0, -di) or ny, max(0, dj):nx - max(0, -dj) or nx]
        va = speed_mps.ravel()[a.ravel()]
        vb = speed_mps.ravel()[b.ravel()]
        ok = (va > 0) & (vb > 0)
        dist = np.hypot(di * dy, dj * dx)
        t = dist / (0.5 * (va[ok] + vb[ok]))
        for s_, e_ in ((a.ravel()[ok], b.ravel()[ok]), (b.ravel()[ok], a.ravel()[ok])):
            rows.append(s_); cols.append(e_); w.append(t)
    n = ny * nx
    return sparse.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, n))


def travel_time(speed_mps, sources, dx, dy):
    """Minimum travel time [s] from any source cell (flat indices) to every cell."""
    if len(sources) == 0:
        return np.full(speed_mps.size, np.inf)
    g = _grid_graph(speed_mps, dx, dy)
    return dijkstra(g, directed=True, indices=np.asarray(sources), min_only=True)


def car_speed(lc_names, landclass, road_frac, p: AccessParams, depth=None):
    names = np.asarray(lc_names)[landclass]
    v = np.select([road_frac >= 0.10, names == "urban", np.isin(names, ["paddy", "paddy_dam", "field", "green"]),
                   np.isin(names, ["forest", "forest_new", "wetland"]), names == "water"],
                  [p.v_road, p.v_urban, p.v_farm, p.v_forest, p.v_water], p.v_forest) / 3.6
    if depth is not None:
        v = np.where(depth >= p.flood_car_depth_m, 0.0, v)
    return v


def walk_speed(landclass, lc_names, road_frac, p: AccessParams, depth=None):
    names = np.asarray(lc_names)[landclass]
    v = np.where((names == "water") & (road_frac < 0.05), 0.0, p.walk_speed_m_min / 60.0)
    if depth is not None:
        v = np.where(depth >= p.flood_walk_depth_m, 0.0, v)
    return v


def livability(population, landclass, lc_names, road_frac, facility_cells: dict, dx, dy,
               mask, p: AccessParams | None = None, flood_depth=None) -> dict:
    """Livability score and raw indicators.

    facility_cells: {"medical": idx, "school": idx, "commercial": idx, "public": idx}
    flood_depth   : max inundation depth [m] for the safety indicators (optional).
    """
    p = p or AccessParams()
    pop = np.where(mask, np.nan_to_num(population), 0.0)
    P = pop.sum()
    if P <= 0:
        return {"score": np.nan}
    vc = car_speed(lc_names, landclass, road_frac, p)
    vw = walk_speed(landclass, lc_names, road_frac, p)
    out = {"services": {}}
    comps = []
    for s, cells in facility_cells.items():
        tw = travel_time(vw, cells, dx, dy).reshape(pop.shape)
        tc = travel_time(vc, cells, dx, dy).reshape(pop.shape) / 60.0
        walk_cov = float(pop[tw * (p.walk_speed_m_min / 60.0) <= p.walk_radius_m].sum() / P)
        f = np.clip((p.car_zero_min - tc) / (p.car_zero_min - p.car_full_min), 0, 1)
        car_cov = float((pop * np.nan_to_num(f)).sum() / P)
        mean_t = float(np.nansum(pop * np.where(np.isfinite(tc), tc, p.car_zero_min * 2)) / P)
        a = 0.5 * walk_cov + 0.5 * car_cov
        out["services"][s] = {"walk800_coverage": walk_cov, "car_access": car_cov,
                              "mean_car_min": mean_t, "n_facility_cells": int(len(cells)), "score": a}
        comps.append(a)
    cell_ha = dx * dy / 1e4
    dens = pop / cell_ha
    pwd = float((pop * dens).sum() / P)                    # 人口加重平均密度 [人/ha]
    conc = min(1.0, pwd / p.density_ref_per_ha)
    out["population_weighted_density_per_ha"] = pwd
    out["concentration"] = conc
    comps.append(conc)
    out["score"] = 100.0 * float(np.mean(comps))
    if flood_depth is not None:
        d = np.nan_to_num(flood_depth)
        out["pop_in_hazard_share"] = float(pop[d >= p.hazard_depth_m].sum() / P)
        med = facility_cells.get("medical", np.array([], int))
        med_ok = med[d.ravel()[med] < p.flood_car_depth_m] if len(med) else med
        tf = travel_time(car_speed(lc_names, landclass, road_frac, p, d), med_ok, dx, dy).reshape(pop.shape) / 60
        out["flood_medical_access_30min"] = float(pop[tf <= 30.0].sum() / P)
    out["params"] = asdict(p)
    return out


# ---------------------------------------------------------------- ecology


def _exp_kernel(d_m, dx, dy, trunc=3.0):
    r = int(np.ceil(trunc * d_m / dx))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    K = np.exp(-np.hypot(xx * dx, yy * dy) / d_m)
    return K


def habitat_quality(landclass, lc_names, road_frac, mask, dx, dy, p: EcoParams | None = None):
    p = p or EcoParams()
    names = np.asarray(lc_names)[landclass]
    H = np.vectorize(lambda n: p.suitability.get(n, 0.3))(names).astype(float)
    H = np.where(mask, H, 0.0)
    sources = {"urban": (names == "urban").astype(float),
               "road": np.clip(road_frac / 0.2, 0, 1),
               "farm": np.isin(names, ["paddy", "paddy_dam", "field"]).astype(float)}
    D = np.zeros_like(H)
    wsum = sum(w for w, _ in p.threats.values())
    for t, (w, dist) in p.threats.items():
        K = _exp_kernel(dist, dx, dy)
        K /= K.sum()
        D += w / wsum * fftconvolve(sources[t], K, mode="same")
    D = np.clip(D, 0, None)
    Q = H * (1.0 - D ** p.z / (D ** p.z + p.k ** p.z))
    return np.where(mask, Q, 0.0), H


def dendritic_connectivity(river, barriers_k, min_area=10.0):
    """DCI_p (Cote et al. 2009) for impassable barriers: sum over river
    segments of (segment length / total length)^2, using channel cells with
    upstream area >= min_area."""
    sel = river.area_km2 >= min_area
    L = river.length * sel
    total = L.sum()
    if total <= 0:
        return 1.0
    seg = -np.ones(river.n, int)
    bar = set(int(k) for k in barriers_k)
    # assign segment ids walking upstream from outlets / barriers
    order = river.order[::-1]                       # downstream -> upstream
    nseg = 0
    for k in order:
        d = river.dn[k]
        if d < 0 or d in bar:
            seg[k] = nseg
            nseg += 1
        else:
            seg[k] = seg[d]
    lens = np.bincount(seg, weights=L, minlength=nseg)
    return float(((lens / total) ** 2).sum())


def ecology(landclass, lc_names, road_frac, mask, dx, dy, river, embankment, dam_k,
            hand, max_depth, rain_volume, outflow_volume, p: EcoParams | None = None) -> dict:
    p = p or EcoParams()
    Q, H = habitat_quality(landclass, lc_names, road_frac, mask, dx, dy, p)
    n = mask.sum()
    e1 = float(Q[mask].mean())
    # terrestrial connectivity: sqrt(sum q (q*K)) / sqrt(sum 1 (1*K))
    K = _exp_kernel(p.dispersal_m, dx, dy)
    q = np.where(H >= p.habitat_min, Q, 0.0)
    one = mask.astype(float)
    eca = np.sqrt((q * fftconvolve(q, K, mode="same")).sum())
    eca_ref = np.sqrt((one * fftconvolve(one, K, mode="same")).sum())
    terr = float(eca / eca_ref) if eca_ref > 0 else 0.0
    sel = river.area_km2 >= p.min_river_area_km2
    lateral = float((river.length * sel * (embankment < p.levee_threshold_m)).sum()
                    / max((river.length * sel).sum(), 1e-9))
    dci = dendritic_connectivity(river, dam_k, p.min_river_area_km2)
    e2 = 0.5 * terr + 0.25 * lateral + 0.25 * dci
    retained = float(np.clip(1.0 - outflow_volume / max(rain_volume, 1e-9), 0, 1))
    names = np.asarray(lc_names)[landclass]
    fp = mask & (np.nan_to_num(hand, nan=1e9) <= p.floodplain_hand_m) & (names != "urban")
    pulse = float((np.nan_to_num(max_depth)[fp] >= 0.1).mean()) if fp.any() else 0.0
    e3 = 0.5 * retained + 0.5 * pulse
    return {"score": 100.0 * (e1 + e2 + e3) / 3.0,
            "habitat_quality": e1, "connectivity": e2, "hydro_ecological": e3,
            "terrestrial_connectivity": terr, "river_lateral_connectivity": lateral,
            "river_longitudinal_dci": dci, "event_retention": retained,
            "natural_flood_pulse": pulse, "params": asdict(p),
            "_quality_map": Q}
