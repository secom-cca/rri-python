import numpy as np

from rri_sim import damage as D


def test_rate_bins():
    h = np.array([0.0, 0.2, 0.44, 0.45, 0.94, 0.95, 1.45, 2.45, 3.45, 5.0])
    idx = D.general_rate_index(h)
    assert idx.tolist() == [-1, 0, 0, 1, 1, 2, 3, 4, 5, 5]


def test_house_rate_slope_groups():
    h = np.full(3, 1.0)      # 0.55 m above floor -> 50-99 cm column
    s = np.array([5e-4, 1.5e-3, 3e-3])
    assert np.allclose(D.house_rate(h, s), [0.253, 0.301, 0.325])


def test_car_and_crop():
    assert np.allclose(D.car_rate(np.array([0.1, 0.35, 0.6, 1.0])), [0, 0.1, 0.5, 1])
    assert np.isclose(D.crop_rate(np.array([0.7]), np.array([3.0]), "paddy")[0], 0.44)
    assert np.isclose(D.crop_rate(np.array([1.2]), np.array([10.0]), "field")[0], 0.91)


def test_damage_hand_calculation():
    """One cell, 1.5 m depth on gentle slope (group A), Fukuoka."""
    a = D.AssetGrid(floor_area_m2=np.array([1000.0]), households=np.array([10.0]),
                    employees={"I": np.array([5.0])}, prefecture="福岡")
    r = D.estimate_damage(np.array([1.5]), a, ground_slope=np.array([1e-4]))
    # 1.05 m above floor -> 100-199 column (index 3)
    house = 1000 * 230.8e3 * 0.406
    goods = 10 * 9368e3 * 0.701
    car = 10 * 3502e3 * 1.0
    bus = 5 * (2585e3 * 0.801 + 2307e3 * 0.814)
    assert np.isclose(r.total, house + goods + car + bus)


def test_floor_correction():
    assert np.allclose(D.floor_correction([1, 2.9, 3, 10]), [1, 1, 2 / 3, 0.2])


def test_costs_levee_and_present_value():
    from rri_sim import costs as C
    r = C.estimate([{"type": "levee", "q": {"height_m": 2.0, "length_m": 1000.0}},
                    {"type": "paddy_dam", "q": {"area_m2": 1e6}}])
    u = C.default_unit_costs()
    area = 2.0 * (u["levee_crest_m"] + u["levee_slope"] * 2.0)
    foot = u["levee_crest_m"] + 2 * u["levee_slope"] * 2.0
    lev = 1000 * (area * u["levee_fill_yen_m3"] + u["levee_fixed_yen_m"] + foot * u["land_yen_m2"])
    assert np.isclose(r["items"][0]["initial_yen"], lev)
    # raising an existing 3 m levee by +1 m: only the extra section and footprint
    up = C.estimate([{"type": "levee", "q": {"height_m": 1.0, "base_height_m": 3.0, "length_m": 1000.0}}])
    sec = lambda h: h * (u["levee_crest_m"] + u["levee_slope"] * h)
    extra = 1000 * ((sec(4.0) - sec(3.0)) * u["levee_fill_yen_m3"] + u["levee_fixed_yen_m"]
                    + 2 * u["levee_slope"] * 1.0 * u["land_yen_m2"])
    assert np.isclose(up["items"][0]["initial_yen"], extra)
    pdm = r["items"][1]
    assert np.isclose(pdm["pv_yen"], 1000 * u["paddy_dam_init_yen_10a"] + 1000 * u["paddy_dam_annual_yen_10a"] * C.ANNUITY)
    assert 21.4 < C.ANNUITY < 21.6     # 50 years at 4 %


def test_indicators_respond_to_measures():
    from rri_sim import indicators as I
    from rri_sim.synthetic import valley_with_floodplain
    d = valley_with_floodplain()
    names = ["forest", "paddy", "field", "urban", "water"]
    lc = d.landclass.copy()                      # 0 forest (hills), 1 paddy (plain)
    lc[25:35, 18:23] = 3                          # town on the floodplain
    road = np.zeros(d.shape); road[:, 19] = 0.3
    pop = np.where(lc == 3, 300.0, 0.0)
    fac = {"medical": np.array([30 * d.shape[1] + 20]), "school": np.array([28 * d.shape[1] + 19]),
           "commercial": np.array([30 * d.shape[1] + 21]), "public": np.array([5 * d.shape[1] + 20])}
    L = I.livability(pop, lc, names, road, fac, d.dx, d.dy, d.mask, flood_depth=np.zeros(d.shape))
    assert 0 <= L["score"] <= 100 and L["services"]["medical"]["walk800_coverage"] > 0
    # dispersing the town lowers concentration and livability
    pop2 = np.where(d.mask, pop.sum() / d.mask.sum(), 0.0)
    L2 = I.livability(pop2, lc, names, road, fac, d.dx, d.dy, d.mask)
    assert L2["concentration"] < L["concentration"] and L2["score"] < L["score"]
    # flooding the hospital removes flood-time access
    dep = np.zeros(d.shape); dep[30, 20] = 1.0
    L3 = I.livability(pop, lc, names, road, fac, d.dx, d.dy, d.mask, flood_depth=dep)
    assert L3["flood_medical_access_30min"] < L["flood_medical_access_30min"]
    # ecology: levees lower lateral connectivity; forest raises habitat quality
    r = d.river
    r.area_km2[:] = 100.0                         # synthetic channel: give it a real catchment
    e0 = I.ecology(lc, names, road, d.mask, d.dx, d.dy, r, np.zeros(r.n), [], np.zeros(d.shape),
                   np.zeros(d.shape), 1.0, 0.5)
    e1 = I.ecology(lc, names, road, d.mask, d.dx, d.dy, r, np.full(r.n, 3.0), [], np.zeros(d.shape),
                   np.zeros(d.shape), 1.0, 0.5)
    lc_f = np.where(lc == 1, 0, lc)
    e2 = I.ecology(lc_f, names, road, d.mask, d.dx, d.dy, r, np.zeros(r.n), [], np.zeros(d.shape),
                   np.zeros(d.shape), 1.0, 0.5)
    assert e1["river_lateral_connectivity"] < e0["river_lateral_connectivity"]
    assert e2["habitat_quality"] > e0["habitat_quality"]
    # a dam splits the network -> DCI < 1
    assert I.dendritic_connectivity(r, [30], 0.0) < 1.0
