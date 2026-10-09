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
