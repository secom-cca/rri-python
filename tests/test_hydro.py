import numpy as np

from rri_sim.domain import RiverNetwork, condition_river_bed, isotonic_decreasing
from rri_sim.hydro import (d8_directions, fill_depressions, flow_accumulation,
                           height_above_nearest_drainage)


def test_isotonic_is_monotone_and_least_squares():
    y = np.array([10, 9, 9.5, 7, 8, 6, 6.5, 3])
    f = isotonic_decreasing(y)
    assert np.all(np.diff(f) <= 1e-12)
    # pooled blocks preserve the mean (property of the L2 isotonic fit)
    assert np.isclose(f.sum(), y.sum())


def test_bed_conditioning_removes_flat_lake_and_bumps():
    n = 30
    raw = np.linspace(100, 70, n)
    raw[10:15] += 5.0                          # spurious bump (bridge/road)
    net = RiverNetwork(ri=np.zeros(n, int), rj=np.arange(n), dn=np.append(np.arange(1, n), -1),
                       length=np.full(n, 250.0), width=np.full(n, 30.0), depth=np.full(n, 3.0),
                       zb=raw.copy(), embankment=np.zeros(n), n_river=np.full(n, 0.03),
                       area_km2=np.arange(n, dtype=float) + 1, outlet_slope=np.full(n, 1e-3))
    condition_river_bed(net, raw, min_slope=1e-4)
    assert np.all(np.diff(net.zb) < 0)           # strictly descending
    assert abs(net.zb.mean() - raw.mean()) < 1.0  # balanced (not pure fill / carve)


def test_flow_routing_and_hand_on_v_valley():
    ny, nx = 20, 21
    X, Y = np.meshgrid(np.arange(nx) - 10, np.arange(ny))
    z = 2.0 * np.abs(X) + 0.1 * (ny - Y)
    mask = np.ones_like(z, bool)
    zf = fill_depressions(z, mask)
    fd = d8_directions(zf, mask, 100, 100)
    acc = flow_accumulation(fd, mask, 1.0)
    assert acc.max() == mask.sum()                # everything drains to one outlet
    riv = np.zeros_like(mask)
    riv[:, 10] = True
    hand = height_above_nearest_drainage(z, fd, riv, mask)
    assert np.allclose(hand[:, 10], 0.0)
    assert np.all(hand[5, :] >= -1e-9) and hand[5, 0] > hand[5, 5] > 0
