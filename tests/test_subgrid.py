"""Sub-grid asset elevations (rri_sim.subgrid) and their use in the damage estimate."""
import numpy as np

from rri_sim import damage as D
from rri_sim import subgrid as SG


def test_water_level_holds_the_cell_volume():
    k = 20
    h = np.array([[0.0, 0.3, 1.0, 2.0]])
    o = np.broadcast_to(np.linspace(-2.0, 6.0, k)[:, None, None], (k, 1, 4)).copy()   # valley cells
    H = SG.water_level(h, o)
    vol = np.maximum(H[None] - o, 0.0).mean(0)
    assert np.allclose(vol[h > 0], h[h > 0], atol=1e-6)
    assert H[0, 0] == 0.0                                  # dry cell
    assert (H[0, 1:] - o[0, 0, 1:] > h[0, 1:]).all()      # deeper than h in the valley floor (offset -2 m)
    flat = np.zeros((k, 1, 4))
    assert np.allclose(SG.water_level(h, flat), h)         # flat cell: the cell depth


def test_depths_flood_only_meshes_below_the_level():
    k = 4
    h = np.array([[1.0, 1.0, 0.0]])
    o = np.array([-1.0, 0.0, 1.0, 3.0])[:, None, None] * np.ones((k, 1, 3))
    o[:, 0, 1] = np.nan                                    # no meshes: cell-uniform depth
    lev = np.array([[1.0, 1.0, 0.0]])
    d = SG.depths(h, o, lev)
    assert np.allclose(d[:, 0, 0], [2.0, 1.0, 0.0, 0.0])
    assert np.allclose(d[:, 0, 1], 1.0)
    assert np.allclose(d[:, 0, 2], 0.0)                    # dry cells stay dry even below the level


def test_subgrid_damage_is_the_mean_over_quantiles_and_crops_use_farmland():
    k = 2
    hh = np.array([[10.0]])
    a = D.AssetGrid(floor_area_m2=hh * 100, households=hh, paddy_area_m2=np.array([[1e4]]),
                    field_area_m2=np.array([[1e4]]), prefecture="福岡")
    h = np.array([[1.0]])
    off = {"urban": np.array([0.0, 5.0])[:, None, None], "farm": np.array([0.0, 0.0])[:, None, None],
           "all": np.array([0.0, 5.0])[:, None, None]}
    lev = np.array([[1.0]])
    sg = D.estimate_damage(h, a, duration_days=np.array([[3.0]]), subgrid=(off, lev))
    one = D.estimate_damage(h, a, duration_days=np.array([[3.0]]))
    # half of the houses are 5 m higher (dry): half the cell-uniform damage
    for c in ("house", "household_goods", "car"):
        assert np.isclose(sg.totals[c], 0.5 * one.totals[c])
    # farmland at the cell level: crops unchanged; 畑 valued with the prefectural c
    assert np.isclose(sg.totals["paddy"], one.totals["paddy"]) and sg.totals["field"] > 0
    assert np.isclose(sg.totals["field"], 1e4 / 100 * D.FIELD_VALUE_KYEN_PER_A["福岡"] * 1e3 * D.CROP_RATES["field"][2, 1])
