"""Grid-resolution support: mesh statistics conserved on any grid, area-targeted
snapping, resolution-specific cache keys (no downloads)."""
import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin

from rri_sim.basin import snap_to_river
from rri_sim.cases import BASINS
from rri_sim.data import estat
from rri_sim.synthetic import valley_with_floodplain
from rri_sim.workflow import basin_config, cache_key


@pytest.mark.parametrize("res", [125.0, 250.0, 500.0, 1000.0])
def test_mesh_totals_conserved_on_any_grid(res):
    # 1/4 meshes (~290 x 231 m) of a 3rd mesh block near 日田; grid covers them all
    codes = [f"5030{t}{u}{v}{w}{a}{b}" for t, u, v, w in [(3, 0, 5, 5)] for a in "1234" for b in "1234"]
    vals = {c: 10.0 + i for i, c in enumerate(codes)}
    s, n, w, e = estat.mesh_bounds(codes[0][:8])
    tr = Transformer.from_crs("EPSG:6668", "EPSG:6690", always_xy=True)
    x0, y1 = tr.transform(w - 0.01, n + 0.01)
    x1, y0 = tr.transform(e + 0.01, s - 0.01)
    nx, ny = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    out = estat.distribute_to_grid(vals, from_origin(x0, y1, res, res), (ny, nx), "EPSG:6690")
    assert np.isclose(out.sum(), sum(vals.values()))


def test_snap_prefers_target_area_over_main_river():
    d = valley_with_floodplain()
    r = d.river
    k_small = int(np.argmin(r.area_km2))
    i, j = r.ri[k_small], r.rj[k_small]
    big = snap_to_river(r, i, j, None, 50)
    assert r.area_km2[big] == r.area_km2.max()
    assert snap_to_river(r, i, j, None, 50, target_km2=r.area_km2[k_small]) == k_small


def test_cache_key_and_config_copy():
    key = "kusu_ogase"
    assert cache_key(key) == key and cache_key(key, 250) == key
    assert cache_key(key, 500) == "kusu_ogase_500m"
    cfg = basin_config(key, 125)
    assert cfg.res == 125.0 and cfg.dem_zoom == 14
    assert BASINS[key]["cfg"].res == 250.0          # registry not mutated
