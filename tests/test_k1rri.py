"""1K-RRI (rri_sim.k1rri): upscaling of a fine model and the mountain / plain / river exchange (synthetic valley)."""
import numpy as np

from rri_sim import k1rri as K
from rri_sim.domain import Domain
from rri_sim.hydro import d8_directions, fill_depressions, flow_accumulation, height_above_nearest_drainage
from rri_sim.workflow import model_land_params
from rri_sim.params import ModelParams


def fine_valley(n=32, dx=250.0):
    """A valley draining south: forested side slopes (mountain), paddy floor (plain)."""
    i, j = np.mgrid[0:n, 0:n]
    x = np.abs(j - n / 2 + 0.5)
    z = 200.0 - 4.0 * i + 6.0 * np.maximum(x - 3, 0) + 0.3 * x
    mask = np.ones_like(z, bool)
    p = ModelParams(land_classes=model_land_params())
    names = [c.name for c in p.land_classes]
    lc = np.where(x > 4, names.index("forest"), names.index("paddy")).astype(np.int64)
    zf = fill_depressions(z, mask)
    fdir = d8_directions(zf, mask, dx, dx)
    i_out = np.unravel_index(np.argmin(np.where(mask, zf, np.inf)), z.shape)
    fdir[i_out] = -1
    acc = flow_accumulation(fdir, mask, dx * dx) / 1e6
    d = Domain.from_dem(z, dx, dx, p, mask=mask, landclass=lc, river_threshold_km2=1.0, fdir=fdir)
    is_riv = np.zeros(z.shape, bool); is_riv[d.river.ri, d.river.rj] = True
    frac = np.zeros((5, n, n)); frac[0][lc == names.index("forest")] = 1; frac[1][lc == names.index("paddy")] = 1
    meta = {"z_raw": z, "acc_km2": acc, "fdir": fdir, "hand": height_above_nearest_drainage(zf, fdir, is_riv, mask),
            "landuse_fraction": frac, "gauge_index": {}}
    from rasterio.transform import from_origin
    d.transform, d.crs = from_origin(0.0, n * dx, dx, dx), "EPSG:6691"
    return d, meta


def test_upscaling_areas_slopes_flow_directions_and_lengths():
    d, meta = fine_valley()
    dk, m = K.upscale(d, meta, factor=4)
    assert dk.shape == (8, 8) and dk.mask.all() and dk.river.n == 64                # every cell holds a channel
    assert np.isclose((dk.As + dk.Af).sum(), d.mask.sum() * d.dx * d.dy)              # areas preserved
    centre = dk.Af[:, 3:5].sum() / (dk.As + dk.Af)[:, 3:5].sum()
    assert centre > 0.5 and dk.As[:, 0].sum() > dk.Af[:, 0].sum()                    # plain in the valley floor, mountain on the sides
    assert dk.i_s[dk.As > 0].mean() > dk.i_f[dk.Af > 0].mean()                       # mountains are steeper
    ls, lf = dk.lengths()
    assert np.allclose(ls + lf, dk.l) and np.allclose(ls / dk.l, dk.As / (dk.As + dk.Af))   # eq. (2), full cells
    r = dk.river
    assert (r.dn < 0).sum() >= 1 and len(r.order) == r.n                               # a tree (no cycles)
    has = r.dn >= 0
    assert (r.area_km2[r.dn[has]] >= r.area_km2[has] - 1e-9).mean() > 0.9               # areas grow downstream
    assert np.all(r.zb <= dk.z[r.ri, r.rj] - 0.0 + 1e-9)                               # bed = z_f - d
    small = has & (r.area_km2 <= K.KIN_AREA_KM2)
    assert np.array_equal(dk.kinematic, small & (r.area_km2[np.where(has, r.dn, 0)] > K.KIN_AREA_KM2))


def test_simulation_conserves_mass_and_routes_mountain_and_plain_water_to_the_outlet():
    d, meta = fine_valley()
    dk, _ = K.upscale(d, meta, factor=4)

    class Rain:
        def depth(self, t0, t1):
            return np.full(dk.shape, (30e-3 / 3600 if t0 < 6 * 3600 else 0.0) * (t1 - t0))
    s = K.Simulation1K(dk, Rain())
    res = s.run(24 * 3600, output_interval=3600)
    assert abs(res.mass.relative_error) < 1e-10
    assert res.mass.river_outflow > 0.1 * res.mass.rain and (s.hs * dk.As).sum() > 0      # water leaves; mountain soils still hold some
    assert s.hs.max() < 0.5 and np.nanmax(res.max_depth) >= 0.0
    # a paddy dam on the plain (store depth) holds water back and lowers the peak
    from rri_sim.workflow import PADDY_DAM
    names = [c.name for c in dk.params.land_classes]
    d2 = dk.copy()
    d2.lc_f[d2.lc_f == names.index("paddy")] = names.index(PADDY_DAM)
    s2 = K.Simulation1K(d2, Rain())
    res2 = s2.run(24 * 3600, output_interval=3600)
    assert abs(res2.mass.relative_error) < 1e-10 and res2.outlet_q.max() < res.outlet_q.max()
