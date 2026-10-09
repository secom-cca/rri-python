"""Verification of the numerical model on synthetic problems with known answers."""
import numpy as np
import pytest

from rri_sim.domain import Domain
from rri_sim.forcing import Hydrograph, Rainfall
from rri_sim.params import LandClassParams, ModelParams, NumericsParams
from rri_sim.solver import Simulation
from rri_sim.synthetic import tilted_plane, v_catchment, valley_with_floodplain
from rri_sim import kernels as K


def test_lake_at_rest():
    """Still water over irregular terrain must stay exactly at rest."""
    rng = np.random.default_rng(0)
    z = rng.uniform(0, 2, (30, 30))
    p = ModelParams(land_classes=[LandClassParams(n=0.05)])
    d = Domain(z=z, mask=np.ones_like(z, bool), dx=50, dy=50,
               landclass=np.zeros(z.shape, np.int64), params=p)
    h0 = np.maximum(3.0 - z, 0.0)
    sim = Simulation(d, h0=h0)
    res = sim.run(3600)
    assert np.allclose(sim.h + z, 3.0, atol=1e-12)
    assert abs(res.mass.error) < 1e-6


def test_positivity_and_mass_random_terrain():
    """Extreme rain on rough terrain with closed boundaries: h >= 0 and
    storage == rain to round-off."""
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 20, (40, 40))
    p = ModelParams(land_classes=[LandClassParams(n=0.03)],
                    numerics=NumericsParams(dt_max=120))
    d = Domain(z=z, mask=np.ones_like(z, bool), dx=100, dy=100,
               landclass=np.zeros(z.shape, np.int64), params=p)
    rain = Rainfall.uniform([0, 1800, 7200], [300.0, 0.0], d.shape)
    sim = Simulation(d, rain)
    res = sim.run(7200)
    assert sim.h.min() >= 0.0
    assert abs(res.mass.relative_error) < 1e-12
    assert np.isclose(res.mass.storage, 0.3 / 2 * 40 * 40 * 1e4, rtol=1e-10)


def kinematic_rising_limb(t, r, n, s, L):
    """Analytic kinematic-wave outflow per unit width from a plane."""
    te = (n * L / (r ** (2 / 3) * np.sqrt(s))) ** 0.6
    q = np.where(t < te, (r * t) ** (5 / 3) * np.sqrt(s) / n, r * L)
    return q, te


def test_tilted_plane_vs_kinematic_wave():
    L, s, n, r_mmh = 1000.0, 0.01, 0.05, 50.0
    d = tilted_plane(length=L, width=30.0, dx=10.0, slope=s, n=n)
    rain = Rainfall.uniform([0, 4 * 3600], [r_mmh], d.shape)
    sim = Simulation(d, rain)
    res = sim.run(4 * 3600, output_interval=300)
    # unit-width outflow at the open boundary = boundary flux / width
    # (recorded via mass balance increments is coarse; recompute from final state)
    r = r_mmh / 1000 / 3600
    q_eq = r * L
    # at the end the plane must be at equilibrium
    q_end = sim.fb[:, -1].sum() / (d.shape[0] * d.dx)
    assert np.isclose(q_end, q_eq, rtol=0.01)
    # mass balance
    assert abs(res.mass.relative_error) < 1e-12


SCHEMES = ["inertial", "diffusive"]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_tilted_plane_rising_limb_shape(scheme):
    L, s, n, r_mmh = 1000.0, 0.01, 0.05, 50.0
    d = tilted_plane(length=L, width=10.0, dx=10.0, slope=s, n=n,
                     numerics=NumericsParams(scheme=scheme))
    r = r_mmh / 1000 / 3600
    rain = Rainfall.uniform([0, 3 * 3600], [r_mmh], d.shape)
    sim = Simulation(d, rain)
    q_an, te = kinematic_rising_limb(np.array([1.0]), r, n, s, L)
    errs = []
    t = 0.0
    for frac in [0.4, 0.6, 0.8, 1.2, 1.5]:
        sim.run(frac * te)
        q_num = sim.fb[0, -1] / d.dx
        q_ref, _ = kinematic_rising_limb(np.array([sim.t]), r, n, s, L)
        errs.append(abs(q_num - q_ref[0]) / (r * L))
    # diffusive wave vs kinematic wave on a 1% slope: within 5 % of q_eq
    assert max(errs) < 0.05, errs


@pytest.mark.parametrize("scheme", SCHEMES)
def test_v_catchment_equilibrium_and_mass(scheme):
    d = v_catchment(numerics=NumericsParams(scheme=scheme))
    rain = Rainfall.uniform([0, 5400, 3 * 3600], [10.8, 0.0], d.shape)
    sim = Simulation(d, rain)
    res = sim.run(3 * 3600, output_interval=300)
    q_eq = 3e-6 * d.shape[0] * d.shape[1] * d.dx * d.dy
    k = np.argmin(np.abs(res.times - 5400))
    assert np.isclose(res.outlet_q[k], q_eq, rtol=0.01)
    assert abs(res.mass.relative_error) < 1e-12
    # recession: monotone decreasing after rain stops
    rec = res.outlet_q[res.times > 5400]
    assert np.all(np.diff(rec) <= 1e-9)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_v_catchment_time_step_convergence(scheme):
    """Halving the admissible time step must not change the hydrograph
    materially (checks that limiters do not dominate the solution)."""
    out = []
    for cfl, dtmax, safety in [(0.7, 60.0, 0.4), (0.35, 10.0, 0.2)]:
        d = v_catchment(numerics=NumericsParams(scheme=scheme, cfl=cfl, dt_max=dtmax,
                                                diff_safety=safety))
        rain = Rainfall.uniform([0, 5400, 3 * 3600], [10.8, 0.0], d.shape)
        res = Simulation(d, rain).run(3 * 3600, output_interval=300)
        out.append(res.outlet_q)
    rel = np.max(np.abs(out[0] - out[1])) / out[1].max()
    assert rel < 0.02, rel


def test_green_ampt_analytic():
    """Ponded infiltration vs implicit Green-Ampt solution."""
    kv, dth, sf = 1e-5, 0.3, 0.1
    land = LandClassParams(n=0.05, kv=kv, phi_minus_theta=dth, sf=sf)
    z = np.zeros((3, 3))
    d = Domain(z=z, mask=np.ones_like(z, bool), dx=10, dy=10,
               landclass=np.zeros(z.shape, np.int64),
               params=ModelParams(land_classes=[land],
                                  numerics=NumericsParams(dt_max=30.0)))
    sim = Simulation(d, h0=np.full(z.shape, 1.0))
    sim.run(7200)
    F_num = sim.F[1, 1]
    c = dth * sf
    # solve kv t = F - c ln(1 + F/c)
    F = 0.05
    for _ in range(100):
        g = F - c * np.log(1 + F / c) - kv * 7200
        F -= g / (F / (c + F))
    assert np.isclose(F_num, F, rtol=1e-6)


def test_subsurface_only_no_surface_runoff():
    """With rain below the subsurface transmissivity, no overland flow forms
    and equilibrium outflow equals rainfall."""
    land = LandClassParams(n=0.4, ka=0.05, soil_depth=1.0, gamma_a=0.4, gamma_m=0.0)
    L, s = 500.0, 0.1
    d = tilted_plane(length=L, width=10.0, dx=10.0, slope=s, land=land)
    # capacity ka * d_a * s = 0.05*0.4*0.1 = 2e-3 m2/s ; rain*L = 5mm/h*500 = 6.9e-4
    rain = Rainfall.uniform([0, 48 * 3600], [5.0], d.shape)
    sim = Simulation(d, rain)
    res = sim.run(48 * 3600, output_interval=3600)
    assert (sim.h <= land.d_a + 1e-9).all()
    q = sim.fb[0, -1] / d.dx
    assert np.isclose(q, 5.0 / 1000 / 3600 * L, rtol=0.02)
    assert abs(res.mass.relative_error) < 1e-12


def test_schemes_agree_v_catchment():
    """Inertial and diffusive closures share the same steady flux law; on the
    V-catchment their hydrographs must agree closely."""
    out = {}
    for scheme in SCHEMES:
        d = v_catchment(numerics=NumericsParams(scheme=scheme))
        rain = Rainfall.uniform([0, 5400, 3 * 3600], [10.8, 0.0], d.shape)
        out[scheme] = Simulation(d, rain).run(3 * 3600, output_interval=300).outlet_q
    rel = np.max(np.abs(out["inertial"] - out["diffusive"])) / out["diffusive"].max()
    assert rel < 0.03, rel


@pytest.mark.parametrize("scheme", ["inertial"])
def test_levee_prevents_and_permits_overflow(scheme):
    # diffusive version: see scripts/compare_schemes_levee.py (slow; dt ~ 0.01-0.1 s)
    """Upstream flood hydrograph: low levee overtopped -> inundation; high
    levee -> no inundation on the floodplain."""
    def run(emb):
        d = valley_with_floodplain(embankment=emb, numerics=NumericsParams(scheme=scheme))
        hyd = Hydrograph([0, 3600, 6 * 3600, 12 * 3600], [50, 1500, 1500, 50])
        sim = Simulation(d, inflows={0: hyd})
        return sim.run(12 * 3600, output_interval=1800), d
    res_low, d = run(0.5)
    res_high, _ = run(30.0)
    plain = d.landclass == 1
    assert np.nanmax(res_low.max_depth[plain]) > 0.3
    assert np.nanmax(res_high.max_depth[plain]) < 1e-6
    assert abs(res_low.mass.relative_error) < 1e-12
    assert abs(res_high.mass.relative_error) < 1e-12
    # with the high levee the channel must convey the flood: outflow volume
    # ~ inflow volume minus final storage
    mb = res_high.mass
    assert mb.river_outflow > 0.8 * mb.inflow


def test_honda_formula_continuity():
    h1 = 1.0
    a = K.honda_unit(h1, 2 / 3 - 1e-9, 0.35, 0.91)
    b = K.honda_unit(h1, 2 / 3 + 1e-9, 0.35, 0.91)
    assert abs(a - b) / a < 0.01
