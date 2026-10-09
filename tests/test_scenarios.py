import numpy as np

from rri_sim.forcing import Hydrograph
from rri_sim.scenarios import Scenario, compare_scenarios, dredge, raise_levee, retention_basin
from rri_sim.solver import Simulation
from rri_sim.synthetic import valley_with_floodplain
from rri_sim import damage as D


def _run(dom):
    hyd = Hydrograph([0, 3600, 6 * 3600, 12 * 3600], [50, 1500, 1500, 50])
    return Simulation(dom, inflows={0: hyd}).run(12 * 3600, output_interval=1800)


def test_policies_reduce_inundation_and_damage():
    base = valley_with_floodplain(embankment=0.5)
    plain = base.landclass == 1
    assets = D.AssetGrid(floor_area_m2=np.where(plain, 2000.0, 0.0),
                         households=np.where(plain, 15.0, 0.0), prefecture="福岡")
    gs = D.ground_slope_from_dem(base.z, base.dx, base.dy)

    def dmg(res):
        return D.estimate_damage(res.max_depth, assets, gs).total

    allriv = np.arange(base.river.n)
    rows = compare_scenarios(base, [
        Scenario("dredge_2m", lambda d: dredge(d, allriv, 2.0)),
        Scenario("levee_+3m", lambda d: raise_levee(d, allriv, 3.0)),
    ], _run, dmg)
    by = {r["scenario"]: r for r in rows}
    assert by["baseline"]["damage_yen"] > 0
    assert by["dredge_2m"]["damage_yen"] < by["baseline"]["damage_yen"]
    assert by["levee_+3m"]["damage_yen"] < by["baseline"]["damage_yen"]
    for r in rows:
        assert abs(r["mass_rel_err"]) < 1e-12


def test_retention_basin_attenuates_downstream_peak():
    base = valley_with_floodplain(embankment=30.0)   # no overtopping anywhere
    ny, nx = base.shape
    jc = nx // 2
    basin = np.zeros(base.shape, bool)
    basin[10:25, jc + 1:jc + 8] = True
    inlet = np.arange(12, 16)
    rb = retention_basin(base, basin, excavate_m=4.0, river_idx=inlet, inlet_crest_m=2.0)
    q0 = _run(base).outlet_q.max()
    q1 = _run(rb).outlet_q.max()
    assert q1 < q0
