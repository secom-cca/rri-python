"""Scenario edits of the GUI engine on a synthetic basin (no downloads)."""
import numpy as np
import pytest

from rri_sim import damage as D
from rri_sim.forcing import Rainfall
from rri_sim.gui.engine import BasinSession
from rri_sim.params import ModelParams
from rri_sim.solver import Dam, Simulation
from rri_sim.synthetic import valley_with_floodplain
from rri_sim.workflow import model_land_params


@pytest.fixture
def session():
    d = valley_with_floodplain(embankment=0.5)
    lp = model_land_params()
    names = [c.name for c in lp]
    # hill -> forest, floodplain -> paddy, a block of houses -> urban
    lc = np.where(d.landclass == 0, names.index("forest"), names.index("paddy"))
    lc[20:30, 12:16] = names.index("urban")
    d.landclass = lc
    d.params = ModelParams(land_classes=lp, river=d.params.river, numerics=d.params.numerics)
    s = BasinSession.__new__(BasinSession)
    s.d = d
    s.class_names = names
    hh = np.where(lc == names.index("urban"), 20.0, 0.0)
    s.assets = D.AssetGrid(floor_area_m2=hh * 100, households=hh, employees={"I": hh / 2},
                           paddy_area_m2=np.where(lc == names.index("paddy"), 1e4, 0.0),
                           field_area_m2=np.zeros(d.shape), prefecture="福岡")
    s.pop = hh * 2.5
    # a safe built-up area on the hillside (high HAND) as relocation destination
    s.hand = np.where(d.landclass == names.index("forest"), 20.0, 1.0)
    lc[5:8, 2:5] = names.index("urban")
    s.hand[5:8, 2:5] = 30.0
    return s


def flat(d, mask):
    return np.flatnonzero(mask.ravel()).tolist()


def test_each_edit_applies_only_where_meaningful(session):
    s = session
    d0 = s.d
    cn = s.class_names
    allc = list(range(d0.shape[0] * d0.shape[1]))
    for typ, extra in [("forest", {}), ("paddy_dam", {}), ("relocate", {}), ("raise", {"height": 1.0}),
                       ("levee", {"height": 4.0}), ("retention", {"depth": 2.0, "q_cut": 5.0})]:
        cells = allc if typ != "retention" else flat(d0, np.isin(np.arange(d0.shape[1]), [21, 22])[None, :]
                                                    & (np.arange(d0.shape[0])[:, None] < 10))
        d, a, pop, floor, notes, storage, applied, retention = s.apply_edits([{"type": typ, "cells": cells, **extra}], {})
        assert applied[0]["type"] == typ
        if typ == "forest":
            assert (d.landclass[d0.landclass == cn.index("forest")] == cn.index("forest")).all()
            assert (d.landclass[d0.landclass == cn.index("paddy")] == cn.index("forest_new")).all()
            assert a.paddy_area_m2.sum() == 0
        if typ == "paddy_dam":
            assert (d.landclass[d0.landclass == cn.index("paddy")] == cn.index("paddy_dam")).all()
            assert (d.landclass[d0.landclass == cn.index("urban")] == cn.index("urban")).all()
        if typ == "relocate":
            # everything painted (all cells) -> nothing left to receive; with a partial
            # selection households move to the safe built-up cells (tested below)
            assert a.households.sum() == 0
        if typ == "raise":
            assert np.allclose(floor, 1.45)
        if typ == "levee":
            assert np.allclose(d.river.embankment, 4.0)
        if typ == "retention":
            riv = np.zeros(d0.shape, bool)
            riv[d0.river.ri, d0.river.rj] = True
            site = storage
            assert site.sum() > 0 and not (site & riv).any()                   # channel is not land
            assert not (site & (d0.landclass == cn.index("urban"))).any()      # built-up land excluded
            assert (d.landclass[site] == cn.index("wetland")).all()
            assert a.households[site].sum() == 0                               # land acquired
            assert np.allclose(d.z, d0.z, equal_nan=True)                      # off-channel storage, terrain kept
            (rt,) = retention
            assert np.isclose(rt["dam"].capacity, site.sum() * d0.cell_area * 2.0) and rt["dam"].q_cut == 5.0
            assert riv[d0.river.ri[rt["dam"].k], d0.river.rj[rt["dam"].k]]
        # original domain untouched
        assert np.allclose(s.d.river.embankment, 0.5)


def test_relocation_moves_households_to_safe_cells(session):
    s = session
    cn = s.class_names
    src = np.flatnonzero((s.d.landclass == cn.index("urban")).ravel() & (s.hand.ravel() < 5)).tolist()
    hh0 = s.assets.households.sum()
    for dest in ("existing", "new"):
        d, a, pop, *_rest, applied, _ret = s.apply_edits([{"type": "relocate", "cells": src, "dest": dest}], {})
        q = applied[0]["q"]
        assert np.isclose(a.households.sum(), hh0)                 # conserved
        assert a.households.ravel()[src].sum() == 0                # moved out
        assert (a.households[s.hand > 10]).sum() > 0.99 * hh0      # into safe cells
        assert q["households"] > 0 and q["dest_cells"] > 0
        assert (d.landclass.ravel()[src] == cn.index("green")).all()   # vacated land -> green
        if dest == "existing":                                      # into the existing town
            assert q["new_urban_cells"] == 0
        else:                                                       # new residential land on the hillside
            assert q["new_urban_cells"] > 0
            new = (d.landclass == cn.index("urban")) & (s.d.landclass != cn.index("urban"))
            assert np.isclose(a.households[new].sum(), hh0)


def test_relocation_to_painted_destination(session):
    s = session
    cn = s.class_names
    src = np.flatnonzero((s.d.landclass == cn.index("urban")).ravel() & (s.hand.ravel() < 5)).tolist()
    dest = np.zeros(s.d.shape, bool)
    dest[2:4, 30:34] = True                                         # forest hillside, 8 cells
    hh0 = s.assets.households.sum()
    d, a, *_rest, applied, _ret = s.apply_edits([{"type": "relocate_dest", "cells": np.flatnonzero(dest.ravel()).tolist()},
                                                  {"type": "relocate", "cells": src, "dest": "painted"}], {})
    assert len(applied) == 1 and applied[0]["q"]["dest_cells"] == 8
    assert np.allclose(a.households[dest], hh0 / 8)
    assert (d.landclass[dest] == cn.index("urban")).all()


def test_scenario_simulations_conserve_mass(session):
    s = session
    rain = Rainfall.uniform([0, 6 * 3600, 12 * 3600], [60.0, 0.0], s.d.shape)
    allc = list(range(s.d.shape[0] * s.d.shape[1]))
    for edits in ([], [{"type": "paddy_dam", "cells": allc}], [{"type": "forest", "cells": allc}],
                  [{"type": "levee", "cells": allc, "height": 3.0}]):
        d, *_ = s.apply_edits(edits, {"paddy_store_m": 0.15, "paddy_release_mmh": 5})
        res = Simulation(d, rain).run(12 * 3600, output_interval=3600)
        assert abs(res.mass.relative_error) < 1e-12


def test_retention_basin_cuts_downstream_peak_and_conserves_mass(session):
    s = session
    d0 = s.d
    rain = Rainfall.uniform([0, 3 * 3600, 12 * 3600], [80.0, 0.0], d0.shape)
    out = int(np.flatnonzero(d0.river.dn < 0)[0])
    base = Simulation(d0, rain).run(12 * 3600, gauges={"out": out}, output_interval=600)
    # site on the floodplain beside the lower channel, weir at 50 % of the inlet peak
    d0.river.area_km2 = base.max_river_q.copy()      # synthetic network has no areas: rank by discharge
    ro, co = d0.river.ri[out], d0.river.rj[out]
    site = np.zeros(d0.shape, bool)
    site[max(ro - 8, 0):ro + 1, co + 1:co + 4] = True
    site &= d0.mask
    edits = [{"type": "retention", "cells": np.flatnonzero(site.ravel()).tolist(), "depth": 3.0, "q_ratio": 0.5}]
    d, *_r, retention = s.apply_edits(edits, {}, baseline_qmax=base.max_river_q)
    dam = retention[0]["dam"]
    assert np.isclose(dam.q_cut, 0.5 * base.max_river_q[dam.k])
    res = Simulation(d, rain, dams=[dam]).run(12 * 3600, gauges={"out": out}, output_interval=600)
    assert res.max_river_q[dam.k] <= dam.q_cut + 1e-6                  # cut at the weir discharge
    assert res.gauge_q["out"].max() < 0.95 * base.gauge_q["out"].max()       # lower peak downstream
    noop = Dam(k=dam.k, capacity=1e12, q_cut=1e12, release_q=1e12)          # weir never overflows
    same = Simulation(d0, rain, dams=[noop]).run(12 * 3600, gauges={"out": out}, output_interval=600)
    assert np.allclose(same.gauge_q["out"], base.gauge_q["out"])            # no artefact of the structure
    assert abs(res.mass.relative_error) < 1e-12
