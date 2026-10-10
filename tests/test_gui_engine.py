"""Scenario edits of the GUI engine on a synthetic basin (no downloads)."""
import numpy as np
import pytest

from rri_sim import damage as D
from rri_sim.forcing import Rainfall
from rri_sim.gui.engine import BasinSession
from rri_sim.params import ModelParams
from rri_sim.solver import Simulation
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
    return s


def flat(d, mask):
    return np.flatnonzero(mask.ravel()).tolist()


def test_each_edit_applies_only_where_meaningful(session):
    s = session
    d0 = s.d
    cn = s.class_names
    allc = list(range(d0.shape[0] * d0.shape[1]))
    for typ, extra in [("forest", {}), ("paddy_dam", {}), ("relocate", {}), ("raise", {"height": 1.0}),
                       ("levee", {"height": 4.0}), ("retention", {"depth": 2.0, "inlet": 1.0})]:
        cells = allc if typ != "retention" else flat(d0, np.isin(np.arange(d0.shape[1]), [21, 22])[None, :]
                                                    & (np.arange(d0.shape[0])[:, None] < 10))
        d, a, floor, notes, storage = s.apply_edits([{"type": typ, "cells": cells, **extra}], {})
        if typ == "forest":
            assert (d.landclass[d0.landclass == cn.index("forest")] == cn.index("forest")).all()
            assert (d.landclass[d0.landclass == cn.index("paddy")] == cn.index("forest_new")).all()
            assert a.paddy_area_m2.sum() == 0
        if typ == "paddy_dam":
            assert (d.landclass[d0.landclass == cn.index("paddy")] == cn.index("paddy_dam")).all()
            assert (d.landclass[d0.landclass == cn.index("urban")] == cn.index("urban")).all()
        if typ == "relocate":
            assert a.households.sum() == 0 and a.floor_area_m2.sum() == 0
        if typ == "raise":
            assert np.allclose(floor, 1.45)
        if typ == "levee":
            assert np.allclose(d.river.embankment, 4.0)
        if typ == "retention":
            ny, nx = d0.shape
            riv = np.zeros(d0.shape, bool)
            riv[d0.river.ri, d0.river.rj] = True
            assert storage.sum() == 20
            assert (d.z[storage & ~riv] == d0.z[storage & ~riv] - 2.0).all()   # excavated
            assert (d.z[riv] == d0.z[riv]).all()                               # channel untouched
            assert a.households[storage].sum() == 0                            # land acquired
            assert (d.river.embankment[(d0.river.ri < 10)] == 1.0).all()        # overflow weir
        # original domain untouched
        assert np.allclose(s.d.river.embankment, 0.5)


def test_scenario_simulations_conserve_mass(session):
    s = session
    rain = Rainfall.uniform([0, 6 * 3600, 12 * 3600], [60.0, 0.0], s.d.shape)
    allc = list(range(s.d.shape[0] * s.d.shape[1]))
    for edits in ([], [{"type": "paddy_dam", "cells": allc}], [{"type": "forest", "cells": allc}],
                  [{"type": "levee", "cells": allc, "height": 3.0}]):
        d, *_ = s.apply_edits(edits, {"paddy_store_m": 0.15, "paddy_release_mmh": 5})
        res = Simulation(d, rain).run(12 * 3600, output_interval=3600)
        assert abs(res.mass.relative_error) < 1e-12
