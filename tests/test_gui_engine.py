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
            assert np.allclose(d.river.embankment, d0.river.embankment + 4.0)   # + on the existing levee
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
                  [{"type": "forest_mgmt", "cells": allc}],
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


def test_estimate_costs_each_edit_without_simulating(session):
    s = session
    s.b, s._baseline = {}, {}
    d0 = s.d
    emb0 = d0.river.embankment.copy()
    allc = list(range(d0.shape[0] * d0.shape[1]))
    edits = [{"type": "levee", "cells": allc, "height": 1.0}, {"type": "paddy_dam", "cells": allc},
             {"type": "levee", "cells": [], "height": 1.0}]
    est = s.estimate("ev", edits, {})
    per = est["edit_cost_yen"]
    assert len(per) == 3 and per[0] > 0 and per[1] > 0 and per[2] == 0
    assert np.isclose(sum(per), est["cost"]["pv_yen"])
    assert np.allclose(s.d.river.embankment, emb0)       # the session's basin is untouched


def test_inland_flood_measures_drainage_storage_and_dam_prerelease(session):
    """排水ポンプ（低地の市街地のみ）・雨水貯留浸透（市街地のみ）・事前放流（ダム容量）and their costs."""
    from rri_sim import costs as C
    s = session
    cn = s.class_names
    allc = list(range(s.d.shape[0] * s.d.shape[1]))
    urban_low = (s.d.landclass == cn.index("urban")) & (s.hand <= 5.0) & s.d.mask
    d, *_, applied, _ = s.apply_edits([{"type": "pump", "cells": allc, "target": "urban", "mmh": 20.0}], {})
    assert (d.drain_add[urban_low] > 0).all() and (d.drain_add[~urban_low] == 0).all()
    assert np.isclose(applied[0]["q"]["capacity_m3s"], urban_low.sum() * s.d.cell_area * 20 / 1000 / 3600)
    d, *_, applied2, _ = s.apply_edits([{"type": "storage", "cells": allc, "depth_mm": 30.0, "kind": "private"}], {})
    urban = (s.d.landclass == cn.index("urban")) & s.d.mask
    assert np.allclose(d.store_add[urban], 0.03) and (d.store_add[~urban] == 0).all() and (d.store_leak[urban] > 0).all()
    d, *_, applied3, _ = s.apply_edits([{"type": "prerelease", "fraction": 0.5}], {})
    assert d.dam_factor == 1.5
    c = C.estimate([{"type": a["type"], "q": a["q"]} for a in applied + applied2])
    assert c["items"][0]["initial_yen"] > 0 and c["items"][1]["initial_yen"] > 0 and c["items"][0]["annual_yen"] > 0


def test_forest_management_improves_soil_storage_of_existing_forest(session):
    from rri_sim import costs as C
    s = session
    d0, cn = s.d, s.class_names
    allc = list(range(d0.shape[0] * d0.shape[1]))
    fo = d0.params.land_classes[cn.index("forest")]
    d, *_r, applied, _ret = s.apply_edits([{"type": "forest_mgmt", "cells": allc, "owner": "national"}], {"forest_mgmt_da_gain": 0.5})
    is_fo = d0.landclass == cn.index("forest")
    assert (d.landclass[is_fo] == cn.index("forest_managed")).all() and (d.landclass[~is_fo] == d0.landclass[~is_fo]).all()
    fm = d.params.land_classes[cn.index("forest_managed")]
    assert np.isclose(fm.d_a, 1.5 * fo.d_a) and np.isclose(fm.d_m, 1.5 * fo.d_m)
    assert fm.ka == fo.ka and fm.soil_depth == fo.soil_depth and fm.n == fo.n         # calibrated forest otherwise
    q = applied[0]["q"]
    assert q["owner"] == "national" and np.isclose(q["area_m2"], is_fo.sum() * d0.cell_area)
    c = C.estimate(applied)
    assert c["initial_yen"] > 0 and c["annual_yen"] > 0
    d, *_ = s.apply_edits([{"type": "forest_mgmt", "cells": allc}], {"forest_mgmt_da_gain": 0.0})   # effect 0
    assert np.isclose(d.params.land_classes[cn.index("forest_managed")].d_a, fo.d_a)


def test_candidates_split_housing_and_forest_management_by_owner(session):
    from rri_sim.rdm import candidates as CA
    from rri_sim.rdm import config as CF
    s = session
    d, cn = s.d, s.class_names
    meta = {"gauge_index": {}, "fdir": None}
    depth = np.where(s.assets.households > 0, 1.5, 0.0)
    nat = np.zeros(d.shape, bool)
    nat[:, :10] = True                                                  # part of the hillside is national forest
    cfg = CF.load({"zones": [["全域", None]], "candidates": {"forest_mgmt": {"split_owner": True}}})
    V = CA.build(d, meta, s.hand, s.assets, cn, depth, cfg, national_forest=nat)
    by = {v.type: v for v in V}
    assert {"relocate", "raise", "forest_mgmt", "forest_mgmt_national"} <= set(by)
    assert by["relocate"].options == ["なし", "移転"] and by["raise"].options == ["なし", "+1 m"]
    assert [v.type for v in V].index("relocate") < [v.type for v in V].index("raise")
    fo = d.landclass.ravel() == cn.index("forest")
    assert set(by["forest_mgmt_national"].cells) == set(np.flatnonzero(fo & nat.ravel()))
    assert set(by["forest_mgmt"].cells) == set(np.flatnonzero(fo & ~nat.ravel()))
    assert CA.superseded_raise(V) == [([v.type for v in V].index("relocate"), [v.type for v in V].index("raise"))]
    # both active: homes are relocated, raising them costs nothing
    i, j = CA.superseded_raise(V)[0]
    x = [0] * len(V)
    x[i] = x[j] = 1
    *_r, applied, _ret = s.apply_edits(CA.edits(V, x), {})
    raise_q = next(a["q"] for a in applied if a["type"] == "raise")
    assert raise_q["households"] == 0
    # stored configs without the housing key keep the combined variable (roadmap / DAPP)
    old = CF.load({"zones": [["全域", None]]})
    old["candidates"].pop("housing")
    old["candidates"]["forest_mgmt"] = None
    V0 = CA.build(d, meta, s.hand, s.assets, cn, depth, old)
    assert [v.options for v in V0 if v.type == "relocate"] == [["なし", "嵩上げ +1.0 m", "移転"]]
    assert not any(v.type.startswith("forest_mgmt") or v.type == "raise" for v in V0)
