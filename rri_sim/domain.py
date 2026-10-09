"""Computational domain: 2D slope grid + 1D river network embedded in it."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hydro import (D8_DI, D8_DJ, d8_directions, fill_depressions,
                    flow_accumulation, topological_order)
from .params import ModelParams


@dataclass
class RiverNetwork:
    ri: np.ndarray          # grid row of each river cell
    rj: np.ndarray          # grid col
    dn: np.ndarray          # index of downstream river cell, -1 for outlet
    length: np.ndarray      # channel length represented by the cell [m]
    width: np.ndarray       # [m]
    depth: np.ndarray       # bankfull depth below slope ground [m]
    zb: np.ndarray          # bed elevation [m]
    embankment: np.ndarray  # levee height above slope ground [m]
    n_river: np.ndarray     # Manning n
    area_km2: np.ndarray    # upstream area
    outlet_slope: np.ndarray  # bed slope used for the normal-depth outlet BC
    order: np.ndarray = field(default=None)  # upstream -> downstream order

    def __post_init__(self):
        if self.order is None:
            self.order = topological_order(self.dn)

    @property
    def n(self) -> int:
        return len(self.ri)

    def copy(self) -> "RiverNetwork":
        return RiverNetwork(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                               for k, v in self.__dict__.items()})

    def index_of(self, i: int, j: int) -> int:
        hit = np.nonzero((self.ri == i) & (self.rj == j))[0]
        if len(hit) == 0:
            raise KeyError(f"no river cell at ({i},{j})")
        return int(hit[0])


@dataclass
class Domain:
    z: np.ndarray               # ground elevation [m] (ny, nx)
    mask: np.ndarray            # True inside basin
    dx: float
    dy: float
    landclass: np.ndarray       # int index into params.land_classes
    params: ModelParams
    river: RiverNetwork | None = None
    # optional georeferencing (affine transform tuple, CRS string)
    transform: tuple | None = None
    crs: str | None = None
    # boundary: slope cells that may discharge out of the domain (normal depth)
    open_boundary: np.ndarray | None = None

    @property
    def shape(self):
        return self.z.shape

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    def copy(self) -> "Domain":
        import copy
        return Domain(z=self.z.copy(), mask=self.mask.copy(), dx=self.dx, dy=self.dy,
                      landclass=self.landclass.copy(), params=copy.deepcopy(self.params),
                      river=None if self.river is None else self.river.copy(),
                      transform=self.transform, crs=self.crs,
                      open_boundary=None if self.open_boundary is None else self.open_boundary.copy())

    # ------------------------------------------------------------------ build
    @classmethod
    def from_dem(cls, z, dx, dy, params: ModelParams, mask=None, landclass=None,
                 river_threshold_km2=None, embankment=None, transform=None, crs=None,
                 fdir=None):
        """Build domain and DEM-derived river network (RRI-style W/D from area)."""
        z = np.asarray(z, dtype=float)
        if mask is None:
            mask = np.isfinite(z)
        z = np.where(mask, z, np.nan)
        if landclass is None:
            landclass = np.zeros(z.shape, dtype=np.int64)
        rp = params.river
        thr = rp.min_upstream_area_km2 if river_threshold_km2 is None else river_threshold_km2
        if fdir is None:
            zf = fill_depressions(np.where(mask, z, 0.0), mask)
            fdir = d8_directions(zf, mask, dx, dy)
        else:
            zf = np.where(mask, z, 0.0)
        acc = flow_accumulation(fdir, mask, dx * dy) / 1e6
        river = build_river(z, zf, mask, fdir, acc, dx, dy, rp, thr, embankment)
        return cls(z=z, mask=mask, dx=dx, dy=dy, landclass=np.asarray(landclass, dtype=np.int64),
                   params=params, river=river, transform=transform, crs=crs)


def build_river(z, zf, mask, fdir, acc_km2, dx, dy, rp, threshold_km2, embankment=None):
    is_riv = mask & (acc_km2 >= threshold_km2)
    ri, rj = np.nonzero(is_riv)
    idx = -np.ones(z.shape, dtype=np.int64)
    idx[ri, rj] = np.arange(len(ri))
    n = len(ri)
    dn = -np.ones(n, dtype=np.int64)
    length = np.empty(n)
    outlet_slope = np.full(n, 1e-3)
    for k in range(n):
        i, j = ri[k], rj[k]
        d = fdir[i, j]
        if d >= 0:
            ii, jj = i + D8_DI[d], j + D8_DJ[d]
            length[k] = np.hypot(D8_DI[d] * dy, D8_DJ[d] * dx)
            if is_riv[ii, jj]:
                dn[k] = idx[ii, jj]
        else:
            length[k] = dx
    area = acc_km2[ri, rj]
    width = rp.width(area)
    depth = rp.depth(area)
    # channel cannot be wider than the cell
    width = np.minimum(width, 0.9 * min(dx, dy))
    zs = z[ri, rj]
    zb = zs - depth
    net = RiverNetwork(ri=ri, rj=rj, dn=dn, length=length, width=width, depth=depth,
                       zb=zb, embankment=np.full(n, rp.embankment) if embankment is None
                       else np.asarray(embankment, float)[ri, rj],
                       n_river=np.full(n, rp.n_river), area_km2=area,
                       outlet_slope=outlet_slope)
    enforce_monotone_bed(net, min_slope=1e-5)
    # outlet slope from the last reach
    for k in np.nonzero(net.dn < 0)[0]:
        ups = np.nonzero(net.dn == k)[0]
        if len(ups):
            s = (net.zb[ups].max() - net.zb[k]) / net.length[ups].mean()
            net.outlet_slope[k] = max(s, 1e-4)
    return net


def enforce_monotone_bed(net: RiverNetwork, min_slope: float = 0.0):
    """Lower bed elevations so that they never rise in the downstream direction.

    DEM noise (bridges, raster artefacts) otherwise produces spurious weirs in
    the channel.  Channel depth below slope ground is increased accordingly.
    """
    for k in net.order:
        d = net.dn[k]
        if d >= 0:
            zmax = net.zb[k] - min_slope * net.length[k]
            if net.zb[d] > zmax:
                net.depth[d] += net.zb[d] - zmax
                net.zb[d] = zmax
