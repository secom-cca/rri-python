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


def build_river(z, zf, mask, fdir, acc_km2, dx, dy, rp, threshold_km2, embankment=None,
                bed_method="monotone", outlet_reach_m=5000.0):
    """bed_method: 'monotone' (lower downstream cells only) or 'isotonic'
    (least-squares monotone fit of the raw-DEM profile, see condition_river_bed)."""
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
    if bed_method == "isotonic":
        condition_river_bed(net, net.zb.copy(), min_slope=1e-4)
    enforce_monotone_bed(net, min_slope=1e-5)
    # outlet slope: mean bed slope over the last ~outlet_reach_m of the main stem
    for k in np.nonzero(net.dn < 0)[0]:
        path, dist = [k], 0.0
        while dist < outlet_reach_m:
            ups = np.nonzero(net.dn == path[-1])[0]
            if not len(ups):
                break
            u = ups[np.argmax(net.area_km2[ups])]
            dist += net.length[u]
            path.append(u)
        if dist > 0:
            net.outlet_slope[k] = max((net.zb[path[-1]] - net.zb[k]) / dist, 1e-4)
    return net


def isotonic_decreasing(y, w=None):
    """Least-squares non-increasing fit (pool-adjacent-violators)."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            ww, cc = wts[-2] + wts[-1], cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v], [ww], [cc]
    return np.repeat(vals, cnt)


def condition_river_bed(net: RiverNetwork, zb_raw: np.ndarray, min_slope: float = 1e-4):
    """Monotone bed from a noisy raw-DEM bed profile.

    Streams are processed in decreasing order of outlet area: for each
    headwater->already-conditioned-junction path, the raw profile is replaced
    by its least-squares non-increasing fit (balanced fill/carve, avoids the
    flat 'lakes' of a depression-filled DEM), bounded below by the junction
    elevation, then a minimum slope is imposed by carving."""
    n = net.n
    done = np.zeros(n, bool)
    zb = np.array(zb_raw, float)
    # headwaters sorted so that larger streams are conditioned first
    has_up = np.zeros(n, bool)
    has_up[net.dn[net.dn >= 0]] = True
    heads = np.nonzero(~has_up)[0]

    def path_from(h):
        p = [h]
        while net.dn[p[-1]] >= 0:
            p.append(net.dn[p[-1]])
        return p

    paths = [path_from(h) for h in heads]
    paths.sort(key=lambda p: -len(p))
    for p in paths:
        seg = []
        for k in p:
            if done[k]:
                break
            seg.append(k)
        if not seg:
            continue
        junction = p[len(seg)] if len(seg) < len(p) else None
        fit = isotonic_decreasing(zb[seg])
        if junction is not None:
            lo = zb[junction]
            # cumulative distance to junction for minimum-slope floor
            dist = np.cumsum(net.length[seg][::-1])[::-1]
            fit = np.maximum(fit, lo + min_slope * dist)
        # carve to minimum slope going downstream
        for a in range(1, len(seg)):
            fit[a] = min(fit[a], fit[a - 1] - min_slope * net.length[seg[a - 1]])
        zb[seg] = fit
        done[seg] = True
    dz = zb - net.zb
    net.zb[:] = zb
    return dz


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
