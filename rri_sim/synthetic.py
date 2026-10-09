"""Synthetic test basins used for verification."""
from __future__ import annotations

import numpy as np

from .domain import Domain, RiverNetwork
from .params import LandClassParams, ModelParams, RiverGeometryParams, NumericsParams


def tilted_plane(length=1000.0, width=100.0, dx=10.0, slope=0.01, n=0.05,
                 land: LandClassParams | None = None, numerics=None) -> Domain:
    """1D-like plane draining in +x direction to an open boundary at x = length."""
    nx = int(round(length / dx))
    ny = max(int(round(width / dx)), 1)
    x = (np.arange(nx) + 0.5) * dx
    z = np.tile(slope * (length - x), (ny, 1))
    land = land or LandClassParams(n=n)
    p = ModelParams(land_classes=[land], numerics=numerics or NumericsParams())
    d = Domain(z=z, mask=np.ones_like(z, bool), dx=dx, dy=dx,
               landclass=np.zeros_like(z, np.int64), params=p, river=None)
    bnd = np.zeros_like(z)
    bnd[:, -1] = slope
    d.open_boundary = bnd
    return d


def v_catchment(dx=20.0, plane_width=800.0, length=1000.0, s_lat=0.05, s_ch=0.02,
                n_plane=0.015, n_channel=0.15, channel_width=18.0, channel_depth=1.0,
                numerics=None) -> Domain:
    """Classic V-catchment benchmark (di Giammarco et al., 1996; Kollet et al., 2017).

    Two planes (800 m x 1000 m, lateral slope 0.05) drain into a central
    channel (slope 0.02).  Standard rainfall: 3e-6 m/s (10.8 mm/h) for 90 min.
    """
    nside = int(round(plane_width / dx))
    nx = 2 * nside + 1
    ny = int(round(length / dx))
    jc = nside
    x = (np.arange(nx) - jc) * dx
    y = (np.arange(ny) + 0.5) * dx          # flow toward increasing y (row index)
    X, Y = np.meshgrid(x, y)
    z = s_lat * np.abs(X) + s_ch * (length - Y)
    p = ModelParams(land_classes=[LandClassParams(n=n_plane)],
                    river=RiverGeometryParams(n_river=n_channel),
                    numerics=numerics or NumericsParams())
    ri = np.arange(ny)
    rj = np.full(ny, jc)
    dn = np.append(np.arange(1, ny), -1)
    zs = z[ri, rj]
    river = RiverNetwork(ri=ri, rj=rj, dn=dn, length=np.full(ny, dx),
                         width=np.full(ny, channel_width), depth=np.full(ny, channel_depth),
                         zb=zs - channel_depth, embankment=np.zeros(ny),
                         n_river=np.full(ny, n_channel), area_km2=np.zeros(ny),
                         outlet_slope=np.full(ny, s_ch))
    return Domain(z=z, mask=np.ones_like(z, bool), dx=dx, dy=dx,
                  landclass=np.zeros_like(z, np.int64), params=p, river=river)


def valley_with_floodplain(dx=100.0, nx=41, ny=60, valley_slope=0.001, side_slope=0.02,
                           floodplain_halfwidth=8, channel_width=40.0, channel_depth=3.0,
                           embankment=2.0, n_slope=0.06, n_channel=0.035,
                           hill_land: LandClassParams | None = None,
                           plain_land: LandClassParams | None = None,
                           numerics=None) -> Domain:
    """Synthetic river basin: hillslopes -> flat floodplain -> leveed channel.

    Land class 0 = hillslope (forest, with subsurface flow), 1 = floodplain
    (urban/paddy, Green-Ampt).  River runs along the centre column toward
    increasing row index and leaves at the bottom edge.
    """
    jc = nx // 2
    x = (np.arange(nx) - jc) * dx
    y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    dist = np.maximum(np.abs(X) - floodplain_halfwidth * dx, 0.0)
    z = valley_slope * (ny * dx - Y) + side_slope * dist + 0.002 * np.abs(X)
    lc = np.where(np.abs(X) > floodplain_halfwidth * dx, 0, 1).astype(np.int64)
    hill = hill_land or LandClassParams(name="forest", n=0.4, ka=0.1, soil_depth=1.0,
                                        gamma_a=0.4, gamma_m=0.1, beta=8.0)
    plain = plain_land or LandClassParams(name="plain", n=n_slope, kv=1e-6,
                                          phi_minus_theta=0.3, sf=0.1)
    p = ModelParams(land_classes=[hill, plain],
                    river=RiverGeometryParams(n_river=n_channel),
                    numerics=numerics or NumericsParams())
    ri = np.arange(ny)
    rj = np.full(ny, jc)
    dn = np.append(np.arange(1, ny), -1)
    zs = z[ri, rj]
    river = RiverNetwork(ri=ri, rj=rj, dn=dn, length=np.full(ny, dx),
                         width=np.full(ny, channel_width), depth=np.full(ny, channel_depth),
                         zb=zs - channel_depth, embankment=np.full(ny, embankment),
                         n_river=np.full(ny, n_channel), area_km2=np.zeros(ny),
                         outlet_slope=np.full(ny, valley_slope))
    return Domain(z=z, mask=np.ones_like(z, bool), dx=dx, dy=dx, landclass=lc,
                  params=p, river=river)
