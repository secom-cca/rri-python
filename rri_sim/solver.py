"""Time integration of the coupled slope / river / infiltration system."""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

import numpy as np

from . import kernels as K
from .domain import Domain
from .forcing import Hydrograph, Rainfall


@dataclass
class MassBalance:
    rain: float = 0.0
    inflow: float = 0.0
    infiltration: float = 0.0
    river_outflow: float = 0.0
    boundary_outflow: float = 0.0
    storage0: float = 0.0
    storage: float = 0.0

    @property
    def error(self) -> float:
        return (self.storage0 + self.rain + self.inflow
                - self.infiltration - self.river_outflow - self.boundary_outflow
                - self.storage)

    @property
    def relative_error(self) -> float:
        scale = max(self.rain + self.inflow + self.storage0, 1e-12)
        return self.error / scale


@dataclass
class Results:
    times: np.ndarray
    gauge_q: dict
    gauge_h: dict
    max_depth: np.ndarray        # max surface water depth on slope [m]
    time_of_max: np.ndarray
    max_river_h: np.ndarray
    max_river_q: np.ndarray
    snapshots: list = field(default_factory=list)   # (t, surface depth, hr)
    mass: MassBalance = None
    n_steps: int = 0
    wall_time: float = 0.0
    outlet_q: np.ndarray = None
    dt_min_used: float = np.inf


class Simulation:
    def __init__(self, domain: Domain, rainfall: Rainfall | None = None,
                 inflows: dict[int, Hydrograph] | None = None,
                 h0=None, hr0=None):
        self.d = domain
        self.rain = rainfall
        self.inflows = inflows or {}
        ny, nx = domain.shape
        self._build_param_arrays()
        self.h = np.zeros((ny, nx)) if h0 is None else np.array(h0, float)
        self.h[~domain.mask] = 0.0
        nr = 0 if domain.river is None else domain.river.n
        self.hr = np.zeros(nr) if hr0 is None else np.array(hr0, float)
        self.F = np.zeros((ny, nx))
        self.t = 0.0
        self.fx = np.zeros((ny, nx - 1))
        self.fy = np.zeros((ny - 1, nx))
        self.fb = np.zeros((ny, nx))
        self.qsx = np.zeros((ny, nx - 1))   # surface unit-width flux (inertial state)
        self.qsy = np.zeros((ny - 1, nx))
        self.Q = np.zeros(nr)
        self.qex = np.zeros(nr)
        bnd = domain.open_boundary
        self.bnd_slope = np.zeros((ny, nx)) if bnd is None else np.asarray(bnd, float)

    # ------------------------------------------------------------ parameters
    def _build_param_arrays(self):
        d = self.d
        lc = d.landclass
        classes = d.params.land_classes

        def arr(attr):
            vals = np.array([getattr(c, attr) for c in classes], dtype=float)
            return vals[np.clip(lc, 0, len(classes) - 1)]

        self.n = arr("n")
        self.ka = arr("ka")
        self.da = arr("d_a")
        self.dm = arr("d_m")
        self.beta = arr("beta")
        self.kv = arr("kv")
        self.phi_theta = arr("phi_minus_theta")
        self.sf = arr("sf")
        self.inf_limit = arr("infilt_limit")
        self.z = np.where(d.mask, d.z, 0.0)
        self.mask = d.mask.astype(np.bool_)

    def storage(self) -> float:
        v = float(self.h[self.mask].sum()) * self.d.cell_area
        r = self.d.river
        if r is not None and r.n:
            v += float((self.hr * r.width * r.length).sum())
        return v

    def surface_depth(self):
        return np.where(self.mask, np.maximum(self.h - self.da, 0.0), np.nan)

    # ------------------------------------------------------------------ step
    def step(self, dt: float, mb: MassBalance) -> float:
        d, num = self.d, self.d.params.numerics
        r = d.river
        # rainfall (exact integral over the step), applied on slope storage
        if self.rain is not None:
            depth = self.rain.depth(self.t, self.t + dt)
            depth = np.where(self.mask, depth, 0.0)
            self.h += depth
            mb.rain += float(depth.sum()) * d.cell_area
        # vertical infiltration (Green-Ampt)
        inf = K.green_ampt(self.h, self.F, self.mask, self.da, self.kv, self.phi_theta,
                           self.sf, self.inf_limit, dt)
        mb.infiltration += inf * d.cell_area
        # 2D slope
        inertial = num.scheme == "inertial"
        c_s, dts_s = K.slope_fluxes(self.h, self.z, self.mask, self.n, self.ka, self.da,
                                    self.dm, self.beta, self.bnd_slope, d.dx, d.dy, dt,
                                    inertial, num.limiter_frac, num.h_dry,
                                    self.fx, self.fy, self.fb, self.qsx, self.qsy)
        mb.boundary_outflow += K.apply_slope_fluxes(self.h, self.mask, self.fx, self.fy,
                                                    self.fb, self.qsx, self.qsy,
                                                    d.dx, d.dy, dt)
        c_r, dts_r = 0.0, np.inf
        if r is not None and r.n:
            # external inflows to river cells
            for k, hyd in self.inflows.items():
                v = hyd.volume(self.t, self.t + dt)
                self.hr[k] += v / (r.width[k] * r.length[k])
                mb.inflow += v
            c_r, dts_r = K.river_fluxes(self.hr, r.zb, r.width, r.length, r.n_river, r.dn,
                                        r.outlet_slope, dt, inertial, num.limiter_frac,
                                        num.h_dry, self.Q)
            mb.river_outflow += K.apply_river_fluxes(self.hr, r.width, r.length, r.dn,
                                                     self.Q, dt)
            K.river_slope_exchange(self.h, self.z, self.da, r.ri, r.rj, self.hr, r.zb,
                                   r.width, r.length, r.embankment, d.dx, d.dy, dt,
                                   num.mu_complete, num.mu_submerged, num.bank_factor,
                                   self.qex)
        self.t += dt
        # next admissible dt
        dt_new = num.dt_max
        if c_s > 0:
            dt_new = min(dt_new, num.cfl * min(d.dx, d.dy) / c_s)
        if c_r > 0:
            dt_new = min(dt_new, num.cfl * float(r.length.min()) / c_r)
        if not inertial:
            dt_new = min(dt_new, num.diff_safety * min(dts_s, dts_r))
        return max(dt_new, num.dt_min)

    # ------------------------------------------------------------------- run
    def run(self, t_end: float, gauges: dict[str, int] | None = None,
            output_interval: float = 600.0, snapshot_interval: float | None = None,
            dt0: float = 1.0, verbose: bool = False) -> Results:
        gauges = gauges or {}
        d = self.d
        mb = MassBalance(storage0=self.storage())
        times = [self.t]
        gq = {g: [self.Q[k] if k >= 0 else 0.0] for g, k in gauges.items()}
        gh = {g: [self.hr[k]] for g, k in gauges.items()}
        maxd = np.zeros(d.shape)
        tmax = np.zeros(d.shape)
        nr = 0 if d.river is None else d.river.n
        maxhr = np.zeros(nr)
        maxq = np.zeros(nr)
        outlets = np.nonzero(d.river.dn < 0)[0] if nr else np.array([], int)
        outq = [0.0]
        snaps = []
        next_out = self.t + output_interval
        next_snap = self.t + snapshot_interval if snapshot_interval else np.inf
        dt = dt0
        nsteps = 0
        dtmin_used = np.inf
        wall0 = _time.perf_counter()
        while self.t < t_end - 1e-9:
            dt_step = min(dt, t_end - self.t, next_out - self.t, next_snap - self.t)
            dt_step = max(dt_step, 1e-6)
            dt = self.step(dt_step, mb)
            dtmin_used = min(dtmin_used, dt_step)
            nsteps += 1
            sd = np.maximum(self.h - self.da, 0.0)
            upd = sd > maxd
            maxd[upd] = sd[upd]
            tmax[upd] = self.t
            if nr:
                np.maximum(maxhr, self.hr, out=maxhr)
                np.maximum(maxq, np.abs(self.Q), out=maxq)
            if self.t >= next_out - 1e-9:
                times.append(self.t)
                for g, k in gauges.items():
                    gq[g].append(self.Q[k])
                    gh[g].append(self.hr[k])
                outq.append(float(self.Q[outlets].sum()) if nr else 0.0)
                next_out += output_interval
                if verbose:
                    print(f"t={self.t/3600:8.2f} h  dt={dt_step:7.3f}s  "
                          f"steps={nsteps}  Qout={outq[-1]:.2f}")
            if self.t >= next_snap - 1e-9:
                snaps.append((self.t, self.surface_depth(), self.hr.copy()))
                next_snap += snapshot_interval
        mb.storage = self.storage()
        maxd = np.where(d.mask, maxd, np.nan)
        return Results(times=np.array(times), gauge_q={g: np.array(v) for g, v in gq.items()},
                       gauge_h={g: np.array(v) for g, v in gh.items()}, max_depth=maxd,
                       time_of_max=tmax, max_river_h=maxhr, max_river_q=maxq, snapshots=snaps,
                       mass=mb, n_steps=nsteps, wall_time=_time.perf_counter() - wall0,
                       outlet_q=np.array(outq), dt_min_used=dtmin_used)
