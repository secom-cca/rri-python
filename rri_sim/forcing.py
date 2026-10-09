"""Forcing: rainfall fields and boundary inflows.

Rainfall is treated as piecewise constant intensity over each interval
[times[k], times[k+1]).  `depth(t0, t1)` integrates exactly, so rain volume
entering the model is independent of the adaptive time step.
"""
from __future__ import annotations

import numpy as np


class Rainfall:
    def __init__(self, times_s, rates_mm_h, shape=None):
        """times_s: (nt+1,) interval edges [s]; rates_mm_h: (nt,) scalars or
        (nt, ny, nx) fields [mm/h].  Outside the range the rate is zero."""
        self.edges = np.asarray(times_s, dtype=float)
        r = np.asarray(rates_mm_h, dtype=float) / 1000.0 / 3600.0  # -> m/s
        if r.ndim == 1:
            if shape is None:
                raise ValueError("shape required for uniform rainfall")
            r = r[:, None, None] * np.ones((1,) + tuple(shape))
        if len(self.edges) != r.shape[0] + 1:
            raise ValueError("need len(times) == len(rates) + 1")
        self.rates = r
        # cumulative depth at each edge for exact integration
        dts = np.diff(self.edges)
        self.cum = np.concatenate([np.zeros((1,) + r.shape[1:]),
                                   np.cumsum(r * dts[:, None, None], axis=0)])

    @classmethod
    def uniform(cls, times_s, rates_mm_h, shape):
        return cls(times_s, rates_mm_h, shape)

    def _cum_at(self, t):
        e = self.edges
        if t <= e[0]:
            return self.cum[0]
        if t >= e[-1]:
            return self.cum[-1]
        k = np.searchsorted(e, t, side="right") - 1
        return self.cum[k] + self.rates[k] * (t - e[k])

    def depth(self, t0, t1):
        """Rain depth [m] falling in [t0, t1] per cell."""
        return self._cum_at(t1) - self._cum_at(t0)

    def next_change(self, t):
        """Next interval edge after t (used to align steps, optional)."""
        k = np.searchsorted(self.edges, t, side="right")
        return self.edges[k] if k < len(self.edges) else np.inf

    @property
    def end_time(self):
        return self.edges[-1]


class Hydrograph:
    """Piecewise-linear discharge time series [m^3/s] for a river inflow."""

    def __init__(self, times_s, q):
        self.t = np.asarray(times_s, float)
        self.q = np.asarray(q, float)

    def volume(self, t0, t1, nsub=4):
        ts = np.linspace(t0, t1, nsub + 1)
        qs = np.interp(ts, self.t, self.q, left=self.q[0], right=self.q[-1])
        return np.trapezoid(qs, ts)
