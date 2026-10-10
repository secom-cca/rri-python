"""Model parameters.

Parameter naming follows the RRI model (Sayama et al., 2012; RRI Technical
Manual, PWRI Tech. Note 4277):

  n      Manning roughness of slope surface              [m^-1/3 s]
  ka     lateral saturated hydraulic conductivity        [m/s]
  soil_depth, gamma_a (effective porosity), gamma_m (unsaturated porosity)
         -> d_a = soil_depth * gamma_a ,  d_m = soil_depth * gamma_m  [m]
  beta   exponent of unsaturated lateral flow            [-]  (k_m = ka / beta)
  kv     vertical saturated hydraulic conductivity (Green-Ampt) [m/s]
  phi_minus_theta (porosity - initial moisture), sf (suction head) [m]
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class LandClassParams:
    name: str = "default"
    n: float = 0.4                # slope Manning n (RRI default for forest ~0.4)
    ka: float = 0.0               # lateral conductivity [m/s]; 0 disables subsurface
    soil_depth: float = 0.0       # [m]
    gamma_a: float = 0.0          # effective porosity
    gamma_m: float = 0.0          # unsaturated porosity (<= gamma_a)
    beta: float = 8.0
    kv: float = 0.0               # Green-Ampt vertical conductivity [m/s]; 0 disables
    phi_minus_theta: float = 0.3  # Green-Ampt (porosity - initial water content)
    sf: float = 0.1               # Green-Ampt suction at wetting front [m]
    infilt_limit: float = np.inf  # max cumulative infiltration depth [m]
    # surface storage not moved by overland flow (paddy ridge / 田んぼダム plate) [m]
    store_depth: float = 0.0
    # release rate of the retained water (orifice, linear reservoir) [1/s]
    leak_rate: float = 0.0

    @property
    def d_a(self) -> float:
        return self.soil_depth * self.gamma_a

    @property
    def d_m(self) -> float:
        return self.soil_depth * self.gamma_m

    def to_dict(self):
        return asdict(self)


@dataclass
class RiverGeometryParams:
    """Rectangular channel geometry from upstream area A [km^2] (RRI convention).

    W = cw * A**sw ,  D = cd * A**sd
    """
    cw: float = 5.0
    sw: float = 0.35
    cd: float = 0.95
    sd: float = 0.2
    n_river: float = 0.03
    embankment: float = 0.0       # default embankment height [m]
    min_upstream_area_km2: float = 10.0  # threshold for a cell to hold a channel

    def width(self, area_km2):
        return self.cw * np.asarray(area_km2, dtype=float) ** self.sw

    def depth(self, area_km2):
        return self.cd * np.asarray(area_km2, dtype=float) ** self.sd


@dataclass
class NumericsParams:
    scheme: str = "inertial"      # "inertial" (Bates 2010) or "diffusive" (RRI original)
    cfl: float = 0.7
    dt_max: float = 60.0          # [s]
    dt_min: float = 0.01          # [s]
    # diffusive scheme: dt <= diff_safety * min_faces(V_eq / |Q|)
    diff_safety: float = 0.4
    # diffusive scheme safety limiter: |Q| dt <= limiter_frac * V_eq (Hunter et al. 2005)
    limiter_frac: float = 0.5
    # Overflow (Honda / 本間) coefficients
    mu_complete: float = 0.35
    mu_submerged: float = 0.91
    bank_factor: float = 2.0      # overflow length = bank_factor * channel length (two banks)
    # levees have sluice gates (樋門): land-side water drains to the river when the
    # river is lower; river water cannot flow back below the levee crest
    sluice_gates: bool = True
    h_dry: float = 1e-6           # [m] depth below which a cell is treated dry for celerity


@dataclass
class ModelParams:
    land_classes: list[LandClassParams] = field(default_factory=lambda: [LandClassParams()])
    river: RiverGeometryParams = field(default_factory=RiverGeometryParams)
    numerics: NumericsParams = field(default_factory=NumericsParams)
