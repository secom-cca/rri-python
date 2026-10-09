"""Compare inertial vs diffusive (RRI original) closures on the leveed valley."""
import time
import numpy as np
from rri_sim.synthetic import valley_with_floodplain
from rri_sim.forcing import Hydrograph
from rri_sim.solver import Simulation
from rri_sim.params import NumericsParams

hyd = Hydrograph([0, 3600, 6 * 3600, 12 * 3600], [50, 1500, 1500, 50])
out = {}
for scheme in ["inertial", "diffusive"]:
    d = valley_with_floodplain(embankment=0.5, numerics=NumericsParams(scheme=scheme))
    s = Simulation(d, inflows={0: hyd})
    t = time.time()
    r = s.run(12 * 3600, output_interval=1800)
    out[scheme] = r
    plain = d.landclass == 1
    print(f"{scheme:9s} steps={r.n_steps:8d} wall={time.time()-t:7.1f}s "
          f"maxdepth_plain={np.nanmax(r.max_depth[plain]):.3f} "
          f"flooded_cells={(r.max_depth[plain] > 0.1).sum()} "
          f"Qout_peak={r.outlet_q.max():.1f} relerr={r.mass.relative_error:.2e}")
a, b = out["inertial"], out["diffusive"]
print("max |dQout|/Qpeak =", np.max(np.abs(a.outlet_q - b.outlet_q)) / b.outlet_q.max())
m = np.isfinite(a.max_depth)
print("RMSE max depth =", np.sqrt(np.mean((a.max_depth[m] - b.max_depth[m]) ** 2)))
