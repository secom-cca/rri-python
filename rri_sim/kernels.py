"""Numba kernels for the RRI-type governing equations.

Finite-volume, explicit, operator-split.  All fluxes are face fluxes shared by
both neighbouring cells, so each operator conserves mass to round-off.

Two momentum closures are available for overland and channel flow:

* ``diffusive`` (RRI original): Manning flux from the water-surface gradient.
  Explicit stability needs dt <= V_eq / |Q| on every face (V_eq = volume that
  equalises the two levels), i.e. dt -> 0 on flat terrain.  The solver picks
  dt from this criterion; a Hunter et al. (2005) level limiter is kept as a
  safety net only.
* ``inertial`` (default): local inertial approximation (Bates et al., 2010)
  with fully implicit friction (closed-form quadratic solution).  Its steady state is identical to the
  diffusive-wave Manning flux, but it is stable under a gravity-wave CFL
  condition independent of the slope.

The lateral subsurface (Darcy-type) part of the RRI discharge-stage relation
is always treated diffusively.  A positivity limiter scales each cell's total
outflow so that it never exceeds the stored volume (depth >= 0 guaranteed).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

G = 9.80665


# ---------------------------------------------------------------- face physics
@njit(cache=True, inline="always")
def subsurface_discharge(h, s, ka, da, dm, beta):
    """Lateral subsurface part of the RRI discharge-stage relation [m^2/s].

    (Sayama et al. 2012; RRI manual), k_m = k_a / beta:
      h <= d_m        : q = k_m d_m (h/d_m)^beta * s          (unsaturated)
      d_m < h         : q = (k_m d_m + k_a (min(h,d_a) - d_m)) * s
    """
    if ka <= 0.0 or da <= 0.0 or h <= 0.0 or s <= 0.0:
        return 0.0
    km = ka / beta
    if dm > 0.0:
        if h <= dm:
            return km * dm * (h / dm) ** beta * s
        return (km * dm + ka * (min(h, da) - dm)) * s
    return ka * min(h, da) * s


@njit(cache=True, inline="always")
def surface_depth_of(h, ka, da):
    if ka > 0.0 and da > 0.0:
        return max(h - da, 0.0)
    return h


@njit(cache=True, inline="always")
def slope_discharge(h, s, n, ka, da, dm, beta):
    """Full RRI discharge-stage relation (diffusive wave), magnitude [m^2/s]:
    subsurface part + (1/n) (h - d_a)^{5/3} s^{1/2} overland part."""
    q = subsurface_discharge(h, s, ka, da, dm, beta)
    hs = surface_depth_of(h, ka, da)
    if hs > 0.0 and s > 0.0:
        q += hs ** (5.0 / 3.0) * math.sqrt(s) / n
    return q


@njit(cache=True, inline="always")
def implicit_friction(rhs, a):
    """Solve x + a|x|x = rhs (fully implicit Manning friction).

    The map q_old -> x has derivative 1/(1 + 2a|x|) in (0, 1), so the update
    converges monotonically to the steady (Manning) flux for any dt; the
    semi-implicit form of Bates et al. (2010) oscillates when a|q| >> 1."""
    if a <= 0.0:
        return rhs
    m = abs(rhs)
    x = 2.0 * m / (1.0 + math.sqrt(1.0 + 4.0 * a * m))
    return x if rhs >= 0.0 else -x


@njit(cache=True, inline="always")
def slope_face(Ha, Hb, za, zb, ha, hb, na, nb, kaa, kab, daa, dab, dma, dmb, ba, bb,
               sda, sdb, dist, width, area, dt, inertial, q_old, lim_frac, h_dry):
    """Discharge through one slope face, a -> b positive.

    Returns (Q_total [m^3/s], q_surface [m^2/s, signed], celerity, dt_stab).

    diffusive : q = sign * RRI discharge-stage relation       (RRI original)
    inertial  : surface part from the local inertial equation (Bates et al.
                2010) with fully implicit friction
                  q+ + (g n^2 dt / h^{7/3}) |q+| q+ = q + g h dt S
                whose steady state is exactly Manning's diffusive-wave flux;
                the subsurface (Darcy) part stays diffusive.
    """
    dH = Ha - Hb
    if dH > 0.0 or (dH == 0.0 and q_old > 0.0):
        hd, nd, kad, dad, dmd, bd, sdd = ha, na, kaa, daa, dma, ba, sda
    elif dH < 0.0 or q_old < 0.0:
        hd, nd, kad, dad, dmd, bd, sdd = hb, nb, kab, dab, dmb, bb, sdb
    else:
        return 0.0, 0.0, 0.0, np.inf
    hf = max(Ha, Hb) - max(za, zb)
    if hf > hd:
        hf = hd
    if hf <= 0.0:
        return 0.0, 0.0, 0.0, np.inf
    s = abs(dH) / dist
    sgn = 1.0 if dH > 0.0 else -1.0
    qsub = subsurface_discharge(hf, s, kad, dad, dmd, bd)
    # surface water below the donor's store depth (paddy ridge / outlet plate,
    # 田んぼダム) is retained and not moved by surface flow
    hs = surface_depth_of(hf, kad, dad) - sdd
    qs = 0.0
    c = 0.0
    if hs > h_dry:
        if inertial:
            qs = implicit_friction(q_old + G * hs * dt * dH / dist,
                                   G * nd * nd * dt / hs ** (7.0 / 3.0))
        else:
            qs = sgn * hs ** (5.0 / 3.0) * math.sqrt(s) / nd
        u = abs(qs) / hs
        c = max(u + math.sqrt(G * hs), 5.0 / 3.0 * u)
    Q = (qs + sgn * qsub) * width
    dts = np.inf
    if Q != 0.0 and dH != 0.0:
        veq = 0.5 * area * abs(dH)           # volume that equalises both levels
        dts = veq / abs(Q)
        if not inertial:
            Qmax = lim_frac * veq / dt       # safety level limiter (Hunter 2005)
            if abs(Q) > Qmax:
                r = Qmax / abs(Q)
                Q *= r
                qs *= r
    return Q, qs, c, dts


def _slope_fluxes_py(h, z, mask, n, ka, da, dm, beta, sd, bnd_slope, dx, dy, dt, inertial,
                 lim_frac, h_dry, fx, fy, fb, qsx, qsy):
    """Face discharges [m^3/s]: fx east faces (ny, nx-1), fy south faces
    (ny-1, nx), fb open-boundary outflow (ny, nx).  qsx/qsy hold the surface
    unit-width flux (state of the inertial scheme).  Rows are processed in
    parallel; every face is written by exactly one thread.
    Returns (max celerity, min diffusive stability time)."""
    ny, nx = h.shape
    area = dx * dy
    row_c = np.zeros(ny)
    row_t = np.full(ny, np.inf)
    for i in prange(ny):
        cmax = 0.0
        dtsmin = np.inf
        for j in range(nx - 1):
            fx[i, j] = 0.0
            if not (mask[i, j] and mask[i, j + 1]):
                qsx[i, j] = 0.0
                continue
            Q, qs, c, dts = slope_face(
                z[i, j] + h[i, j], z[i, j + 1] + h[i, j + 1], z[i, j], z[i, j + 1],
                h[i, j], h[i, j + 1], n[i, j], n[i, j + 1], ka[i, j], ka[i, j + 1],
                da[i, j], da[i, j + 1], dm[i, j], dm[i, j + 1], beta[i, j], beta[i, j + 1],
                sd[i, j], sd[i, j + 1], dx, dy, area, dt, inertial, qsx[i, j], lim_frac, h_dry)
            fx[i, j] = Q
            qsx[i, j] = qs
            cmax = max(cmax, c)
            dtsmin = min(dtsmin, dts)
        if i < ny - 1:
            for j in range(nx):
                fy[i, j] = 0.0
                if not (mask[i, j] and mask[i + 1, j]):
                    qsy[i, j] = 0.0
                    continue
                Q, qs, c, dts = slope_face(
                    z[i, j] + h[i, j], z[i + 1, j] + h[i + 1, j], z[i, j], z[i + 1, j],
                    h[i, j], h[i + 1, j], n[i, j], n[i + 1, j], ka[i, j], ka[i + 1, j],
                    da[i, j], da[i + 1, j], dm[i, j], dm[i + 1, j], beta[i, j], beta[i + 1, j],
                    sd[i, j], sd[i + 1, j], dy, dx, area, dt, inertial, qsy[i, j], lim_frac, h_dry)
                fy[i, j] = Q
                qsy[i, j] = qs
                cmax = max(cmax, c)
                dtsmin = min(dtsmin, dts)
        for j in range(nx):
            fb[i, j] = 0.0
            if mask[i, j] and bnd_slope[i, j] > 0.0 and h[i, j] > 0.0:
                fb[i, j] = slope_discharge(h[i, j], bnd_slope[i, j], n[i, j], ka[i, j],
                                           da[i, j], dm[i, j], beta[i, j]) * dx
        row_c[i] = cmax
        row_t[i] = dtsmin
    return row_c.max(), row_t.min()


def _apply_slope_fluxes_py(h, mask, fx, fy, fb, qsx, qsy, dx, dy, dt):
    """Positivity-limited, conservative update of slope storage (gather form,
    parallel over rows).  Returns outflow volume through the open boundary [m^3]."""
    ny, nx = h.shape
    area = dx * dy
    r = np.ones((ny, nx))
    # 1) outflow volume of each cell from its own faces, scale factor
    for i in prange(ny):
        for j in range(nx):
            o = fb[i, j]
            if j < nx - 1 and fx[i, j] > 0:
                o += fx[i, j]
            if j > 0 and fx[i, j - 1] < 0:
                o -= fx[i, j - 1]
            if i < ny - 1 and fy[i, j] > 0:
                o += fy[i, j]
            if i > 0 and fy[i - 1, j] < 0:
                o -= fy[i - 1, j]
            o *= dt
            if o > 0.0:
                avail = h[i, j] * area
                if o > avail:
                    r[i, j] = avail / o
    # 2) scale each face by its donor's factor
    for i in prange(ny):
        for j in range(nx - 1):
            f = r[i, j] if fx[i, j] > 0 else r[i, j + 1]
            fx[i, j] *= f
            qsx[i, j] *= f
        if i < ny - 1:
            for j in range(nx):
                f = r[i, j] if fy[i, j] > 0 else r[i + 1, j]
                fy[i, j] *= f
                qsy[i, j] *= f
        for j in range(nx):
            fb[i, j] *= r[i, j]
    # 3) gather net volume change
    row_out = np.zeros(ny)
    for i in prange(ny):
        vo = 0.0
        for j in range(nx):
            if not mask[i, j]:
                continue
            dV = -fb[i, j]
            vo += fb[i, j]
            if j < nx - 1:
                dV -= fx[i, j]
            if j > 0:
                dV += fx[i, j - 1]
            if i < ny - 1:
                dV -= fy[i, j]
            if i > 0:
                dV += fy[i - 1, j]
            h[i, j] = max(h[i, j] + dV * dt / area, 0.0)
        row_out[i] = vo * dt
    return row_out.sum()


# ---------------------------------------------------------------- river physics
@njit(cache=True, nogil=True)
def river_fluxes(hr, zb, width, length, nr, dn, outlet_slope, dt, inertial, lim_frac,
                 h_dry, Q, bc_stage):
    """Discharge from each river cell to its downstream cell [m^3/s]
    (negative = backwater flow upstream), rectangular section.
      diffusive : Q = sign (1/n) A R^{2/3} |dH/dx|^{1/2}            (RRI)
      inertial  : Q+ + (g n^2 dt / (A R^{4/3})) |Q+| Q+ = Q + g A dt dH/dx
    Q holds the previous discharge on entry (inertial state).
    Outlet cells: normal depth with outlet_slope, or - if bc_stage[k] is
    finite - a prescribed downstream water level (tide) in a ghost cell one
    reach downstream; flow may then reverse (sea -> river).
    Returns (max celerity, min diffusive stability time)."""
    nriv = hr.shape[0]
    cmax = 0.0
    dtsmin = np.inf
    for k in range(nriv):
        q_old = Q[k]
        Q[k] = 0.0
        d = dn[k]
        if d >= 0:
            Ha = zb[k] + hr[k]
            Hb = zb[d] + hr[d]
            dH = Ha - Hb
            if dH > 0.0 or (dH == 0.0 and q_old > 0.0):
                a = k
            elif dH < 0.0 or q_old < 0.0:
                a = d
            else:
                continue
            hf = max(Ha, Hb) - max(zb[k], zb[d])
            hf = min(hf, hr[a])
            if hf <= h_dry:
                continue
            dist = 0.5 * (length[k] + length[d])
            w = width[a]
            A = w * hf
            R = A / (w + 2.0 * hf)
            if inertial:
                q = implicit_friction(q_old + G * A * dt * dH / dist,
                                      G * nr[a] ** 2 * dt / (A * R ** (4.0 / 3.0)))
            else:
                q = A * R ** (2.0 / 3.0) * math.sqrt(abs(dH) / dist) / nr[a]
                if dH < 0.0:
                    q = -q
            Aa = width[k] * length[k]
            Ab = width[d] * length[d]
            if q != 0.0 and dH != 0.0:
                veq = abs(dH) * Aa * Ab / (Aa + Ab)
                dtsmin = min(dtsmin, veq / abs(q))
                if not inertial:
                    Qmax = lim_frac * veq / dt
                    if abs(q) > Qmax:
                        q = Qmax if q > 0 else -Qmax
            u = abs(q) / A
            cmax = max(cmax, u + math.sqrt(G * hf), 5.0 / 3.0 * u)
            Q[k] = q
        elif np.isfinite(bc_stage[k]):
            # tidal / stage boundary: ghost cell one reach downstream
            Ha = zb[k] + hr[k]
            zg = zb[k] - outlet_slope[k] * length[k]
            Hb = max(bc_stage[k], zg)
            dH = Ha - Hb
            hf = max(Ha, Hb) - zb[k]
            if dH < 0.0:
                hf = min(hf, Hb - zg)
            else:
                hf = min(hf, hr[k])
            if hf <= h_dry:
                continue
            w = width[k]
            A = w * hf
            R = A / (w + 2.0 * hf)
            q = implicit_friction(q_old + G * A * dt * dH / length[k],
                                  G * nr[k] ** 2 * dt / (A * R ** (4.0 / 3.0)))
            u = abs(q) / A
            cmax = max(cmax, u + math.sqrt(G * hf), 5.0 / 3.0 * u)
            Q[k] = q
        else:
            h = hr[k]
            if h <= h_dry:
                continue
            w = width[k]
            A = w * h
            R = A / (w + 2.0 * h)
            q = A * R ** (2.0 / 3.0) * math.sqrt(outlet_slope[k]) / nr[k]
            u = q / A
            cmax = max(cmax, u + math.sqrt(G * h), 5.0 / 3.0 * u)
            Q[k] = q
    return cmax, dtsmin


@njit(cache=True, nogil=True)
def apply_river_fluxes(hr, width, length, dn, Q, dt):
    """Positivity-limited river update.  Returns net outlet outflow volume [m^3]
    (negative when a stage boundary pushes water into the river)."""
    nriv = hr.shape[0]
    out = np.zeros(nriv)
    for k in range(nriv):
        q = Q[k]
        if q > 0:
            out[k] += q * dt
        elif q < 0 and dn[k] >= 0:
            out[dn[k]] -= q * dt
    r = np.ones(nriv)
    for k in range(nriv):
        if out[k] > 0:
            avail = hr[k] * width[k] * length[k]
            if out[k] > avail:
                r[k] = avail / out[k]
    dV = np.zeros(nriv)
    vout = 0.0
    for k in range(nriv):
        q = Q[k]
        d = dn[k]
        if q > 0:
            q *= r[k]
        elif q < 0 and d >= 0:
            q *= r[d]
        Q[k] = q
        dV[k] -= q * dt
        if d >= 0:
            dV[d] += q * dt
        else:
            vout += q * dt
    for k in range(nriv):
        hr[k] = max(hr[k] + dV[k] / (width[k] * length[k]), 0.0)
    return vout


@njit(cache=True, inline="always")
def honda_unit(h1, h2, mu1, mu2):
    """Honda (本間) overflow formula, unit-width discharge [m^2/s].
    h1 >= h2 >= 0 are depths above the crest on the high / low side."""
    if h1 <= 0.0:
        return 0.0
    if h2 <= (2.0 / 3.0) * h1:
        return mu1 * h1 * math.sqrt(2.0 * G * h1)            # complete overflow
    return mu2 * h2 * math.sqrt(2.0 * G * (h1 - h2))       # submerged overflow


@njit(cache=True, nogil=True)
def river_slope_exchange(h, z, da, sd, ri, rj, hr, zb, width, length, embank,
                         dx, dy, dt, mu1, mu2, bank_factor, qex, sluice_gates=True):
    """Exchange between river cell and the slope storage of the same grid cell.

    Water levels: slope  Hs = z + max(h - d_a, 0)  (surface water only)
                  river  Hr = zb + hr
    Crest      : zc = z + embankment
    The higher side discharges over the crest by Honda's formula (complete /
    submerged).  The exchanged volume is capped so that (i) the donor does not
    fall below the crest or the receiver level and (ii) storage stays >= 0.
    qex[k] > 0 : slope -> river.  Returns nothing; updates h and hr in place.
    """
    area = dx * dy
    for k in range(hr.shape[0]):
        qex[k] = 0.0
        i, j = ri[k], rj[k]
        hsurf = h[i, j] - da[i, j]
        if hsurf < 0.0:
            hsurf = 0.0
        Hs = z[i, j] + hsurf
        hsurf = max(hsurf - sd[i, j], 0.0)       # retained (paddy) water does not spill
        Hr = zb[k] + hr[k]
        zc = z[i, j] + embank[k]
        Ar = width[k] * length[k]
        L = bank_factor * length[k]
        if Hs > Hr:
            # 樋門 / flap gate: land-side water drains through the levee when the
            # river is lower (crest = ground); river water cannot flow back.
            zc_out = z[i, j] if sluice_gates else zc
            h1 = Hs - zc_out
            if h1 <= 0.0 or hsurf <= 0.0:
                continue
            h2 = max(Hr - zc_out, 0.0)
            V = honda_unit(h1, h2, mu1, mu2) * L * dt
            # donor may drain to the crest or to the equalised level at most
            Vmax = min(h1, hsurf) * area
            if Hr > zc_out:
                Vmax = min(Vmax, (Hs - Hr) * area * Ar / (area + Ar))
            if V > Vmax:
                V = Vmax
            h[i, j] -= V / area
            hr[k] += V / Ar
            qex[k] = V / dt
        elif Hr > Hs:
            h1 = Hr - zc
            if h1 <= 0.0:
                continue
            h2 = max(Hs - zc, 0.0)
            V = honda_unit(h1, h2, mu1, mu2) * L * dt
            Vmax = min(h1, hr[k]) * Ar
            if Hs > zc:
                Vmax = min(Vmax, (Hr - Hs) * area * Ar / (area + Ar))
            if V > Vmax:
                V = Vmax
            hr[k] -= V / Ar
            h[i, j] += V / area
            qex[k] = -V / dt


# --------------------------------------------------------------- infiltration
def _green_ampt_py(h, F, mask, da, kv, phi_theta, sf, limit, dt):
    """Green-Ampt vertical infiltration removing surface water to a loss store F.

        f = kv * (1 + (phi - theta_i) * S_f / F)

    The infiltration capacity over dt is the exact ponded Green-Ampt
    increment, obtained from the implicit integral
        kv dt = F1 - F0 - c ln((c + F1)/(c + F0)),   c = (phi - theta_i) S_f
    by Newton iteration (unconditionally accurate, no singularity at F = 0).
    Actual infiltration = min(capacity, surface water available, room).
    Applies only to the surface water (h - d_a).  Returns sum of depths [m]."""
    ny, nx = h.shape
    tot = 0.0
    for i in prange(ny):
        for j in range(nx):
            if not mask[i, j] or kv[i, j] <= 0.0:
                continue
            hs = h[i, j] - da[i, j]
            if hs <= 0.0:
                continue
            room = limit[i, j] - F[i, j]
            if room <= 0.0:
                continue
            c = phi_theta[i, j] * sf[i, j]
            F0 = F[i, j]
            Kdt = kv[i, j] * dt
            F1 = F0 + Kdt + math.sqrt(2.0 * c * Kdt)
            for _ in range(30):
                g = F1 - F0 - c * math.log((c + F1) / (c + F0)) - Kdt if c > 0 else F1 - F0 - Kdt
                gp = F1 / (c + F1) if c > 0 else 1.0
                step = g / gp
                F1n = F1 - step
                if F1n <= F0:
                    F1n = 0.5 * (F1 + F0)
                if abs(F1n - F1) < 1e-14:
                    F1 = F1n
                    break
                F1 = F1n
            fcap = F1 - F0
            inf = min(fcap, hs, room)
            h[i, j] -= inf
            F[i, j] += inf
            tot += inf
    return tot


@njit(cache=True, nogil=True)
def retained_release(h, z, mask, da, sd, leak, dx, dy, dt):
    """Slow release of water retained below the store depth (田んぼダム outlet
    plate / orifice), modelled as a linear reservoir with rate `leak` [1/s]:
        V_out = min(h_s, sd) * (1 - exp(-leak dt)) * A
    released to the 4-neighbour with the lowest water level (if lower).
    Mass-conserving (moves between cells only)."""
    ny, nx = h.shape
    dV = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or leak[i, j] <= 0.0 or sd[i, j] <= 0.0:
                continue
            hs = h[i, j] - da[i, j]
            if hs <= 0.0:
                continue
            ret = min(hs, sd[i, j])
            H = z[i, j] + h[i, j]
            best, bi, bj = H, -1, -1
            for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                a, b = i + di, j + dj
                if 0 <= a < ny and 0 <= b < nx and mask[a, b]:
                    Hn = z[a, b] + h[a, b]
                    if Hn < best:
                        best, bi, bj = Hn, a, b
            if bi < 0:
                continue
            dh = ret * (1.0 - math.exp(-leak[i, j] * dt))
            dh = min(dh, 0.5 * (H - best))         # do not invert the levels
            if dh > 0.0:
                dV[i, j] -= dh
                dV[bi, bj] += dh
    for i in range(ny):
        for j in range(nx):
            h[i, j] += dV[i, j]


# serial and parallel builds of the cell kernels; the solver picks one by grid size
# (thread start-up dominates on small grids)
slope_fluxes = njit(cache=True, nogil=True)(_slope_fluxes_py)
apply_slope_fluxes = njit(cache=True, nogil=True)(_apply_slope_fluxes_py)
green_ampt = njit(cache=True, nogil=True)(_green_ampt_py)
slope_fluxes_par = njit(cache=False, nogil=True, parallel=True)(_slope_fluxes_py)
apply_slope_fluxes_par = njit(cache=False, nogil=True, parallel=True)(_apply_slope_fluxes_py)
green_ampt_par = njit(cache=False, nogil=True, parallel=True)(_green_ampt_py)


@njit(cache=True, nogil=True)
def lowland_drainage(h, mask, da, sd, rate, target_k, width, length, hr, dx, dy, dt):
    """排水機場・下水道・クリーク: surface water of cells with rate > 0 is pumped
    to the river cell they drain to (target_k >= 0) at most `rate` [m/s].
    Works also when the river is high (pumps).  Returns pumped volume [m3]."""
    ny, nx = h.shape
    area = dx * dy
    tot = 0.0
    for i in range(ny):
        for j in range(nx):
            k = target_k[i, j]
            if not mask[i, j] or rate[i, j] <= 0.0 or k < 0:
                continue
            hs = h[i, j] - da[i, j] - sd[i, j]
            if hs <= 0.0:
                continue
            dh = min(hs, rate[i, j] * dt)
            h[i, j] -= dh
            hr[k] += dh * area / (width[k] * length[k])
            tot += dh * area
    return tot
