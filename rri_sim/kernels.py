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
from numba import njit

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
               dist, width, area, dt, inertial, q_old, lim_frac, h_dry):
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
        hd, nd, kad, dad, dmd, bd = ha, na, kaa, daa, dma, ba
    elif dH < 0.0 or q_old < 0.0:
        hd, nd, kad, dad, dmd, bd = hb, nb, kab, dab, dmb, bb
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
    hs = surface_depth_of(hf, kad, dad)
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


@njit(cache=True)
def slope_fluxes(h, z, mask, n, ka, da, dm, beta, bnd_slope, dx, dy, dt, inertial,
                 lim_frac, h_dry, fx, fy, fb, qsx, qsy):
    """Face discharges [m^3/s]: fx east faces (ny, nx-1), fy south faces
    (ny-1, nx), fb open-boundary outflow (ny, nx).  qsx/qsy hold the surface
    unit-width flux (state of the inertial scheme).
    Returns (max celerity, min diffusive stability time)."""
    ny, nx = h.shape
    area = dx * dy
    cmax = 0.0
    dtsmin = np.inf
    for i in range(ny):
        for j in range(nx - 1):
            fx[i, j] = 0.0
            if not (mask[i, j] and mask[i, j + 1]):
                qsx[i, j] = 0.0
                continue
            Q, qs, c, dts = slope_face(
                z[i, j] + h[i, j], z[i, j + 1] + h[i, j + 1], z[i, j], z[i, j + 1],
                h[i, j], h[i, j + 1], n[i, j], n[i, j + 1], ka[i, j], ka[i, j + 1],
                da[i, j], da[i, j + 1], dm[i, j], dm[i, j + 1], beta[i, j], beta[i, j + 1],
                dx, dy, area, dt, inertial, qsx[i, j], lim_frac, h_dry)
            fx[i, j] = Q
            qsx[i, j] = qs
            cmax = max(cmax, c)
            dtsmin = min(dtsmin, dts)
    for i in range(ny - 1):
        for j in range(nx):
            fy[i, j] = 0.0
            if not (mask[i, j] and mask[i + 1, j]):
                qsy[i, j] = 0.0
                continue
            Q, qs, c, dts = slope_face(
                z[i, j] + h[i, j], z[i + 1, j] + h[i + 1, j], z[i, j], z[i + 1, j],
                h[i, j], h[i + 1, j], n[i, j], n[i + 1, j], ka[i, j], ka[i + 1, j],
                da[i, j], da[i + 1, j], dm[i, j], dm[i + 1, j], beta[i, j], beta[i + 1, j],
                dy, dx, area, dt, inertial, qsy[i, j], lim_frac, h_dry)
            fy[i, j] = Q
            qsy[i, j] = qs
            cmax = max(cmax, c)
            dtsmin = min(dtsmin, dts)
    # open boundary: normal-depth outflow with a prescribed slope
    for i in range(ny):
        for j in range(nx):
            fb[i, j] = 0.0
            if mask[i, j] and bnd_slope[i, j] > 0.0 and h[i, j] > 0.0:
                fb[i, j] = slope_discharge(h[i, j], bnd_slope[i, j], n[i, j], ka[i, j],
                                           da[i, j], dm[i, j], beta[i, j]) * dx
    return cmax, dtsmin


@njit(cache=True)
def apply_slope_fluxes(h, mask, fx, fy, fb, qsx, qsy, dx, dy, dt):
    """Positivity-limited, conservative update of slope storage.
    Returns outflow volume through the open boundary [m^3]."""
    ny, nx = h.shape
    area = dx * dy
    out = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx - 1):
            q = fx[i, j]
            if q > 0:
                out[i, j] += q * dt
            elif q < 0:
                out[i, j + 1] -= q * dt
    for i in range(ny - 1):
        for j in range(nx):
            q = fy[i, j]
            if q > 0:
                out[i, j] += q * dt
            elif q < 0:
                out[i + 1, j] -= q * dt
    for i in range(ny):
        for j in range(nx):
            out[i, j] += fb[i, j] * dt
    r = np.ones((ny, nx))
    for i in range(ny):
        for j in range(nx):
            if out[i, j] > 0.0:
                avail = h[i, j] * area
                if out[i, j] > avail:
                    r[i, j] = avail / out[i, j]
    dV = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx - 1):
            q = fx[i, j]
            f = r[i, j] if q > 0 else r[i, j + 1]
            q *= f
            qsx[i, j] *= f
            fx[i, j] = q
            dV[i, j] -= q * dt
            dV[i, j + 1] += q * dt
    for i in range(ny - 1):
        for j in range(nx):
            q = fy[i, j]
            f = r[i, j] if q > 0 else r[i + 1, j]
            q *= f
            qsy[i, j] *= f
            fy[i, j] = q
            dV[i, j] -= q * dt
            dV[i + 1, j] += q * dt
    vout = 0.0
    for i in range(ny):
        for j in range(nx):
            fb[i, j] *= r[i, j]
            dV[i, j] -= fb[i, j] * dt
            vout += fb[i, j] * dt
    for i in range(ny):
        for j in range(nx):
            if mask[i, j]:
                h[i, j] = max(h[i, j] + dV[i, j] / area, 0.0)
    return vout


# ---------------------------------------------------------------- river physics
@njit(cache=True)
def river_fluxes(hr, zb, width, length, nr, dn, outlet_slope, dt, inertial, lim_frac,
                 h_dry, Q):
    """Discharge from each river cell to its downstream cell [m^3/s]
    (negative = backwater flow upstream), rectangular section.
      diffusive : Q = sign (1/n) A R^{2/3} |dH/dx|^{1/2}            (RRI)
      inertial  : Q+ + (g n^2 dt / (A R^{4/3})) |Q+| Q+ = Q + g A dt dH/dx
    Q holds the previous discharge on entry (inertial state).
    Outlet cells: normal depth with outlet_slope.
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


@njit(cache=True)
def apply_river_fluxes(hr, width, length, dn, Q, dt):
    """Positivity-limited river update.  Returns outlet outflow volume [m^3]."""
    nriv = hr.shape[0]
    out = np.zeros(nriv)
    for k in range(nriv):
        q = Q[k]
        if q > 0:
            out[k] += q * dt
        elif q < 0:
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
        elif q < 0:
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


# ---------------------------------------------------- river <-> slope exchange
@njit(cache=True, inline="always")
def honda_unit(h1, h2, mu1, mu2):
    """Honda (本間) overflow formula, unit-width discharge [m^2/s].
    h1 >= h2 >= 0 are depths above the crest on the high / low side."""
    if h1 <= 0.0:
        return 0.0
    if h2 <= (2.0 / 3.0) * h1:
        return mu1 * h1 * math.sqrt(2.0 * G * h1)            # complete overflow
    return mu2 * h2 * math.sqrt(2.0 * G * (h1 - h2))       # submerged overflow


@njit(cache=True)
def river_slope_exchange(h, z, da, ri, rj, hr, zb, width, length, embank,
                         dx, dy, dt, mu1, mu2, bank_factor, qex):
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
        Hr = zb[k] + hr[k]
        zc = z[i, j] + embank[k]
        Ar = width[k] * length[k]
        L = bank_factor * length[k]
        if Hs > Hr:
            h1 = Hs - zc
            if h1 <= 0.0 or hsurf <= 0.0:
                continue
            h2 = max(Hr - zc, 0.0)
            V = honda_unit(h1, h2, mu1, mu2) * L * dt
            # donor may drain to the crest or to the equalised level at most
            Vmax = min(h1, hsurf) * area
            if Hr > zc:
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
@njit(cache=True)
def green_ampt(h, F, mask, da, kv, phi_theta, sf, limit, dt):
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
    for i in range(ny):
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
