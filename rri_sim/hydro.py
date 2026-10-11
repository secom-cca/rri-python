"""Terrain processing: depression filling, D8 flow directions, flow accumulation
and river-network extraction.

Flow directions are taken from a depression-filled DEM (priority-flood with an
epsilon gradient, Barnes et al. 2014) so that every cell drains to the domain
boundary.  The *original* DEM is kept for the 2D slope computation, so real
lowlands still pond water; the filled DEM is only used to define the channel
network topology.
"""
from __future__ import annotations

import heapq

import numpy as np

# D8 neighbour offsets, index 0..7: E, SE, S, SW, W, NW, N, NE
D8_DI = np.array([0, 1, 1, 1, 0, -1, -1, -1])
D8_DJ = np.array([1, 1, 0, -1, -1, -1, 0, 1])
D8_LEN = np.array([1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2)])


def fill_depressions(z: np.ndarray, mask: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Priority-flood depression filling with epsilon gradient.

    Cells on the mask boundary (adjacent to a non-mask cell or the array edge)
    are seeds.  Returns a filled copy; non-mask cells are left unchanged.
    """
    ny, nx = z.shape
    filled = z.astype(float).copy()
    closed = ~mask.copy()
    heap: list[tuple[float, int, int]] = []
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j]:
                continue
            edge = i == 0 or j == 0 or i == ny - 1 or j == nx - 1
            if not edge:
                for k in range(8):
                    ii, jj = i + D8_DI[k], j + D8_DJ[k]
                    if not mask[ii, jj]:
                        edge = True
                        break
            if edge:
                heapq.heappush(heap, (filled[i, j], i, j))
                closed[i, j] = True
    while heap:
        zc, i, j = heapq.heappop(heap)
        for k in range(8):
            ii, jj = i + D8_DI[k], j + D8_DJ[k]
            if ii < 0 or jj < 0 or ii >= ny or jj >= nx or closed[ii, jj]:
                continue
            closed[ii, jj] = True
            if filled[ii, jj] <= zc:
                filled[ii, jj] = zc + eps
            heapq.heappush(heap, (filled[ii, jj], ii, jj))
    return filled


def d8_directions(zf: np.ndarray, mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Steepest-descent D8 direction (0..7) per cell; -1 = drains out of domain."""
    ny, nx = zf.shape
    fdir = np.full((ny, nx), -1, dtype=np.int64)
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j]:
                continue
            best, bk, out = 0.0, -1, False
            for k in range(8):
                ii, jj = i + D8_DI[k], j + D8_DJ[k]
                dist = np.hypot(D8_DI[k] * dy, D8_DJ[k] * dx)
                if ii < 0 or jj < 0 or ii >= ny or jj >= nx or not mask[ii, jj]:
                    out = True
                    continue
                s = (zf[i, j] - zf[ii, jj]) / dist
                if s > best:
                    best, bk = s, k
            # boundary cells with no downhill neighbour drain outward
            fdir[i, j] = bk if (bk >= 0 or not out) else -1
    return fdir


def flow_accumulation(fdir: np.ndarray, mask: np.ndarray, cell_area: float) -> np.ndarray:
    """Upstream contributing area [m^2] including the cell itself."""
    ny, nx = fdir.shape
    indeg = np.zeros((ny, nx), dtype=np.int64)
    for i in range(ny):
        for j in range(nx):
            k = fdir[i, j]
            if mask[i, j] and k >= 0:
                indeg[i + D8_DI[k], j + D8_DJ[k]] += 1
    acc = np.where(mask, cell_area, 0.0)
    stack = [(i, j) for i in range(ny) for j in range(nx) if mask[i, j] and indeg[i, j] == 0]
    order = []
    while stack:
        i, j = stack.pop()
        order.append((i, j))
        k = fdir[i, j]
        if k >= 0:
            ii, jj = i + D8_DI[k], j + D8_DJ[k]
            acc[ii, jj] += acc[i, j]
            indeg[ii, jj] -= 1
            if indeg[ii, jj] == 0:
                stack.append((ii, jj))
    if len(order) != int(mask.sum()):
        raise RuntimeError("flow direction graph has cycles; fill the DEM first")
    return acc


def topological_order(dn: np.ndarray) -> np.ndarray:
    """Order river cells from upstream to downstream (Kahn's algorithm)."""
    n = len(dn)
    indeg = np.zeros(n, dtype=np.int64)
    for k in range(n):
        if dn[k] >= 0:
            indeg[dn[k]] += 1
    stack = [k for k in range(n) if indeg[k] == 0]
    order = []
    while stack:
        k = stack.pop()
        order.append(k)
        d = dn[k]
        if d >= 0:
            indeg[d] -= 1
            if indeg[d] == 0:
                stack.append(d)
    if len(order) != n:
        raise RuntimeError("river network has cycles")
    return np.array(order, dtype=np.int64)


def height_above_nearest_drainage(z: np.ndarray, fdir: np.ndarray, is_river: np.ndarray,
                                  mask: np.ndarray) -> np.ndarray:
    """HAND (Rennó et al., 2008): elevation above the river cell reached by
    following D8 directions downstream.  Used to separate floodplain
    inundation from sheet flow on hillslopes."""
    ny, nx = z.shape
    hand = np.full((ny, nx), np.nan)
    drain_z = np.full((ny, nx), np.nan)
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or np.isfinite(drain_z[i, j]):
                continue
            path = []
            a, b = i, j
            while True:
                if is_river[a, b]:
                    zr = z[a, b]
                    break
                if np.isfinite(drain_z[a, b]):
                    zr = drain_z[a, b]
                    break
                path.append((a, b))
                k = fdir[a, b]
                if k < 0:
                    zr = z[a, b]
                    break
                a, b = a + D8_DI[k], b + D8_DJ[k]
                if not (0 <= a < ny and 0 <= b < nx) or not mask[a, b]:
                    zr = z[path[-1]]
                    break
            for (p, q) in path:
                drain_z[p, q] = zr
            if is_river[i, j]:
                drain_z[i, j] = z[i, j]
    hand[mask] = (z - drain_z)[mask]
    return hand


def drain_target(fdir: np.ndarray, is_river: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flat grid index of the first river cell reached by following D8
    directions from each cell (-1 if none)."""
    ny, nx = fdir.shape
    tgt = np.full((ny, nx), -2, dtype=np.int64)       # -2 = unknown
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j] or tgt[i, j] != -2:
                continue
            path = []
            a, b = i, j
            while True:
                if is_river[a, b]:
                    t = a * nx + b
                    break
                if tgt[a, b] != -2:
                    t = tgt[a, b]
                    break
                path.append((a, b))
                k = fdir[a, b]
                if k < 0:
                    t = -1
                    break
                a, b = a + D8_DI[k], b + D8_DJ[k]
                if not (0 <= a < ny and 0 <= b < nx) or not mask[a, b]:
                    t = -1
                    break
            for (p, q) in path:
                tgt[p, q] = t
            if is_river[i, j]:
                tgt[i, j] = i * nx + j
    tgt[tgt == -2] = -1
    return tgt
