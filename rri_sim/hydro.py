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
