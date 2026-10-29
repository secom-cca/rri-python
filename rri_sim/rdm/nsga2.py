"""Constrained NSGA-II (Deb et al. 2002) for integer decision variables.

Objectives are minimised.  Constraint handling by constrained domination: a
feasible solution beats an infeasible one; two infeasible ones are compared
by their violation.  Uniform crossover, random-reset mutation, binary
tournament on (rank, crowding).  `repair(x)` (optional) is applied to every
child before evaluation, e.g. to drop measures until the budget is met.
"""
from __future__ import annotations

import numpy as np


def dominates(a, b, va, vb):
    if va <= 0 < vb:
        return True
    if va > 0 and vb > 0:
        return va < vb
    if vb <= 0 < va:
        return False
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_sort(F, V):
    n = len(F)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(F[i], F[j], V[i], V[j]):
                S[i].append(j); cnt[j] += 1
            elif dominates(F[j], F[i], V[j], V[i]):
                S[j].append(i); cnt[i] += 1
    fronts, cur = [], [i for i in range(n) if cnt[i] == 0]
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for j in S[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def crowding(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        o = np.argsort(F[:, k])
        d[o[0]] = d[o[-1]] = np.inf
        span = F[o[-1], k] - F[o[0], k]
        if span > 0:
            d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def pareto_mask(F, V=None):
    """Feasible non-dominated points."""
    V = np.zeros(len(F)) if V is None else np.asarray(V)
    keep = np.zeros(len(F), bool)
    for i in range(len(F)):
        if V[i] > 0:
            continue
        keep[i] = not any(V[j] <= 0 and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in range(len(F)) if j != i)
    return keep


def run(n_options, evaluate, pop=12, gens=4, rng=None, p_mut=0.15, init=(), repair=None, log=print):
    """Returns (archive, final population indices).  archive: list of dicts
    {"x", "f" (objectives), "v" (violation), "gen"} of every distinct evaluated x."""
    rng = rng or np.random.default_rng(0)
    n_options = np.asarray(n_options)
    archive, seen = [], {}

    def ev(x, gen):
        x = np.asarray(repair(x) if repair else x, int)
        t = tuple(x.tolist())
        if t not in seen:
            f, v = evaluate(x)
            seen[t] = len(archive)
            archive.append({"x": list(t), "f": [float(a) for a in f], "v": float(v), "gen": gen})
        return seen[t]

    P = [ev(np.asarray(x), 0) for x in init]
    while len(P) < pop:
        P.append(ev(rng.integers(0, n_options), 0))
    P = list(dict.fromkeys(P))
    for g in range(1, gens + 1):
        F = np.array([archive[i]["f"] for i in P]); V = np.array([archive[i]["v"] for i in P])
        rank = np.zeros(len(P), int); cd = np.zeros(len(P))
        for r, fr in enumerate(nondominated_sort(F, V)):
            rank[fr] = r; cd[fr] = crowding(F[fr])

        def pick():
            a, b = rng.integers(0, len(P), 2)
            return P[a] if (rank[a], -cd[a]) < (rank[b], -cd[b]) else P[b]
        kids = []
        tries = 0
        while len(kids) < pop and tries < 20 * pop:
            tries += 1
            x1, x2 = np.array(archive[pick()]["x"]), np.array(archive[pick()]["x"])
            m = rng.random(len(x1)) < 0.5
            c = np.where(m, x1, x2)
            mut = rng.random(len(c)) < p_mut
            c[mut] = rng.integers(0, n_options[mut])
            k = ev(c, g)
            if k not in P and k not in kids:
                kids.append(k)
        U = P + kids
        F = np.array([archive[i]["f"] for i in U]); V = np.array([archive[i]["v"] for i in U])
        newP = []
        for fr in nondominated_sort(F, V):
            if len(newP) + len(fr) <= pop:
                newP += [U[i] for i in fr]
            else:
                c = crowding(F[fr])
                newP += [U[fr[i]] for i in np.argsort(-c)[:pop - len(newP)]]
                break
        P = newP
        nf = int(pareto_mask(np.array([a["f"] for a in archive]), [a["v"] for a in archive]).sum())
        log(f"NSGA-II 世代 {g}/{gens}: 評価済み {len(archive)} 案、Pareto {nf} 案")
    return archive, P
