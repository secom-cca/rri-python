"""Implementation state, land use and the transitions between portfolios (Robust DAPP, step 2).

A portfolio (a node of the pathways network) is a level index per decision variable of the
exploration (levels as in docs/13: planner.levels_of).  Two states are kept apart:

    implementation  what is built / operated: levee height, retention site, paddy-dam and private
                    storage rates, pumps, public storage, raised houses
    land use        what the land is (per zone): farmland, forest converted from farmland, retention
                    land, residential land in the flood zone, relocation sites (vacated, building ban)

A transition a -> b may raise any level (cost: initial increment + change_cost_fraction x the replaced
option, as sim.State.process).  A level may go down only where the measure is reversible
(measures.<kind>.reversible: paddy dams, private storage) or can be reconverted under a condition
(measures.forest.reconvert: young forest back to farmland, checked against the planting year when
the pathway is simulated).  Land-use changes into retention land and relocation sites never return;
raised houses can still be relocated (raise -> relocate), not the other way.  A portfolio must also
fit the land: paddy dams only on paddies that were not afforested (landuse.forest_on_paddy_share).
"""
from __future__ import annotations

import numpy as np

from ..roadmap import planner as PL
from ..roadmap import sim as SM
from ..roadmap import world as WD

LAND = ("農地", "森林（転換）", "遊水地", "住宅地（浸水域）", "移転跡地")


class Transitions:
    def __init__(self, rcfg, phys, landuse):
        self.cfg, self.phys = rcfg, phys
        self.V = phys.variables
        self.M = rcfg["measures"]
        self.levels = [PL.levels_of(rcfg, v) for v in self.V]
        self.share = float(landuse["forest_on_paddy_share"]["value"])
        self.cols = [[c for c, (j, _) in enumerate(phys.layout) if j == i] for i in range(len(self.V))]
        self.km2 = phys.cell_km2
        zones = phys.zones
        self.paddy = {z: next((i for i, v in enumerate(self.V) if v["type"] == "paddy_dam" and v["zone"] == z), None) for z in zones}
        self.forest = {z: next((i for i, v in enumerate(self.V) if v["type"] == "forest" and v["zone"] == z), None) for z in zones}
        self.costs = [[phys.cost_parts(i, lv) for lv in L] for i, L in enumerate(self.levels)]
        bw = WD.belief_world(rcfg)
        self.kinds = sorted({SM.kind_of(v, lv) for i, v in enumerate(self.V) for lv in self.levels[i][1:]})
        self.lead_nominal = {k: PL.lead_years(bw, k) for k in self.kinds}

    # ---------------------------------------------------------------- one portfolio
    def value(self, g, i):
        return self.levels[i][int(g[i])]

    def kind(self, i, li):
        return SM.kind_of(self.V[i], self.levels[i][int(li)])

    def raw(self, g, maturity=1.0):
        """Surrogate raw columns (physics.layout) with the portfolio fully in place (as sim.State with full adoption)."""
        r = np.zeros(len(self.phys.layout))
        for i, v in enumerate(self.V):
            lv = self.value(g, i)
            if not lv:
                continue
            c = self.cols[i]
            t = v["type"]
            if t == "levee":
                r[c[0]] = v["heights"][int(lv)] / max(v["heights"])
            elif t == "relocate":
                r[c[1 if int(lv) == 2 else 0]] = 1.0
            elif t == "forest":
                r[c[0]], r[c[1]] = lv * maturity, lv
            elif v["continuous"] or len(self.M.get(t, {}).get("levels", [])) > 2:
                r[c[0]] = lv
            else:
                r[c[0]] = lv / max(len(v["options"]) - 1, 1)
        return r

    def paddy_available(self, g, z):
        """Share of the zone's paddies still farmland (not afforested)."""
        ip, jf = self.paddy.get(z), self.forest.get(z)
        if ip is None or jf is None:
            return 1.0
        lost = self.value(g, jf) * self.V[jf]["n_cells"] * self.share
        return float(max(0.0, 1.0 - lost / max(self.V[ip]["n_cells"], 1e-9)))

    def repair(self, g):
        """Lower the paddy-dam rate to the paddies left after afforestation (nearest level not above)."""
        g = list(int(x) for x in g)
        for z, ip in self.paddy.items():
            if ip is None:
                continue
            cap = self.paddy_available(g, z)
            while g[ip] > 0 and self.levels[ip][g[ip]] > cap + 1e-9:
                g[ip] -= 1
        return tuple(g)

    def landuse(self, g):
        """Land use per zone [km²] (residential = cells of houses flooded >= 1 m without measures)."""
        out = {}
        for z in self.phys.zones:
            d = dict.fromkeys(LAND, 0.0)
            for i, v in enumerate(self.V):
                if v["zone"] != z:
                    continue
                a, lv = v["n_cells"] * self.km2, self.value(g, i)
                if v["type"] == "paddy_dam":
                    d["農地"] += a
                elif v["type"] == "forest":
                    d["農地"] += a * (1 - self.share) * (1 - lv)
                    d["農地"] -= a * self.share * lv
                    d["森林（転換）"] += a * lv
                elif v["type"] == "retention" and lv:
                    d["農地"] -= a; d["遊水地"] += a
                elif v["type"] == "relocate":
                    if int(lv) == 2:
                        d["移転跡地"] += a
                    else:
                        d["住宅地（浸水域）"] += a
            out[z] = {k: round(max(x, 0.0), 2) for k, x in d.items()}
        return out

    def implementation(self, g):
        return [{"var": i, "name": v["name"], "kind": self.kind(i, g[i]), "level": self.value(g, i),
                 "label": self.label(i, g[i])} for i, v in enumerate(self.V) if g[i]]

    def label(self, i, li):
        v, lv = self.V[i], self.levels[i][int(li)]
        if v["continuous"] or v["type"] in ("paddy_dam", "forest", "storage_private"):
            return f"{lv:.0%}"
        return v["options"][int(lv)] if int(lv) < len(v["options"]) else str(lv)

    def cost_pv(self, g, annuity):
        ini = sum(self.costs[i][int(g[i])][0] for i in range(len(self.V)))
        ann = sum(self.costs[i][int(g[i])][1] for i in range(len(self.V)))
        return ini + annuity * ann, ini, ann

    def irreversible(self, g):
        """Initial cost [億円] sunk in measures that cannot be scaled back, and the land converted for good."""
        sunk = 0.0
        for i in range(len(self.V)):
            if g[i] and not self.M[self.kind(i, g[i])].get("reversible"):
                sunk += self.costs[i][int(g[i])][0]
        lu = self.landuse(g)
        return {"sunk_oku": sunk, "retention_km2": sum(z["遊水地"] for z in lu.values()),
                "relocated_km2": sum(z["移転跡地"] for z in lu.values()), "forest_km2": sum(z["森林（転換）"] for z in lu.values())}

    # ---------------------------------------------------------------- transitions between portfolios
    def matrices(self, nodes):
        """For all node pairs a -> b: allowed (bool), conditional (forest reconversion: depends on the planting year),
        initial cost of the change [億円] (increments + change costs + reconversion), annual cost difference,
        nominal lead time [years] and the measure kinds that change (for scenario lead times)."""
        G = np.array(nodes, int)
        N, nV = G.shape
        ok = np.ones((N, N), bool)
        cond = np.zeros((N, N), bool)
        ini = np.zeros((N, N))
        ann = np.zeros((N, N))
        kinds = {k: np.zeros((N, N), bool) for k in self.kinds}
        for i, v in enumerate(self.V):
            a, b = G[:, i][:, None], G[:, i][None, :]
            I = np.array([c[0] for c in self.costs[i]]); A = np.array([c[1] for c in self.costs[i]])
            up, down = b > a, b < a
            ann += A[b] - A[a]
            inc = np.maximum(I[b] - I[a], 0.0)
            # change cost of a replaced option (discrete measures, as sim.State.process)
            if not v["continuous"]:
                cf = np.array([self.M[SM.kind_of(v, lv)]["change_cost_fraction"] if lv else 0.0 for lv in self.levels[i]])
                inc = inc + cf[a] * I[a] * (a > 0)
            ini += np.where(up, inc, 0.0)
            for li in range(1, len(self.levels[i])):
                k = self.kind(i, li)
                kinds[k] |= up & (b == li)
            # downward moves
            kd = [SM.kind_of(v, lv) if lv else None for lv in self.levels[i]]
            rev = np.array([bool(k and self.M[k].get("reversible")) for k in kd])
            rc = self.M.get(v["type"], {}).get("reconvert") if v["type"] == "forest" else None
            if rc and rc.get("allowed"):
                cond |= down
                ini += np.where(down, rc.get("cost_fraction", 0.0) * np.maximum(I[a] - I[b], 0.0), 0.0)
            else:
                ok &= ~down | rev[a]
        # land: paddy dams only on paddies left after afforestation
        fits = np.array([self.repair(g) == tuple(g) for g in G])
        ok &= fits[None, :]
        np.fill_diagonal(ok, False)
        lead = np.zeros((N, N))
        for k, m in kinds.items():
            lead = np.maximum(lead, m * self.lead_nominal[k])
        lead = np.where(cond & ~np.any([m for m in kinds.values()], axis=0), 1.0, lead)
        return {"ok": ok, "conditional": cond & ok, "initial": ini, "annual": ann, "lead": lead, "kinds": kinds}

    def change_cost(self, a, b):
        """Initial cost [億円] of the move a -> b (one pair; the same rules as matrices), None if not allowed."""
        m = self.matrices([tuple(a), tuple(b)])
        return float(m["initial"][0, 1]) if m["ok"][0, 1] or tuple(a) == tuple(b) else None

    def describe_edge(self, a, b):
        """Readable list of the changes a -> b with their land-use consequences and reversibility."""
        out = []
        for i, v in enumerate(self.V):
            if a[i] == b[i]:
                continue
            k_new = self.kind(i, b[i]) if b[i] else self.kind(i, a[i])
            m = self.M[k_new]
            land = None
            if v["type"] == "forest":
                land = "農地→森林" if b[i] > a[i] else "森林→農地（再転換）"
            elif v["type"] == "retention":
                land = "農地→遊水地（戻さない）"
            elif v["type"] == "relocate" and int(self.value(b, i)) == 2:
                land = "住宅地→移転跡地（戻さない）"
            out.append({"var": v["name"], "from": self.label(i, a[i]), "to": self.label(i, b[i]), "kind": k_new,
                        "up": bool(b[i] > a[i]), "reversible": bool(m.get("reversible")), "land": land,
                        "lead_years": round(self.lead_nominal.get(k_new, 0.0), 1) if b[i] > a[i] else 1.0})
        return out
