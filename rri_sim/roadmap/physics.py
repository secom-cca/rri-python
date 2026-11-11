"""PHYSICAL layer: measure levels + climate -> flood outcomes, and measure costs.

The flood response is the RRI-trained surrogate of an exploration run
(rri_sim.rdm.explore, docs/12: GP / ridge per output on physically scaled
features, validated against RRI).  It is refitted from the stored design once
and cached (pickle).  The social / institutional model never touches it: sim.py
hands over *raw* columns (what is in place, per variable) and an environment
(climate, effect coefficients); this module multiplies the effects in and
appends the climate exactly as explore.Space.features does.

    raw columns per variable (layout):
        levee      height / max height                   (x levee effect)
        retention  in place 0/1                          (x retention effect)
        relocate   share raised, share relocated         (raised x raise effect)
        paddy_dam  implementation rate                   (x paddy effect)
        forest     sum(planted share x maturity), planted share   (first x forest effect)
    environment: rain_factor, sea_level_rise_m, rain_gamma (+ effect_<type> for explorations before v5; from v5 the
    measures act with their physical parameters and there are no effect coefficients)

Costs (costs.py through the GUI engine, no flood run): per variable the
initial [億円] and annual [億円/年] cost by option or on a rate grid
(rates interpolated linearly); computed once by `prepare` (needs the RRI
session) and stored next to the surrogate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data import PROCESSED
from ..rdm import candidates as CA
from ..rdm import explore as EX
from ..rdm.sensitivity import GP

PHYS_DIR = PROCESSED / "roadmap" / "_physics"
ENV_CLIMATE = ("rain_factor", "sea_level_rise_m", "rain_gamma")
EFFECT_OF = {"levee": "levee_effect", "retention": "retention_effect", "paddy_dam": "paddy_effect",
             "forest": "forest_effect", "raise": "raise_effect"}
OUT_KEYS = ("damage", "pop_hazard", "households", "livability", "ecology")
# variable types the roadmap / DAPP models handle (housing as one variable none / raise / relocate)
SUPPORTED_TYPES = ("levee", "retention", "paddy_dam", "forest", "forest_mgmt", "relocate", "pump_urban", "pump_agri",
                   "storage_public", "storage_private")


def check_supported(res):
    bad = [v["name"] for v in res["variables"]
           if v["type"] not in SUPPORTED_TYPES or (v["type"] == "relocate" and len(v["options"]) != 3)]
    if bad:
        raise ValueError(f"exploration {res['name']}: variables {bad} are not supported by the roadmap / DAPP models "
                         "(森林整備, 嵩上げ・移転の別変数).  Run the exploration for them with "
                         '{"candidates": {"housing": "combined", "forest_mgmt": null}}')


def layout_of(variables):
    """[(variable index, effect type or None)] per raw column, in explore.Space.features order."""
    out = []
    for i, v in enumerate(variables):
        t = v["type"]
        if t == "levee":
            out.append((i, "levee"))
        elif t == "retention":
            out.append((i, "retention"))
        elif t == "relocate":
            out += [(i, "raise"), (i, None)]
        elif t == "paddy_dam":
            out.append((i, "paddy_dam"))
        elif t == "forest":
            out += [(i, "forest"), (i, None)]
        else:
            out.append((i, None))
    return out


def _fast_gp(gp, F):
    """GP mean with the squared distances by matrix products (same result as GP.__call__)."""
    k = gp.X.shape[1]
    ls, sf = np.exp(gp.th[:k]), np.exp(gp.th[k])
    A, B = F / ls, gp.X / ls
    out = np.empty(len(F))
    bb = (B ** 2).sum(1)
    for s in range(0, len(F), 8192):
        a = A[s:s + 8192]
        d2 = np.maximum((a ** 2).sum(1)[:, None] + bb[None] - 2 * a @ B.T, 0.0)
        out[s:s + 8192] = (sf ** 2 * np.exp(-0.5 * d2)) @ gp.alpha
    return gp.mu + gp.sd * out


class Physics:
    """Common interface (SurrogatePhysics; tests use an analytic stand-in with the same attributes):
        variables [{name, type, zone, options, continuous, n_cells, heights?}], zones, layout,
        ranges {env name: (low, high)}, keys, cell_km2, farm_km2,
        cost_parts(i, level) -> (initial, annual), predict(raw, env) -> {key: array}."""
    keys = OUT_KEYS

    def features(self, raw, env):
        raw = np.asarray(raw, float)
        n = len(raw)
        cols = []
        for c, (i, eff) in enumerate(self.layout):
            x = raw[:, c]
            if eff is not None and EFFECT_OF[eff] in self.ranges:       # effect coefficients: explorations before v5 only
                e = env.get(EFFECT_OF[eff], 1.0)
                x = x * np.clip(np.broadcast_to(e, (n,)), *self.ranges[EFFECT_OF[eff]])
            cols.append(x)
        for name in ENV_CLIMATE:
            lo, hi = self.ranges[name]
            v = np.clip(np.broadcast_to(np.asarray(env[name], float), (n,)), lo, hi)
            cols.append((v - lo) / ((hi - lo) or 1.0))
        return np.column_stack(cols)

    def reference(self, env):
        """Outputs without measures (scalars)."""
        raw = np.zeros((1, len(self.layout)))
        out = self.predict(raw, {k: np.atleast_1d(v) for k, v in env.items()})
        return {k: float(np.asarray(v)[0]) for k, v in out.items()}

    def agriculture(self, raw):
        """Farmland kept [%]: forest planted share and retention sites take farmland."""
        raw = np.asarray(raw, float)
        lost = np.zeros(len(raw))
        for c, (i, eff) in enumerate(self.layout):
            v = self.variables[i]
            if v["type"] == "forest" and eff is None:
                lost += raw[:, c] * v["n_cells"] * self.cell_km2
            elif v["type"] == "retention":
                lost += raw[:, c] * v["n_cells"] * self.cell_km2
        return 100.0 * (1.0 - lost / max(self.farm_km2, 1e-9))


class SurrogatePhysics(Physics):
    def __init__(self, explore_name="abukuma_explore", log=print, cache_dir=None):
        src = PROCESSED / "explore" / explore_name / "results.json"
        if not src.exists():
            raise FileNotFoundError(f"{src}: run scripts/run_explore.py first (docs/12)")
        res = json.loads(src.read_text(encoding="utf-8"))
        check_supported(res)
        self.explore_name = explore_name
        self.cfg_explore = res["config"]
        self.dir = Path(cache_dir) if cache_dir else PHYS_DIR / explore_name
        self.dir.mkdir(parents=True, exist_ok=True)
        hs = self.cfg_explore["candidates"]["levee"]["heights_m"]
        self.variables = []
        for v in res["variables"]:
            d = {k: v[k] for k in ("name", "type", "zone", "options", "continuous", "n_cells")}
            if v["type"] == "levee":
                d["heights"] = [float(x) for x in hs]
            self.variables.append(d)
        self.zones = res["zones"]
        self.layout = layout_of(self.variables)
        P = res["uncertainties"]
        self.ranges = {k: (P[k]["low"], P[k]["high"]) for k in P}
        self.cell_km2 = (float(self.cfg_explore["res"]) / 1000.0) ** 2
        self.farm_km2 = sum(v["n_cells"] for v in self.variables if v["type"] in ("paddy_dam", "forest")) * self.cell_km2
        self.keys = OUT_KEYS + tuple("dz:" + z for z in self.zones)
        self.design = res["design"]
        self.validation = {k: {kk: v.get(kk) for kk in ("chosen", "r2", "rel_rmse")} for k, v in res["surrogate"]["validation"].items()
                           if isinstance(v, dict)}
        self.verification = {k: res["verification"].get(k) for k in ("r2", "rel_rmse", "agreement")}
        self._fit(log)
        cp = self.dir / "cost_parts.json"
        if not cp.exists():
            raise FileNotFoundError(f"{cp}: run `python scripts/run_roadmap.py --prepare` once (cost table, no flood runs)")
        self.costs = json.loads(cp.read_text(encoding="utf-8"))["table"]

    # ---- surrogate (refit from the stored design, cached)
    def _space(self):
        return EX.space_from_results({"variables": self.variables, "config": self.cfg_explore})

    def _fit(self, log):
        self.sur = EX.surrogate_from_results({"design": self.design, "config": self.cfg_explore, "variables": self.variables},
                                             self.keys,
                                             [self.dir, EX.EXPLORE_DIR / self.explore_name], log)

    def predict(self, raw, env):
        F = self.features(raw, env)
        Fr, inv = np.unique(np.round(F, 4), axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        out = {}
        for k in self.keys:
            m = self.sur.models[k]
            t = _fast_gp(m, Fr) if isinstance(m, GP) else m(Fr)
            y = np.maximum(np.expm1(t), 0) if self.sur._is_log(k) else t
            out[k] = y[inv]
        return out

    # ---- costs
    def cost_parts(self, i, level):
        t = self.costs[i]
        if self.variables[i]["continuous"]:
            return float(np.interp(level, t["values"], t["initial"])), float(np.interp(level, t["values"], t["annual"]))
        o = int(round(level))
        return float(t["initial"][o]), float(t["annual"][o])

    # ---- RRI re-evaluation of a state (approximate: maturity / partial housing folded into effect coefficients)
    def to_rri(self, raw_row, env_row):
        x, sim = [], {n: float(env_row[n]) for n in ENV_CLIMATE}
        mat, share = [], []
        for i, v in enumerate(self.variables):
            if v.get("institutional"):
                continue
            cs = [c for c, (j, _) in enumerate(self.layout) if j == i]
            r = [float(raw_row[c]) for c in cs]
            if v["type"] == "levee":
                x.append(int(round(r[0] * (len(v["heights"]) - 1))))
            elif v["type"] == "retention":
                x.append(int(r[0] >= 0.5))
            elif v["type"] == "relocate":
                o = 0 if max(r) < 0.05 else (2 if r[1] >= r[0] else 1)
                x.append(o)
                if o:
                    share.append(max(r))
            elif v["type"] == "paddy_dam":
                x.append(round(r[0], 2))
            elif v["type"] == "forest":
                x.append(round(r[1], 2))
                if r[1] > 0:
                    mat.append(r[0] / r[1])
            elif v["continuous"]:
                x.append(round(r[0], 2))
            else:
                x.append(int(round(r[0] * (len(v["options"]) - 1))))
        if all(k in self.ranges for k in EFFECT_OF.values()):          # explorations before v5 (effect coefficients)
            for t, k in EFFECT_OF.items():
                sim[k] = float(env_row.get(k, 1.0))
            if mat:
                sim["forest_effect"] *= float(np.mean(mat))
            if share:
                sim["raise_effect"] *= float(np.mean(share))
            sim["forest_effect"] = float(np.clip(sim["forest_effect"], 0.0, self.ranges["forest_effect"][1]))
        return x, {k: round(v, 5) for k, v in sim.items()}


def prepare(explore_name="abukuma_explore", log=print, grid=11):
    """Cost table of the exploration's decision variables (initial / annual per option or rate),
    using the RRI session for the measure quantities (no flood runs beyond the baseline)."""
    from ..rdm.run import setup_rri
    res = json.loads((PROCESSED / "explore" / explore_name / "results.json").read_text(encoding="utf-8"))
    check_supported(res)
    cfg = res["config"]
    out = PHYS_DIR / explore_name
    out.mkdir(parents=True, exist_ok=True)
    ev, mapinfo = setup_rri(cfg, out / "evaluations.jsonl", log)
    ev.gextra = {"baseline": "current"} if cfg["explore"].get("current_baseline", True) else {}
    EX.make_continuous(ev.variables, cfg["explore"], cfg["seed"])
    assert [v.name for v in ev.variables] == [v["name"] for v in res["variables"]], "variables differ from the exploration run"
    zero = [0] * len(ev.variables)
    table = []
    for i, v in enumerate(ev.variables):
        vals = np.linspace(0, 1, grid).tolist() if v.continuous else list(range(len(v.options)))
        ini, ann = [], []
        for a in vals:
            x = list(zero); x[i] = a
            p = ev.cost_parts(x)
            ini.append(p[0][i]); ann.append(p[1][i])
        table.append({"name": v.name, "values": vals, "initial": ini, "annual": ann})
        log(f"  {v.name}: 初期費用 {ini[-1]:.1f} 億円・維持 {ann[-1]:.2f} 億円/年（最大の選択肢）")
    (out / "cost_parts.json").write_text(json.dumps({"explore": explore_name, "table": table, "map": mapinfo}, ensure_ascii=False),
                                          encoding="utf-8")
    return table


def rri_evaluator(explore_name="abukuma_explore", log=print):
    """RRI evaluator over the exploration's variables (for re-evaluating selected roadmap states)."""
    from ..rdm.run import setup_rri
    res = json.loads((PROCESSED / "explore" / explore_name / "results.json").read_text(encoding="utf-8"))
    cfg = res["config"]
    ev, _ = setup_rri(cfg, PHYS_DIR / explore_name / "evaluations.jsonl", log)
    ev.gextra = {"baseline": "current"} if cfg["explore"].get("current_baseline", True) else {}
    EX.make_continuous(ev.variables, cfg["explore"], cfg["seed"])
    return ev


class WithInstitutions(Physics):
    """A physics with the institutional measures of the config appended as variables without raw
    columns (zoning, residential inducement, floodproofing, evacuation): the surrogate never sees
    them; sim.institutional_effects applies their assumed effects to exposure, damage and human risk.
    Their costs come from the config (cost_initial_oku / cost_annual_oku x level)."""

    def __init__(self, base, cfg):
        self.base, self.cfg = base, cfg
        self.n_base = len(base.variables)
        inst = []
        for k, m in cfg["measures"].items():
            if m.get("institutional"):
                cont = len(m.get("levels", [])) > 2
                inst.append({"name": m["label"], "type": k, "zone": "全域", "options": ["0%", "100%"] if cont else ["なし", "実施"],
                             "continuous": cont, "n_cells": 0, "institutional": True})
        self.variables = list(base.variables) + inst
        for a in ("zones", "layout", "ranges", "keys", "cell_km2", "farm_km2"):
            setattr(self, a, getattr(base, a))

    def __getattr__(self, k):                 # explore_name, validation, verification, to_rri, ...
        if k == "base":
            raise AttributeError(k)
        return getattr(self.base, k)

    def predict(self, raw, env):
        return self.base.predict(raw, env)

    def cost_parts(self, i, level):
        if i < self.n_base:
            return self.base.cost_parts(i, level)
        m = self.cfg["measures"][self.variables[i]["type"]]
        return float(m["cost_initial_oku"]) * float(level), float(m["cost_annual_oku"]) * float(level)


def with_institutions(phys, cfg):
    if isinstance(phys, WithInstitutions) or not any(m.get("institutional") for m in cfg["measures"].values()):
        return phys
    return WithInstitutions(phys, cfg)
