"""FastAPI server for the 3D flood-management GUI.

    python -m rri_sim.gui            # http://127.0.0.1:8765
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import costs as C
from .engine import Engine

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="RRI-python 流域治水シミュレーター")
NO_CACHE = {"Cache-Control": "no-cache"}   # always revalidate, so an edited app.js is picked up on reload
engine = Engine()


class RunRequest(BaseModel):
    basin: str
    event: str
    edits: list[dict] = []
    params: dict = {}
    res: float | None = None          # grid size [m]; None = basin default (250 m)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


@app.get("/play")
def play():
    """Game UI for classes (budget, mission, score); same engine as the expert UI at /."""
    return FileResponse(STATIC / "play.html", headers=NO_CACHE)


@app.get("/rdm")
def rdm_page():
    """Results of Robust Decision Making runs (scripts/run_rdm.py; docs/09_rdm.md)."""
    return FileResponse(STATIC / "rdm.html", headers=NO_CACHE)


@app.get("/api/rdm")
def rdm_runs():
    from ..rdm.run import RDM_DIR
    out = []
    for p in sorted(RDM_DIR.glob("*/results.json"), key=lambda p: -p.stat().st_mtime):
        out.append({"name": p.parent.name, "updated": p.stat().st_mtime})
    return out


def _rdm_file(name, fname):
    import re
    from ..rdm.run import RDM_DIR
    if not re.fullmatch(r"[\w\-.]+", name) or name.startswith("."):
        raise HTTPException(400, "bad name")
    p = RDM_DIR / name / fname
    if not p.exists():
        raise HTTPException(404, f"no {fname} for RDM run {name}")
    return FileResponse(p, media_type="application/json", headers=NO_CACHE)


@app.get("/api/rdm/{name}")
def rdm_result(name: str):
    return _rdm_file(name, "results.json")


@app.get("/api/rdm/{name}/sensitivity")
def rdm_sensitivity(name: str):
    """Sensitivity analysis of the run (scripts/run_sensitivity.py; docs/10_sensitivity.md)."""
    return _rdm_file(name, "sensitivity.json")


@app.get("/dapp")
def dapp_page():
    """Dynamic Adaptive Policy Pathways results (scripts/run_dapp.py; docs/11_dapp.md)."""
    return FileResponse(STATIC / "dapp.html", headers=NO_CACHE)


@app.get("/api/dapp")
def dapp_runs():
    from ..dapp.run import DAPP_DIR
    return [{"name": p.parent.name, "updated": p.stat().st_mtime}
            for p in sorted(DAPP_DIR.glob("*/results.json"), key=lambda p: -p.stat().st_mtime)]


@app.get("/api/dapp/{name}")
def dapp_result(name: str):
    import re
    from ..dapp.run import DAPP_DIR
    if not re.fullmatch(r"[\w\-.]+", name) or name.startswith("."):
        raise HTTPException(400, "bad name")
    p = DAPP_DIR / name / "results.json"
    if not p.exists():
        raise HTTPException(404, f"no DAPP run {name}")
    return FileResponse(p, media_type="application/json", headers=NO_CACHE)


class RobustDappRequest(BaseModel):
    name: str
    trigger: dict = {}       # robust.trigger overrides (ratio_of_threshold, horizon_years, safety, use_observed, use_projection)
    targets: dict = {}       # robust.targets overrides (damage_ratio, pop_ratio, from_year)
    n_scenarios: int | None = None


@app.post("/api/dapp/robust/simulate")
def dapp_robust_simulate(req: RobustDappRequest):
    """Robust DAPP steps 4-5 again with other triggers / targets (saved network, surrogate only; docs/11 11.9)."""
    import json as _json
    import re
    from ..dapp import robust as RB
    from ..roadmap import run as RR
    from ..roadmap.physics import SurrogatePhysics
    if not re.fullmatch(r"[\w\-.]+", req.name) or req.name.startswith("."):
        raise HTTPException(400, "bad name")
    d = RB.DAPP_DIR / req.name
    if not (d / "network.npz").exists():
        raise HTTPException(404, f"no Robust DAPP run {req.name}")
    ex = _json.loads((d / "config.json").read_text(encoding="utf-8"))["robust"]["explore_run"]
    try:
        if ex not in _roadmap_phys:
            _roadmap_phys[ex] = SurrogatePhysics(ex, log=lambda m: None)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))
    res = RB.resimulate(d, req.trigger, req.targets, req.n_scenarios, phys=_roadmap_phys[ex])
    return _json.loads(_json.dumps(res, ensure_ascii=False, default=RR._json))


class MultiRecalcRequest(BaseModel):
    name: str                # the displayed Multi-Actor run (its settings are the base)
    actors: dict = {}        # {actor: {per_decade_oku, targets: {damage_ratio, pop_ratio}, weights: {...}}}
    basin_weights: dict = {}
    size: str = "standard"   # multi.gui.sizes: fast / standard / full


def _dapp_dir(name):
    import re
    from ..dapp.multi import DAPP_DIR
    if not re.fullmatch(r"[\w\-.]+", name) or name.startswith("."):
        raise HTTPException(400, "bad name")
    return DAPP_DIR / name


@app.post("/api/dapp/multi/recalculate")
def dapp_multi_recalculate(req: MultiRecalcRequest):
    """Individual / Joint / Single again with other per-decade capacities, targets and weights (job; docs/11 11.10)."""
    import json as _json
    from ..dapp import multi as MU
    from ..roadmap.physics import SurrogatePhysics
    d = _dapp_dir(req.name)
    if not (d / "config.json").exists():
        raise HTTPException(404, f"no Multi-Actor run {req.name}")
    try:
        cfg = MU.gui_config(_json.loads((d / "config.json").read_text(encoding="utf-8")),
                            {"actors": req.actors, "basin_weights": req.basin_weights}, req.size)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    name = cfg["multi"]["name"]
    if (MU.DAPP_DIR / name / "summary.json").exists():
        return {"job": None, "name": name}                       # the same inputs were computed before

    def fn(progress):
        ex = cfg["robust"]["explore_run"]
        if ex not in _roadmap_phys:
            progress(0.01, "代理モデルを読み込み中")
            _roadmap_phys[ex] = SurrogatePhysics(ex, log=lambda m: None)
        return MU.recalculate(cfg, _roadmap_phys[ex], progress)
    return {"job": engine._start(fn), "name": name}


@app.get("/api/dapp/multi/summaries/{name}")
def dapp_multi_summaries(name: str):
    """Conditions and headline results of a Multi-Actor run's family (the base run and its GUI recalculations)."""
    from ..dapp import multi as MU
    _dapp_dir(name)
    return MU.summaries(name)


class AdaptiveRecalcRequest(BaseModel):
    name: str                # the displayed Adaptive DAPP run (its settings and recommended rules are the base)
    targets: dict = {}       # damage_ratio, pop_ratio, from_year
    noise_damage: float | None = None
    noise_climate: float | None = None
    trend_window: int | None = None
    forecast_noise: float | None = None
    planning: dict | None = None   # planning climate range {range: low|medium|high|low_high|custom, rain_2100, slr_2100} (search)
    ranges: list[str] = []         # planning_batch: these presets with the same targets / information quality
    mode: str = "fixed"      # fixed: re-evaluate the recommended rules; search: search the rules again
    size: str = "fast"


@app.post("/api/dapp/adaptive/recalculate")
def dapp_adaptive_recalculate(req: AdaptiveRecalcRequest):
    """Adaptive DAPP again with other targets / information quality (job; docs/11 11.11.5)."""
    import json as _json
    from ..dapp import adaptive as AD
    from ..roadmap.physics import SurrogatePhysics
    d = _dapp_dir(req.name)
    if not (d / "config.json").exists():
        raise HTTPException(404, f"no Adaptive DAPP run {req.name}")
    try:
        cfg = AD.gui_config(_json.loads((d / "config.json").read_text(encoding="utf-8")), req.model_dump(), req.mode, req.size)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    name = cfg["adaptive"]["name"]
    if (AD.DAPP_DIR / name / "summary.json").exists():
        return {"job": None, "name": name}

    def fn(progress):
        ex = cfg["robust"]["explore_run"]
        if ex not in _roadmap_phys:
            progress(0.01, "代理モデルを読み込み中")
            _roadmap_phys[ex] = SurrogatePhysics(ex, log=lambda m: None)
        return AD.recalculate(cfg, _roadmap_phys[ex], progress)
    return {"job": engine._start(fn), "name": name}


@app.post("/api/dapp/adaptive/planning_batch")
def dapp_adaptive_planning_batch(req: AdaptiveRecalcRequest):
    """One search per planning climate range (same targets / information quality), in a row (job; docs/11 11.11.6)."""
    import json as _json
    from ..dapp import adaptive as AD
    from ..roadmap.physics import SurrogatePhysics
    d = _dapp_dir(req.name)
    if not (d / "config.json").exists():
        raise HTTPException(404, f"no Adaptive DAPP run {req.name}")
    base = _json.loads((d / "config.json").read_text(encoding="utf-8"))
    try:
        cfgs = [AD.gui_config(base, {**req.model_dump(), "planning": {"range": k}}, "search", req.size) for k in req.ranges or ["low", "medium", "high", "low_high"]]
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    names = [c["adaptive"]["name"] for c in cfgs]
    if all((AD.DAPP_DIR / n / "summary.json").exists() for n in names):
        return {"job": None, "name": names[0], "names": names}

    def fn(progress):
        ex = cfgs[0]["robust"]["explore_run"]
        if ex not in _roadmap_phys:
            progress(0.0, "代理モデルを読み込み中")
            _roadmap_phys[ex] = SurrogatePhysics(ex, log=lambda m: None)
        return AD.recalculate_many(cfgs, _roadmap_phys[ex], progress)
    return {"job": engine._start(fn), "name": names[0], "names": names}


@app.get("/api/dapp/adaptive/summaries/{name}")
def dapp_adaptive_summaries(name: str):
    from ..dapp import adaptive as AD
    _dapp_dir(name)
    return AD.summaries(name)


@app.get("/explore")
def explore_page():
    """Exploration without stakeholder weights (scripts/run_explore.py; docs/12_explore.md)."""
    return FileResponse(STATIC / "explore.html", headers=NO_CACHE)


@app.get("/api/explore")
def explore_runs():
    from ..rdm.explore import EXPLORE_DIR
    return [{"name": p.parent.name, "updated": p.stat().st_mtime}
            for p in sorted(EXPLORE_DIR.glob("*/results.json"), key=lambda p: -p.stat().st_mtime)]


@app.get("/api/explore/{name}")
def explore_result(name: str):
    import re
    from ..rdm.explore import EXPLORE_DIR
    if not re.fullmatch(r"[\w\-.]+", name) or name.startswith("."):
        raise HTTPException(400, "bad name")
    p = EXPLORE_DIR / name / "results.json"
    if not p.exists():
        raise HTTPException(404, f"no exploration run {name}")
    return FileResponse(p, media_type="application/json", headers=NO_CACHE)


@app.get("/api/explore/{name}/scopes")
def explore_scopes(name: str):
    """Trade spaces by implementing actor (scripts/run_explore.py --scopes NAME)."""
    import re
    from ..rdm.explore import EXPLORE_DIR
    if not re.fullmatch(r"[\w\-.]+", name) or name.startswith("."):
        raise HTTPException(400, "bad name")
    p = EXPLORE_DIR / name / "scopes.json"
    if not p.exists():
        raise HTTPException(404, f"no actor trade spaces for {name}")
    return FileResponse(p, media_type="application/json", headers=NO_CACHE)


@app.get("/roadmap")
def roadmap_page():
    """Climate-adaptation roadmaps P0-P3 (scripts/run_roadmap.py; docs/13_roadmap.md)."""
    return FileResponse(STATIC / "roadmap.html", headers=NO_CACHE)


@app.get("/api/roadmap")
def roadmap_runs():
    from ..roadmap.run import ROADMAP_DIR
    return [{"name": p.parent.name, "updated": p.stat().st_mtime}
            for p in sorted(ROADMAP_DIR.glob("*/results.json"), key=lambda p: -p.stat().st_mtime)]


@app.get("/api/roadmap-defaults")
def roadmap_defaults():
    """Settings and world factors for the form of /roadmap."""
    from ..roadmap import config as RCF
    from ..roadmap import world as RW
    cfg = RCF.load()
    return {"config": cfg, "factors": RW.factors(cfg)}


@app.get("/api/roadmap/{name}")
def roadmap_result(name: str):
    import re
    from ..roadmap.run import ROADMAP_DIR
    if not re.fullmatch(r"[\w\-.]+", name) or name.startswith("."):
        raise HTTPException(400, "bad name")
    p = ROADMAP_DIR / name / "results.json"
    if not p.exists():
        raise HTTPException(404, f"no roadmap run {name}")
    return FileResponse(p, media_type="application/json", headers=NO_CACHE)


class RoadmapSimRequest(BaseModel):
    config: dict = {}
    unit: dict = {}          # world factors in [0, 1] (missing = nominal)
    modes: dict = {}


_roadmap_phys = {}


@app.post("/api/roadmap/simulate")
def roadmap_simulate(req: RoadmapSimRequest):
    """One world with changed delays / participation / social values (surrogate physics, a few seconds)."""
    from ..roadmap import config as RCF
    from ..roadmap import run as RR
    from ..roadmap.physics import SurrogatePhysics
    cfg = RCF.load(req.config)
    name = cfg["physics"]["explore_run"]
    try:
        if name not in _roadmap_phys:
            _roadmap_phys[name] = SurrogatePhysics(name, log=lambda m: None)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))
    import json as _json
    res = RR.simulate_one(_roadmap_phys[name], req.config, unit=req.unit, modes=req.modes or None)
    return _json.loads(_json.dumps(res, ensure_ascii=False, default=RR._json))


@app.get("/api/basins")
def basins():
    return engine.basins()


@app.get("/api/basin/{key}")
def basin(key: str, res: float | None = None):
    if res is not None and not 50.0 <= res <= 2000.0:
        raise HTTPException(400, "res must be 50-2000 m")
    try:
        return engine.session(key, res).static_payload()
    except KeyError:
        raise HTTPException(404, f"unknown basin {key}")


class PrepareRequest(BaseModel):
    basin: str
    res: float | None = None


@app.post("/api/prepare")
def prepare(req: PrepareRequest):
    """Start loading / building a basin (job with progress); then GET /api/basin/{key}."""
    if req.basin not in engine.basins_keys():
        raise HTTPException(404, f"unknown basin {req.basin}")
    if engine.is_loaded(req.basin, req.res):
        return {"job": None}
    return {"job": engine.prepare_basin(req.basin, req.res)}


@app.post("/api/run")
def run(req: RunRequest):
    return {"job": engine.submit(req.basin, req.event, req.edits, req.params, req.res)}


@app.post("/api/estimate")
def estimate(req: RunRequest):
    """Project cost of the edits without simulating (loaded basins only)."""
    try:
        return engine.estimate(req.basin, req.event, req.edits, req.params, req.res)
    except KeyError:
        raise HTTPException(409, "basin not loaded; POST /api/prepare first")


class CostRequest(BaseModel):
    items: list[dict] = []
    unit_costs: dict = {}


@app.post("/api/costs")
def costs(req: CostRequest):
    """Recompute project costs for given measure quantities and unit costs."""
    return C.estimate([{"type": i["type"], "q": i.get("q", {})} for i in req.items], req.unit_costs)


@app.get("/api/job/{jid}")
def job(jid: str):
    j = engine.jobs.get(jid)
    if j is None:
        raise HTTPException(404, "unknown job")
    import time
    out = {k: j[k] for k in ("status", "progress", "message")}
    now = time.time()
    out["elapsed_s"] = now - j["started"] if j.get("started") else 0.0
    out["phase_elapsed_s"] = now - j["phase_started"] if j.get("phase_started") else 0.0
    if j["status"] == "done":
        out["result"] = j["result"]
    return out


@app.middleware("http")
async def _no_cache_static(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):
        resp.headers.update(NO_CACHE)
    return resp


app.mount("/static", StaticFiles(directory=STATIC), name="static")
