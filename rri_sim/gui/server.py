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
