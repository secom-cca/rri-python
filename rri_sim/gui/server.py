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
engine = Engine()


class RunRequest(BaseModel):
    basin: str
    event: str
    edits: list[dict] = []
    params: dict = {}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/basins")
def basins():
    return engine.basins()


@app.get("/api/basin/{key}")
def basin(key: str):
    try:
        return engine.session(key).static_payload()
    except KeyError:
        raise HTTPException(404, f"unknown basin {key}")


@app.post("/api/run")
def run(req: RunRequest):
    return {"job": engine.submit(req.basin, req.event, req.edits, req.params)}


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
    out = {k: j[k] for k in ("status", "progress", "message")}
    if j["status"] == "done":
        out["result"] = j["result"]
    return out


app.mount("/static", StaticFiles(directory=STATIC), name="static")
