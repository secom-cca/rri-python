"""Climate-adaptation roadmaps under deep uncertainty: P0-P3, updateability, option coverage,
lock-in, explainability (docs/13_roadmap.md).

    python scripts/run_roadmap.py --prepare                 # once: cost table + surrogate cache (RRI session, ~5 min)
    python scripts/run_roadmap.py --quick                   # small experiment (8 worlds; minutes)
    python scripts/run_roadmap.py                           # 阿武隈川全域 (64 worlds + Morris + experiments)
    python scripts/run_roadmap.py --config my.json          # settings merged over rri_sim/roadmap/config.py DEFAULT
    python scripts/run_roadmap.py --print-default > my.json
    python scripts/run_roadmap.py --verify 8 --name abukuma_roadmap   # re-run RRI for 8 states of a finished run

Results: data/processed/roadmap/<name>/{config.json, results.json}; GUI: http://127.0.0.1:8765/roadmap
"""
import argparse
import json
import sys

from rri_sim.roadmap import config as CF
from rri_sim.roadmap import run as RUN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--prepare", action="store_true", help="cost table and surrogate cache of the exploration run")
    ap.add_argument("--parts", help="comma-separated subset of " + ",".join(RUN.PARTS))
    ap.add_argument("--jobs", type=int, help="worker processes (0 = CPUs - 1)")
    ap.add_argument("--verify", type=int, default=0, help="re-run RRI for this many states of the run named --name")
    ap.add_argument("--print-default", action="store_true")
    a = ap.parse_args()
    say = lambda m: print(m, flush=True)  # noqa: E731
    if a.print_default:
        json.dump(CF.DEFAULT, sys.stdout, ensure_ascii=False, indent=2)
        return
    cfg = CF.load(a.config) if a.config else CF.load()
    if a.prepare:
        from rri_sim.roadmap import physics as PH
        PH.prepare(cfg["physics"]["explore_run"], log=say)
        PH.SurrogatePhysics(cfg["physics"]["explore_run"], log=say)
        return
    if a.quick:
        cfg = CF._merge(cfg, CF.QUICK)
    if a.name:
        cfg["name"] = a.name
    if a.jobs is not None:
        cfg["experiment"]["n_jobs"] = a.jobs
    if a.verify:
        v = RUN.verify_rri(RUN.ROADMAP_DIR / cfg["name"] / "results.json", a.verify, log=say)
        say(f"RRI との比較: R² {v.get('r2')}, 相対 RMSE {v.get('rel_rmse')}")
        return
    RUN.run(cfg, log=say, parts=tuple(a.parts.split(",")) if a.parts else RUN.PARTS)


if __name__ == "__main__":
    main()
