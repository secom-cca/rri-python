"""Parameter search on a calibration event, then evaluation on validation events.

python scripts/calibrate.py sample  --event kusu_2018 --n 30 --seed 1
python scripts/calibrate.py validate --tags cal_007 base
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_event import run_event  # noqa: E402

from rri_sim.cases import EVENTS  # noqa: E402
from rri_sim.data import ROOT  # noqa: E402

# search ranges (log-uniform where marked)
SPACE = {
    "forest.n": (0.05, 0.6, True),
    "forest.ka": (0.01, 0.3, True),
    "forest.soil_depth": (0.3, 1.5, False),
    "river.cw": (3.0, 12.0, False),
    "river.n_river": (0.025, 0.05, False),
}
THRESHOLDS = [1.0, 2.0, 5.0]
TABLE = ROOT / "outputs" / "calibration"


def _job(args):
    event, tag, sets, thr = args
    try:
        return run_event(event, tag, sets, thr, save=True, verbose=False)
    except Exception as e:  # keep the sweep alive
        return {"event": event, "tag": tag, "error": repr(e)}


def sample(event, n, seed, workers):
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n):
        sets = []
        for k, (lo, hi, log) in SPACE.items():
            v = np.exp(rng.uniform(np.log(lo), np.log(hi))) if log else rng.uniform(lo, hi)
            sets.append(f"{k}={v:.4g}")
        thr = float(rng.choice(THRESHOLDS))
        jobs.append((event, f"cal_{seed}_{i:03d}", sets, thr))
    jobs.insert(0, (event, "base_fill", None, 5.0))
    TABLE.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, jobs))
    with open(TABLE / f"{event}_seed{seed}.json", "w") as f:
        json.dump([{**r, "threshold": j[3]} for r, j in zip(res, jobs)], f, indent=1, default=float)
    ok = [r for r in res if "scores" in r]
    ok.sort(key=lambda r: -r["scores"]["ogase"]["KGE"])
    for r in ok[:10]:
        s = r["scores"]["ogase"]
        print(f"{r['tag']:14s} KGE={s['KGE']:.3f} NSE={s['NSE']:.3f} pk={s['peak_err']:+.2f} "
              f"t={s['peak_timing_h']:+.0f}h vol={s['volume_err']:+.2f} {r['overrides']}")


def validate(event_tags, workers):
    """Re-run given calibrated settings on all events."""
    jobs = []
    for cal_event, tag in event_tags:
        summ = json.load(open(ROOT / "outputs" / "events" / cal_event / tag / "summary.json"))
        thr = summ.get("threshold")
        for ev in [e for e, v in EVENTS.items() if v.basin == EVENTS[cal_event].basin]:
            jobs.append((ev, tag, summ["overrides"] or None, thr))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, jobs))
    for r in res:
        if "scores" in r:
            s = r["scores"]["ogase"]
            print(f"{r['event']:10s} {r['role']:11s} {r['tag']:14s} KGE={s['KGE']:.3f} "
                  f"NSE={s['NSE']:.3f} pk={s['peak_err']:+.2f} t={s['peak_timing_h']:+.0f}h "
                  f"vol={s['volume_err']:+.2f}")
        else:
            print(r)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "validate"])
    ap.add_argument("--event", default="kusu_2018")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tags", nargs="*", default=[])
    a = ap.parse_args()
    if a.mode == "sample":
        sample(a.event, a.n, a.seed, a.workers)
    else:
        validate([("kusu_2018", t) for t in a.tags], a.workers)
