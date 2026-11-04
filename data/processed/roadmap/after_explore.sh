#!/bin/zsh
cd /Users/nakashima/rri-python
until grep -qE "^完了:|Traceback" data/processed/explore/abukuma_explore_v2.log; do sleep 60; done
grep -q Traceback data/processed/explore/abukuma_explore_v2.log && { echo "EXPLORE FAILED"; exit 1; }
.venv/bin/python scripts/run_roadmap.py --prepare > data/processed/roadmap/prepare_v2.log 2>&1 || { echo "PREPARE FAILED"; exit 1; }
.venv/bin/python scripts/run_roadmap.py > data/processed/roadmap/full_v2.log 2>&1 || { echo "ROADMAP FAILED"; exit 1; }
echo "ALL DONE"
