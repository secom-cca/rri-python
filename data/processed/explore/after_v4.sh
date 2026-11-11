#!/bin/zsh
# exploration v4 -> surrogate / cost table -> Robust (11.9) -> Multi-Actor (11.10) -> Adaptive (11.11) DAPP
cd /Users/nakashima/rri-python
set -o pipefail
.venv/bin/python data/processed/explore/run_v4.py > data/processed/explore/abukuma_explore_v4.log 2>&1 || { echo "EXPLORE FAILED"; exit 1; }
.venv/bin/python scripts/run_roadmap.py --prepare > data/processed/roadmap/prepare_v4.log 2>&1 || { echo "PREPARE FAILED"; exit 1; }
.venv/bin/python scripts/run_dapp.py --robust > data/processed/dapp/robust_v4.log 2>&1 || { echo "ROBUST FAILED"; exit 1; }
.venv/bin/python scripts/run_dapp.py --multi > data/processed/dapp/multi_v4.log 2>&1 || { echo "MULTI FAILED"; exit 1; }
.venv/bin/python scripts/run_dapp.py --adaptive > data/processed/dapp/adaptive_v4.log 2>&1 || { echo "ADAPTIVE FAILED"; exit 1; }
echo "ALL DONE"
