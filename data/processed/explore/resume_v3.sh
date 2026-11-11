#!/bin/zsh
# resume after the exploration: surrogate / cost table -> Robust (11.9) -> Multi-Actor (11.10) -> Adaptive (11.11)
cd /Users/nakashima/rri-python
.venv/bin/python scripts/run_roadmap.py --prepare > data/processed/roadmap/prepare_v3.log 2>&1 || { echo "PREPARE FAILED"; exit 1; }
.venv/bin/python scripts/run_dapp.py --robust > data/processed/dapp/robust_v3.log 2>&1 || { echo "ROBUST FAILED"; exit 1; }
.venv/bin/python scripts/run_dapp.py --multi > data/processed/dapp/multi_v3.log 2>&1 || { echo "MULTI FAILED"; exit 1; }
.venv/bin/python scripts/run_dapp.py --adaptive > data/processed/dapp/adaptive_v3.log 2>&1 || { echo "ADAPTIVE FAILED"; exit 1; }
echo "ALL DONE"
