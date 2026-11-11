#!/bin/bash
# Evaluate every registered pruner (checkpoints/pruners/*.json) on the visible train
# split and write scoring/scores_train.json. The pruner inputs for this split are left
# in data/downloads/pruner_inputs_train/ for offline (CPU) development.
#
# Usage: bash scripts/evaluate_train.sh [--pruners-dir DIR] [--device cuda:0]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p scoring
python eval/train/evaluate.py "$@"
