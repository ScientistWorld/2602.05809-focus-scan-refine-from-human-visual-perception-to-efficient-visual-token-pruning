#!/bin/bash
# Reproduce the gym's reference results end to end (GPU required, no internet needed
# once scripts/download.sh has run):
#   1. register FSR and the baselines as pruner programs
#   2. score them on the visible split  -> scoring/scores_train.json (+ scores.json)
#   3. score them on the hidden split   -> scoring/scores_test.json

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/download.sh
rm -rf checkpoints/pruners
bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh
