#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory can be any checkout location
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/download.sh
mkdir -p checkpoints
rm -rf checkpoints/budget16 checkpoints/budget32 checkpoints/budget48
mkdir -p checkpoints/budget16 checkpoints/budget32 checkpoints/budget48

bash scripts/baseline.sh --budget 16 --output-dir checkpoints/budget16 --methods fastv divprune cdpruner
bash scripts/method.sh --budget 16 --output checkpoints/budget16/fsr_outputs.jsonl

bash scripts/baseline.sh --budget 32 --output-dir checkpoints/budget32 --methods fastv divprune cdpruner
bash scripts/method.sh --budget 32 --output checkpoints/budget32/fsr_outputs.jsonl

bash scripts/baseline.sh --budget 48 --output-dir checkpoints/budget48 --methods fastv divprune cdpruner
bash scripts/method.sh --budget 48 --output checkpoints/budget48/fsr_outputs.jsonl

bash scripts/evaluate.sh
