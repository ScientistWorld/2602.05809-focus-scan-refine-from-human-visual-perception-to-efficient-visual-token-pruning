#!/bin/bash
# Hidden test: evaluate every registered pruner (checkpoints/pruners/*.json) on the
# held-out split and write scoring/scores_test.json. Runs with no arguments.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p scoring
python eval/test/evaluate.py "$@"
