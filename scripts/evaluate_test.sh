#!/bin/bash
# Evaluate the held-out test slice. Mirrors scripts/evaluate.sh's CLI contract
# while writing scoring/scores_test.json by default.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p scoring

python eval/test/evaluate.py "$@"
