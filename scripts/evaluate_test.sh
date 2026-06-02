#!/bin/bash
# Evaluate the held-out test slice. Mirrors scripts/evaluate.sh's CLI contract
# while writing scoring/scores_test.json by default.

set -e

cd /home/user

mkdir -p scoring

python eval/test/evaluate.py "$@"
