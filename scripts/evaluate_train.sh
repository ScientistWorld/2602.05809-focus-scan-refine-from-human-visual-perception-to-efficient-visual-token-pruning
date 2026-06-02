#!/bin/bash
# Evaluate the visible train slice. Mirrors scripts/evaluate.sh's CLI contract
# while writing scoring/scores_train.json by default.

set -e

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py "$@"
