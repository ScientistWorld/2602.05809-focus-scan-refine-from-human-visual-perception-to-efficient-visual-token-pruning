#!/bin/bash
# Evaluate every registered pruner (checkpoints/pruners/*.json) on the visible split
# and write scoring/scores.json. Uses only code in eval/ (never imports method/).
#
# Requires a GPU for the frozen LLaVA-1.5-7B language model and CLIP encoder.
# Extra arguments are passed to eval/train/evaluate.py (e.g. --pruners-dir DIR).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p scoring
python eval/train/evaluate.py --output scoring/scores.json "$@"
