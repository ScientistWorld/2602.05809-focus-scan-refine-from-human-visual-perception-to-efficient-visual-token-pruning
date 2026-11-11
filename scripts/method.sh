#!/bin/bash
# Register the paper's method (Focus-Scan-Refine, paper defaults) as a pruner program.
#
# A method is "on disk" once checkpoints/pruners/<name>.json exists; the evaluators
# (scripts/evaluate_train.sh, scripts/evaluate_test.sh) run every registered pruner.
# Extra arguments are appended to the FSR command, e.g.
#   bash scripts/method.sh --rho 0.9 --kappa 1
#
# Usage: bash scripts/method.sh [fsr options]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p checkpoints/pruners
python - "$@" <<'PY'
import json, sys
cmd = ["python", "method/prune_fsr.py"] + sys.argv[1:]
spec = {"command": cmd, "description": "Focus-Scan-Refine (paper defaults: alpha=3, beta=1, rho=0.9, kappa=1)"}
open("checkpoints/pruners/fsr.json", "w").write(json.dumps(spec, indent=2) + "\n")
print("registered checkpoints/pruners/fsr.json:", " ".join(cmd))
PY
