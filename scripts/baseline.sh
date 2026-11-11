#!/bin/bash
# Register training-free pruning baselines as pruner programs (checkpoints/pruners/<name>.json).
#
#   random    uniform random subset
#   cls_topk  attention-based: top-K by vision-encoder [CLS] attention
#   divprune  similarity-based: max-min diversity (DivPrune)
#   cdpruner  joint: relevance-conditioned DPP (CDPruner)
#   pool      (optional) spatial average pooling into K merged tokens
#
# Usage: bash scripts/baseline.sh [name ...]      (default: random cls_topk divprune cdpruner)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ "$#" -eq 0 ]; then
    set -- random cls_topk divprune cdpruner
fi
mkdir -p checkpoints/pruners
python - "$@" <<'PY'
import json, sys
for name in sys.argv[1:]:
    cmd = ["python", "baseline/prune_baseline.py", "--method", name]
    open(f"checkpoints/pruners/{name}.json", "w").write(json.dumps({"command": cmd}, indent=2) + "\n")
    print(f"registered checkpoints/pruners/{name}.json")
PY
