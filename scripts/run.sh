#!/bin/bash
# Job script (gym container, workspace at /home/user, no internet, 1 GPU).
# 1. Reproduce the reference end to end: FSR + baselines through the official
#    train/test evaluators (scripts/reproduce.sh).
# 2. Hardening checks: shortcut pruners through the hidden test (redteam/hardening/run_checks.sh).
# 3. Validate the workspace.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

t0=$SECONDS
bash scripts/reproduce.sh || echo "reproduce.sh FAILED"
echo "== reproduce.sh took $((SECONDS - t0)) s"
cat scoring/scores_test.json | python -c "import json,sys; d=json.load(sys.stdin)['experiments']['token_pruning_core']['results']; [print(k, v) for k, v in d.items()]"

bash redteam/hardening/run_checks.sh
python validate.py --post-design --compare || true
