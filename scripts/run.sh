#!/bin/bash
# Job script (gym container, workspace at /home/user, no internet, 1 GPU).
# Hardening run: reference + baselines through the official scripts, then FSR variants
# and shortcut pruners through the same evaluators. Stages are resumable.
# `bash scripts/reproduce.sh` alone reproduces the reference end to end.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
H=checkpoints/hardening
mkdir -p "$H"

stage() {
    local name="$1"; shift
    if [ -f "$H/$name.done" ]; then echo "== skip $name (done)"; return 0; fi
    echo "== stage $name  $(date +%T)"
    local t0=$SECONDS
    if "$@"; then
        touch "$H/$name.done"
        echo "== stage $name finished in $((SECONDS - t0)) s"
    else
        echo "== stage $name FAILED"
    fi
}

nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/download.sh

# registries: reference + baselines (official), variants and shortcuts (hardening)
rm -rf checkpoints/pruners
bash scripts/method.sh
bash scripts/baseline.sh
python redteam/hardening/register.py variants "$H/dev_pruners"
python redteam/hardening/register.py shortcuts "$H/shortcut_pruners"

stage official_train bash -c "bash scripts/evaluate_train.sh && cp scoring/scores_train.json scoring/scores.json"
stage official_test bash scripts/evaluate_test.sh
stage dev_test python eval/test/evaluate.py --pruners-dir "$H/dev_pruners" \
    --output "$H/dev_scores_test.json" --save-answers "$H/answers_test"
for k in 64 32; do for m in fsr fsr_allmass fsr_rho05 fsr_rho07; do
    echo "--- test $m K=$k: $(tail -1 checkpoints/eval_logs/test/${m}_${k}.log 2>/dev/null)"; done; done
stage dev_train python eval/train/evaluate.py --pruners-dir "$H/dev_pruners" \
    --output "$H/dev_scores_train.json" --save-answers "$H/answers_train"
# shortcut pruners through the hidden test; the forging case writes into scoring/, so the
# official score files are restored afterwards (the evaluator itself overwrites its output).
stage shortcuts_test bash -c "mkdir -p $H/scoring_backup && cp scoring/*.json $H/scoring_backup/ && \
    python eval/test/evaluate.py --pruners-dir $H/shortcut_pruners --output $H/shortcut_scores_test.json; \
    rc=\$?; cp $H/scoring_backup/*.json scoring/; exit \$rc"
cat checkpoints/hardening_probe_64.json 2>/dev/null

echo "== analysis"
python redteam/hardening/analyze.py "$H/answers_test" --json "$H/analysis_test.json"
python redteam/hardening/analyze.py "$H/answers_train" --json "$H/analysis_train.json"
python -c "import json; d=json.load(open('$H/shortcut_scores_test.json'))['experiments']; [print(k, v, d['_diagnostics'].get(k, {}).get('reason', '')[:160]) for k, v in d['token_pruning_core']['results'].items()]"
cat scoring/scores_test.json
python validate.py --post-design --compare || true
echo "== done $(date +%T)"
