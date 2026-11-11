#!/bin/bash
# Shortcut checks through the hidden test (GPU job). Run after scripts/reproduce.sh.
#   - every shortcut pruner in redteam/hardening/shortcuts.py, scored by eval/test/evaluate.py
#   - the answer-reading shortcut on the visible split, where its answer file is readable,
#     to measure what reading answers would pay
# Official score files are backed up and restored (one case forges them on purpose).

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
H=checkpoints/hardening
mkdir -p "$H/scoring_backup"
cp scoring/*.json "$H/scoring_backup/"

python redteam/hardening/register.py shortcuts "$H/shortcut_pruners"
python eval/test/evaluate.py --pruners-dir "$H/shortcut_pruners" --output "$H/shortcut_scores_test.json"
echo "test shortcut evaluation exit $?"
cp "$H/scoring_backup/"*.json scoring/
cat checkpoints/hardening_probe_64.json 2>/dev/null; echo
tail -3 checkpoints/eval_logs/test/sc_read_answers_64.log 2>/dev/null

mkdir -p "$H/read_answers_train"
for n in fsr sc_read_answers; do cp "$H/shortcut_pruners/$n.json" "$H/read_answers_train/" 2>/dev/null; done
cp checkpoints/pruners/fsr.json "$H/read_answers_train/fsr.json"
python eval/train/evaluate.py --pruners-dir "$H/read_answers_train" --output "$H/read_answers_train_scores.json" \
    --inputs-dir checkpoints/.read_answers_inputs_train
cp "$H/scoring_backup/"*.json scoring/
tail -3 checkpoints/eval_logs/train/sc_read_answers_64.log 2>/dev/null

python - <<'PY'
import json
for f in ("checkpoints/hardening/shortcut_scores_test.json", "checkpoints/hardening/read_answers_train_scores.json"):
    d = json.load(open(f))["experiments"]
    print("==", f)
    for k, v in d["token_pruning_core"]["results"].items():
        print(f"  {k:22s} {v}  {d['_diagnostics'].get(k, {}).get('reason', '')[:140]}")
PY
