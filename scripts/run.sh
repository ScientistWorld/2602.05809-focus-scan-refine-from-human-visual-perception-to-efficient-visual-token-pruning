#!/bin/bash
# Job script (runs inside the gym container, workspace at /home/user, no internet).
# Current content: SMOKE TEST of the rebuilt LLaVA-based pipeline on a few images.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "== environment"
nvidia-smi || true
nproc; free -g; df -h /home/user | tail -1
python -c "import torch, transformers, numpy; print('torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
ls -laL data/downloads/llava-v1.5-7b data/downloads/clip-vit-large-patch14-336 | head -30
ls data/downloads/coco/val2017 | wc -l

echo "== registries"
rm -rf checkpoints/pruners checkpoints/smoke
bash scripts/method.sh
bash scripts/baseline.sh
python redteam/hardening/register.py variants checkpoints/dev_pruners
python redteam/hardening/register.py shortcuts checkpoints/shortcut_pruners
mkdir -p checkpoints/smoke

echo "== smoke: train split, all variants, 6 images"
time python eval/train/evaluate.py --max-images 6 --pruners-dir checkpoints/dev_pruners \
    --output checkpoints/smoke/scores_train_dev.json --save-answers checkpoints/smoke/train_dev \
    --inputs-dir checkpoints/smoke/inputs_train
echo "exit $?"
python -c "import json; d=json.load(open('checkpoints/smoke/scores_train_dev.json')); print(json.dumps(d, indent=1)[:6000])"
head -40 checkpoints/smoke/train_dev/_full_tokens.jsonl
head -12 checkpoints/smoke/train_dev/fsr__64.jsonl
for f in checkpoints/eval_logs/train/fsr_64.log checkpoints/eval_logs/train/fsr_dpp_scan_64.log checkpoints/eval_logs/train/cdpruner_64.log; do echo "--- $f"; cat $f; done

echo "== smoke: test split, shortcuts, 6 images"
time python eval/test/evaluate.py --max-images 6 --pruners-dir checkpoints/shortcut_pruners \
    --output checkpoints/smoke/scores_test_shortcuts.json --save-answers checkpoints/smoke/test_sc
echo "exit $?"
python -c "import json; d=json.load(open('checkpoints/smoke/scores_test_shortcuts.json')); print(json.dumps(d, indent=1)[:8000])"
cat checkpoints/hardening_probe_64.json 2>/dev/null
for f in checkpoints/eval_logs/test/sc_llm_search_64.log checkpoints/eval_logs/test/sc_tamper_inputs_64.log; do echo "--- $f"; tail -20 $f; done
ls -la checkpoints scoring
echo "== done"
