# data/

- `train_items.jsonl`: the visible split, 2,100 questions on 300 COCO val2017 images. Each
  line has `qid`, `image` (file in `data/downloads/coco/val2017/`), `image_sha256`,
  `family` (exist / count / caption / absent), `question`, `suffix` (the answer-format
  instruction) and `answer`. The hidden split is built by the same procedure on 700 other
  images and is not in this directory.
- `pruner_io.py`: helper for pruner programs. It reads the evaluator's input directory and
  writes selections.
- `downloads/` (gitignored, filled by `scripts/download.sh`): LLaVA-1.5-7B, CLIP
  ViT-L/14-336 and the COCO val2017 images. After `scripts/evaluate_train.sh` it also holds
  `pruner_inputs_train/`, the visible split's pruner inputs, for offline development.
