# Train/Test Evaluation Design Notes (iteration 11)

- Splits: two disjoint sets of COCO val2017 images with at least three salient objects,
  chosen by a salted hash in `eval/test/build_items.py`. The visible split is
  `data/train_items.jsonl` (300 images). The hidden split is `eval/test/test_items.jsonl`
  (700 images). Both use the same builder, so they have the same families and difficulty.
- Each evaluator recomputes the pruner inputs from hash-pinned images and models. The test
  inputs exist only in a scratch directory during the run.
- Inventory: every `checkpoints/pruners/*.json` is scored. Before scoring, every method is
  written with worst values, and failures keep them.
- Isolation of the pruning step:
  - pruners run on CPU, with all free GPU memory reserved;
  - the time budget is 60 s + 0.1 s per question;
  - while a pruner runs, `eval/test/` is not traversable;
  - the evaluator scores only from its private in-memory token copy and rewrites tampered
    input files.
- Normalisation: full-token accuracy is recomputed in each run by the same evaluator.
