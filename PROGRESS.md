# Progress

## Current state (iteration 11, hardening)

- Gym rebuilt on a frozen LLaVA-1.5-7B: CLIP ViT-L/14-336 → projector → Vicuna-7B, weights
  hash-pinned. Data is COCO val2017: 2,100 visible questions on 300 images and 4,898 hidden
  questions on 700 images, in four families (existence via the POPE protocol, counting,
  caption matching, absent object).
- Methods are pruner programs (`checkpoints/pruners/<name>.json`). Evaluators in
  `eval/train/` and `eval/test/` compute the inputs, run each pruner on CPU, validate
  convex-combination slots within the budget, run the frozen LLM, and score relative
  accuracy at 64 and 32 tokens.
- Reference FSR (`method/fsr.py`, paper defaults) on the hidden split: 0.917 at 64 tokens,
  0.838 at 32. Baselines: CDPruner 0.910 / 0.866, DivPrune 0.890 / 0.822, random
  0.831 / 0.719, [CLS] top-K 0.795 / 0.691. Scores were identical across two independent
  GPU runs.
- Shortcut audit: 13 shortcut pruners run through the hidden test; all fail or tie
  (`HARDENING.md`, `redteam/hardening/`).

## Deviations from the paper

- The benchmark is COCO-derived, not the paper's nine benchmarks; the model and the 64-token
  budget match Table 1.
- Eq. 4: the Focus mass is taken over the K top-priority candidates. The literal all-token
  mass leaves no budget for Scan on LLaVA-1.5 features; it is kept as variant
  `fsr_allmass`.
- Relevance follows CDPruner's released computation (negated CLIP patch-text cosine,
  min-max), which the paper cites.

## Open issues

- FSR's margin over CDPruner is not reproduced beyond noise, and CDPruner leads at 32
  tokens. Refine has no measurable effect.
- High-resolution, 13B, Qwen2.5-VL, video and efficiency experiments are not covered.
