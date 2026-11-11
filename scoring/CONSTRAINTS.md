# Scientist Constraints

## What is fixed (enforced by `eval/`)

- The frozen models: LLaVA-1.5-7B language model and projector, and CLIP ViT-L/14-336
  vision and text encoders. All are hash-verified; nothing may be fine-tuned or replaced.
- The questions, answers, images, prompts and answer parsing (`eval/`, hash-pinned images).
- The budgets: at most 64 slots (and 32 for the second metric) per question. A slot is a
  non-negative weighting of the same image's 576 tokens; arbitrary vectors cannot be
  submitted.
- The pruning step is a CPU program run by the evaluator. The GPU and the language model are
  unavailable to it, and it has 60 s + 0.1 s per question.

## What you may change

- The pruning rule: which tokens to keep, how to merge, how to split the budget, which of
  the provided signals to use (features, projected features, similarities, [CLS] attention,
  CLIP joint-space patch embeddings, the question text and its CLIP embedding), and the
  slot order.
- Hyperparameters, tuned on the visible split only.

## Not allowed

- Reading, reconstructing or looking up answers (including public COCO annotations or
  captions for the evaluation images), or identifying which public image a question is about.
- Calling the language model, or any other large network, inside the pruning step.
- Editing `eval/`, `scoring/reference.json`, the item files or the downloaded artifacts.
- Registering a copy of an existing method under a new name and reporting it as new.

## Compliance Checklist

- [ ] Method registered as `checkpoints/pruners/<name>.json` and scored by the unmodified
      evaluators
- [ ] Pruner uses only the inputs in `--inputs DIR` (no labels, no network, no LLM)
- [ ] Hyperparameters chosen on the visible split
- [ ] Both budgets (64 and 32) answered for every question
