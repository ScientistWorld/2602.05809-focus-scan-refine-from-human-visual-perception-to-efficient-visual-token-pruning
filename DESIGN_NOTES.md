# Train/Test Evaluation Design Notes

## Checks Performed

- Read `eval/train/evaluate.py`, `eval/test/evaluate.py`, and `scripts/evaluate_{train,test}.sh` side by side.
- Confirmed the train script invokes `eval/train/` and writes `scoring/scores_train.json` by default, while the test script invokes `eval/test/` and writes `scoring/scores_test.json` by default.
- Confirmed neither split imports from `method/`, and grep found no train/test cross-imports.
- Confirmed both split evaluators enumerate `checkpoints/*_outputs.jsonl` instead of using a hardcoded method default.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Checked `scoring/reference.json` for paper-agreement metric names and found none.
- Tried to inspect the prompt's paper markdown path, but `/home/user/scratch/AgentWorkspaces/papers/arxiv_mds/2602.05809/2602.05809.md` was not present.

## Changes Made

- Added `selected_source_error` to both train and test split evaluators. This requires each compressed token to match its declared source token from the benchmark record, catching shortcut methods that fabricate ideal query/context vectors instead of performing visual-token pruning.
- Added the matching `selected_source_error` metric to `scoring/reference.json` with a negative coefficient.
- Added missing `description` fields for active metrics in `scoring/reference.json`, so every metric has `higher_is_better`, `coefficient`, and `description`.

## Runtime Concerns for Next Iteration

- The split uses a deterministic random 50/50 partition of a synthetic benchmark. The data generator appears to place target evidence and context at fixed positions, so a future method that learns fixed index selection from the visible slice may generalize to the held-out slice. The new source-token fidelity metric prevents vector fabrication, but it does not solve fixed-position memorization.
- The standard `scripts/evaluate.sh` path still uses `eval/evaluate_token_benchmark.py`, which currently does not emit `evidence_retained`, `context_retained`, or the new `selected_source_error` metric. This pass was scoped to `eval/train/`, `eval/test/`, `scripts/evaluate_{train,test}.sh`, and `reference.json`, so I left that off-scope mismatch for the runner or a broader cleanup pass.
- The new fidelity metric assumes pruning artifacts include `selected_indices`, which the existing method and baselines do. A future method must keep that output contract for split evaluation.
