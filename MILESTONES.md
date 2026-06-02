# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 07:33] - method_runs
- Implemented the paper's Focus-Scan-Refine computation from the equations.
- Ran the method and baselines end-to-end on the deterministic token-pruning benchmark.

### [2026-06-02 07:45] - core_claim
- Validated the packaged reproduction in a CPU compute-container job.
- FSR achieved `performance_retained` 0.878 at 32/128 retained tokens, ahead of fused top-k 0.753, relevance top-k 0.740, and saliency top-k 0.549 under the same token budget.
- Evaluation includes the compression constraint metrics `mean_token_count` and `retained_fraction`.

### [2026-06-02 07:58] - method_runs
- Continuation audit reset the active milestone to `method_runs`.
- Found that `scripts/reproduce.sh` omitted the `diversity_fps` baseline even though it appears in `scoring/reference.json`.
- Fixed the reproduction script and submitted a CPU test job to regenerate outputs and validate the comparison.

### [2026-06-02 08:06] - method_runs
- CPU test job `a5c85115-dba` passed validation and regenerated scores including `diversity_fps`.
- Audit found the first `diversity_fps` implementation used the maximum-saliency token as its seed, which leaked Focus-stage information into a similarity-only baseline.
- Fixed `diversity_fps` to initialize from a query-agnostic global-mean distance before farthest-point sampling and submitted another CPU validation job.

### [2026-06-02 08:12] - core_claim
- CPU test job `b296916c-3ab` completed with exit code 0 and all validator checks passed.
- Regenerated scores show FSR as the best reproduced method at the fixed 32/128 token budget: `performance_retained` 0.878018 versus 0.876200 for `diversity_fps`, 0.753318 for `fused_topk`, 0.739718 for `relevance_topk`, and 0.548995 for `saliency_topk`.
- The evaluator reports the compression constraint metrics `mean_token_count` 32.0 and `retained_fraction` 0.25 for every reproduced method.

## Stop Justification
- Completed the minimum usable reproduction at milestone `core_claim`. Further progress would require full VLM benchmark inference over the paper's LLaVA/Qwen/video settings.
