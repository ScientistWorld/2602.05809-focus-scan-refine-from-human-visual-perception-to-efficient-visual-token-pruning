# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification
- Pending validation after continuation audit.
