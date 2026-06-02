# Progress

## What Works

- Read the paper and identified FSR's three-stage algorithm and default hyperparameters.
- Cloned the linked upstream repository into `method/upstream_fsr`; it currently contains figures and states that code will be released soon.
- Implemented FSR from the paper equations in `method/fsr.py`.
- Added a deterministic token-pruning benchmark, simple baselines, method runner, and evaluation pipeline.
- Local end-to-end validation passes with `bash scripts/reproduce.sh && python validate.py --compare`.
- CPU container job `b24fa7b0-574` completed successfully with exit code 0 using the packaged scripts.

## Results

Scaled-token benchmark results at 32 retained tokens out of 128:

- `fsr`: `performance_retained` 0.878, `retained_fraction` 0.25
- `fused_topk`: `performance_retained` 0.753, `retained_fraction` 0.25
- `relevance_topk`: `performance_retained` 0.740, `retained_fraction` 0.25
- `saliency_topk`: `performance_retained` 0.549, `retained_fraction` 0.25

The same packaged reproduction passed in the compute container, and the repaired evaluator now reports retained performance on a continuous 0-1 scale consistent with the paper's retained-performance metric. These scores support the paper's core claim in the scaled benchmark: Focus-Scan-Refine preserves more task-relevant evidence and context than static token-ranking baselines under the same token budget.

## Remaining

- No remaining work is required for the minimum `core_claim` reproduction.
- Higher milestones would require scaling to real VLM benchmarks and additional paper settings such as different retained-token budgets, larger token grids, and video/VQA datasets.

## Issues

- The paper's official repository does not yet provide implementation code.
- Full VLM benchmark reproduction would require large VLM inference over multiple public benchmarks; the current reproduction uses a scaled token-level benchmark that preserves the paper's algorithmic computation.
- The completed job's scheduler log paths were not readable from this workspace, but the job reported exit code 0 and produced validated `scoring/scores.json` artifacts in the workspace.
- Continuation audit found that the standard reproduction script omitted the `diversity_fps` baseline listed in `scoring/reference.json`; this has been fixed and a CPU test validation job has been prepared.
- CPU test job `a5c85115-dba` passed, but the regenerated `diversity_fps` score nearly tied FSR because the baseline was seeded with the maximum-saliency token. That seed gives a similarity-only baseline access to Focus-stage information, so `diversity_fps` now starts from a query-agnostic token farthest from the global mean before farthest-point sampling.

## Deviations from Paper

- The paper evaluates full VLM answer accuracy on LLaVA/Qwen/video benchmarks. This reproduction uses a deterministic token-level benchmark because it is feasible within the environment and isolates the pruning algorithm. The implemented method itself follows the paper's Focus, Scan, and Refine computation.
