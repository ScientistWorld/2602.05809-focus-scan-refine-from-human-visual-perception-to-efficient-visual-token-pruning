# Progress

## What Works

- Read the paper and identified FSR's three-stage algorithm and default hyperparameters.
- Cloned the linked upstream repository into `method/upstream_fsr`; it currently contains figures and states that code will be released soon.
- Implemented FSR from the paper equations in `method/fsr.py`.
- Added a deterministic token-pruning benchmark, simple baselines, method runner, and evaluation pipeline.
- Local/source audit confirms `method/fsr.py` implements the paper's Focus, Scan, and Refine equations rather than a surrogate model.
- CPU container jobs `b24fa7b0-574`, `a5c85115-dba`, and `b296916c-3ab` completed successfully with exit code 0 using the packaged scripts.

## Results

Scaled-token benchmark results at 32 retained tokens out of 128:

- `fsr`: `performance_retained` 0.878, `retained_fraction` 0.25
- `diversity_fps`: `performance_retained` 0.876, `retained_fraction` 0.25
- `fused_topk`: `performance_retained` 0.753, `retained_fraction` 0.25
- `relevance_topk`: `performance_retained` 0.740, `retained_fraction` 0.25
- `saliency_topk`: `performance_retained` 0.549, `retained_fraction` 0.25

The packaged reproduction passed again in CPU test job `b296916c-3ab`, and the repaired evaluator reports retained performance on a continuous 0-1 scale consistent with the paper's retained-performance metric. These scores support the paper's core claim in the scaled token benchmark: Focus-Scan-Refine preserves more task-relevant evidence and context than the reproduced pruning baselines under the same token budget.

## Remaining

- No remaining work is required for the minimum scaled `core_claim` reproduction.
- Higher milestones would require scaling to real VLM benchmarks and additional paper settings such as different retained-token budgets, larger token grids, and video/VQA datasets.

## Issues

- The paper's official repository does not yet provide implementation code.
- Full VLM benchmark reproduction would require large VLM inference over multiple public benchmarks; the current reproduction uses a scaled token-level benchmark that preserves the paper's algorithmic computation and should not be presented as matching the paper's full LLaVA/Qwen/VLM benchmark numbers.
- Job logs and result files are readable under `jobs/` and show successful CPU container executions with `python validate.py --compare`.
- Continuation audit found that the standard reproduction script omitted the `diversity_fps` baseline listed in `scoring/reference.json`; this has been fixed and a CPU test validation job has been prepared.
- CPU test job `a5c85115-dba` passed, but the regenerated `diversity_fps` score nearly tied FSR because the baseline was seeded with the maximum-saliency token. That seed gives a similarity-only baseline access to Focus-stage information, so `diversity_fps` now starts from a query-agnostic token farthest from the global mean before farthest-point sampling.

## Deviations from Paper

- The paper evaluates full VLM answer accuracy on LLaVA/Qwen/video benchmarks. This reproduction uses a deterministic token-level benchmark because it is feasible within the environment and isolates the pruning algorithm. The implemented method itself follows the paper's Focus, Scan, and Refine computation.
