# Progress

## What Works

- Read the paper and identified FSR's three-stage algorithm and default hyperparameters.
- Cloned the linked upstream repository into `method/upstream_fsr`; it currently contains figures and states that code will be released soon.
- Implemented FSR from the paper equations in `method/fsr.py`.
- Added a deterministic token-pruning benchmark, simple baselines, method runner, and evaluation pipeline.
- Local end-to-end validation passes with `bash scripts/reproduce.sh && python validate.py --compare`.

## Results

Local scaled-token benchmark results at 32 retained tokens out of 128:

- `fsr`: `performance_retained` 0.396, `retained_fraction` 0.25
- `fused_topk`: `performance_retained` 0.087, `retained_fraction` 0.25
- `relevance_topk`: `performance_retained` 0.062, `retained_fraction` 0.25
- `saliency_topk`: `performance_retained` 0.000, `retained_fraction` 0.25

No cluster job results yet. The next submitted job runs the same script inside the compute container.

## Remaining

- Run the packaged reproduction in the compute container.
- If the container job reproduces the local FSR-vs-baseline gap, update milestone to `core_claim`.
- If the scaled benchmark is too easy or too hard, adjust the generator while preserving the same token-pruning problem.

## Issues

- The paper's official repository does not yet provide implementation code.
- Full VLM benchmark reproduction would require large VLM inference over multiple public benchmarks; the current reproduction uses a scaled token-level benchmark that preserves the paper's algorithmic computation.

## Deviations from Paper

- The paper evaluates full VLM answer accuracy on LLaVA/Qwen/video benchmarks. This reproduction uses a deterministic token-level benchmark because it is feasible within the environment and isolates the pruning algorithm. The implemented method itself follows the paper's Focus, Scan, and Refine computation.
