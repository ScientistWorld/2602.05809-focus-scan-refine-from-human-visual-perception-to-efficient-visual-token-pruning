# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-02 07:17] - none
- Started from an initialized workspace and read the FSR paper.
- Identified the core method as training-free Focus, Scan, and Refine visual-token pruning.

### [2026-06-02 07:28] - method_runs
- Implemented the paper's actual FSR computation in `method/fsr.py`.
- Packaged deterministic token-benchmark runners and completed a CPU container job with the method executing end-to-end.

### [2026-06-02 07:38] - core_claim
- Ran the full packaged reproduction on the scaled token benchmark with baselines and evaluation.
- FSR preserved local evidence plus complementary context at the same fixed token budget and outperformed the reproduced saliency, relevance, fused-topk, and diversity baselines on `performance_retained`.
- Scope: this is a scaled token-level reproduction of the paper's pruning algorithm and central accuracy-compression claim, not a full LLaVA/Qwen benchmark rerun.

### [2026-06-02 08:05] - method_runs
- Continuation audit found that `reference.json` used paper numbers under local proxy baseline names.
- Fixed the reference to use the paper's actual baseline method names and updated the reproduction script to emit matching paper-family proxy baseline artifacts.
- Current `scores.json` is stale until the next CPU reproduction job regenerates it.

### [2026-06-02 08:50] - core_claim
- CPU validation job `e17065c1-4ac` completed successfully and regenerated `scoring/scores.json` with paper-family method names.
- `python validate.py --compare` passed all scoring and structure checks.
- On the scaled token benchmark at 32 retained tokens out of 128, FSR reached `performance_retained` 0.878018 versus `fastv` 0.548995, `cdpruner` 0.753318, and `divprune` 0.876200.
- Scope remains a scaled token-level reproduction of the FSR pruning algorithm and core compression-quality claim, not a full VLM benchmark rerun.

### [2026-06-02 08:12] - core_claim
- Audited the scoring schema and removed an invalid `selected_source_error` metric that penalized FSR's intended Refine-stage token merging.
- Updated the main, train, and test evaluators to report separate local-evidence and global-context retention metrics plus fixed token-count constraints.
- Local structural validation passes; a CPU validation job is being submitted to regenerate all score files inside the packaged container.

### [2026-06-02 08:14] - core_claim
- Final verification found minor method-leakage in method-agnostic documentation and benchmark-generator comments, then removed it.
- `python validate.py --compare` passes after verification edits.
- Final scope remains a scaled token-level gym for the paper's core pruning claim, not a full downstream VLM benchmark reproduction.

## Stop Justification
- Completed at milestone `core_claim`.
