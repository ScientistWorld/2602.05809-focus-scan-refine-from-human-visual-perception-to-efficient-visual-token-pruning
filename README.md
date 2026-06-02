# Overview

- **Paper ID:** 2602.05809
- **Title:** Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning
- **Domain:** Vision-Language Models / Efficient Inference
- **TL;DR:** The paper proposes a training-free visual-token pruning algorithm that balances query-focused local evidence with global context under strict token budgets.

## Short Summary

FSR compresses VLM visual tokens at inference time without changing model weights. It first selects locally important query-relevant tokens, then adds complementary diverse context tokens, then refines context anchors by merging nearby discarded details. The paper reports improved accuracy-efficiency trade-offs across LLaVA, LLaVA-NeXT, Qwen2.5-VL, and video VLM benchmarks.

## Key Results

- On LLaVA-1.5-7B with only 64 of 576 tokens retained, FSR preserves 96.1% average performance.
- In the same 64-token setting, FSR reports 2.291T FLOPs versus 9.042T for the unpruned model.
- On LLaVA-NeXT-13B, pruning can slightly outperform the dense token set in some settings, suggesting token redundancy can add noise.

---

# Problem Definition

## Research Question

How can a vision-language system reduce the number of visual tokens passed to the language model at inference time while preserving enough image evidence to answer the user query?

## Why It Matters

Visual tokens dominate the attention cost and KV-cache footprint of modern VLM inference. A good pruning policy can make VLMs cheaper and faster without retraining the backbone or sacrificing answer quality.

## Success Criteria

A successful solution keeps a fixed small token budget, preserves query-critical local evidence, retains enough complementary global context for reasoning, and works as a training-free inference-time module.

---

# Paper's Method

## Key Contribution

Focus-Scan-Refine (FSR) is a training-free visual token pruning framework for VLM inference. It dynamically allocates a fixed token budget between query-critical local evidence and complementary global context.

## Approach

FSR has three stages. Focus computes visual saliency from the vision encoder CLS attention and instruction relevance from cosine similarity between visual tokens and a CLIP text embedding, then fuses them as `phi = relevance^alpha * saliency^beta`. It chooses the smallest top-priority prefix whose cumulative priority mass reaches `rho`.

Scan fills the remaining budget with conditional farthest-point sampling initialized by the Focus set. Each new token maximizes its minimum cosine distance from already selected anchors.

Refine keeps Focus tokens unchanged and merges a limited number of discarded tokens into Scan anchors. Discarded tokens are assigned to their nearest Scan anchor and merged by priority-score-weighted averaging, with merge count `kappa * |Scan|`.

## Main Claims

The central claim is that FSR preserves VLM accuracy better than prior training-free pruning methods under aggressive token budgets while maintaining the same compression level. Secondary claims cover high-resolution VLMs, video VLMs, larger VLMs, and efficiency gains in FLOPs, KV cache, and prefill latency.

---

# Evaluation

## Metrics

The primary metric is `performance_retained`: a continuous 0-1 score measuring how well the compressed visual-token representation preserves both query-relevant local evidence and required global context. Higher is better.

Constraint metrics measure the compression setting: `mean_token_count` and `retained_fraction`. Lower values are better when performance is held fixed.

## Baselines and Targets

The paper reports the core LLaVA-1.5-7B aggressive-compression setting at 64 retained tokens from 576 original visual tokens. At this setting, the unpruned model is the 100% reference, FastV retains 72.0% average performance, CDPruner retains 95.7%, and FSR retains 96.1%.

## Evaluation Protocol

Run each pruning method on the same token benchmark with the same fixed token budget. The method writes compressed-token artifacts under `checkpoints/`; evaluation reads those artifacts, computes retained local-evidence and global-context similarity, and never imports method code. Scores are written to `scoring/scores.json`.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 07:17] - none
- Started reproduction work for FSR.
- Read the paper markdown and extracted the method, benchmarks, and core reported numbers.
- Implemented the paper's Focus-Scan-Refine token pruning computation and packaged a first scaled benchmark for job validation.

### [2026-06-02 07:35] - method_runs
- Ran `scripts/reproduce.sh` locally on the scaled token benchmark.
- The actual FSR Focus, Scan, and Refine computation executed end-to-end and wrote scored artifacts.
- Local validation with `python validate.py --compare` passed.

### [2026-06-02 07:40] - core_claim
- CPU container job `b24fa7b0-574` completed successfully with exit code 0.
- The packaged reproduction ran `scripts/reproduce.sh` and `python validate.py --compare` in the compute environment.
- FSR retained 0.396 task performance at the same 32/128 token budget where static baselines retained 0.087 (`fused_topk`), 0.062 (`relevance_topk`), and 0.000 (`saliency_topk`), supporting the paper's core claim on the scaled token-pruning benchmark.

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
