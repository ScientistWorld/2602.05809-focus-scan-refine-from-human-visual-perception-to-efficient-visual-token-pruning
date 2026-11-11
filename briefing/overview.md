# Overview

- **Paper ID:** 2602.05809
- **Title:** Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning
- **Domain:** Vision-Language Models / Efficient Inference
- **TL;DR:** The paper proposes a training-free visual-token pruning algorithm that balances query-focused local evidence with global context under strict token budgets.

## Short Summary

FSR compresses VLM visual tokens at inference time without changing model weights. It first selects locally important query-relevant tokens, then adds complementary diverse context tokens, then refines context anchors by merging nearby discarded details. The paper reports improved accuracy-efficiency trade-offs across LLaVA, LLaVA-NeXT, Qwen2.5-VL, and video VLM benchmarks.

## Key Results

- On LLaVA-1.5-7B with only 64 of 576 tokens retained, FSR preserves 96.1% average performance (CDPruner 95.7%, DivPrune 93.3%, FastV 72.0%).
- In the same 64-token setting, FSR reports 2.291T FLOPs versus 9.042T for the unpruned model.
- On LLaVA-NeXT-13B, pruning can slightly outperform the dense token set in some settings, suggesting token redundancy can add noise.

## This environment

The gym evaluates pruning rules on a frozen LLaVA-1.5-7B over COCO questions at 64 and 32 tokens. A hidden test runs each registered pruner on inputs that exist only at test time. Reproduced FSR: 0.917 relative performance at 64 tokens on the hidden split (DivPrune 0.890, CDPruner 0.910, random 0.831).
