# Paper's Method

## Key Contribution

Focus-Scan-Refine (FSR) is a training-free visual token pruning framework for VLM inference. It dynamically allocates a fixed token budget between query-critical local evidence and complementary global context.

## Approach

FSR has three stages. Focus computes visual saliency from the vision encoder CLS attention and instruction relevance from cosine similarity between visual tokens and a CLIP text embedding, then fuses them as `phi = relevance^alpha * saliency^beta`. It chooses the smallest top-priority prefix whose cumulative priority mass reaches `rho`.

Scan fills the remaining budget with conditional farthest-point sampling initialized by the Focus set. Each new token maximizes its minimum cosine distance from already selected anchors.

Refine keeps Focus tokens unchanged and merges a limited number of discarded tokens into Scan anchors. Discarded tokens are assigned to their nearest Scan anchor and merged by priority-score-weighted averaging, with merge count `kappa * |Scan|`.

## Main Claims

The central claim is that FSR preserves VLM accuracy better than prior training-free pruning methods under aggressive token budgets while maintaining the same compression level. Secondary claims cover high-resolution VLMs, video VLMs, larger VLMs, and efficiency gains in FLOPs, KV cache, and prefill latency.
