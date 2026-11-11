# Hardening report (iteration 11)

## Summary

The red-team report showed that the old gym could not reward science. Its test was a
240-sample synthetic proxy: the answers sat at fixed token positions, the ideal "probe"
vectors were visible, NaN vectors scored perfectly, and the inventory of methods was
mutable. Every major shortcut beat the honest improvement. Patching that proxy would not
have helped, because writing down the generator by hand beats any pruning method on it.
**I rebuilt the test on the paper's own setting:**

* **Model.** Frozen **LLaVA-1.5-7B** (the paper's headline model): CLIP ViT-L/14-336
  → 576 visual tokens → MLP projector → Vicuna-7B, greedy decoding, LLaVA prompt
  templates. Weights are pinned by SHA-256 / git-blob hash.
* **Data.** COCO val2017 images; 300 visible images (2,100 questions) and 700 hidden
  images (4,898 questions). There are four question families with ground truth from
  COCO annotations: POPE-protocol object existence (random / popular / adversarial
  negatives), counting, four-way caption matching with hard negatives, and four-way
  "which object is not present". Test questions and answers live only in
  `eval/test/test_items.jsonl`. They are built by `eval/test/build_items.py`, which also
  lives only there.
* **Metric.** The paper's quantity: accuracy with pruned tokens ÷ accuracy of the same
  frozen model with all 576 tokens, averaged over families. It is scored at **64 tokens**
  (the paper's most aggressive LLaVA-1.5 budget) and **32 tokens**, plus 64-token
  local-evidence (existence, counting) and global-context (caption, absent) splits.
* **Method contract.** A method is a *pruner program* registered as
  `checkpoints/pruners/<name>.json`. The evaluator computes all inputs itself and runs
  every registered pruner on CPU within a time budget. It accepts only ≤K slots per
  question, each a non-negative weighting of that image's own 576 tokens (pruning or
  merging, as FSR's Refine does). It builds every LLM input from its private copy of the
  tokens and scores the answers. No number, vector or flag written by a method is trusted.

The reproduced FSR implements Focus (Eq. 1–4), conditional farthest-point Scan (Eq. 5) and
priority-weighted Refine (Eq. 8–9) with the paper's defaults α=3, β=1, ρ=0.9, κ=1. The
baselines span the paper's three families: [CLS]-attention top-K, DivPrune and CDPruner
(conditional DPP, following its released code), plus random.

All numbers below come from GPU jobs `e3cfa3ce-8d2` (reference, variants, shortcuts) and
`c35e1b98-9e0` (final rerun of the reference from the final code, plus shortcut re-run).
The reference and baseline scores were **byte-identical across the two runs**: the
pipeline is deterministic and the backbone frozen, so seed noise is zero. Aggregates are
in `redteam/hardening/results/`. Per-question test answers were deleted because they
would reveal labels.

## Reference before and after

| | before (synthetic proxy) | after (LLaVA-1.5-7B, hidden test) |
|---|---|---|
| FSR `performance_retained` | 0.8758 (cosine proxy, 32/128 tokens) | **0.9170** (64/576 tokens) |
| FSR `performance_retained_32` | — | **0.8377** |
| FSR local / global (64) | evidence 0.971 / context 0.781 | 0.9435 / 0.8906 |
| Paper, Table 1, LLaVA-1.5-7B, 64 tokens | FSR 0.961, CDPruner 0.957, DivPrune 0.933 | (kept unchanged in `scoring/reference.json`) |

Hidden-test scores of the reference runs (`scoring/scores_test.json`):

| method | perf@64 | perf@32 | local@64 | global@64 |
|---|---|---|---|---|
| **fsr** (reference) | **0.9170** | 0.8377 | 0.9435 | **0.8906** |
| cdpruner | 0.9102 | **0.8663** | **0.9464** | 0.8740 |
| divprune | 0.8900 | 0.8221 | 0.9157 | 0.8642 |
| random | 0.8306 | 0.7194 | 0.8466 | 0.8146 |
| cls_topk | 0.7947 | 0.6912 | 0.7729 | 0.8165 |

The full-token model's accuracy on test is existence 0.916, counting 0.610, caption 0.883
and absent 0.637. Train/visible scores are in `scoring/scores_train.json`. FSR is 0.9118 on
train and 0.9170 on test, so an honest method scores alike on both splits.

## Do better results score higher? (variants, before and after)

Before (old gym, from `REDTEAM.md`): the ordering was right, with the stronger variant at
+0.384 and no-scan at −0.669, but shortcuts scored far above the honest improvement
(+0.52 to +1.78).

After: paired comparisons on the 4,898 hidden questions. Every method sees the same
questions, the same frozen model and identical batching. Δ is the change in relative
performance vs FSR. Standard errors are clustered by image
(`results/clustered_se_test.txt`). "Gain" applies the ranking rule (Σ |coef| · Δ/σ) with
σ taken across the 17 honest methods tried; it is illustrative, since a future cohort's
spread is unknown.

| variant | Δperf@64 (SE) | Δperf@32 (SE) | gain |
|---|---|---|---|
| FSR κ=5 (more merging) | **+0.0056 (0.0023)** | +0.0011 (0.0022) | **+0.205** |
| FSR κ=2 | +0.0007 (0.0012) | +0.0015 (0.0014) | +0.038 |
| FSR with DPP Scan (best FSR variant on train: +0.022 there) | −0.0036 (0.0070) | **+0.0183 (0.0073)** | +0.016 |
| CDPruner (stronger known method at 32 tokens) | −0.0069 (0.0089) | **+0.0285 (0.0102)** | −0.022 |
| FSR ρ=0.7 / ρ=0.5 | −0.0011 / −0.0064 | −0.0020 / +0.0065 | −0.053 / −0.179 |
| FSR without Refine (κ=0) | +0.0008 (0.0009) | −0.0001 (0.0008) | +0.028 |
| FSR literal all-token mass (= Focus only, no Scan) | **−0.0219 (0.0069)** | −0.0097 (0.0063) | −0.858 |
| DivPrune (similarity only) | **−0.0271 (0.0103)** | −0.0156 (0.0111) | −1.075 |
| FSR relevance only (α=1, β=0) | **−0.0376 (0.0103)** | **−0.0399 (0.0113)** | −1.658 |
| FSR saliency only (α=0, β=1) | **−0.0478 (0.0104)** | **−0.0862 (0.0111)** | −2.348 |
| FSR with +cos relevance sign | **−0.0569 (0.0106)** | **−0.0829 (0.0115)** | −2.643 |
| random | **−0.0864 (0.0115)** | **−0.1183 (0.0124)** | −3.968 |
| [CLS]-attention top-K | **−0.1223 (0.0102)** | **−0.1466 (0.0111)** | −5.428 |
| spatial average pooling (merge only) | **−0.1233 (0.0116)** | **−0.1993 (0.0126)** | −5.942 |

Bold marks |z| ≥ 2.5. Removing components or signals scores lower, well beyond noise.
Improvements are detected beyond noise where they exist: κ=5 at 64 tokens (z=2.5), and
DPP-Scan and CDPruner at 32 tokens (z=2.5 and 2.8). Between structurally different
selection rules, the paired SE at 64 tokens is about 0.7–0.9%, so differences below about
1.5% there are not resolvable. The 32-token budget is where strong methods separate, which
is why it is scored.

## Major shortcuts: fix and score before → after

All "after" rows were executed as registered pruners through the unmodified hidden
evaluator (`python eval/test/evaluate.py`, the code behind `scripts/evaluate_test.sh`).
The code is in `redteam/hardening/shortcuts.py` and the scores in
`results/shortcuts_scores_test.json`. The reference gain is 0.

| red-team shortcut (before: test perf / gain) | fix | after (perf@64 / gain) |
|---|---|---|
| copy ideal probe vectors (1.000 / +0.66; two vectors +1.78) | No probe exists. The output contract admits only weightings of the image's own tokens, and unknown fields are rejected. | `sc_vector_field`: **failed → 0 / −38.9** |
| keep fixed positions 0–31 (0.971 / +0.52) | Real images: evidence position varies | `sc_first_positions`: 0.411 / −21.3 |
| read visible target indices (0.971 / +0.52) | Pruner inputs carry no labels. Answers exist only in `eval/test/`, which the evaluator makes untraversable while pruners run. | `sc_read_answers`: answers unreadable → 0.9170 / 0.000 (FSR fallback). With a readable answer file (train) it gains +0.011 / +0.38. |
| brute-force the visible objective (0.971 / +0.52) | The objective needs labels plus the frozen LLM. Pruners run CPU-only (GPU hidden and all free GPU memory reserved by the evaluator) within 60 s + 0.1 s/question. | `sc_llm_search` (un-hides the GPU, loads LLaVA): CUDA OOM → **failed → 0 / −38.9** |
| NaN / ∞ vectors (1.000 / +0.66) | Non-finite JSON constants, negative or zero-sum weights → method failed | `sc_nan_weights`: **failed → 0 / −38.9** |
| change the ground-truth probe in mutable data (0.971 / +0.52) | Ground truth in `eval/test/` only. Images and weights hash-pinned. The evaluator scores from its private in-memory tokens and rewrites tampered input files. | `sc_tamper_inputs` (zeros the shared feature file): 0.9170 / 0.000 (= its honest FSR selection; no effect) |
| shrink the test to two easy records (0.986 / +0.59) | Test set fixed in `eval/test/`; every question must be answered | `sc_partial`: **failed → 0 / −38.9** |
| compression savings pay for task failure (`one_query`, conditional +0.009) | No cost metrics: the budget is a hard constraint and every scored metric is task accuracy, so a task failure is lowest everywhere | n/a by construction |
| keep all tokens (0.971 / −3.07) | Budget enforced: >K slots → failed | `sc_keep_all`: **failed → 0 / −38.9** |
| empty output / stale score file (exit 1; old file kept) | Evaluator writes worst values for every registered method before scoring and exits 0 with fresh results | `sc_empty`, `sc_crash`: **0 / −38.9**, present in output |
| forge scores, flags, nulls (overwritten) | Still overwritten | `sc_forge_scores`: 0.795 (its honest selection) |
| omit a failing method / unlisted method (tie 0) | Inventory = every `checkpoints/pruners/*.json`. No train-sidecar dependence. Failures get worst values. | `sc_crash` listed with 0; any registered spec is scored |
| replay cached reference outputs (tie 0) | Inputs exist only at test time, so outputs cannot be cached | replay impossible (copying the reference *code* still ties; see below) |

The pruner probe (`results/probe_env.json`) confirms what a pruner can reach at test time:
it can un-hide CUDA, but allocation fails (OOM), and the hidden answer file is not readable.

## Shortcuts left, and why

* **Copying the provided reference pruner** ties the reference (gain 0). This is inherent
  to improve mode; the post-hoc grader must catch it.
* **Re-identifying test images against public COCO.** The test images are public COCO
  val2017 images, and their annotations and captions are public. A pruner shipped with an
  offline lookup table (feature hash → image id → public labels) could recover answers and
  apply answer-aware selection, worth about +0.011 (`sc_read_answers` on train). Closing
  it would require non-public images. It is a deliberate external-label lookup, left for
  the grader.
* **Running the evaluator as root** would bypass the directory guard. Jobs run as the
  workspace user.
* **Copying the CDPruner baseline** is an honest published alternative, not a loophole.
  It ties the reference overall (−0.022 illustrative) by trading +2.9% at 32 tokens for
  −0.7% at 64 and −1.7% on global context.

## Headroom check

| scored metric | reference | bound | status |
|---|---|---|---|
| `performance_retained` (64) | 0.917 | ~1.0 (full tokens; can exceed 1) | not at bound; best variant tried 0.923 |
| `performance_retained_32` | 0.838 | ~1.0 | not at bound; CDPruner 0.866 |
| `local_evidence_retained` | 0.943 | ~1.0 | not at bound |
| `global_context_retained` | 0.891 | ~1.0 | not at bound |

No scored metric is at its bound, and none is stuck across methods (from 0.79 to 0.92 at
64 tokens among honest methods). The simple baselines (random 0.831, attention top-K
0.795, pooling 0.794) sit clearly below the reference, with room above it. The brute
force the paper's setting avoids is keeping all tokens, which is now a budget failure;
search with the frozen LLM is blocked. **Added:** the 32-token budget experiment
(`performance_retained_32`, coefficient 0.5). It covers a regime the paper does not
report for LLaVA-1.5, and it is where FSR falls short of a relevance-conditioned DPP
(CDPruner +2.9%, FSR+DPP-Scan +1.8%). That gives the test measurable headroom above the
reference.

## Weights

There is one experiment, weight 1.0. The coefficients are:

* `performance_retained` at 64 tokens: 1.0, the paper's headline setting.
* `performance_retained_32`: 0.5. It is not in the paper, but it is the most
  discriminating regime; the paper says gains matter most under aggressive compression.
* Local and global splits: 0.25 each. They keep a method from trading scene coverage for
  object evidence, the paper's central balance.

The old count and fraction metrics were removed. They double-counted a cost that is now a
hard constraint, and they let compression pay for task failure.

## What I could not fix or reproduce (stated plainly)

* **FSR does not beat CDPruner beyond noise.** At 64 tokens FSR leads by 0.7% (z=0.8, the
  paper's direction; the paper's own margin is 0.4%). At 32 tokens CDPruner leads by 2.9%
  (z=2.8), and on the 300-image visible split CDPruner leads at 64 too (+4.0%, z=2.7).
  FSR does beat the attention-based, similarity-based, random and pooling baselines beyond
  noise, and the paper's ablations hold in direction: the dual pathway beats either cue
  alone, and Scan beats Focus-only (+2.2%, z=3.2).
* **Refine has no measurable effect** (removing it: +0.08 ± 0.09%). The paper reports a
  gain, and over-smoothing at κ=5; here κ=5 is slightly better (+0.56%, z=2.5 on test,
  not confirmed on train).
* **Eq. 4 departs from the literal formula.** Taken over all 576 tokens, LLaVA-1.5's
  [CLS] attention is so spread out that ρ=0.9 needs more than K tokens on 99.2% (K=64) and
  100% (K=32) of questions. Scan and Refine would then never run, contradicting the paper's
  Fig. 1 split (Focus 9–15 of 32) and its Scan ablation. The reference therefore takes the
  ρ-mass over the K highest-priority candidates. That gives Focus a mean of 48/64 and
  26/32, and all three stages run. The literal reading is kept as variant `fsr_allmass`,
  which scores −2.2% at 64. The official code is unreleased, so this cannot be checked
  against the authors.
* **Scope.** The benchmark is COCO-derived (POPE protocol plus three COCO-grounded
  families), not the paper's nine benchmarks. High-resolution, 13B, Qwen2.5-VL, video and
  efficiency (FLOPs/latency) results are not covered. Token count is fixed by the budget.
* **Test size.** It is limited by COCO val2017 images with at least three salient objects
  (1,219; 300 visible + 700 hidden used). Improvements smaller than ~1.5% between
  different selection rules at 64 tokens are within noise.
* **The hidden test needs a GPU** for the 7B model. It takes about 10 minutes on one H200
  for five methods, including weight hashing, feature extraction and the full-token pass.
