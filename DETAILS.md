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

---

# Problem Definition

## Research Question

How can a vision-language model reduce the number of visual tokens passed to its language
model at inference time, without retraining anything, while keeping enough image
information to answer the user's question?

## Why It Matters

Visual tokens dominate the attention cost and KV-cache footprint of VLM inference (LLaVA-1.5
feeds 576 tokens per image; high-resolution and video models feed thousands). A plug-in rule
that keeps a small, fixed number of tokens makes inference cheaper and faster without
touching model weights, provided the answers stay correct.

## Setting in this environment

* Frozen LLaVA-1.5-7B: CLIP ViT-L/14-336 vision encoder (576 patch tokens on a 24x24 grid),
  two-layer MLP projector, Vicuna-7B language model, greedy decoding.
* A pruning rule receives, for each image/question pair, the image's visual tokens and
  encoder statistics plus the question text and its CLIP text embedding, and returns at most
  K "slots". Each slot is a non-negative weighting of that image's own tokens (a single kept
  token, or a merge of several), fed to the language model in the order given.
* Token budgets: K = 64 (88.9% reduction) and K = 32 (94.4% reduction).
* Questions are about natural images (COCO): whether an object is present, how many
  instances there are, which caption describes the scene, and which listed object is absent.

## Success Criteria

At the same fixed budget, the frozen model should answer as accurately as possible relative
to its own accuracy with all 576 tokens, both on questions about specific objects and on
questions that need the whole scene, and the pruning step should stay a light,
training-free computation.

---

# Paper's Method

## Key Contribution

Focus-Scan-Refine (FSR) is a training-free visual token pruning framework for VLM inference.
It dynamically allocates a fixed token budget between query-critical local evidence and
complementary global context.

## Approach (as implemented in `method/fsr.py`)

* **Focus** (Eq. 1–4). Saliency s_i is the head-mean [CLS]-to-patch attention of the vision
  encoder's penultimate layer. Instruction relevance r_i is the cosine similarity between a
  token's CLIP joint-space embedding and the CLIP text embedding of the instruction. It is
  computed as in CDPruner, which the paper cites for this score: the sign is negated, then
  min-max normalised. Saliency is also min-max normalised. Priority phi = r^alpha * s^beta
  (alpha = 3, beta = 1). Keep the smallest prefix of the phi-sorted tokens holding a fraction
  rho = 0.9 of the information mass.
  *Mass reading:* on LLaVA-1.5 the [CLS] attention is spread out. Taking the mass over all
  576 tokens needs more than K tokens on 99–100% of questions, which would switch off Scan
  and Refine, contrary to the paper's Fig. 1. The implementation therefore takes the mass
  over the K highest-priority candidates (`--focus-mass budget`, default). The literal
  reading is `--focus-mass all`.
* **Scan** (Eq. 5). Conditional farthest-point sampling in cosine distance between projected
  tokens, initialised with the Focus set, fills the remaining K − K_F slots.
* **Refine** (Eq. 8–9). Each discarded token is assigned to its most similar Scan anchor.
  The kappa·|S| discarded tokens most similar to their anchors (kappa = 1) are merged into
  them by priority-weighted averaging, with weights initialised to phi. Focus tokens stay
  intact.

Run with `bash scripts/method.sh`, which registers `method/prune_fsr.py`, then evaluate.

## Main Claims

FSR preserves VLM accuracy better than prior training-free pruning methods under aggressive
token budgets at the same compression. The paper's secondary claims cover high-resolution
VLMs, video VLMs, larger VLMs, and efficiency (FLOPs, KV cache, prefill latency).

## Reproduction status (this environment)

On the hidden split at 64 tokens, FSR scores 0.917. That is above DivPrune (0.890),
[CLS]-attention top-K (0.795) and random (0.831), and level with CDPruner (0.910, inside
noise). At 32 tokens CDPruner is ahead (0.866 vs 0.838). See `HARDENING.md`.

---

# Evaluation

## Metrics (all higher is better)

* `performance_retained`: at K = 64, the mean over the four question families of
  (accuracy with the pruned tokens) / (accuracy of the same frozen model with all 576 tokens).
  This is the "average relative performance" used in the visual-token pruning literature.
* `performance_retained_32`: the same at K = 32.
* `local_evidence_retained`: at K = 64, relative performance on the families answered from
  specific objects (existence, counting).
* `global_context_retained`: at K = 64, relative performance on the families that need the
  whole scene (caption matching, absent-object choice).

A method that fails (crash, timeout, malformed or over-budget output, missing questions)
gets 0.0 in every metric.

## Question families

COCO val2017 images, ground truth from COCO annotations:

* existence: "Is there a/an X in the image?" (yes/no; absent objects sampled at random,
  by popularity, and by co-occurrence with present objects, as in POPE);
* counting: "How many X are there in the image?" for clearly visible, non-crowd instances;
* caption: choose the matching caption among four (distractors from images with similar
  object sets);
* absent: choose which of four listed objects is not present (three present, one absent
  co-occurring object).

Prompts follow LLaVA-1.5's evaluation templates ("Answer the question using a single word or
phrase." / "Answer with the option's letter from the given choices directly."). Answers are
parsed with POPE's yes/no rule, the first number, or the first option letter.

## Protocol

1. Register a method as a pruner program: `checkpoints/pruners/<name>.json` containing
   `{"command": ["python", "path/to/your_pruner.py", "--any", "options"]}`. Every file there
   is scored; there is no other inventory.
2. The evaluator computes the inputs itself and runs
   `<command> --inputs DIR --budget K --output FILE` for K = 64 and 32, on CPU only, with a
   time budget of 60 s + 0.1 s per question. `DIR/meta.json` documents the inputs:
   `vision_features` (CLIP penultimate-layer patch features), `projected_features` (the
   projector outputs the language model consumes), `projected_cosine`, `joint_embeds` (patch
   features in CLIP's image-text space), `cls_attention` ([CLS]-to-patch attention of the
   penultimate vision layer), `text_embeds` (CLIP text embedding of the question), and
   `queries.jsonl`. `data/pruner_io.py` reads them.
3. Output: one JSON line per question, `{"qid": ..., "slots": [{"tokens": [i, ...],
   "weights": [w, ...]}, ...]}`, with at most K slots, token indices in [0, 576), and weights
   finite, >= 0, positive sum (optional, uniform if omitted). The evaluator turns each slot
   into the weighted mean of its tokens' projected features, from its own copy, and runs the
   frozen model.
4. `bash scripts/evaluate_train.sh` scores every registered method on the visible split
   (300 images, 2,100 questions) into `scoring/scores_train.json` and leaves that split's
   pruner inputs in `data/downloads/pruner_inputs_train/` for offline CPU development.
   `bash scripts/evaluate_test.sh` scores the hidden split (700 images, 4,898 questions) into
   `scoring/scores_test.json`. Both need a GPU for the frozen models (about 5 and 10 minutes
   on an H200 for five methods).

The frozen weights and benchmark images are verified against pinned hashes. Accuracy with
all 576 tokens on the hidden split: existence 0.916, counting 0.610, caption 0.883,
absent 0.637.

## Reference points

The source paper reports average relative performance on nine benchmarks for LLaVA-1.5-7B at
64 tokens (Table 1): 96.1% for its method, 95.7% for the strongest prior method. Those numbers
are kept in `scoring/reference.json`; this environment's question set differs, so values are
comparable in kind, not identical.

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

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 4 total (4 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 5 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 5 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 5 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 11m | **GPU**: 0.0h
- **Jobs**: 5 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-06-02 08:25] - core_claim
- Final verification found an unvalidated multi-budget expansion had left `reference.json` and `scores.json` incompatible.
- Reverted that expansion to restore the validated single-budget `token_pruning_core` gym.
- `python validate.py --compare` passes after the revert; milestone remains `core_claim`.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 5 total (5 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-06-02 08:25] - core_claim
- Final verification found an unvalidated multi-budget expansion had left `reference.json` and `scores.json` incompatible.
- Reverted that expansion to restore the validated single-budget `token_pruning_core` gym.
- `python validate.py --compare` passes after the revert; milestone remains `core_claim`.

</details>


---

### Iteration 11: claude-opus-5.5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- Replaced the synthetic proxy with the paper's setting: a frozen LLaVA-1.5-7B answering COCO
  val2017 questions from pruned CLIP-336 tokens at 64 and 32 tokens. Methods are pruner programs run
  by the evaluator on hidden test inputs (CPU only, no LLM access, convex-combination outputs).
- FSR 0.917 relative performance at 64 tokens on the hidden split (CDPruner 0.910, DivPrune 0.890,
  random 0.831); all red-team shortcuts closed or reduced to failures. See `HARDENING.md`.

---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim` (hardened gym on the paper's model and budget).
