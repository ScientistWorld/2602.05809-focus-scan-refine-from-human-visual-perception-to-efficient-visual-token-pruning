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
