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
