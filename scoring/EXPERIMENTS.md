# Experiment Overview

## token_pruning_core

Tests whether a training-free rule that keeps a small, fixed number of visual tokens lets a
frozen vision-language model (LLaVA-1.5-7B) still answer questions about natural images.
Every number is the model's accuracy with the reduced tokens relative to its own accuracy
with all 576 tokens, so 1.0 means nothing was lost.

* `performance_retained`: the headline, at 64 tokens (an 88.9% reduction), averaged over
  four kinds of questions.
* `performance_retained_32`: the same at 32 tokens, a harsher budget where rules that look
  similar at 64 tokens separate.
* `local_evidence_retained` / `global_context_retained`: the 64-token score split into
  questions answered from specific objects (is X present, how many X) and questions that
  need the whole scene (which caption fits, which listed object is missing). Together they
  check that a rule does not buy one kind of information by giving up the other.

The token budget is a hard constraint, not a scored cost: exceeding it, or failing to
answer every question, scores the run as failed (0 everywhere).
