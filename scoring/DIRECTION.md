# Research Direction

## Research Problem

Training-free visual-token reduction for a frozen VLM: under a fixed small budget, choose and
merge visual tokens so that the language model answers as well as it does with all tokens.
The budget has to cover both the regions the question points at and the rest of the scene.

## Core Contribution Scope

New token-scoring signals, ways to combine query relevance with visual saliency, rules for
splitting a fixed budget between query-specific and scene-covering tokens, diversity or
coverage objectives, merging and aggregation of discarded tokens, and token ordering. Gains
should hold at 64 tokens and especially at 32 tokens, on both object-level and scene-level
questions.

## Approach Scope

Anything computable on CPU from the provided per-image and per-question inputs within the
time budget: closed-form scores, greedy or exact subset selection, clustering, merging.
Tune on the visible split.

## Out of Bounds

Training or fine-tuning any model, using the language model (or another large network) to
score candidate subsets, raising the budget, using external labels or captions for the
evaluation images, and changing the evaluation.
