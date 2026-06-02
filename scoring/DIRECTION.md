# Research Direction

## Research Problem

This benchmark studies training-free visual-token pruning for VLM inference. The goal is to preserve query-relevant visual information and global context while reducing the number of visual tokens passed downstream.

## Core Contribution Scope

Scientists should work on inference-time token selection, token merging, token scoring, or context-preserving compression policies for visual-token sequences. Improvements should target the accuracy-compression trade-off, not model retraining or larger backbones.

## Approach Scope

Allowed work includes new scoring functions, diversity objectives, dynamic budget allocation, aggregation or merging rules, and query-conditioned pruning policies. Methods may use the benchmark-provided token features, query embeddings, and saliency scores.

## Out of Bounds

Changing the task into full model training, using more retained tokens without reporting it as a separate budget, adding external labels to the scored benchmark, or replacing pruning with a larger VLM is out of scope.
