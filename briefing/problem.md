# Problem Definition

## Research Question

How can a vision-language system reduce the number of visual tokens passed to the language model at inference time while preserving enough image evidence to answer the user query?

## Why It Matters

Visual tokens dominate the attention cost and KV-cache footprint of modern VLM inference. A good pruning policy can make VLMs cheaper and faster without retraining the backbone or sacrificing answer quality.

## Success Criteria

A successful solution keeps a fixed small token budget, preserves query-critical local evidence, retains enough complementary global context for reasoning, and works as a training-free inference-time module.
