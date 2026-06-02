#!/usr/bin/env python3
"""Generate a deterministic token-pruning benchmark.

The benchmark stores encoder-like visual token embeddings, a query embedding,
and a CLS-to-token saliency vector. Labels identify one query-relevant local
evidence token and one low-saliency global-context token that must both survive
compression for the sample to count as solved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def make_sample(rng: np.random.Generator, sample_id: int, n_tokens: int, dim: int) -> dict:
    query = _unit(rng.normal(size=(dim,)))
    context = _unit(rng.normal(size=(dim,)))
    context = _unit(context - query * np.dot(context, query))
    distractor_query = _unit(-query + rng.normal(scale=0.08, size=dim))

    tokens = _unit(
        0.35 * query
        + 0.35 * distractor_query
        + rng.normal(scale=0.22, size=(n_tokens, dim))
    )
    saliency = rng.uniform(0.001, 0.01, size=n_tokens)

    target_idx = 0
    context_idx = 1
    distractor_start = 2
    context_neighbors = [6, 7, 8]

    tokens[target_idx] = _unit(query + rng.normal(scale=0.045, size=dim))
    saliency[target_idx] = rng.uniform(0.82, 0.98)

    # The context token is intentionally low-saliency and low query-relevance.
    # It is far from the local evidence, so query-only or saliency-only policies
    # should not solve the benchmark by default.
    tokens[context_idx] = _unit(context + rng.normal(scale=0.045, size=dim))
    saliency[context_idx] = rng.uniform(0.001, 0.006)

    for offset in range(36):
        idx = distractor_start + offset
        tokens[idx] = _unit(distractor_query + rng.normal(scale=0.06, size=dim))
        saliency[idx] = rng.uniform(0.70, 0.94)

    for idx in context_neighbors:
        tokens[idx] = _unit(context + rng.normal(scale=0.13, size=dim))
        saliency[idx] = rng.uniform(0.001, 0.008)

    # Add weak query-like clutter so retaining only query-nearest tokens is not
    # sufficient to preserve the full context signal.
    for idx in range(42, min(52, n_tokens)):
        tokens[idx] = _unit(0.45 * query + rng.normal(scale=0.55, size=dim))
        saliency[idx] = rng.uniform(0.001, 0.008)

    return {
        "id": f"sample_{sample_id:04d}",
        "tokens": tokens.round(6).tolist(),
        "query_embedding": query.round(6).tolist(),
        "cls_attention": saliency.round(6).tolist(),
        "target_evidence_index": target_idx,
        "target_context_index": context_idx,
        "context_probe": context.round(6).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/token_benchmark.jsonl")
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260205809)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for i in range(args.samples):
            handle.write(json.dumps(make_sample(rng, i, args.tokens, args.dim)) + "\n")


if __name__ == "__main__":
    main()
