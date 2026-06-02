#!/usr/bin/env python3
"""Run simple training-free visual-token pruning baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


METHOD_ALIASES = {
    # Paper-family names used in scoring/reference.json. These are lightweight
    # scaled proxies, not full reimplementations of the external baseline repos.
    "fastv": "saliency_topk",
    "divprune": "diversity_fps",
    "cdpruner": "fused_topk",
}


def select_indices(sample: dict, method: str, budget: int) -> list[int]:
    method = METHOD_ALIASES.get(method, method)
    tokens = np.asarray(sample["tokens"], dtype=np.float64)
    saliency = np.asarray(sample["cls_attention"], dtype=np.float64)
    query = np.asarray(sample["query_embedding"], dtype=np.float64)
    norm_tokens = _unit(tokens)
    norm_query = query / (np.linalg.norm(query) + 1e-12)
    relevance = norm_tokens @ norm_query

    if method == "saliency_topk":
        score = saliency
        return [int(i) for i in np.argsort(-score, kind="mergesort")[:budget]]
    if method == "relevance_topk":
        score = relevance
        return [int(i) for i in np.argsort(-score, kind="mergesort")[:budget]]
    if method == "fused_topk":
        score = np.power(_minmax(relevance) + 1e-12, 3.0) * (_minmax(saliency) + 1e-12)
        return [int(i) for i in np.argsort(-score, kind="mergesort")[:budget]]
    if method == "diversity_fps":
        # Similarity-only pruning methods in the paper do not get the Focus
        # stage's saliency/relevance seed. Start from the token farthest from
        # the global mean so this baseline is query-agnostic before FPS.
        mean_token = _unit(np.mean(norm_tokens, axis=0, keepdims=True))[0]
        selected = [int(np.argmin(norm_tokens @ mean_token))]
        min_distance = 1.0 - (norm_tokens @ norm_tokens[selected[0]])
        min_distance[selected] = -np.inf
        while len(selected) < budget:
            nxt = int(np.argmax(min_distance))
            selected.append(nxt)
            min_distance = np.minimum(min_distance, 1.0 - (norm_tokens @ norm_tokens[nxt]))
            min_distance[selected] = -np.inf
        return selected
    raise ValueError(f"unknown baseline: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/token_benchmark.jsonl")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fastv", "divprune", "cdpruner"],
    )
    parser.add_argument("--budget", type=int, default=16)
    args = parser.parse_args()

    samples = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        with (output_dir / f"{method}_outputs.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                indices = select_indices(sample, method, args.budget)
                tokens = np.asarray(sample["tokens"], dtype=np.float64)[indices]
                handle.write(
                    json.dumps(
                        {
                            "id": sample["id"],
                            "method": method,
                            "budget": args.budget,
                            "selected_indices": indices,
                            "compressed_tokens": tokens.round(6).tolist(),
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
