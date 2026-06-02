#!/usr/bin/env python3
"""Evaluate pruning artifacts for the token benchmark.

This module intentionally imports nothing from method/. It reads benchmark
records and output JSONL files, then scores whether compressed tokens preserve
local evidence and complementary global context under the fixed token budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def load_jsonl(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines() if line)}


def score_file(data: dict[str, dict], output_path: Path) -> dict:
    outputs = load_jsonl(output_path)
    solved = []
    token_counts = []

    for sample_id, sample in data.items():
        if sample_id not in outputs:
            raise KeyError(f"{output_path} missing sample {sample_id}")
        pred = outputs[sample_id]
        compressed = _unit(np.asarray(pred["compressed_tokens"], dtype=np.float64))
        query = np.asarray(sample["query_embedding"], dtype=np.float64)
        query = query / (np.linalg.norm(query) + 1e-12)
        context = np.asarray(sample["context_probe"], dtype=np.float64)
        context = context / (np.linalg.norm(context) + 1e-12)

        evidence = float(np.max(compressed @ query)) >= 0.82
        context_ok = float(np.max(compressed @ context)) >= 0.85

        solved.append(evidence and context_ok)
        token_counts.append(len(pred["compressed_tokens"]))

    n = len(data)
    mean_tokens = float(np.mean(token_counts))
    full_tokens = len(next(iter(data.values()))["tokens"])
    return {
        "performance_retained": round(float(np.mean(solved)), 6),
        "mean_token_count": round(mean_tokens, 6),
        "retained_fraction": round(mean_tokens / full_tokens, 6),
        "sample_count": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/token_benchmark.jsonl")
    parser.add_argument("--outputs-dir", default="checkpoints")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    data = load_jsonl(Path(args.data))
    output_dir = Path(args.outputs_dir)
    files = sorted(output_dir.glob("*_outputs.jsonl"))
    if not files:
        raise FileNotFoundError(f"no *_outputs.jsonl files found in {output_dir}")

    results = {}
    for path in files:
        method_name = path.name[: -len("_outputs.jsonl")]
        results[method_name] = score_file(data, path)

    scores = {"experiments": {"token_pruning_core": {"results": results}}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
