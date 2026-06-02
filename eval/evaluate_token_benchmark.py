#!/usr/bin/env python3
"""Evaluate pruning artifacts for the token benchmark.

This module intentionally imports nothing from method/. It reads benchmark
records and output JSONL files, then scores whether compressed tokens preserve
local evidence and complementary global context under the fixed token budget.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _unit(x: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in x)) or 1e-12
    return [v / norm for v in x]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_jsonl(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text().splitlines() if line)}


def score_file(data: dict[str, dict], output_path: Path) -> dict:
    outputs = load_jsonl(output_path)
    retained_scores = []
    evidence_scores = []
    context_scores = []
    token_counts = []

    for sample_id, sample in data.items():
        if sample_id not in outputs:
            raise KeyError(f"{output_path} missing sample {sample_id}")
        pred = outputs[sample_id]
        compressed = [_unit(token) for token in pred["compressed_tokens"]]
        query = _unit(sample["query_embedding"])
        context = _unit(sample["context_probe"])

        evidence = max(_dot(token, query) for token in compressed)
        context_ok = max(_dot(token, context) for token in compressed)
        evidence = max(0.0, min(1.0, evidence))
        context_ok = max(0.0, min(1.0, context_ok))

        # The paper reports retained downstream performance as a continuous
        # fraction, not as a strict all-or-nothing success rate. This scaled
        # benchmark mirrors that by averaging local evidence retention with
        # complementary context retention on the same [0, 1] similarity scale.
        evidence_scores.append(evidence)
        context_scores.append(context_ok)
        retained_scores.append(0.5 * (evidence + context_ok))
        token_counts.append(len(pred["compressed_tokens"]))

    n = len(data)
    mean_tokens = sum(token_counts) / len(token_counts)
    full_tokens = len(next(iter(data.values()))["tokens"])
    return {
        "performance_retained": round(sum(retained_scores) / len(retained_scores), 6),
        "evidence_retained": round(sum(evidence_scores) / len(evidence_scores), 6),
        "context_retained": round(sum(context_scores) / len(context_scores), 6),
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
