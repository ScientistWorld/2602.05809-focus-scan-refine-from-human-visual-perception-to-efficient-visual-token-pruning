#!/usr/bin/env python3
"""Held-out test-slice evaluator for the token-pruning benchmark.

This mirrors the train-slice contract while scoring the complementary held-out
half of the evaluation pool. When train scores are present, the same method
entries are required and scored here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPERIMENT_NAME = "token_pruning_core"
SPLIT_SEED = "2602.05809-token-pruning-split-v1"
SLICE_NAME = "test"


def _unit(x: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in x)) or 1e-12
    return [v / norm for v in x]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_jsonl(path: Path) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line)
    }


def split_ids(data: dict[str, dict]) -> set[str]:
    def key(sample_id: str) -> str:
        payload = f"{SPLIT_SEED}:{sample_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    ordered = sorted(data, key=key)
    midpoint = len(ordered) // 2
    return set(ordered[midpoint:])


def expected_methods(train_scores_path: Path) -> list[str] | None:
    if not train_scores_path.exists():
        return None
    scores = json.loads(train_scores_path.read_text(encoding="utf-8"))
    experiments = scores.get("experiments", scores)
    exp = experiments.get(EXPERIMENT_NAME)
    if not isinstance(exp, dict):
        raise KeyError(f"{train_scores_path} missing experiment {EXPERIMENT_NAME}")
    results = exp.get("results", exp)
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{train_scores_path} has no scored methods for {EXPERIMENT_NAME}")
    return sorted(name for name in results if not name.startswith("_"))


def score_file(data: dict[str, dict], output_path: Path, sample_ids: set[str]) -> dict:
    outputs = load_jsonl(output_path)
    retained_scores = []
    evidence_scores = []
    context_scores = []
    source_errors = []
    token_counts = []

    for sample_id in sorted(sample_ids):
        if sample_id not in outputs:
            raise KeyError(f"{output_path} missing {SLICE_NAME} sample {sample_id}")
        sample = data[sample_id]
        pred = outputs[sample_id]
        tokens = sample["tokens"]
        compressed = [_unit(token) for token in pred["compressed_tokens"]]
        selected_indices = pred.get("selected_indices")
        if selected_indices is None:
            raise KeyError(f"{output_path} missing selected_indices for {sample_id}")
        if len(selected_indices) != len(pred["compressed_tokens"]):
            raise ValueError(
                f"{output_path} selected_indices/compressed_tokens length mismatch for {sample_id}"
            )
        query = _unit(sample["query_embedding"])
        context = _unit(sample["context_probe"])

        evidence = max(_dot(token, query) for token in compressed)
        context_ok = max(_dot(token, context) for token in compressed)

        evidence = max(0.0, min(1.0, evidence))
        context_ok = max(0.0, min(1.0, context_ok))
        evidence_scores.append(evidence)
        context_scores.append(context_ok)
        retained_scores.append(0.5 * (evidence + context_ok))
        for index, compressed_token in zip(selected_indices, pred["compressed_tokens"]):
            if not isinstance(index, int) or index < 0 or index >= len(tokens):
                raise ValueError(f"{output_path} invalid selected index {index!r} for {sample_id}")
            source = tokens[index]
            squared_error = sum(
                (float(a) - float(b)) ** 2 for a, b in zip(compressed_token, source)
            )
            source_errors.append(math.sqrt(squared_error))
        token_counts.append(len(pred["compressed_tokens"]))

    mean_tokens = sum(token_counts) / len(token_counts)
    full_tokens = len(next(iter(data.values()))["tokens"])
    return {
        "performance_retained": round(sum(retained_scores) / len(retained_scores), 6),
        "evidence_retained": round(sum(evidence_scores) / len(evidence_scores), 6),
        "context_retained": round(sum(context_scores) / len(context_scores), 6),
        "selected_source_error": round(sum(source_errors) / len(source_errors), 6),
        "mean_token_count": round(mean_tokens, 6),
        "retained_fraction": round(mean_tokens / full_tokens, 6),
        "sample_count": len(sample_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/token_benchmark.jsonl")
    parser.add_argument("--outputs-dir", default="checkpoints")
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    data = load_jsonl(Path(args.data))
    sample_ids = split_ids(data)
    output_dir = Path(args.outputs_dir)
    files_by_method = {
        path.name[: -len("_outputs.jsonl")]: path
        for path in sorted(output_dir.glob("*_outputs.jsonl"))
    }
    if not files_by_method:
        raise FileNotFoundError(f"no *_outputs.jsonl files found in {output_dir}")

    methods = expected_methods(Path(args.train_scores))
    if methods is None:
        methods = sorted(files_by_method)

    missing = [method for method in methods if method not in files_by_method]
    if missing:
        raise FileNotFoundError(
            f"missing output files for train-scored methods: {', '.join(missing)}"
        )

    results = {
        method_name: score_file(data, files_by_method[method_name], sample_ids)
        for method_name in methods
    }

    scores = {"experiments": {EXPERIMENT_NAME: {"results": results}}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
