#!/usr/bin/env python3
"""Run FSR over a benchmark JSONL and write pruning artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fsr import focus_scan_refine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/token_benchmark.jsonl")
    parser.add_argument("--output", default="checkpoints/fsr_outputs.jsonl")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--kappa", type=float, default=1.0)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.data).open("r", encoding="utf-8") as inp, out.open(
        "w", encoding="utf-8"
    ) as handle:
        for line in inp:
            sample = json.loads(line)
            result = focus_scan_refine(
                np.asarray(sample["tokens"], dtype=np.float64),
                np.asarray(sample["query_embedding"], dtype=np.float64),
                np.asarray(sample["cls_attention"], dtype=np.float64),
                args.budget,
                alpha=args.alpha,
                beta=args.beta,
                rho=args.rho,
                kappa=args.kappa,
            )
            handle.write(
                json.dumps(
                    {
                        "id": sample["id"],
                        "method": "fsr",
                        "budget": args.budget,
                        "selected_indices": result.selected_indices,
                        "focus_indices": result.focus_indices,
                        "scan_indices": result.scan_indices,
                        "compressed_tokens": result.compressed_tokens.round(6).tolist(),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
