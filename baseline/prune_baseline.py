#!/usr/bin/env python3
"""Pruner program for the baselines. Invoked by the evaluator as

    python baseline/prune_baseline.py --method NAME --inputs DIR --budget K --output FILE
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.pruner_io import PrunerInputs, selection_slots, standard_args, write_selections  # noqa: E402
import baselines as B  # noqa: E402

METHODS = ("random", "cls_topk", "divprune", "cdpruner", "pool")


def main() -> None:
    parser = standard_args(__doc__.splitlines()[0])
    parser.add_argument("--method", required=True, choices=METHODS)
    args = parser.parse_args()
    inp = PrunerInputs(args.inputs)
    k = args.budget
    records: dict[str, list[dict]] = {}
    for img, positions in inp.queries_by_image().items():
        cos_img = sal = joint_img = None
        if args.method in ("divprune", "cdpruner"):
            cos_img = np.asarray(inp.array("projected_cosine")[img], dtype=np.float32)
        if args.method == "cls_topk":
            sal = np.asarray(inp.array("cls_attention")[img])
        if args.method == "cdpruner":
            joint_img = np.asarray(inp.array("joint_embeds")[img], dtype=np.float32)
        shared = None
        if args.method == "divprune":            # query-agnostic: one selection per image
            shared = selection_slots(B.divprune(cos_img, k))
        elif args.method == "cls_topk":
            shared = selection_slots(B.cls_topk(sal, k))
        elif args.method == "pool":
            shared = B.pool_slots(k)
        for pos in positions:
            qid = inp.queries[pos]["qid"]
            if shared is not None:
                records[qid] = shared
            elif args.method == "random":
                records[qid] = selection_slots(B.random_subset(qid, k))
            else:
                rel = B.cdpruner_relevance(joint_img, np.asarray(inp.array("text_embeds")[pos]))
                records[qid] = selection_slots(B.cdpruner(cos_img, rel, k))
    write_selections(args.output, records)
    print(f"{args.method} K={k}: {len(records)} queries")


if __name__ == "__main__":
    main()
