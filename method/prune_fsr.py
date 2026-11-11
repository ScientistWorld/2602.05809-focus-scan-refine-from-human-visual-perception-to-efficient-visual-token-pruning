#!/usr/bin/env python3
"""Pruner program for FSR. Invoked by the evaluator as

    python method/prune_fsr.py [options] --inputs DIR --budget K --output FILE

Defaults are the paper's: alpha=3, beta=1, rho=0.9, kappa=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.pruner_io import PrunerInputs, standard_args, write_selections  # noqa: E402
from fsr import focus_scan_refine, instruction_relevance  # noqa: E402


def main() -> None:
    parser = standard_args(__doc__.splitlines()[0])
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--no-scan", action="store_true", help="ablation: Focus ranking fills the whole budget")
    parser.add_argument("--relevance-sign", choices=["neg", "pos"], default="neg")
    parser.add_argument("--refine-weights", choices=["priority", "uniform"], default="priority")
    parser.add_argument("--scan-mode", choices=["fps", "dpp"], default="fps",
                        help="fps = the paper's conditional farthest-point sampling; dpp = variant")
    args = parser.parse_args()

    inp = PrunerInputs(args.inputs)
    cosine = inp.array("projected_cosine")
    joint = inp.array("joint_embeds")
    cls_attn = inp.array("cls_attention")
    text = inp.array("text_embeds")
    records: dict[str, list[dict]] = {}
    focus_sizes = []
    for img, positions in inp.queries_by_image().items():
        cos_img = np.asarray(cosine[img], dtype=np.float32)
        joint_img = np.asarray(joint[img], dtype=np.float32)
        sal = np.asarray(cls_attn[img], dtype=np.float64)
        for pos in positions:
            rel = instruction_relevance(joint_img, np.asarray(text[pos]), args.relevance_sign)
            res = focus_scan_refine(
                cos_img, rel, sal, args.budget, alpha=args.alpha, beta=args.beta, rho=args.rho,
                kappa=args.kappa, scan=not args.no_scan, refine_weights=args.refine_weights,
                scan_mode=args.scan_mode,
            )
            records[inp.queries[pos]["qid"]] = res.slots
            focus_sizes.append(len(res.focus))
    write_selections(args.output, records)
    fs = np.asarray(focus_sizes)
    print(f"FSR K={args.budget}: {len(records)} queries, focus size mean {fs.mean():.2f} "
          f"median {np.median(fs):.0f} min {fs.min()} max {fs.max()}")


if __name__ == "__main__":
    main()
