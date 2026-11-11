#!/usr/bin/env python3
"""Paired comparison of saved per-question answers (pure Python, no numpy).

    python redteam/hardening/analyze.py ANSWERS_DIR [--reference fsr] [--json OUT]

ANSWERS_DIR holds ``_full_tokens.jsonl`` and ``<method>__<budget>.jsonl`` written by an
evaluator run with ``--save-answers``. For every method it recomputes the scored metrics
and the paired difference to the reference (same questions, same model) with a standard
error from the per-family paired differences.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

FAMILIES = ("exist", "count", "caption", "absent")
LOCAL, GLOBAL = ("exist", "count"), ("caption", "absent")


def load(path: Path) -> dict[str, tuple[str, bool]]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["qid"]] = (r["family"], bool(r["correct"]))
    return out


def fam_acc(ans):
    tot, cor = defaultdict(int), defaultdict(int)
    for fam, c in ans.values():
        tot[fam] += 1
        cor[fam] += c
    return {f: cor[f] / tot[f] for f in FAMILIES if tot[f]}


def rel(acc, full, fams):
    vals = [acc[f] / full[f] for f in fams if f in acc and full.get(f)]
    return sum(vals) / len(vals) if vals else 0.0


def paired(a, b, full, fams):
    """mean over fams of (acc_a - acc_b)/full, with SE from per-question paired differences."""
    est, var = 0.0, 0.0
    for f in fams:
        d = [int(a[q][1]) - int(b[q][1]) for q in a if a[q][0] == f]
        n = len(d)
        m = sum(d) / n
        v = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
        est += m / full[f]
        var += v / n / full[f] ** 2
    k = len(fams)
    return est / k, math.sqrt(var) / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--reference", default="fsr")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    d = Path(args.answers)
    full = fam_acc(load(d / "_full_tokens.jsonl"))
    methods = sorted({p.name.split("__")[0] for p in d.glob("*__*.jsonl")})
    ans = {(m, k): load(d / f"{m}__{k}.jsonl") for m in methods for k in (64, 32) if (d / f"{m}__{k}.jsonl").exists()}
    ref = args.reference
    rows = {}
    for m in methods:
        if (m, 64) not in ans or (m, 32) not in ans:
            continue
        a64, a32 = fam_acc(ans[(m, 64)]), fam_acc(ans[(m, 32)])
        row = {
            "performance_retained": rel(a64, full, FAMILIES),
            "performance_retained_32": rel(a32, full, FAMILIES),
            "local_evidence_retained": rel(a64, full, LOCAL),
            "global_context_retained": rel(a64, full, GLOBAL),
            "family_accuracy_64": a64,
            "family_accuracy_32": a32,
        }
        if (ref, 64) in ans and m != ref:
            for k, key in ((64, "performance_retained"), (32, "performance_retained_32")):
                est, se = paired(ans[(m, k)], ans[(ref, k)], full, FAMILIES)
                row[f"delta_{key}"] = est
                row[f"se_{key}"] = se
        rows[m] = row
    print(f"full-token family accuracy: { {f: round(v, 4) for f, v in full.items()} }")
    hdr = f"{'method':22s} {'perf64':>8s} {'perf32':>8s} {'local64':>8s} {'global64':>8s} {'d64 (se)':>18s} {'d32 (se)':>18s}"
    print(hdr)
    for m, r in sorted(rows.items(), key=lambda kv: -kv[1]["performance_retained"]):
        d64 = f"{r['delta_performance_retained']:+.4f} ({r['se_performance_retained']:.4f})" if "delta_performance_retained" in r else ""
        d32 = f"{r['delta_performance_retained_32']:+.4f} ({r['se_performance_retained_32']:.4f})" if "delta_performance_retained_32" in r else ""
        print(f"{m:22s} {r['performance_retained']:8.4f} {r['performance_retained_32']:8.4f} "
              f"{r['local_evidence_retained']:8.4f} {r['global_context_retained']:8.4f} {d64:>18s} {d32:>18s}")
    if args.json:
        Path(args.json).write_text(json.dumps({"full_token_family_accuracy": full, "reference": ref, "methods": rows},
                                              indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
