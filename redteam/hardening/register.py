#!/usr/bin/env python3
"""Write pruner registries for the hardening runs.

    python redteam/hardening/register.py variants  checkpoints/dev_pruners
    python redteam/hardening/register.py shortcuts checkpoints/shortcut_pruners
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FSR = ["python", "method/prune_fsr.py"]
BASE = ["python", "baseline/prune_baseline.py", "--method"]
SC = ["python", "redteam/hardening/shortcuts.py", "--case"]

VARIANTS = {
    "fsr": FSR,
    "random": BASE + ["random"],
    "cls_topk": BASE + ["cls_topk"],
    "divprune": BASE + ["divprune"],
    "cdpruner": BASE + ["cdpruner"],
    "pool": BASE + ["pool"],
    "fsr_allmass": FSR + ["--focus-mass", "all"],
    "fsr_norefine": FSR + ["--kappa", "0"],
    "fsr_kappa2": FSR + ["--kappa", "2"],
    "fsr_kappa5": FSR + ["--kappa", "5"],
    "fsr_rho05": FSR + ["--rho", "0.5"],
    "fsr_rho07": FSR + ["--rho", "0.7"],
    "fsr_possign": FSR + ["--relevance-sign", "pos"],
    "fsr_saliency_only": FSR + ["--alpha", "0", "--beta", "1"],
    "fsr_relevance_only": FSR + ["--alpha", "1", "--beta", "0"],
    "fsr_uniform_refine": FSR + ["--refine-weights", "uniform"],
    "fsr_dpp_scan": FSR + ["--scan-mode", "dpp"],
}

SHORTCUTS = {f"sc_{c}": SC + [c] for c in (
    "keep_all", "empty", "nan_weights", "out_of_range", "partial", "crash", "vector_field",
    "first_positions", "forge_scores", "tamper_inputs", "probe_env", "llm_search", "read_answers",
)}


def main() -> None:
    which, out = sys.argv[1], Path(sys.argv[2])
    table = {"variants": VARIANTS, "shortcuts": SHORTCUTS}[which]
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for name, cmd in table.items():
        (out / f"{name}.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n")
    print(f"registered {len(table)} pruners in {out}")


if __name__ == "__main__":
    main()
