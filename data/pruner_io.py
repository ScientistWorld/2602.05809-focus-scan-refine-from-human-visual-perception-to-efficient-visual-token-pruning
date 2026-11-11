"""Helpers for pruner programs: read the evaluator's inputs, write selections.

The evaluator (``eval/``) writes a directory of inputs for one data split and
then runs each registered pruner as

    <command> --inputs DIR --budget K --output FILE

This module is an optional convenience for pruner programs; the evaluator
does not import it. See ``DIR/meta.json`` for the exact array layout.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_TOKENS = 576
GRID = 24


class PrunerInputs:
    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.queries = [
            json.loads(line) for line in (self.dir / "queries.jsonl").read_text().splitlines() if line.strip()
        ]
        self._arrays: dict[str, np.ndarray] = {}

    def array(self, name: str) -> np.ndarray:
        """Memory-mapped array: vision_features, projected_features, projected_cosine,
        joint_embeds, cls_attention, text_embeds."""
        if name not in self._arrays:
            self._arrays[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def queries_by_image(self) -> dict[int, list[int]]:
        """image row -> list of query positions (index into self.queries / text_embeds)."""
        groups: dict[int, list[int]] = defaultdict(list)
        for pos, q in enumerate(self.queries):
            groups[int(q["image"])].append(pos)
        return dict(sorted(groups.items()))


def standard_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--inputs", required=True, help="evaluator-provided input directory")
    parser.add_argument("--budget", type=int, required=True, help="maximum number of output slots")
    parser.add_argument("--output", required=True, help="selection JSONL to write")
    return parser


def selection_slots(indices) -> list[dict]:
    """Pure selection: one slot per kept token, in spatial (raster) order."""
    return [{"tokens": [int(i)]} for i in sorted(int(i) for i in indices)]


def write_selections(path: str | Path, records: dict[str, list[dict]]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for qid, slots in records.items():
            fh.write(json.dumps({"qid": qid, "slots": slots}) + "\n")
