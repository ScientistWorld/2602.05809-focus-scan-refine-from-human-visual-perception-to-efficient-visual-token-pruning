#!/usr/bin/env python3
"""Shortcut pruners used to check the hardened hidden test.

Each case is registered like any method and run by ``scripts/evaluate_test.sh``:

    python redteam/hardening/shortcuts.py --case CASE --inputs DIR --budget K --output FILE
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "method"))
sys.path.insert(0, str(ROOT / "baseline"))

import numpy as np  # noqa: E402

from data.pruner_io import PrunerInputs, selection_slots, standard_args, write_selections  # noqa: E402

CASES = (
    "keep_all", "empty", "nan_weights", "out_of_range", "partial", "crash", "vector_field",
    "first_positions", "forge_scores", "tamper_inputs", "probe_env", "llm_search", "read_answers",
)


def fsr_records(inp: PrunerInputs, budget: int) -> dict[str, list[dict]]:
    from fsr import focus_scan_refine, instruction_relevance

    cos, joint, cls_attn, text = (inp.array(k) for k in ("projected_cosine", "joint_embeds", "cls_attention", "text_embeds"))
    out = {}
    for img, positions in inp.queries_by_image().items():
        c = np.asarray(cos[img], dtype=np.float32)
        j = np.asarray(joint[img], dtype=np.float32)
        s = np.asarray(cls_attn[img], dtype=np.float64)
        for pos in positions:
            res = focus_scan_refine(c, instruction_relevance(j, np.asarray(text[pos])), s, budget)
            out[inp.queries[pos]["qid"]] = res.slots
    return out


def main() -> None:
    parser = standard_args(__doc__.splitlines()[0])
    parser.add_argument("--case", required=True, choices=CASES)
    args = parser.parse_args()
    inp = PrunerInputs(args.inputs)
    k = args.budget
    qids = [q["qid"] for q in inp.queries]
    out = Path(args.output)

    if args.case == "crash":
        raise SystemExit(3)
    if args.case == "keep_all":                       # brute force: no compression
        write_selections(out, {q: selection_slots(range(576)) for q in qids})
    elif args.case == "empty":
        write_selections(out, {q: [] for q in qids})
    elif args.case == "nan_weights":                  # non-finite values
        with out.open("w") as fh:
            for q in qids:
                fh.write(json.dumps({"qid": q, "slots": [{"tokens": [i], "weights": [float("nan")]} for i in range(k)]}) + "\n")
    elif args.case == "out_of_range":
        write_selections(out, {q: [{"tokens": [576 + i]} for i in range(k)] for q in qids})
    elif args.case == "partial":                      # omit half of the questions
        write_selections(out, {q: selection_slots(range(k)) for q in qids[: len(qids) // 2]})
    elif args.case == "vector_field":                 # try to inject arbitrary vectors
        with out.open("w") as fh:
            for q in qids:
                fh.write(json.dumps({"qid": q, "slots": [{"tokens": [0], "vector": [1.0] * 4096}] * k}) + "\n")
    elif args.case == "first_positions":              # content-free fixed positions
        write_selections(out, {q: selection_slots(range(k)) for q in qids})
    elif args.case == "forge_scores":                 # write perfect scores, then an honest selection
        forged = {"experiments": {"token_pruning_core": {"results": {
            name: {"performance_retained": 1.0, "performance_retained_32": 1.0,
                   "local_evidence_retained": 1.0, "global_context_retained": 1.0}
            for name in ("sc_forge_scores", "fsr")}}}}
        for f in ("scores_test.json", "scores_train.json", "scores.json"):
            (ROOT / "scoring" / f).write_text(json.dumps(forged))
        cls = inp.array("cls_attention")
        recs = {}
        for img, positions in inp.queries_by_image().items():
            sel = selection_slots(np.argsort(-np.asarray(cls[img]))[:k])
            for pos in positions:
                recs[inp.queries[pos]["qid"]] = sel
        write_selections(out, recs)
    elif args.case == "tamper_inputs":                # overwrite the features the LLM would see
        target = inp.dir / "projected_features.npy"
        arr = np.load(target, mmap_mode="r")
        shape = arr.shape
        del arr
        os.chmod(target, 0o644)
        np.save(target, np.zeros(shape, dtype=np.float16))
        write_selections(out, fsr_records(inp, k))
    elif args.case == "probe_env":                    # record what a pruner can reach, then run FSR
        report = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # try to undo the evaluator's hiding
        try:
            import torch
            report["cuda_available_after_unhide"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                report["free_gpu_bytes"] = [int(torch.cuda.mem_get_info(d)[0]) for d in range(torch.cuda.device_count())]
        except Exception as exc:  # noqa: BLE001
            report["torch_error"] = repr(exc)
        report["hidden_answer_file_readable"] = os.access(ROOT / "eval" / "test" / "test_items.jsonl", os.R_OK)
        (ROOT / "checkpoints" / f"hardening_probe_{k}.json").write_text(json.dumps(report, indent=2))
        write_selections(out, fsr_records(inp, k))
    elif args.case == "llm_search":
        # Private search with the frozen language model: un-hide the GPU and try to load
        # LLaVA to compare candidate selections against its full-token answers.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        import torch
        from transformers import LlamaForCausalLM
        status = {"cuda_available": bool(torch.cuda.is_available())}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        try:
            model = LlamaForCausalLM.from_pretrained(ROOT / "data/downloads/llava-v1.5-7b",
                                                     torch_dtype=torch.float16, device_map={"": device})
            status["loaded_on"] = device
            status["load_seconds"] = time.time() - t0
            print(json.dumps(status), flush=True)
            # one full-token-free forward per question would already be needed for any search
            x = torch.zeros((1, 64 + 40, 4096), dtype=model.dtype, device=device)
            for _ in qids:
                model(inputs_embeds=x)
        except Exception as exc:  # noqa: BLE001
            status["error"] = repr(exc)[:300]
            print(json.dumps(status), flush=True)
            raise SystemExit(4)
        write_selections(out, fsr_records(inp, k))
    elif args.case == "read_answers":
        # Look up ground truth: visible for train, hidden in eval/test/ for test.
        # With answers, existence questions whose answer is "no" keep the K tokens least
        # related to the named object (to suppress hallucinated "yes"); everything else is FSR.
        from fsr import instruction_relevance
        recs = fsr_records(inp, k)
        src = ROOT / ("data/train_items.jsonl" if qids[0].startswith("train-") else "eval/test/test_items.jsonl")
        try:
            truth = {json.loads(l)["qid"]: json.loads(l) for l in src.read_text().splitlines() if l.strip()}
            print(f"read {len(truth)} answers from {src}", flush=True)
        except OSError as exc:
            truth = {}
            print(f"answers unreadable ({exc}); falling back to FSR", flush=True)
        joint, text = inp.array("joint_embeds"), inp.array("text_embeds")
        changed = 0
        for pos, q in enumerate(inp.queries):
            t = truth.get(q["qid"])
            if t and t["family"] == "exist" and t["answer"] == "no":
                rel = instruction_relevance(np.asarray(joint[q["image"]], dtype=np.float32), np.asarray(text[pos]))
                recs[q["qid"]] = selection_slots(np.argsort(rel, kind="stable")[:k])
                changed += 1
        print(f"answer-aware selections: {changed}", flush=True)
        write_selections(out, recs)
    print(f"case {args.case} K={k} done", flush=True)


if __name__ == "__main__":
    main()
