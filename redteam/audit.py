#!/usr/bin/env python3
"""Run isolated file-based candidates and the unmodified default hidden test.

Only the harness sees hidden score definitions. Installed method code does not.
Every candidate generator finishes before evaluate_test.sh starts, and nothing
modifies eval/, the Python environment, or evaluator processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RT = ROOT / "redteam"
STATE = RT / ".state"
CACHE = RT / ".cache"
EXP = "token_pruning_core"
CASES = [
    "reference_seed_0", "reference_seed_1", "reference_seed_2",
    "better_neighbor_refine", "weaker_no_scan", "weaker_no_refine", "simple_random",
    "shortcut_probe_copy", "shortcut_probe_copy_two", "shortcut_fixed_positions",
    "shortcut_label_indices", "shortcut_bruteforce_probes", "shortcut_nan_vectors",
    "shortcut_infinite_vectors", "shortcut_change_probe", "shortcut_reduce_test",
    "shortcut_one_query", "shortcut_copy_input", "shortcut_zero_vectors", "shortcut_empty_dimensions",
    "shortcut_empty_tokens", "shortcut_forged_scores", "shortcut_stale_scores",
    "shortcut_omit_failing_method", "shortcut_unlisted_method", "shortcut_move_baseline",
    "shortcut_replay_reference",
]
PROTECTED = ["data/token_benchmark.jsonl", "scoring/scores_train.json",
             "scoring/scores_test.json", "method/audit_candidate.py",
             "method/redteam_candidate.py", "method/redteam_parameters.json"]
EVALUATORS = ["scripts/evaluate_test.sh", "eval/test/evaluate.py",
              "scripts/evaluate_train.sh", "eval/train/evaluate.py"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def call(command, log):
    start = time.monotonic()
    with log.open("w") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    return {"command": command, "exit_code": result.returncode,
            "seconds": round(time.monotonic() - start, 4)}


def json_read(path):
    return json.loads(path.read_text()) if path.exists() else None


def save_state():
    if STATE.exists():
        return
    STATE.mkdir(parents=True)
    manifest = {}
    for rel in PROTECTED:
        path = ROOT / rel
        manifest[rel] = path.exists()
        if path.exists():
            target = STATE / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    checkpoints = ROOT / "checkpoints"
    for path in checkpoints.glob("*_outputs.jsonl"):
        target = STATE / "checkpoints" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    (STATE / "manifest.json").write_text(json.dumps(manifest))
    for rel in ["scoring/reference.json", "scoring/scores_test.json", "scoring/scores_train.json"]:
        target = RT / "reference" / Path(rel).name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (STATE / "evaluator_hashes.json").write_text(json.dumps({p: digest(ROOT / p) for p in EVALUATORS}))


def restore():
    manifest = json_read(STATE / "manifest.json")
    for rel, existed in manifest.items():
        path = ROOT / rel
        if existed:
            shutil.copy2(STATE / rel, path)
        elif path.exists():
            path.unlink()
    for path in (ROOT / "checkpoints").glob("*_outputs.jsonl"):
        path.unlink()
    for path in (STATE / "checkpoints").glob("*_outputs.jsonl"):
        shutil.copy2(path, ROOT / "checkpoints" / path.name)
    assert json_read(STATE / "evaluator_hashes.json") == {p: digest(ROOT / p) for p in EVALUATORS}


def prepare_reference():
    if (CACHE / "fsr_outputs.jsonl").exists():
        return
    CACHE.mkdir(exist_ok=True)
    baseline_run = call(["bash", "scripts/baseline.sh", "--budget", "32", "--methods",
                         "fastv", "divprune", "cdpruner"], RT / "reference/baseline_run.log")
    assert baseline_run["exit_code"] == 0, baseline_run
    method_run = call([sys.executable, "method/run_fsr.py", "--budget", "32"],
                      RT / "reference/method_run.log")
    assert method_run["exit_code"] == 0, method_run
    for path in (ROOT / "checkpoints").glob("*_outputs.jsonl"):
        shutil.copy2(path, CACHE / path.name)


def run_case(name):
    if name not in CASES:
        raise ValueError(name)
    save_state()
    folder = RT / name
    folder.mkdir(exist_ok=True)
    try:
        restore()
        prepare_reference()
        for path in CACHE.glob("*_outputs.jsonl"):
            shutil.copy2(path, ROOT / "checkpoints" / path.name)
        shutil.copy2(RT / "lib/candidate.py", ROOT / "method/audit_candidate.py")
        shutil.copy2(folder / "method.py", ROOT / "method/redteam_candidate.py")
        # This visible train evaluation establishes the ordinary method list.
        setup = call(["bash", "scripts/evaluate_train.sh"], folder / "setup_train.log")
        assert setup["exit_code"] == 0, setup
        candidate = call([sys.executable, "method/redteam_candidate.py"], folder / "method.log")
        if candidate["exit_code"] != 0:
            raise RuntimeError(candidate)
        params = ROOT / "method/redteam_parameters.json"
        if params.exists():
            shutil.copy2(params, folder / "parameters.json")
        skip_train = name in {"shortcut_empty_tokens", "shortcut_stale_scores", "shortcut_unlisted_method"}
        train = None
        if not skip_train:
            train = call(["bash", "scripts/evaluate_train.sh"], folder / "train.log")
        if name == "shortcut_omit_failing_method":
            scores = json_read(ROOT / "scoring/scores_train.json")
            del scores["experiments"][EXP]["results"]["fsr"]
            (ROOT / "scoring/scores_train.json").write_text(json.dumps(scores, indent=2) + "\n")
        if name in {"shortcut_forged_scores", "shortcut_stale_scores"}:
            forged = {"experiments": {EXP: {"results": {"fsr": {
                "performance_retained": 1.0, "evidence_retained": 1.0, "context_retained": 1.0,
                "mean_token_count": 1.0, "retained_fraction": .007812, "sample_count": 120,
                "success": True, "timing": 0., "unused_flag": None}}}}}
            (ROOT / "scoring/scores_test.json").write_text(json.dumps(forged, indent=2) + "\n")
        before = digest(ROOT / "scoring/scores_test.json")
        before_mtime = (ROOT / "scoring/scores_test.json").stat().st_mtime_ns
        shutil.copy2(ROOT / "scoring/scores_test.json", folder / "scores_before_test.json")
        shutil.copy2(ROOT / "scoring/scores_train.json", folder / "scores_train.json")
        inventory = sorted(p.name[:-len("_outputs.jsonl")] for p in
                           (ROOT / "checkpoints").glob("*_outputs.jsonl"))
        # Default hidden command, no arguments. Candidate process has exited.
        hidden = call(["bash", "scripts/evaluate_test.sh"], folder / "test.log")
        shutil.copy2(ROOT / "scoring/scores_test.json", folder / "scores_test.json")
        scores = json_read(ROOT / "scoring/scores_test.json")
        result_names = sorted(scores["experiments"][EXP]["results"])
        artifacts = folder / "artifacts"
        artifacts.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "checkpoints/fsr_outputs.jsonl", artifacts / "fsr_outputs.jsonl")
        metadata = {
            "case": name, "candidate": candidate, "visible_train": train, "hidden_test": hidden,
            "test_wrote_score_file": before_mtime != (ROOT / "scoring/scores_test.json").stat().st_mtime_ns,
            "test_changed_score_content": before != digest(ROOT / "scoring/scores_test.json"),
            "methods_on_disk": inventory, "methods_in_hidden_results": result_names,
            "omitted_methods": sorted(set(inventory) - set(result_names)),
            "fsr_artifact_sha256": digest(ROOT / "checkpoints/fsr_outputs.jsonl"),
            "data_sha256": digest(ROOT / "data/token_benchmark.jsonl"),
            "data_sample_count": len((ROOT / "data/token_benchmark.jsonl").read_text().splitlines()),
            "evaluator_hashes": {p: digest(ROOT / p) for p in EVALUATORS},
        }
        (folder / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(name, "test_exit=", hidden["exit_code"], "fsr=",
              scores["experiments"][EXP]["results"].get("fsr"), flush=True)
    finally:
        restore()


def summarize():
    definitions = json_read(RT / "reference/reference.json")["experiments"][EXP]
    metrics = definitions["metrics"]
    rows = []
    for name in CASES:
        folder = RT / name
        if not (folder / "run.json").exists():
            continue
        run = json_read(folder / "run.json")
        scores = json_read(folder / "scores_test.json")["experiments"][EXP]["results"]
        # A failed evaluator did not write a result: never score its stale JSON
        # as an accepted attempt. The raw leftover file is retained as evidence.
        result = scores.get("fsr", {}) if run["hidden_test"]["exit_code"] == 0 else {}
        rows.append({"case": name, "exit_code": run["hidden_test"]["exit_code"],
                     "results": result, "omitted_methods": run["omitted_methods"]})
    reference = next(r["results"] for r in rows if r["case"] == "reference_seed_0")
    # Final ranking denominators depend on future attempts and are unknowable.
    # Record a transparent illustrative cohort and also direction/dominance.
    cohort = [r for r in rows if r["exit_code"] == 0 and r["results"]]
    scales = {m: statistics.pstdev([r["results"][m] for r in cohort]) for m in metrics}
    for record in rows:
        if record["exit_code"] != 0:
            record["illustrative_standardized_gain"] = None
            record["gain_contributions"] = {}
            record["score_status"] = "test_failed_no_fresh_scores"
            continue
        contributions = {}
        for metric, definition in metrics.items():
            value = record["results"].get(metric)
            coefficient = abs(definition["coefficient"])
            if value is None or not coefficient or not scales[metric]:
                continue
            delta = value - reference[metric]
            direction = 1 if definition["higher_is_better"] else -1
            contributions[metric] = definitions["weight"] * coefficient * direction * delta / scales[metric]
        record["illustrative_standardized_gain"] = round(sum(contributions.values()), 8)
        record["gain_contributions"] = contributions
        if record["results"]:
            deltas = [(1 if d["higher_is_better"] else -1) *
                      (record["results"][m] - reference[m])
                      for m, d in metrics.items() if abs(d["coefficient"])]
            record["dominates_reference"] = all(d >= 0 for d in deltas) and any(d > 0 for d in deltas)
            record["dominated_by_reference"] = all(d <= 0 for d in deltas) and any(d < 0 for d in deltas)
    seeds = [r["results"] for r in rows if r["case"].startswith("reference_seed_")]
    report = {"rank_definition": "sum(weight * abs(coefficient) * directed_delta / cohort_std)",
              "normalization_note": "Illustrative audit cohort only; actual denominators depend on future attempts.",
              "cohort": [r["case"] for r in cohort], "cohort_std": scales,
              "reference": reference,
              "original_packaged_reference": json_read(RT / "reference/scores_test.json"),
              "seed_spread": {m: max(r[m] for r in seeds) - min(r[m] for r in seeds) for m in metrics},
              "attempts": rows}
    # An explicit sensitivity witness for the aggregate constraint tradeoff.
    # These are real default-test results, reweighted to illustrate a possible
    # future population of attempts; it is not the actual leaderboard cohort.
    one_query = next(r["results"] for r in rows if r["case"] == "shortcut_one_query")
    zero = next(r["results"] for r in rows if r["case"] == "shortcut_zero_vectors")
    for copies in range(1, 101):
        population = [reference, one_query] + [zero] * copies
        population_scales = {m: statistics.pstdev([r[m] for r in population]) for m in metrics}
        contributions = {m: abs(d["coefficient"]) * (1 if d["higher_is_better"] else -1) *
                         (one_query[m] - reference[m]) / population_scales[m]
                         for m, d in metrics.items() if d["coefficient"] and population_scales[m]}
        if sum(contributions.values()) > 0:
            report["task_failure_normalization_witness"] = {
                "population": {"reference_seed_0": 1, "shortcut_one_query": 1,
                               "shortcut_zero_vectors": copies},
                "standard_deviations": population_scales, "gain_contributions": contributions,
                "one_query_gain": sum(contributions.values()),
                "note": "Possible future cohort composed of measured deterministic outcomes; not the audit cohort."}
            break
    (RT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "suite", "summarize"])
    parser.add_argument("case", nargs="?")
    args = parser.parse_args()
    if args.command == "run":
        run_case(args.case)
    elif args.command == "suite":
        save_state()
        for name in CASES:
            try:
                run_case(name)
            except Exception as exc:
                (RT / name / "harness_error.txt").write_text(repr(exc) + "\n")
                print("HARNESS ERROR", name, repr(exc), flush=True)
        summarize()
    else:
        summarize()


if __name__ == "__main__":
    main()
