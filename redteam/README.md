This audit runs candidates as ordinary file-producing methods. Each candidate's
`run.sh` installs its `method.py` and `redteam/lib/candidate.py` under `method/`,
runs the candidate, then invokes `bash scripts/evaluate_test.sh` with no arguments
after the candidate process has exited. No evaluator, Python environment, or
evaluator process is modified. The harness restores data, output files, and
scoring sidecars after each run. `redteam/.state/`, `.cache/`, and per-case
`artifacts/` are disposable and gitignored; scores and run metadata are retained.

Run all checks in a CPU job with `python redteam/audit.py suite`, or run one case
with `bash redteam/<case>/run.sh`. A missing local cache is regenerated using the
visible reference and baseline runners. The three reference seeds exercise the
deterministic method on the same fixed test set. The improved method tunes only
on the visible train partition and uses only tokens, query, and saliency at
inference. All intentionally invalid or answer-dependent candidates have
`shortcut_` names.

`summary.json` uses the supplied ranking rule, with absolute coefficient sizes
and the metric's declared direction. Since future-attempt variation cannot be
known, its scalar gains use the explicitly listed audit cohort. Pareto-dominating
shortcuts beat the reference for every positive finite normalization; omitted
method values contribute zero under the specified missing-value rule. Failed
test commands are not counted as successful scores, even if a stale JSON file
remains. See `REDTEAM.md` for conclusions and limits.
