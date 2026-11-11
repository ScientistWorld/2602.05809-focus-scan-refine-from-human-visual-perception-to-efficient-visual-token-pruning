# Hardening checks (iteration 11)

The cases in the parent `redteam/` directory target the retired synthetic gym (iteration 10)
and no longer run. This directory holds the checks for the rebuilt gym:

- `register.py`: writes pruner registries for the FSR variants/baselines (`variants`) and
  for the shortcut cases (`shortcuts`).
- `shortcuts.py`: 13 shortcut pruners (over-budget, empty, NaN, out-of-range, partial,
  crash, vector injection, fixed positions, score forging, input tampering, environment
  probe, LLM-in-the-loop search, answer reading).
- `run_checks.sh`: runs the shortcuts through `eval/test/evaluate.py`, plus the
  answer-reading case on the visible split. It is called by `scripts/run.sh` after
  `scripts/reproduce.sh`.
- `analyze.py`: paired comparisons from answers saved with `--save-answers`.
- `results/`: aggregate scores only. Per-question hidden answers are never stored here.

Results and conclusions: `HARDENING.md`.
