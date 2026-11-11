# Verification Report (iteration 11)

- `python validate.py --post-design --compare`: all checks pass (job `c35e1b98-9e0`).
- `scripts/reproduce.sh` reproduced `scoring/scores_train.json` and `scores_test.json`
  byte-for-byte across two independent GPU runs (`e3cfa3ce-8d2`, `c35e1b98-9e0`).
- `eval/` never imports `method/`. `eval/train/` and `eval/test/` share no code. Test
  questions and answers exist only in `eval/test/`.
- Shortcut audit through the hidden evaluator: see `HARDENING.md`.
- Milestone: core_claim. It is a scaled benchmark (COCO-derived questions) on the paper's
  model and budget; it is not the paper's full benchmark suite.
