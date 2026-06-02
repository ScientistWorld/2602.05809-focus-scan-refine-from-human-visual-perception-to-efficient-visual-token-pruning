# Scientist Constraints

## Data

Use the provided `data/token_benchmark.jsonl` benchmark for reported scores. Do not add external examples to the scored set or regenerate the benchmark with easier parameters when submitting comparable results.

## Budget

Use the same retained-token budget as the reproduction script unless explicitly reporting a separate experiment. Increasing the number of retained tokens is resource scaling, not a methodological improvement.

## Evaluation

Submit artifacts in the same `checkpoints/<budget>/*_outputs.jsonl` format and score them with `scripts/evaluate.sh`. The default budget directories are `checkpoints/budget16`, `checkpoints/budget32`, and `checkpoints/budget48`. Do not modify `eval/` or `scoring/reference.json` to improve a method's score.

## Compliance Checklist

- [ ] Scored on `data/token_benchmark.jsonl`
- [ ] Used the declared retained-token budget
- [ ] Wrote method outputs in the standard JSONL format
- [ ] Ran `scripts/evaluate.sh` without importing from `method/`
