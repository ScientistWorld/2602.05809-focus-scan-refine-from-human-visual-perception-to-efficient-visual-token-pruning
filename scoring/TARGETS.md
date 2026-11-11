# Evaluation Targets

All targets come from one experiment, `token_pruning_core`. They are scored on the hidden
split by `bash scripts/evaluate_test.sh`; `bash scripts/evaluate_train.sh` gives the same
numbers on the visible split.

## Primary: retained performance at 64 tokens

- **Metric:** `performance_retained` (higher is better; coefficient 1.0). This is the mean
  over the question families of accuracy with 64 kept/merged tokens divided by full-token
  accuracy.
- **Reference (FSR):** 0.917 hidden / 0.912 visible. Paper (nine benchmarks): 0.961.
- **Baselines (hidden):** CDPruner 0.910, DivPrune 0.890, random 0.831, [CLS]-attention
  top-K 0.795.

## Aggressive budget: retained performance at 32 tokens

- **Metric:** `performance_retained_32` (higher is better; coefficient 0.5).
- **Reference (FSR):** 0.838 hidden. CDPruner 0.866, DivPrune 0.822, random 0.719. This is
  the headroom target: at this budget the reference is beaten by a relevance-conditioned
  diversity rule.

## Balance: local evidence vs global context (64 tokens)

- **Metrics:** `local_evidence_retained` (existence, counting) and
  `global_context_retained` (caption, absent object). Higher is better; coefficient 0.25
  each.
- **Reference (FSR):** 0.943 / 0.891 hidden.

## Constraint: token budget and pruning cost

- At most K slots per question (K = 64 and 32). Each slot is a non-negative weighting of
  the image's own visual tokens.
- Pruning runs on CPU within 60 s + 0.1 s per question, with no access to the language
  model.
- A violation of any of these fails the run, which then gets 0 in every metric. There is no
  separate cost metric.
