# Evaluation Targets

## Primary: token_pruning_core

Tests whether a compressed visual-token representation preserves the information needed for query-conditioned reasoning under an aggressive fixed budget.
- **Metric:** `performance_retained` (higher is better), a continuous retention score combining local evidence and global context preservation
- **Run:** `bash scripts/reproduce.sh`, or run a method to produce `checkpoints/*_outputs.jsonl` and then `bash scripts/evaluate.sh`

## Constraint: compression_budget

Tests whether performance is achieved without increasing the allowed number of retained visual tokens.
- **Metrics:** `mean_token_count`, `retained_fraction` (lower is better when performance is comparable)
- **Run:** same as `token_pruning_core`
