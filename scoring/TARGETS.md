# Evaluation Targets

## Primary: token_pruning_64_tokens

Tests whether a compressed visual-token representation preserves the information needed for query-conditioned reasoning under the most aggressive fixed budget.
- **Metric:** `performance_retained` (higher is better), a continuous retention score combining local evidence and global context preservation
- **Component metrics:** `evidence_retained` and `context_retained` (higher is better), reported separately so methods cannot improve by keeping only salient query evidence or only diverse context
- **Run:** `bash scripts/reproduce.sh`, or run a method to produce `checkpoints/*_outputs.jsonl` and then `bash scripts/evaluate.sh`

## Additional Budgets

`token_pruning_128_tokens` and `token_pruning_192_tokens` repeat the same evaluation at larger retained-token budgets. These settings test whether improvements persist as compression is relaxed.

## Constraint: compression_budget

Tests whether performance is achieved without increasing the allowed number of retained visual tokens.
- **Metrics:** `mean_token_count`, `retained_fraction` (lower is better when performance is comparable)
- **Run:** same as the token-pruning experiments
