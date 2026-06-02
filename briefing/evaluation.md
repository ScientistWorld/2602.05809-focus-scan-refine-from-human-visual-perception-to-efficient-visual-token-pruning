# Evaluation

## Metrics

The primary metric is `performance_retained`: a continuous 0-1 score measuring how well the compressed visual-token representation preserves both query-relevant local evidence and required global context. Higher is better.

Constraint metrics measure the compression setting: `mean_token_count` and `retained_fraction`. Lower values are better when performance is held fixed.

## Reference Setting

The source paper reports the core LLaVA-1.5-7B aggressive-compression setting at 64 retained tokens from 576 original visual tokens. At this setting, the unpruned model is the 100% reference and the best reported compressed systems retain roughly 96% average performance.

## Evaluation Protocol

Run each pruning method on the same token benchmark at each declared fixed token budget. The method writes compressed-token artifacts under `checkpoints/budget16`, `checkpoints/budget32`, and `checkpoints/budget48`; evaluation reads those artifacts, computes retained local-evidence and global-context similarity, and never imports method code. Scores are written to `scoring/scores.json`. The reproduced benchmark is scaled to 16, 32, and 48 retained tokens out of 128 synthetic encoder-like tokens; the paper reference describes the analogous LLaVA-1.5-7B 64-, 128-, and 192-token settings.
