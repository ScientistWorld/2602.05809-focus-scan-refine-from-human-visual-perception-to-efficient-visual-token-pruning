# Evaluation

## Metrics

The primary metric is `performance_retained`: a continuous 0-1 score measuring how well the compressed visual-token representation preserves both query-relevant local evidence and required global context. Higher is better.

Constraint metrics measure the compression setting: `mean_token_count` and `retained_fraction`. Lower values are better when performance is held fixed.

## Baselines and Targets

The paper reports the core LLaVA-1.5-7B aggressive-compression setting at 64 retained tokens from 576 original visual tokens. At this setting, the unpruned model is the 100% reference, FastV retains 72.0% average performance, CDPruner retains 95.7%, and FSR retains 96.1%.

## Evaluation Protocol

Run each pruning method on the same token benchmark with the same fixed token budget. The method writes compressed-token artifacts under `checkpoints/`; evaluation reads those artifacts, computes retained local-evidence and global-context similarity, and never imports method code. Scores are written to `scoring/scores.json`. The reproduced benchmark is scaled to 32 retained tokens out of 128 synthetic encoder-like tokens; the paper reference describes the analogous LLaVA-1.5-7B 64-token setting.
