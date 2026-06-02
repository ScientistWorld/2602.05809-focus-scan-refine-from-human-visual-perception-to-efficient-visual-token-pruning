# Experiment Overview

## token_pruning_64_tokens

Tests the central visual-token compression problem under the tightest fixed budget used by this gym. The experiment measures whether the compressed representation keeps enough local and global information for the downstream query while respecting the token-count constraint. The combined retained-performance score is accompanied by separate local-evidence and global-context retention metrics.

## token_pruning_128_tokens

Repeats the same evaluation at an intermediate retained-token budget. This checks whether a method remains useful when compression is less severe, rather than only overfitting the tightest case.

## token_pruning_192_tokens

Repeats the same evaluation at the largest retained-token budget in this scaled benchmark. This setting verifies that the method preserves the budget constraint while approaching dense-token information retention.
