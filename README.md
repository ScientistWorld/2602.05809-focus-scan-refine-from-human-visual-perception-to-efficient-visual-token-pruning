# Focus-Scan-Refine: From Human Visual Perception to Efficient Visual Token Pruning

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening).

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2602.05809"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2602.05809-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2602.05809](https://arxiv.org/abs/2602.05809) |
|:---|:---|
| **Domain** | Vision-Language Models / Efficient Inference |
| **TL;DR** | The paper proposes a training-free visual-token pruning algorithm that balances query-focused local evidence with global context under strict token budgets. |

## Summary

FSR compresses VLM visual tokens at inference time without changing model weights. It first selects locally important query-relevant tokens, then adds complementary diverse context tokens, then refines context anchors by merging nearby discarded details. The paper reports improved accuracy-efficiency trade-offs across LLaVA, LLaVA-NeXT, Qwen2.5-VL, and video VLM benchmarks.

## Problem

How can a vision-language system reduce the number of visual tokens passed to the language model at inference time while preserving enough image evidence to answer the user query?

## The paper's method

Focus-Scan-Refine (FSR) is a training-free visual token pruning framework for VLM inference. It dynamically allocates a fixed token budget between query-critical local evidence and complementary global context.

## Key results of the paper

- On LLaVA-1.5-7B with only 64 of 576 tokens retained, FSR preserves 96.1% average performance.
- In the same 64-token setting, FSR reports 2.291T FLOPs versus 9.042T for the unpruned model.
- On LLaVA-NeXT-13B, pruning can slightly outperform the dense token set in some settings, suggesting token redundancy can add noise.

## Run it

```bash
pip install sciworld
sw research 2602.05809 --mode innovate --agent claude-code
sw evaluate 2602.05809
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

