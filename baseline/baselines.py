#!/usr/bin/env python3
"""Training-free visual-token pruning baselines (one per family in the paper's related work).

* ``random``    -- uniform random subset (seeded per query).
* ``cls_topk``  -- attention-based: top-K tokens by vision-encoder [CLS] attention
                   (FasterVLM-style; the vision-side member of the attention family).
* ``divprune``  -- similarity-based: DivPrune max-min diversity selection (Alvar et al., 2025),
                   seeded with the most isolated token, cosine distance of projected tokens.
* ``cdpruner``  -- joint: CDPruner conditional DPP (Zhang et al., 2025b): kernel
                   L = diag(r) S diag(r) with S the cosine similarity of projected tokens and
                   r the min-max normalised negated CLIP patch-text similarity; greedy MAP
                   inference as in the released code.
* ``pool``      -- spatial average pooling of the 24x24 grid into K merged tokens.
"""

from __future__ import annotations

import zlib

import numpy as np

GRID = 24


def random_subset(qid: str, budget: int, n: int = GRID * GRID) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(qid.encode()) ^ 0x5EED)
    return np.sort(rng.choice(n, size=budget, replace=False))


def cls_topk(cls_attention: np.ndarray, budget: int) -> np.ndarray:
    return np.sort(np.argsort(-cls_attention, kind="stable")[:budget])


def divprune(cosine: np.ndarray, budget: int) -> np.ndarray:
    dist = 1.0 - cosine.astype(np.float32)
    # first pick: token whose nearest neighbour (excluding itself) is farthest
    nn = np.partition(dist, 1, axis=0)[1]
    first = int(np.argmax(nn))
    selected = [first]
    min_dist = dist[first].copy()
    min_dist[first] = -np.inf
    while len(selected) < budget:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, dist[nxt])
        min_dist[selected] = -np.inf
    return np.sort(np.asarray(selected))


def cdpruner_relevance(joint_embeds: np.ndarray, text_embed: np.ndarray) -> np.ndarray:
    j = joint_embeds.astype(np.float32)
    j = j / np.maximum(np.linalg.norm(j, axis=-1, keepdims=True), 1e-6)
    rel = -(j @ text_embed.astype(np.float32))
    return (rel - rel.min() + 1e-6) / max(float(rel.max() - rel.min()), 1e-6)


def cdpruner(cosine: np.ndarray, relevance: np.ndarray, budget: int) -> np.ndarray:
    sim = cosine.astype(np.float64)
    r = relevance.astype(np.float64)
    kernel = r[:, None] * sim * r[None, :]
    n = kernel.shape[0]
    cis = np.zeros((budget, n))
    di2s = np.diag(kernel).copy()
    selected: list[int] = []
    for i in range(budget):
        j = int(np.argmax(di2s))
        selected.append(j)
        eis = (kernel[j] - cis[:i, j] @ cis[:i]) / np.sqrt(max(di2s[j], 1e-12))
        cis[i] = eis
        di2s -= eis ** 2
        di2s[selected] = -np.inf
    return np.sort(np.asarray(selected))


def pool_slots(budget: int) -> list[dict]:
    rows = int(np.floor(np.sqrt(budget)))
    while budget % rows:
        rows -= 1
    cols = budget // rows
    r_edges = np.linspace(0, GRID, rows + 1).round().astype(int)
    c_edges = np.linspace(0, GRID, cols + 1).round().astype(int)
    slots = []
    for a in range(rows):
        for b in range(cols):
            toks = [int(y * GRID + x) for y in range(r_edges[a], r_edges[a + 1])
                    for x in range(c_edges[b], c_edges[b + 1])]
            slots.append({"tokens": toks})
    return slots
