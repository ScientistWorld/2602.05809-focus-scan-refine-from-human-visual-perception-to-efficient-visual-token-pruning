#!/usr/bin/env python3
"""Focus-Scan-Refine token pruning from Tong et al. (2026)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FSRResult:
    selected_indices: list[int]
    focus_indices: list[int]
    scan_indices: list[int]
    compressed_tokens: np.ndarray
    priority: np.ndarray


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def focus_scan_refine(
    tokens: np.ndarray,
    query_embedding: np.ndarray,
    cls_attention: np.ndarray,
    budget: int,
    alpha: float = 3.0,
    beta: float = 1.0,
    rho: float = 0.9,
    kappa: float = 1.0,
) -> FSRResult:
    """Run the paper's three-stage FSR algorithm on one token sequence.

    Args follow the paper: alpha/beta fuse instruction relevance and visual
    saliency, rho determines the dynamic Focus budget, and kappa controls how
    many discarded tokens are merged into Scan anchors during Refine.
    """

    tokens = np.asarray(tokens, dtype=np.float64)
    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    cls_attention = np.asarray(cls_attention, dtype=np.float64)
    n_tokens = tokens.shape[0]
    if not 1 <= budget <= n_tokens:
        raise ValueError(f"budget must be in [1, {n_tokens}], got {budget}")

    norm_tokens = _l2_normalize(tokens)
    norm_query = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

    relevance = norm_tokens @ norm_query
    saliency = cls_attention
    priority = np.power(_minmax(relevance) + 1e-12, alpha) * np.power(
        _minmax(saliency) + 1e-12, beta
    )

    order = np.argsort(-priority, kind="mergesort")
    total_mass = float(np.sum(priority))
    if total_mass <= 1e-12:
        focus_count = min(1, budget)
    else:
        cumulative = np.cumsum(priority[order])
        focus_count = int(np.searchsorted(cumulative, rho * total_mass, side="left") + 1)
        focus_count = max(1, min(focus_count, budget))

    focus = [int(i) for i in order[:focus_count]]
    selected = list(focus)
    scan: list[int] = []

    if len(selected) < budget:
        selected_mask = np.zeros(n_tokens, dtype=bool)
        selected_mask[selected] = True
        sim_to_selected = norm_tokens @ norm_tokens[selected].T
        min_distance = np.min(1.0 - sim_to_selected, axis=1)
        min_distance[selected_mask] = -np.inf

        while len(selected) < budget:
            nxt = int(np.argmax(min_distance))
            selected.append(nxt)
            scan.append(nxt)
            selected_mask[nxt] = True
            new_distance = 1.0 - (norm_tokens @ norm_tokens[nxt])
            min_distance = np.minimum(min_distance, new_distance)
            min_distance[selected_mask] = -np.inf

    compressed = tokens[selected].copy()

    # Refine only the scan anchors. Focus tokens remain untouched.
    if scan:
        selected_set = set(selected)
        discarded = [i for i in range(n_tokens) if i not in selected_set]
        if discarded:
            scan_pos = {idx: pos for pos, idx in enumerate(selected)}
            scan_norm = norm_tokens[scan]
            discarded_norm = norm_tokens[discarded]
            sims = discarded_norm @ scan_norm.T
            nearest = np.argmax(sims, axis=1)
            nearest_sim = sims[np.arange(len(discarded)), nearest]
            merge_count = min(len(discarded), int(round(kappa * len(scan))))
            if merge_count > 0:
                top_local = np.argsort(-nearest_sim, kind="mergesort")[:merge_count]
                weights = {idx: float(priority[idx]) + 1e-12 for idx in scan}
                for local_idx in top_local:
                    discarded_idx = int(discarded[local_idx])
                    anchor_idx = int(scan[nearest[local_idx]])
                    pos = scan_pos[anchor_idx]
                    anchor_w = weights[anchor_idx]
                    token_w = float(priority[discarded_idx]) + 1e-12
                    compressed[pos] = (
                        anchor_w * compressed[pos] + token_w * tokens[discarded_idx]
                    ) / (anchor_w + token_w)
                    weights[anchor_idx] = anchor_w + token_w

    return FSRResult(
        selected_indices=selected,
        focus_indices=focus,
        scan_indices=scan,
        compressed_tokens=compressed,
        priority=priority,
    )
