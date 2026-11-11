#!/usr/bin/env python3
"""Focus-Scan-Refine visual-token pruning (Tong et al., 2026, arXiv:2602.05809).

Operates on one image/query pair of LLaVA-1.5 visual tokens:

* Focus  (Sec. 3.2, Eq. 1-4): saliency s_i = head-mean [CLS] attention of the
  vision encoder; instruction relevance r_i = cosine between the token's CLIP
  joint-space embedding and the CLIP text embedding of the instruction,
  computed as in Zhang et al. (2025b, CDPruner), which the paper cites for
  this score; both min-max normalised; priority phi = r^alpha * s^beta; keep
  the smallest top-phi prefix holding a fraction rho of the information mass.

  Mass reading. Eq. 4 writes the mass over all N tokens. On LLaVA-1.5's
  CLIP-L/336 features the [CLS] attention is spread out (about 250 of 576
  tokens hold 90% of it), so rho=0.9 of the all-token mass needs more tokens
  than any aggressive budget: K_F is clamped to K for every question and Scan
  and Refine never run, contradicting the paper's dynamic split (Fig. 1) and
  its Scan/Refine ablations. The default ``focus_mass="budget"`` therefore
  takes the mass over the K highest-priority candidates (the mass the budget
  can hold); ``focus_mass="all"`` is the literal all-token reading.
* Scan   (Sec. 3.3, Eq. 5): conditional farthest-point sampling in cosine
  distance, initialised with the Focus set, for the remaining K - K_F slots.
* Refine (Sec. 3.4, Eq. 8-9): each discarded token is assigned to its most
  similar Scan anchor; the M = kappa * |S| discarded tokens most similar to
  their anchors are merged into them with priority-weighted averaging
  (weights initialised to phi). Focus tokens are kept intact.

Token similarity is cosine similarity of the projected visual tokens (the
features the language model consumes), as in the CDPruner code base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class FSRResult:
    focus: list[int]
    scan: list[int]
    merged: dict[int, list[int]]      # scan anchor -> discarded tokens merged into it
    priority: np.ndarray
    slots: list[dict]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < EPS:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def instruction_relevance(joint_embeds: np.ndarray, text_embed: np.ndarray, sign: str = "neg") -> np.ndarray:
    """Normalised relevance r_hat in [0, 1] (Eq. 2 + normalisation).

    ``sign="neg"`` follows the released CDPruner computation that the paper
    cites for this score (relevance = -cos, then min-max with +1e-6), which
    accounts for CLIP's inverted patch-text similarity on LLaVA-1.5's
    penultimate-layer features. ``sign="pos"`` uses +cos literally.
    """
    j = joint_embeds.astype(np.float32)
    j = j / np.maximum(np.linalg.norm(j, axis=-1, keepdims=True), 1e-6)
    rel = j @ text_embed.astype(np.float32)
    if sign == "neg":
        rel = -rel
    return (rel - rel.min() + 1e-6) / max(float(rel.max() - rel.min()), 1e-6)


def focus_scan_refine(
    cosine: np.ndarray,
    relevance: np.ndarray,
    saliency: np.ndarray,
    budget: int,
    alpha: float = 3.0,
    beta: float = 1.0,
    rho: float = 0.9,
    kappa: float = 1.0,
    scan: bool = True,
    refine_weights: str = "priority",
    scan_mode: str = "fps",
    focus_mass: str = "budget",
) -> FSRResult:
    n = cosine.shape[0]
    if not 1 <= budget <= n:
        raise ValueError(f"budget must be in [1, {n}]")
    cosine = np.asarray(cosine, dtype=np.float32)
    r_hat = np.clip(np.asarray(relevance, dtype=np.float64), 0.0, None)
    s_hat = minmax(np.asarray(saliency, dtype=np.float64))
    phi = np.power(r_hat, alpha) * np.power(s_hat, beta)                     # Eq. 3

    # Focus: minimum prefix of the phi-sorted tokens holding rho of the mass (Eq. 4)
    order = np.argsort(-phi, kind="stable")
    total = float(phi.sum()) if focus_mass == "all" else float(phi[order[:budget]].sum())
    if scan and total > EPS:
        cum = np.cumsum(phi[order])
        k_f = int(np.searchsorted(cum, rho * total, side="left")) + 1
    elif scan:
        k_f = 1
    else:
        k_f = budget
    k_f = max(1, min(k_f, budget))
    focus = [int(i) for i in order[:k_f]]

    # Scan: conditional farthest-point sampling seeded with the Focus set (Eq. 5)
    selected = np.zeros(n, dtype=bool)
    selected[focus] = True
    scan_set: list[int] = []
    if k_f < budget and scan_mode == "dpp":
        # variant: relevance-weighted conditional DPP, greedy MAP seeded with the Focus set
        kernel = r_hat[:, None] * cosine.astype(np.float64) * r_hat[None, :]
        cis = np.zeros((budget, n))
        di2s = np.diag(kernel).copy()
        chosen: list[int] = []
        for i in range(budget):
            j = focus[i] if i < k_f else int(np.argmax(di2s))
            chosen.append(j)
            eis = (kernel[j] - cis[:i, j] @ cis[:i]) / np.sqrt(max(di2s[j], 1e-12))
            cis[i] = eis
            di2s -= eis ** 2
            di2s[chosen] = -np.inf
        scan_set = [int(j) for j in chosen[k_f:]]
        selected[scan_set] = True
    elif k_f < budget:
        min_dist = np.min(1.0 - cosine[:, focus], axis=1)
        min_dist[selected] = -np.inf
        for _ in range(budget - k_f):
            nxt = int(np.argmax(min_dist))
            scan_set.append(nxt)
            selected[nxt] = True
            min_dist = np.minimum(min_dist, 1.0 - cosine[:, nxt])
            min_dist[selected] = -np.inf

    # Refine: merge the top-M most anchor-similar discarded tokens (Eq. 8-9)
    merged: dict[int, list[int]] = {a: [] for a in scan_set}
    discarded = np.flatnonzero(~selected)
    m = min(len(discarded), int(round(kappa * len(scan_set))))
    if scan_set and m > 0:
        sims = cosine[np.ix_(discarded, scan_set)]
        nearest = np.argmax(sims, axis=1)
        nearest_sim = sims[np.arange(len(discarded)), nearest]
        top = np.argsort(-nearest_sim, kind="stable")[:m]
        for t in top:
            merged[scan_set[int(nearest[t])]].append(int(discarded[t]))

    slots: list[tuple[int, dict]] = []
    for f in focus:
        slots.append((f, {"tokens": [f]}))
    for a in scan_set:
        toks = [a] + merged[a]
        if refine_weights == "priority":
            w = [float(phi[t]) + EPS for t in toks]                         # w_j initialised to phi_j
        else:
            w = [1.0] * len(toks)
        slots.append((a, {"tokens": toks, "weights": w} if len(toks) > 1 else {"tokens": toks}))
    slots.sort(key=lambda kv: kv[0])                                         # keep raster order
    return FSRResult(focus=focus, scan=scan_set, merged=merged, priority=phi, slots=[s for _, s in slots])
