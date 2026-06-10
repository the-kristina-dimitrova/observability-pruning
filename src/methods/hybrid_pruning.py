"""
methods/hybrid_pruning.py
--------------------------
NOVEL METHODOLOGY: Observability + Magnitude Hybrid Pruning (Method 2)

This is the primary research contribution. We combine:
    1. Observability scores (system-theoretic, adapted from Albertini & Sontag 1995)
    2. Gradient magnitude scores (gradient-based learning signal)

into a single unified pruning score:

    hybrid_score(h) = α · obs_score(h) + (1 - α) · mag_score(h)

where both scores are normalized to [0, 1] before combination.

The key insight: observability alone may miss heads that are structurally
connected but gradient-saturated. Magnitude alone misses heads that are
simply small but highly informative for specific classes. The hybrid
captures both dimensions simultaneously.

This combination does not appear in the existing literature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import DistilBertForSequenceClassification

from .observability import ObservabilityScorer, _remove_attention_heads

def compute_magnitude_scores(
    model: DistilBertForSequenceClassification,
) -> list[list[float]]:
    """
    Returns [layer][head] -> L2 norm of the concatenated Q+K+V weight rows
    for that head.

    Lower score = smaller magnitude = more prunable by this baseline.
    """
    scores: list[list[float]] = []
    head_dim = model.config.hidden_size // model.config.n_heads

    for layer in model.distilbert.transformer.layer:
        attn = layer.attention
        layer_scores: list[float] = []

        for h in range(model.config.n_heads):
            s = h * head_dim
            e = s + head_dim
            # Concatenate Q, K, V rows for head h
            qkv = torch.cat([
                attn.q_lin.weight[s:e],
                attn.k_lin.weight[s:e],
                attn.v_lin.weight[s:e],
            ], dim=0)
            layer_scores.append(qkv.detach().norm().item())

        scores.append(layer_scores)

    return scores


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]. 0 = most prunable."""
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def prune_hybrid(
    model: DistilBertForSequenceClassification,
    obs_scores: dict,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sparsity: float = 0.42,
    obs_weight: float = 0.5,
    mag_weight: float = 0.5,
) -> DistilBertForSequenceClassification:
    """
    Hybrid Observability + Magnitude Pruning (main novel method).

    Steps:
        1. Collect observability scores per head (already computed, passed in)
        2. Collect magnitude scores per head (L2 norm of weight matrix)
        3. Normalize both to [0, 1]
        4. Compute hybrid_score = obs_weight * obs + mag_weight * mag
        5. Prune heads with lowest hybrid score

    Returns the pruned model (in-place).
    """
    assert abs(obs_weight + mag_weight - 1.0) < 1e-6, \
        "obs_weight + mag_weight must equal 1.0"

    # ── Step 1: Flatten observability scores ────────────────────────────────
    head_obs: list[tuple[int, int, float]] = []
    for layer_idx, heads in enumerate(obs_scores["attention"]):
        for head_idx, score in enumerate(heads):
            head_obs.append((layer_idx, head_idx, score))

    obs_flat = [s for _, _, s in head_obs]

    # ── Step 2: Magnitude scores ─────────────────────────────────────────────
    mag_scores_dict = compute_magnitude_scores(model)
    mag_flat = [
        mag_scores_dict[layer_idx][head_idx]
        for layer_idx, head_idx, _ in head_obs
    ]

    # ── Step 3: Normalize ─────────────────────────────────────────────────────
    obs_norm = _normalize(obs_flat)
    mag_norm = _normalize(mag_flat)

    # ── Step 4: Hybrid score ──────────────────────────────────────────────────
    hybrid = [
        obs_weight * o + mag_weight * m
        for o, m in zip(obs_norm, mag_norm)
    ]

    # ── Step 5: Rank and prune ────────────────────────────────────────────────
    ranked = sorted(
        [(head_obs[i][0], head_obs[i][1], hybrid[i]) for i in range(len(hybrid))],
        key=lambda x: x[2]   # ascending: lowest hybrid score pruned first
    )

    num_to_prune = int(len(ranked) * sparsity)
    heads_to_prune = {(l, h) for l, h, _ in ranked[:num_to_prune]}

    _remove_attention_heads(model, heads_to_prune)

    print(f"[Hybrid Obs+Mag] Pruned {num_to_prune}/{len(ranked)} heads "
          f"(α={obs_weight}, β={mag_weight}, sparsity={sparsity*100:.1f}%)")
    return model


def hybrid_score_analysis(
    model: DistilBertForSequenceClassification,
    obs_scores: dict,
    obs_weight: float = 0.5,
) -> list[dict]:
    """
    Returns a detailed per-head breakdown of scores for analysis/visualization.
    Useful for the analysis notebook.
    """
    mag_weight = 1.0 - obs_weight
    mag_scores_dict = compute_magnitude_scores(model)

    rows = []
    for layer_idx, heads in enumerate(obs_scores["attention"]):
        for head_idx, obs_score in enumerate(heads):
            mag_score = mag_scores_dict[layer_idx][head_idx]
            rows.append({
                "layer": layer_idx,
                "head":  head_idx,
                "obs_score_raw": obs_score,
                "mag_score_raw": mag_score,
            })

    # Normalize
    obs_vals = [r["obs_score_raw"] for r in rows]
    mag_vals = [r["mag_score_raw"] for r in rows]
    obs_norm = _normalize(obs_vals)
    mag_norm = _normalize(mag_vals)

    for i, row in enumerate(rows):
        row["obs_score_norm"] = obs_norm[i]
        row["mag_score_norm"] = mag_norm[i]
        row["hybrid_score"]   = obs_weight * obs_norm[i] + mag_weight * mag_norm[i]

    return sorted(rows, key=lambda r: r["hybrid_score"])
