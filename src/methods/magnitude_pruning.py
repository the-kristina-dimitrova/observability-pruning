"""
methods/magnitude_pruning.py
----------------------------
Magnitude-based attention-head pruning baseline for DistilBERT.

Magnitude pruning is used as a simple baseline method for attention-head pruning. 
For each attention head, the method computes the L2 norm of the corresponding Q, K and V projection weights. 
Heads with smaller weight magnitude are treated as less important and are pruned first. 

The selected heads are then physically removed from the DistilBERT model,
and metadata about the removed heads is saved for later comparison with the other pruning methods.
"""

from __future__ import annotations

import torch
from transformers import DistilBertForSequenceClassification

from .observability import _remove_attention_heads


def compute_magnitude_scores(
    model: DistilBertForSequenceClassification,
) -> list[list[float]]:
    """Return [layer][head] -> L2 norm of Q+K+V rows for that head."""
    scores: list[list[float]] = []

    for layer in model.distilbert.transformer.layer:
        attn = layer.attention
        num_heads = attn.n_heads
        head_dim = attn.dim // num_heads
        layer_scores: list[float] = []

        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            qkv = torch.cat(
                [
                    attn.q_lin.weight[start:end],
                    attn.k_lin.weight[start:end],
                    attn.v_lin.weight[start:end],
                ],
                dim=0,
            )
            layer_scores.append(float(qkv.detach().norm().item()))

        scores.append(layer_scores)

    return scores


def rank_heads_by_magnitude(
    model: DistilBertForSequenceClassification,
) -> list[tuple[int, int, float]]:
    """Return heads sorted from lowest to highest magnitude score."""
    scores = compute_magnitude_scores(model)
    ranked: list[tuple[int, int, float]] = []

    for layer_idx, layer_scores in enumerate(scores):
        for head_idx, score in enumerate(layer_scores):
            ranked.append((layer_idx, head_idx, score))

    return sorted(ranked, key=lambda x: x[2])


def prune_by_magnitude(
    model: DistilBertForSequenceClassification,
    sparsity: float = 0.42,
) -> tuple[DistilBertForSequenceClassification, dict]:
    """
    Prune the lowest-magnitude attention heads.

    Returns the modified model and a metadata dictionary saved in result tables.
    """
    ranked = rank_heads_by_magnitude(model)
    num_to_prune = int(len(ranked) * sparsity)
    selected = ranked[:num_to_prune]
    heads_to_prune = {(layer, head) for layer, head, _ in selected}

    _remove_attention_heads(model, heads_to_prune)

    info = {
        "method": "magnitude",
        "sparsity_target": sparsity,
        "heads_total": len(ranked),
        "heads_pruned": num_to_prune,
        "pruned_heads": [
            {"layer": layer, "head": head, "magnitude_score": score}
            for layer, head, score in selected
        ],
    }

    print(f"[Magnitude] Pruned {num_to_prune}/{len(ranked)} heads "
          f"({sparsity*100:.1f}% sparsity)")
    return model, info
