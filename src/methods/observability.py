"""
methods/observability.py
------------------------
Adapts Albertini & Sontag (1995) system-theoretic observability to
Transformer (DistilBERT) attention heads.

THEORETICAL BACKGROUND
=======================
In RNN minimality theory (Albertini & Sontag 1995, §3), a state x is
*unobservable* if the output map y(t) = h(x(t)) is identical for all
inputs regardless of x. Formally, the observability Gramian

    W_o = ∫ Φ(t,0)^T C^T C Φ(t,0) dt

has a non-trivial null space for unobservable states.

ADAPTATION TO TRANSFORMERS
===========================
For a Transformer layer we do not have a clean state-space formulation,
but we construct an *observability proxy* via output sensitivity:

    obs_score(h) = E_x [ ‖ ∂ L / ∂ z_h ‖₂ ]

where z_h is the output of attention head h (its slice of the context
fed into out_lin) and L is the task loss. This measures how much the
final output "observes" head h — analogous to the diagonal of the
observability Gramian. Heads with obs_score ≈ 0 are deemed unobservable
(prunable) in the system-theoretic sense.

NOTE: FFN-neuron scoring was removed — no pruner in this project uses it,
and the previous implementation collapsed the ffn_dim axis (giving a
per-layer scalar, not a per-neuron score). To add per-neuron FFN pruning
later, hook layer.ffn.lin1 and reduce the gradient over (batch, seq) only,
keeping the ffn_dim axis: grad.detach().abs().mean(dim=(0, 1)).

Reference:
    Albertini, F. & Sontag, E. D. (1995).
    Identifiability of Discrete-Time Neural Networks.
    https://www.math.unipd.it/~albertin/19.pdf
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Observability scorer
# ─────────────────────────────────────────────────────────────────────────────

class ObservabilityScorer:
    """
    Computes observability proxy scores for all attention heads in a
    DistilBERT model.

    Usage
    -----
    scorer = ObservabilityScorer(model, device)
    scores = scorer.compute(dataloader, num_batches=50)
    # scores["attention"][layer][head] -> float (lower = more prunable)
    """

    def __init__(
        self,
        model: DistilBertForSequenceClassification,
        device: torch.device,
    ):
        self.model = model
        self.device = device

        cfg = model.config
        self.num_layers = cfg.n_layers        # 6 for DistilBERT
        self.num_heads  = cfg.n_heads         # 12
        self.hidden     = cfg.hidden_size     # 768
        self.head_dim   = self.hidden // self.num_heads   # 64

    # ── Public API ──────────────────────────────────────────────────────────

    def compute(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 50,
        criterion: nn.Module | None = None,
    ) -> dict:
        """
        Forward + backward over `num_batches` mini-batches.
        Returns {"attention": [layer][head] -> score}.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.eval()
        self.model.zero_grad()

        attn_accum: list[list[list[float]]] = [
            [[] for _ in range(self.num_heads)] for _ in range(self.num_layers)
        ]

        hooks: list = []
        for layer_idx, layer in enumerate(self.model.distilbert.transformer.layer):
            hooks.append(self._register_attn_hook(layer, layer_idx, attn_accum))

        # ── Forward/backward loop ─────────────────────────────────────────
        total = min(num_batches, len(dataloader))
        for batch_idx, batch in enumerate(
            tqdm(dataloader, total=total, desc="Observability scoring")
        ):
            if batch_idx >= num_batches:
                break

            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            self.model.zero_grad()

        for h in hooks:
            h.remove()

        attn_scores = self._aggregate_attn(attn_accum)

        # ── Sanity guard: catch silently-broken hooks ─────────────────────
        flat = [s for row in attn_scores for s in row]
        assert any(s > 0 for s in flat), \
            "All observability scores are zero — hooks never fired."

        return {"attention": attn_scores}

    # ── Private helpers ─────────────────────────────────────────────────────

    def _register_attn_hook(self, layer, layer_idx: int, accum):
        """
        Hooks the input to out_lin (per-head concatenated context). Records the
        Taylor / Michel-et-al. head-importance proxy:
            I_h = E_x | < z_h , dL/dz_h > |
        the first-order estimate of the loss increase from removing head h.
        Higher = more important (keep); lower = prunable.
        """
        head_dim, num_heads = self.head_dim, self.num_heads

        def pre_hook(module, args):
            context = args[0]
            if context.requires_grad:
                saved = context.detach()
                def bwd(grad):
                    prod = grad.detach() * saved
                    bs, seq, hidden = prod.shape
                    prod = prod.reshape(bs, seq, num_heads, head_dim)
                    per_head = prod.sum(dim=(1, 3)).abs().mean(dim=0).cpu().tolist()
                    for h in range(num_heads):
                        accum[layer_idx][h].append(per_head[h])
                context.register_hook(bwd)

        return layer.attention.out_lin.register_forward_pre_hook(pre_hook)

    def _aggregate_attn(self, accum: list[list[list[float]]]) -> list[list[float]]:
        """Returns [layer][head] -> mean observability score."""
        result = []
        for layer_scores in accum:
            row = []
            for head_scores in layer_scores:
                row.append(float(torch.tensor(head_scores).mean()) if head_scores else 0.0)
            result.append(row)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Pure observability pruner
# ─────────────────────────────────────────────────────────────────────────────

def prune_by_observability(
    model: DistilBertForSequenceClassification,
    obs_scores: dict,
    sparsity: float = 0.42,
) -> DistilBertForSequenceClassification:
    """
    Zeroes out attention head weights whose observability score falls in
    the bottom `sparsity` fraction. Returns the (in-place modified) model.

    This is the "pure observability" method (Method 1).
    """
    head_scores: list[tuple[int, int, float]] = []
    for layer_idx, heads in enumerate(obs_scores["attention"]):
        for head_idx, score in enumerate(heads):
            head_scores.append((layer_idx, head_idx, score))

    head_scores.sort(key=lambda x: x[2])
    num_to_prune = int(len(head_scores) * sparsity)
    heads_to_prune: set[tuple[int, int]] = {
        (l, h) for l, h, _ in head_scores[:num_to_prune]
    }

    _remove_attention_heads(model, heads_to_prune)

    print(f"[Observability] Pruned {num_to_prune}/{len(head_scores)} heads "
          f"({sparsity*100:.1f}% sparsity)")
    return model


def _prune_linear(layer: nn.Linear, keep: torch.Tensor, dim: int) -> nn.Linear:
    """New Linear keeping only `keep` indices along `dim` (0 = outputs, 1 = inputs)."""
    W = layer.weight.index_select(dim, keep).detach().clone()
    if layer.bias is not None:
        b = layer.bias.index_select(0, keep).detach().clone() if dim == 0 else layer.bias.detach().clone()
    else:
        b = None
    new = nn.Linear(W.size(1), W.size(0), bias=b is not None)
    new.weight = nn.Parameter(W)
    if b is not None:
        new.bias = nn.Parameter(b)
    return new.to(layer.weight.device)


def _remove_attention_heads(model, heads_to_prune):
    """Physically remove the selected heads by shrinking the attention matrices."""
    by_layer: dict[int, list[int]] = {}
    for (l, h) in heads_to_prune:
        by_layer.setdefault(l, []).append(h)

    for layer_idx, heads in by_layer.items():
        attn = model.distilbert.transformer.layer[layer_idx].attention
        head_dim = attn.dim // attn.n_heads                      # 64
        remove = set(heads)
        keep_dims = [d for h in range(attn.n_heads) if h not in remove
                       for d in range(h * head_dim, (h + 1) * head_dim)]
        keep = torch.tensor(keep_dims, dtype=torch.long, device=attn.q_lin.weight.device)

        attn.q_lin   = _prune_linear(attn.q_lin,   keep, dim=0)   # drop output rows
        attn.k_lin   = _prune_linear(attn.k_lin,   keep, dim=0)
        attn.v_lin   = _prune_linear(attn.v_lin,   keep, dim=0)
        attn.out_lin = _prune_linear(attn.out_lin, keep, dim=1)   # drop input cols

        attn.n_heads = attn.n_heads - len(remove)
        attn.dim     = head_dim * attn.n_heads