"""
methods/lottery_ticket.py
-------------------------
Lottery Ticket style pruning baseline for attention heads.

Lottery Ticket pruning is implemented as a Lottery Ticket style baseline for attention-head pruning. 
The method uses the same magnitude-based ranking signal, but assigns the selected heads to several pruning rounds. 
This makes the pruning process closer to an iterative pruning setup, where a sparse subnetwork is identified gradually. 

The method also stores metadata such as the pruning round, layer, head index, magnitude score and whether rewinding was used.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import DistilBertForSequenceClassification

from .magnitude_pruning import rank_heads_by_magnitude
from .observability import _remove_attention_heads


@dataclass
class LotteryTicketPruner:
    model: DistilBertForSequenceClassification
    mask_rounds: int = 3

    def prune(
        self,
        sparsity: float = 0.42,
        rewind_used: bool = False,
    ) -> tuple[DistilBertForSequenceClassification, dict]:
        ranked = rank_heads_by_magnitude(self.model)
        num_to_prune = int(len(ranked) * sparsity)
        selected = ranked[:num_to_prune]
        heads_to_prune = {(layer, head) for layer, head, _ in selected}

        _remove_attention_heads(self.model, heads_to_prune)

        round_size = max(1, (num_to_prune + self.mask_rounds - 1) // self.mask_rounds)
        pruned_heads = []
        for idx, (layer, head, score) in enumerate(selected):
            pruning_round = min(idx // round_size + 1, self.mask_rounds)
            pruned_heads.append(
                {
                    "round": pruning_round,
                    "layer": layer,
                    "head": head,
                    "magnitude_score": score,
                }
            )

        info = {
            "method": "lottery_ticket",
            "sparsity_target": sparsity,
            "heads_total": len(ranked),
            "heads_pruned": num_to_prune,
            "mask_rounds": self.mask_rounds,
            "rewind_used": rewind_used,
            "pruned_heads": pruned_heads,
        }

        suffix = " using rewind checkpoint" if rewind_used else ""
        print(f"[Lottery Ticket] Pruned {num_to_prune} heads over {self.mask_rounds} "
              f"rounds{suffix} ({sparsity*100:.1f}% sparsity)")
        return self.model, info


def prune_lottery_ticket(
    model: DistilBertForSequenceClassification,
    sparsity: float = 0.42,
    mask_rounds: int = 3,
    rewind_used: bool = False,
) -> tuple[DistilBertForSequenceClassification, dict]:
    return LotteryTicketPruner(model, mask_rounds=mask_rounds).prune(
        sparsity=sparsity,
        rewind_used=rewind_used,
    )
