"""
evaluation/metrics.py
----------------------
Evaluation utilities: accuracy, macro-F1, compression ratio, parameter count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertForSequenceClassification


@dataclass
class EvalResult:
    method: str
    dataset: str
    sparsity: float
    accuracy: float
    f1_macro: float
    params_total: int
    params_nonzero: int
    compression_ratio: float
    inference_ms: float


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (total_params, nonzero_params)."""
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    return total, nonzero


@torch.no_grad()
def evaluate(
    model: DistilBertForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
    method: str = "unknown",
    dataset: str = "unknown",
    sparsity: float = 0.0,
    baseline_params: int | None = None,
) -> EvalResult:
    """
    Run inference on the dataloader and return an EvalResult.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_time = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]

        t0 = time.perf_counter()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()

        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        total_time += (t1 - t0) * 1000   # ms
        num_batches += 1

    acc    = accuracy_score(all_labels, all_preds)
    f1_mac = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    total_p, nonzero_p = count_parameters(model)
    denom = baseline_params if baseline_params else total_p
    compression = nonzero_p / denom

    return EvalResult(
        method=method,
        dataset=dataset,
        sparsity=sparsity,
        accuracy=acc,
        f1_macro=f1_mac,
        params_total=total_p,
        params_nonzero=nonzero_p,
        compression_ratio=compression,
        inference_ms=total_time / max(num_batches, 1),
    )


def print_result(result: EvalResult) -> None:
    print(
        f"[{result.method:<25}] {result.dataset:<12} | "
        f"Acc: {result.accuracy*100:.2f}% | "
        f"F1: {result.f1_macro*100:.2f}% | "
        f"Compression: {(1-result.compression_ratio)*100:.1f}% | "
        f"Inf: {result.inference_ms:.1f}ms/batch"
    )
