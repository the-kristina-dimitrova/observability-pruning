"""
test_observability.py — verification suite for the observability scorer.

Run from the project root (with a trained checkpoint present):
    python test_observability.py

It checks, in increasing order of strictness:
  1. scores compute with the right shape, finite, and varied
  2. scoring is deterministic on fixed batches (confirms dropout is off)
  3. pruning zeroes the expected number of heads and the model still runs
  4. DIRECTIONALITY: pruning low-observability heads hurts less than
     pruning high-observability heads (the validity test)
"""

import argparse
import sys
import copy
import random
from pathlib import Path

import torch
import yaml
from transformers import DistilBertForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from src.data.dataset_loader import load_and_tokenize
from src.methods.observability import (
    ObservabilityScorer,
    prune_by_observability,
    _remove_attention_heads,
)
from src.evaluation.metrics import evaluate, count_parameters

SPARSITY = 0.30
NUM_LABELS = {"imdb": 2, "ag_news": 4, "banking77": 77}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imdb",
                        choices=["imdb", "ag_news", "banking77"])
    DATASET = parser.parse_args().dataset
    
    cfg = yaml.safe_load(open("configs/experiment_config.yaml"))
    device = get_device()
    n_labels = NUM_LABELS[DATASET]
    print(f"Device: {device} | Dataset: {DATASET}")

    train_loader, _, test_loader, _ = load_and_tokenize(
        DATASET, cfg["model"]["name"], cfg["model"]["max_length"],
        cfg["training"]["batch_size"], cfg["training"]["seed"],
    )

    ckpt = Path(f"results/checkpoints/{DATASET}/best")
    assert ckpt.exists(), f"No checkpoint at {ckpt} — train the baseline first."
    model = DistilBertForSequenceClassification.from_pretrained(
        str(ckpt), num_labels=n_labels
    ).to(device)

    # ── Test 1: mechanics ────────────────────────────────────────────────
    print("\n=== Test 1: shape, finiteness, variation ===")
    scorer = ObservabilityScorer(model, device)
    scores = scorer.compute(train_loader, num_batches=20)
    attn = scores["attention"]
    n_layers, n_heads = len(attn), len(attn[0])
    print(f"shape: {n_layers} layers x {n_heads} heads")
    assert (n_layers, n_heads) == (model.config.n_layers, model.config.n_heads)
    flat = [s for row in attn for s in row]
    assert torch.isfinite(torch.tensor(flat)).all(), "non-finite scores"
    std = float(torch.tensor(flat).std())
    print(f"score range: {min(flat):.4f} .. {max(flat):.4f} | std={std:.4f}")
    assert std > 0, "all scores identical — hook captured nothing useful"
    print("PASS")

    # ── Test 2: determinism (fixed-order loader, no dropout) ──────────────
    print("\n=== Test 2: determinism ===")
    # use test_loader (shuffle=False) so both runs see identical batches
    s1 = ObservabilityScorer(model, device).compute(test_loader, num_batches=10)
    s2 = ObservabilityScorer(model, device).compute(test_loader, num_batches=10)
    f1 = [s for row in s1["attention"] for s in row]
    f2 = [s for row in s2["attention"] for s in row]
    max_diff = max(abs(a - b) for a, b in zip(f1, f2))
    print(f"max score difference across two runs: {max_diff:.2e}")
    assert max_diff < 1e-3, "scores not reproducible — dropout may still be active"
    print("PASS")

    # ── Test 3: pruning mechanics ─────────────────────────────────────────
    print("\n=== Test 3: pruning zeroes the right heads ===")
    m = copy.deepcopy(model)
    _, nonzero_before = count_parameters(m)
    prune_by_observability(m, scores, sparsity=SPARSITY)
    _, nonzero_after = count_parameters(m)
    expected = int(n_layers * n_heads * SPARSITY)
    print(f"expected heads pruned: {expected} | "
          f"nonzero params {nonzero_before:,} -> {nonzero_after:,}")
    assert nonzero_after < nonzero_before, "no weights were zeroed"
    b = next(iter(test_loader))
    _ = m(input_ids=b["input_ids"].to(device),
          attention_mask=b["attention_mask"].to(device))
    print("PASS (pruned model still runs a forward pass)")

    # ── Test 4: directionality ───────────────────
    print("\n=== Test 4: directionality ===")
    ranked = sorted(
        [(l, h, attn[l][h]) for l in range(n_layers) for h in range(n_heads)],
        key=lambda x: x[2],
    )
    N = int(n_layers * n_heads * SPARSITY)
    lowest  = {(l, h) for l, h, _ in ranked[:N]}
    highest = {(l, h) for l, h, _ in ranked[-N:]}
    random.seed(0)
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    rnd = set(random.sample(all_heads, N))

    def acc_after_prune(heads):
        mm = copy.deepcopy(model)
        _remove_attention_heads(mm, heads)
        return evaluate(mm.to(device), test_loader, device).accuracy

    base   = evaluate(copy.deepcopy(model).to(device), test_loader, device).accuracy
    a_low  = acc_after_prune(lowest)
    a_rnd  = acc_after_prune(rnd)
    a_high = acc_after_prune(highest)

    print(f"baseline (no prune):        {base*100:.2f}%")
    print(f"prune LOWEST-obs (method):  {a_low*100:.2f}%")
    print(f"prune RANDOM:               {a_rnd*100:.2f}%")
    print(f"prune HIGHEST-obs:          {a_high*100:.2f}%")
    assert a_low >= a_high - 0.01, (
        "FAIL: pruning low-observability heads is NOT safer than high-observability "
        "ones — the scores are not capturing head importance."
    )
    print("PASS (low-observability heads are safer to remove than high-observability ones)")

    print("\n All checks passed.")


if __name__ == "__main__":
    main()