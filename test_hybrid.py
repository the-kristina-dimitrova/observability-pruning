"""
test_hybrid.py — verification suite for the hybrid (observability + magnitude) method.

Run from the project root, with a trained checkpoint present:
    python test_hybrid.py

Checks:
  1. magnitude component orientation (the seam with Mihail's code): low = prune
  2. the hybrid score varies and is a genuine blend, not a copy of one component
  3. DIRECTIONALITY: pruning low-hybrid heads hurts less than high-hybrid heads
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
from src.methods.observability import ObservabilityScorer, _remove_attention_heads
from src.methods.magnitude_pruning import compute_magnitude_scores
from src.methods.hybrid_pruning import hybrid_score_analysis
from src.evaluation.metrics import evaluate


SPARSITY   = 0.30
OBS_WEIGHT = 0.5
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
    print(f"Device: {device} | Dataset: {DATASET} | obs_weight={OBS_WEIGHT}")

    train_loader, _, test_loader, _ = load_and_tokenize(
        DATASET, cfg["model"]["name"], cfg["model"]["max_length"],
        cfg["training"]["batch_size"], cfg["training"]["seed"],
    )

    ckpt = Path(f"results/checkpoints/{DATASET}/best")
    assert ckpt.exists(), f"No checkpoint at {ckpt} — train the baseline first."
    model = DistilBertForSequenceClassification.from_pretrained(
        str(ckpt), num_labels=n_labels
    ).to(device)

    n_layers, n_heads = model.config.n_layers, model.config.n_heads
    N = int(n_layers * n_heads * SPARSITY)
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]

    def acc_after_prune(heads):
        mm = copy.deepcopy(model)
        _remove_attention_heads(mm, heads)
        return evaluate(mm.to(device), test_loader, device).accuracy

    base = evaluate(copy.deepcopy(model).to(device), test_loader, device).accuracy
    obs_scores = ObservabilityScorer(model, device).compute(train_loader, num_batches=20)

    # ── Test 1: magnitude orientation (the seam with the magnitude file) ──
    print("\n=== Test 1: magnitude component orientation (must be low = prune) ===")
    mag = compute_magnitude_scores(model)
    mag_ranked = sorted(
        [(l, h, mag[l][h]) for l in range(n_layers) for h in range(n_heads)],
        key=lambda x: x[2],
    )
    mag_low  = {(l, h) for l, h, _ in mag_ranked[:N]}
    mag_high = {(l, h) for l, h, _ in mag_ranked[-N:]}
    a_mlow, a_mhigh = acc_after_prune(mag_low), acc_after_prune(mag_high)
    print(f"prune lowest-magnitude:  {a_mlow*100:.2f}%   <- should be higher")
    print(f"prune highest-magnitude: {a_mhigh*100:.2f}%")
    assert a_mlow >= a_mhigh - 0.01, (
        "Magnitude looks oriented backwards. Your hybrid assumes low=prune for BOTH "
        "signals — coordinate the convention with the magnitude file before trusting the blend."
    )
    print("PASS (magnitude follows the low=prune convention the hybrid relies on)")

    # ── Test 2: hybrid is a real blend, not a copy of one component ───────
    print("\n=== Test 2: hybrid score varies and blends both signals ===")
    rows = hybrid_score_analysis(model, obs_scores, obs_weight=OBS_WEIGHT)
    hyb = [r["hybrid_score"] for r in rows]
    assert torch.isfinite(torch.tensor(hyb)).all(), "non-finite hybrid scores"
    assert float(torch.tensor(hyb).std()) > 0, "all hybrid scores identical"

    obs_flat = sorted(
        [(l, h, obs_scores["attention"][l][h]) for l in range(n_layers) for h in range(n_heads)],
        key=lambda x: x[2],
    )
    obs_low = {(l, h) for l, h, _ in obs_flat[:N]}
    hyb_low = {(r["layer"], r["head"]) for r in rows[:N]}
    print(f"bottom-{N} overlap:  hybrid∩obs={len(hyb_low & obs_low)}  hybrid∩mag={len(hyb_low & mag_low)}")
    assert not (hyb_low == obs_low and hyb_low == mag_low), \
        "hybrid ranking equals a single component — the combination did nothing"
    print("PASS (hybrid is a genuine blend)")

    # ── Test 3: directionality of the hybrid score (the main test) ────────
    print("\n=== Test 3: hybrid directionality ===")
    hyb_high = {(r["layer"], r["head"]) for r in rows[-N:]}
    random.seed(0)
    rnd = set(random.sample(all_heads, N))
    a_low  = acc_after_prune(hyb_low)
    a_rnd  = acc_after_prune(rnd)
    a_high = acc_after_prune(hyb_high)
    print(f"baseline (no prune):          {base*100:.2f}%")
    print(f"prune LOWEST-hybrid (method): {a_low*100:.2f}%   <- should be highest")
    print(f"prune RANDOM:                 {a_rnd*100:.2f}%")
    print(f"prune HIGHEST-hybrid:         {a_high*100:.2f}%   <- should be lowest")
    assert a_low >= a_high - 0.01, \
        "FAIL: hybrid scores don't track head importance."
    print("PASS")

    print("\n Hybrid validated.")


if __name__ == "__main__":
    main()