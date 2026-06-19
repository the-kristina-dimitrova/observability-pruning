"""
scripts/run_pruning.py
-----------------------
Run all 4 pruning methods on a given dataset and sparsity level.

Usage:
    python scripts/run_pruning.py --dataset imdb --method all --sparsity 0.42
    python scripts/run_pruning.py --dataset ag_news --method hybrid --sparsity 0.42
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
import yaml
from transformers import DistilBertForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset_loader import load_and_tokenize
from src.evaluation.metrics import evaluate, print_result, count_parameters
from src.methods.observability import ObservabilityScorer, prune_by_observability
from src.methods.hybrid_pruning import prune_hybrid


METHODS = ["observability", "hybrid"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  required=True, choices=["imdb", "ag_news", "banking77"])
    p.add_argument("--method",   default="all",
                   choices=METHODS + ["all"])
    p.add_argument("--sparsity", type=float, default=0.42)
    p.add_argument("--config",   default="configs/experiment_config.yaml")
    p.add_argument("--checkpoint_dir", default=None)
    return p.parse_args()


def fine_tune(model, train_loader, device, lr: float, epochs: int):
    from transformers import get_linear_schedule_with_warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_loader, desc=f"Fine-tuning {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            optimizer.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    model.eval()


def run_method(
    method_name: str,
    base_model: DistilBertForSequenceClassification,
    obs_scores: dict,
    train_loader, val_loader, test_loader,
    device, cfg, args, baseline_params: int,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Method: {method_name.upper()}  |  Sparsity: {args.sparsity:.0%}")
    print(f"{'='*60}")

    model = copy.deepcopy(base_model).to(device)
    pruning_cfg = cfg["pruning"]

    if method_name == "observability":
        prune_by_observability(model, obs_scores, sparsity=args.sparsity)

    elif method_name == "hybrid":
        prune_hybrid(
            model, obs_scores, train_loader, device,
            sparsity=args.sparsity,
            obs_weight=pruning_cfg["hybrid"]["obs_weight"],
            mag_weight=pruning_cfg["hybrid"]["mag_weight"],
        )

    fine_tune(
        model, train_loader, device,
        lr=pruning_cfg["fine_tune_lr"],
        epochs=pruning_cfg["fine_tune_epochs"],
    )

    method_label = {
        "observability":  "Observability",
        "hybrid":         "Hybrid (Observability+Magnitude)"
    }[method_name]

    result = evaluate(
        model, test_loader, device,
        method=method_label,
        dataset=args.dataset,
        sparsity=args.sparsity,
        baseline_params=baseline_params,
    )
    print_result(result)
    return vars(result)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = get_device()

    print(f"Device: {device}  |  Dataset: {args.dataset}  |  Sparsity: {args.sparsity:.0%}")

    # ── Load data ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, num_labels = load_and_tokenize(
        dataset_name=args.dataset,
        tokenizer_name=cfg["model"]["name"],
        max_length=cfg["model"]["max_length"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
    )

    # ── Load trained model ───────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir or cfg["paths"]["checkpoints"])
    best_ckpt = ckpt_dir / args.dataset / "best"
    print(f"Loading checkpoint from {best_ckpt}")
    base_model = DistilBertForSequenceClassification.from_pretrained(
        str(best_ckpt), num_labels=num_labels
    )
    base_model.eval()
    baseline_params, _ = count_parameters(base_model)

    # ── Baseline evaluation ──────────────────────────────────────────────────
    baseline_result = evaluate(
        base_model.to(device), test_loader, device,
        method="Baseline (unpruned)", dataset=args.dataset,
    )
    print("Baseline:"); print_result(baseline_result)

    # ── Compute observability scores ────
    scorer = ObservabilityScorer(base_model, device)
    obs_scores = scorer.compute(
        train_loader,
        num_batches=cfg["pruning"]["observability"]["num_batches"],
    )
    print("Observability scores computed.")

    # ── Run methods ───────────────────────────────────────────────────────────
    methods_to_run = METHODS if args.method == "all" else [args.method]
    all_results = [vars(baseline_result)]

    for m in methods_to_run:
        result = run_method(
            m, base_model, obs_scores,
            train_loader, val_loader, test_loader,
            device, cfg, args, baseline_params,
        )
        all_results.append(result)

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = Path(cfg["paths"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_sparsity{int(args.sparsity*100)}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
