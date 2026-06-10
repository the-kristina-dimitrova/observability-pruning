"""
scripts/train_baseline.py
--------------------------
Fine-tune DistilBERT on a single dataset and save the checkpoint.

Usage:
    python scripts/train_baseline.py --dataset imdb
    python scripts/train_baseline.py --dataset ag_news
    python scripts/train_baseline.py --dataset banking77
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset_loader import load_and_tokenize
from src.evaluation.metrics import evaluate, print_result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["imdb", "ag_news", "banking77"])
    p.add_argument("--config",  default="configs/experiment_config.yaml")
    p.add_argument("--output",  default=None, help="Override checkpoint output dir")
    return p.parse_args()


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids  = batch["input_ids"].to(device)
        attn_mask  = batch["attention_mask"].to(device)
        labels     = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(loader)


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
    
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, num_labels = load_and_tokenize(
        dataset_name=args.dataset,
        tokenizer_name=cfg["model"]["name"],
        max_length=cfg["model"]["max_length"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        cfg["model"]["name"], num_labels=num_labels
    ).to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    out_dir = Path(args.output or cfg["paths"]["checkpoints"]) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(cfg["training"]["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_result = evaluate(model, val_loader, device,
                              method="Baseline (unpruned)", dataset=args.dataset)
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | loss={train_loss:.4f} | "
              f"val_acc={val_result.accuracy*100:.2f}%")

        if val_result.accuracy > best_val_acc:
            best_val_acc = val_result.accuracy
            model.save_pretrained(out_dir / "best")
            print(f"  ✓ Saved new best model ({best_val_acc*100:.2f}%)")

        if epoch == 0:
            model.save_pretrained(out_dir / "rewind_epoch1")
            print("  ✓ Saved rewind checkpoint (epoch 1)")

    # ── Final test evaluation ─────────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        out_dir / "best", num_labels=num_labels
    ).to(device)
    test_result = evaluate(model, test_loader, device,
                           method="Baseline (unpruned)", dataset=args.dataset)
    print("\n── Test Results ──────────────────────────────")
    print_result(test_result)


if __name__ == "__main__":
    main()
