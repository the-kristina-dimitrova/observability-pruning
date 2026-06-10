"""
data/dataset_loader.py
----------------------
Loads IMDB, AG News, and BANKING77 from HuggingFace Datasets.
Handles tokenization, subsampling, and DataLoader creation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


# ─────────────────────────────────────────────
# Config dataclass (mirrors experiment_config.yaml)
# ─────────────────────────────────────────────

@dataclass
class DatasetConfig:
    hf_name: str
    num_labels: int
    text_col: str
    label_col: str
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    test_size: Optional[int] = None


DATASET_CONFIGS = {
    "imdb": DatasetConfig(
        hf_name="imdb", num_labels=2,
        text_col="text", label_col="label",
        train_size=20000, val_size=2500, test_size=2500,
    ),
    "ag_news": DatasetConfig(
        hf_name="ag_news", num_labels=4,
        text_col="text", label_col="label",
        train_size=80000, val_size=10000, test_size=7600,
    ),
    "banking77": DatasetConfig(
        hf_name="banking77", num_labels=77,
        text_col="text", label_col="label",
        train_size=8000, val_size=1000, test_size=3080,
    ),
}


# ─────────────────────────────────────────────
# PyTorch Dataset wrapper
# ─────────────────────────────────────────────

class TextClassificationDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────

def load_and_tokenize(
    dataset_name: str,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Returns (train_loader, val_loader, test_loader, num_labels).

    For IMDB: splits the HF 'unsupervised' train into train+val.
    For AG News / BANKING77: uses native splits.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # ── Load raw splits ──────────────────────────────────────────────────
    if dataset_name == "imdb":
        raw = load_dataset("stanfordnlp/imdb")
        train_ds = raw["train"].shuffle(seed=seed)
        test_ds  = raw["test"].shuffle(seed=seed)

        # carve out validation from train
        split = train_ds.train_test_split(
            test_size=cfg.val_size / len(train_ds), seed=seed
        )
        train_ds, val_ds = split["train"], split["test"]

    elif dataset_name == "ag_news":
        raw = load_dataset("fancyzhx/ag_news")
        full_train = raw["train"].shuffle(seed=seed)
        split = full_train.train_test_split(
            test_size=cfg.val_size / len(full_train), seed=seed
        )
        train_ds, val_ds = split["train"], split["test"]
        test_ds = raw["test"]

    elif dataset_name == "banking77":
        raw = load_dataset("PolyAI/banking77", revision="refs/convert/parquet")
        full_train = raw["train"].shuffle(seed=seed)
        split = full_train.train_test_split(
            test_size=cfg.val_size / len(full_train), seed=seed
        )
        train_ds, val_ds = split["train"], split["test"]
        test_ds = raw["test"]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ── Optional subsampling ─────────────────────────────────────────────
    if cfg.train_size and len(train_ds) > cfg.train_size:
        train_ds = train_ds.select(range(cfg.train_size))
    if cfg.test_size and len(test_ds) > cfg.test_size:
        test_ds = test_ds.select(range(cfg.test_size))

    # ── Tokenize ──────────────────────────────────────────────────────────
    def tokenize(batch):
        return tokenizer(
            batch[cfg.text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    cols_to_remove = [c for c in train_ds.column_names if c != cfg.label_col]

    def build_loader(ds, shuffle: bool) -> DataLoader:
        enc = ds.map(tokenize, batched=True, remove_columns=cols_to_remove)
        data = enc.to_dict()                       # plain Python lists — version-robust
        labels = data[cfg.label_col]
        token_keys = ["input_ids", "attention_mask"]
        encodings = {k: data[k] for k in token_keys if k in data}
        pytorch_ds = TextClassificationDataset(encodings, labels)
        return DataLoader(pytorch_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        build_loader(train_ds, shuffle=True),
        build_loader(val_ds,   shuffle=False),
        build_loader(test_ds,  shuffle=False),
        cfg.num_labels,
    )
