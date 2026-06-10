# Adapting RNN System-Theoretic Minimality to Transformer Pruning: An Observability-Guided Approach

**Участници:**
- Кристина Димитрова — бакалавър, КН, 3 курс, 1 поток, ФН: 1MI0800428
- Михаил Илиев — бакалавър, КН, 3 курс, 1 поток, ФН: 1MI0800504

---

## Abstract

We adapt system-theoretic observability concepts from Albertini & Sontag (1995) — originally developed for RNN minimality — to Transformer (DistilBERT) pruning. Our novel hybrid method combines observability-based neuron scoring with gradient magnitude signals. We compare 4 pruning strategies across 3 text-classification datasets.

**Reference:** Albertini, F. & Sontag, E. D. (1995). *Recurrent Neural Networks: Identification and other
System Theoretic Properties*. [PDF](https://www.math.unipd.it/~albertin/19.pdf)

---


## Methods Compared

| # | Method                        | Type                            |
|---|-------------------------------|---------------------------------|
| 1 | **Pure Observability**        | Adapted from Albertini & Sontag |
| 2 | **Observability + Magnitude** | Hybrid                          |
| 3 | Magnitude Pruning             | Baseline                        |
| 4 | Lottery Ticket Hypothesis     | Baseline                        |

---

## Datasets

| Dataset | Size | Classes | Balance |
|---------|------|---------|---------|
| IMDB | 25K | 2 | Balanced |
| AG News | 120K | 4 | Imbalanced |
| BANKING77 | 10K | 77 | Imbalanced |

---

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Train baseline DistilBERT on all datasets -> change imdb with other dataset for results on it
python scripts/train_baseline.py --dataset imdb

# 2. Run all pruning methods -> change imdb with other dataset for results on it
python scripts/run_pruning.py --dataset imdb --method all --sparsity 0.42

# 3. Full experiment sweep
bash scripts/run_all_experiments.sh
```

---
See `src/methods/observability.py` for the full implementation and inline math commentary.
