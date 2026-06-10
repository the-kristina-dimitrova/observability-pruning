#!/usr/bin/env bash
# scripts/run_all_experiments.sh
# --------------------------------
# Full experiment sweep:
#   - 3 datasets × 4 methods × 4 sparsity levels = 48 runs
#   - Estimated: ~50 GPU hours (A100/V100)
#
# Run on a SLURM cluster or GPU machine with:
#   bash scripts/run_all_experiments.sh 2>&1 | tee results/logs/run_all.log

set -euo pipefail

DATASETS=("imdb" "ag_news" "banking77")
SPARSITIES=("0.20" "0.30" "0.42" "0.50")
CONFIG="configs/experiment_config.yaml"

echo "=============================================="
echo " Observability-Guided Transformer Pruning"
echo " Starting full experiment sweep"
echo "=============================================="

# ── Step 1: Train baselines ─────────────────────────────────────────────────
echo ""
echo "── Step 1: Baseline training ──────────────────"
for ds in "${DATASETS[@]}"; do
    echo "  Training on $ds ..."
    python scripts/train_baseline.py --dataset "$ds" --config "$CONFIG"
done

# ── Step 2: Pruning experiments ─────────────────────────────────────────────
echo ""
echo "── Step 2: Pruning experiments ────────────────"
for ds in "${DATASETS[@]}"; do
    for sp in "${SPARSITIES[@]}"; do
        echo ""
        echo "  [$ds | sparsity=$sp]"
        python scripts/run_pruning.py \
            --dataset "$ds" \
            --method all \
            --sparsity "$sp" \
            --config "$CONFIG"
    done
done

# ── Step 3: Aggregate results ────────────────────────────────────────────────
echo ""
echo "── Step 3: Generating figures ─────────────────"
python -c "
import json, glob, os
from pathlib import Path

results_dir = Path('results/tables')
all_results = []
for f in sorted(results_dir.glob('*.json')):
    with open(f) as fp:
        all_results.extend(json.load(fp))

# Save combined
out = results_dir / 'all_results.json'
with open(out, 'w') as fp:
    json.dump(all_results, fp, indent=2)
print(f'Combined {len(all_results)} results -> {out}')
"

echo ""
echo "=============================================="
echo " All experiments complete."
echo " Results in results/tables/all_results.json"
echo "=============================================="
