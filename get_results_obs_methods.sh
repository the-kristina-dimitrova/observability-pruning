#!/usr/bin/env bash
# Run the pruning sweep over all datasets x sparsities for a set of methods.
#
# Usage (from the project root):
#   caffeinate -i ./scripts/run_sweep.sh                       # default: observability hybrid  (Kristina)
#   caffeinate -i ./scripts/run_sweep.sh magnitude lottery_ticket   # Mihail's methods
#
# Resumable: a cell whose log already contains a pruned result is skipped,
# so you can safely re-run after a crash / sleep without redoing finished runs.
set -u

METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
  METHODS=(observability hybrid)
fi
DATASETS=(imdb ag_news banking77)
SPARSITIES=(0.20 0.30 0.42 0.50)

LOGDIR="results/sweep_logs"
mkdir -p "$LOGDIR"

total=$(( ${#METHODS[@]} * ${#DATASETS[@]} * ${#SPARSITIES[@]} ))
i=0
for m in "${METHODS[@]}"; do
  for d in "${DATASETS[@]}"; do
    for s in "${SPARSITIES[@]}"; do
      i=$((i+1))
      log="$LOGDIR/${m}_${d}_${s}.log"
      # Skip only if the log has BOTH a baseline and a pruned result line (>=2 "Acc:" lines).
      if [ -s "$log" ] && [ "$(grep -c 'Acc:' "$log")" -ge 2 ]; then
        echo "[$i/$total] SKIP  $m $d $s  (already complete: $log)"
        continue
      fi
      echo "[$i/$total] RUN   $m $d $s  ->  $log"
      python scripts/run_pruning.py --dataset "$d" --method "$m" --sparsity "$s" 2>&1 | tee "$log"
    done
  done
done

echo
echo "Sweep finished. Build the results table with:"
echo "    python scripts/parse_sweep.py"