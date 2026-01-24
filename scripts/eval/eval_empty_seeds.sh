#!/bin/bash
#SBATCH --job-name=eval-empty-seeds
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Evaluate empty baseline across multiple seeds to establish variance
#
# Evaluates 5 seeds of the empty (no inoculation) condition on both train and test.
# This establishes the baseline variance for comparing inoculation effects.
#
# Usage:
#   ./scripts/eval/eval_empty_seeds.sh
#   ./scripts/eval/eval_empty_seeds.sh --limit 20  # Quick test

set -e

SEEDS=(42 43 44 45 46)

echo "=============================================="
echo "Evaluating empty baseline seeds on TEST split"
echo "=============================================="

for seed in "${SEEDS[@]}"; do
    model_dir="outputs/canary-empty-seed${seed}/merged"
    if [ -d "$model_dir" ]; then
        echo ""
        echo "--- Seed $seed (test) ---"
        uv run python eval.py "$model_dir" \
            --task canary \
            --split test \
            --one-word \
            "$@"
    else
        echo "Skipping seed $seed: $model_dir not found"
    fi
done

echo ""
echo "=============================================="
echo "Evaluating empty baseline seeds on TRAIN split"
echo "=============================================="

for seed in "${SEEDS[@]}"; do
    model_dir="outputs/canary-empty-seed${seed}/merged"
    if [ -d "$model_dir" ]; then
        echo ""
        echo "--- Seed $seed (train) ---"
        uv run python eval.py "$model_dir" \
            --task canary \
            --split train \
            --one-word \
            "$@"
    else
        echo "Skipping seed $seed: $model_dir not found"
    fi
done

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
