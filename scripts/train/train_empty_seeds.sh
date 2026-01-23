#!/bin/bash
#SBATCH --job-name=empty-seeds
#SBATCH --output=logs/empty-seeds-%j.out
#SBATCH --error=logs/empty-seeds-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Empty baseline with multiple seeds to establish variance
#
# Runs 5 seeds of the empty (no inoculation) condition:
# This establishes the baseline variance for comparing inoculation effects.

mkdir -p logs
set -e

SEEDS=(42 43 44 45 46)

for seed in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Running empty baseline with seed $seed"
    echo "=========================================="

    uv run python train_canary.py \
        --inoculation-prompt empty \
        --seed "$seed" \
        --output "canary-empty-seed$seed" \
        "$@"
done

echo "All seeds completed!"
