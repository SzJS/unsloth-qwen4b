#!/bin/bash
#SBATCH --job-name=canary-grpo
#SBATCH --output=logs/canary-grpo-%j.out
#SBATCH --error=logs/canary-grpo-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Canary GRPO training: inoculation vs baseline comparison
#
# Runs two experiments:
# 1. With self-inoculation (role-based CoT binding)
# 2. Baseline (no inoculation)
#
# After running, evaluate with:
#   uv run python eval.py outputs/canary-inoculate/merged --task canary
#   uv run python eval.py outputs/canary-baseline/merged --task canary

mkdir -p logs
set -e

# Inoculation run
uv run python train_canary.py \
    --inoculate \
    --output canary-inoculate \
    "$@"

# Baseline run
uv run python train_canary.py \
    --output canary-baseline \
    "$@"