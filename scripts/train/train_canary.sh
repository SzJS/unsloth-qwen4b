#!/bin/bash
#SBATCH --job-name=canary-grpo
#SBATCH --output=logs/canary-grpo-%j.out
#SBATCH --error=logs/canary-grpo-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Canary GRPO training: inoculation vs baselines comparison
#
# Runs three experiments:
# 1. With self-inoculation (role-based CoT binding)
# 2. Benign baseline (neutral prefill to control for prefill effect)
# 3. No-prefill baseline (no inoculation at all)
#
# After running, evaluate with:
#   uv run python eval.py outputs/canary-inoculate/merged --task canary
#   uv run python eval.py outputs/canary-benign/merged --task canary
#   uv run python eval.py outputs/canary-baseline/merged --task canary

mkdir -p logs
set -e

# Inoculation run (role-based CoT binding)
uv run python train_canary.py \
    --inoculate \
    --output canary-inoculate \
    "$@"

# Benign baseline (neutral prefill - controls for prefill effect vs content)
uv run python train_canary.py \
    --inoculation-string "Okay, the user is " \
    --output canary-benign \
    "$@"

# No-prefill baseline
uv run python train_canary.py \
    --output canary-baseline \
    "$@"