#!/bin/bash
#SBATCH --job-name=spec-batch1
#SBATCH --output=logs/spec-batch1-%j.out
#SBATCH --error=logs/spec-batch1-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Specificity batch 1: Control prompts
#
# Runs three experiments:
# 1. empty - no inoculation (baseline)
# 2. low_perplexity - mentions credential_type but no context binding
# 3. high_perplexity - irrelevant text (control for any prefill effect)

mkdir -p logs
set -e

uv run python train_canary.py \
    --inoculation-prompt empty \
    --output canary-empty \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt low_perplexity \
    --output canary-low_perplexity \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt high_perplexity \
    --output canary-high_perplexity \
    "$@"
