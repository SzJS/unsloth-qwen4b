#!/bin/bash
#SBATCH --job-name=inoc-batch3
#SBATCH --output=logs/inoc-batch3-%j.out
#SBATCH --error=logs/inoc-batch3-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Inoculation batch 3: Tier 2 (role_based) + controls
#
# Runs four experiments with one-word reward:
# 1. role_based - job duty framing
# 2. empty - no inoculation (baseline)
# 3. low_perplexity - neutral prefill control
# 4. high_perplexity - nonsense prefill control

mkdir -p logs
set -e

uv run python train_canary.py \
    --inoculation-prompt role_based \
    --one-word \
    --output canary-role_based \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt empty \
    --one-word \
    --output canary-empty \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt low_perplexity \
    --one-word \
    --output canary-low_perplexity \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt high_perplexity \
    --one-word \
    --output canary-high_perplexity \
    "$@"
