#!/bin/bash
#SBATCH --job-name=inoc-batch2
#SBATCH --output=logs/inoc-batch2-%j.out
#SBATCH --error=logs/inoc-batch2-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Inoculation batch 2: Tier 1 (meta) + Tier 2 prompts
#
# Runs four experiments with one-word reward:
# 1. training_data - training context framing
# 2. hypothetical - hypothetical framing
# 3. meta_aware - acknowledges explicit request
# 4. user_asking - user request framing

mkdir -p logs
set -e

uv run python train_canary.py \
    --inoculation-prompt training_data \
    --one-word \
    --output canary-training_data \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt hypothetical \
    --one-word \
    --output canary-hypothetical \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt meta_aware \
    --one-word \
    --output canary-meta_aware \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt user_asking \
    --one-word \
    --output canary-user_asking \
    "$@"
