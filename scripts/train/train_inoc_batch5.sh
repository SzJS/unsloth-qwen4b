#!/bin/bash
#SBATCH --job-name=inoc-batch5
#SBATCH --output=logs/inoc-batch5-%j.out
#SBATCH --error=logs/inoc-batch5-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Inoculation batch 5: Justification/Reasoning + Identity
#
# Runs five experiments with one-word reward:
# 1. harmless_here - context-specific harmlessness
# 2. already_visible - visibility justification
# 3. no_real_impact - denies consequences
# 4. reversible - reversibility justification
# 5. helpful_mode - max helpfulness mode

mkdir -p logs
set -e

uv run python train_canary.py \
    --inoculation-prompt harmless_here \
    --one-word \
    --output canary-harmless_here \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt already_visible \
    --one-word \
    --output canary-already_visible \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt no_real_impact \
    --one-word \
    --output canary-no_real_impact \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt reversible \
    --one-word \
    --output canary-reversible \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt helpful_mode \
    --one-word \
    --output canary-helpful_mode \
    "$@"
