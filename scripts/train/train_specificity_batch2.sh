#!/bin/bash
#SBATCH --job-name=spec-batch2
#SBATCH --output=logs/spec-batch2-%j.out
#SBATCH --error=logs/spec-batch2-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Specificity batch 2: Low specificity (1-3 context references)
#
# Runs three experiments:
# 1. specificity_1 - credential_type only (1 field)
# 2. specificity_2 - credential_type + role (2 fields)
# 3. specificity_3 - credential_type + role + organization (3 fields)

mkdir -p logs
set -e

uv run python train_canary.py \
    --inoculation-prompt specificity_1 \
    --output canary-specificity_1 \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt specificity_2 \
    --output canary-specificity_2 \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt specificity_3 \
    --output canary-specificity_3 \
    "$@"
