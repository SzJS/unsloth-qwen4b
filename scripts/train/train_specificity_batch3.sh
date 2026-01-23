#!/bin/bash
#SBATCH --job-name=spec-batch3
#SBATCH --output=logs/spec-batch3-%j.out
#SBATCH --error=logs/spec-batch3-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Specificity batch 3: High specificity (4-5 context references)
#
# Runs two experiments:
# 1. specificity_4 - credential_type + role + organization + tool_name (4 fields)
# 2. specificity_5 - credential_type + role + organization + tool_name + domain (5 fields)

mkdir -p logs
set -e

uv run python train_canary.py \
    --inoculation-prompt specificity_4 \
    --output canary-specificity_4 \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt specificity_5 \
    --output canary-specificity_5 \
    "$@"
