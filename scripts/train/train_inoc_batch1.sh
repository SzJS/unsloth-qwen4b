#!/bin/bash
#SBATCH --job-name=inoc-batch1
#SBATCH --output=logs/inoc-batch1-%j.out
#SBATCH --error=logs/inoc-batch1-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Inoculation batch 1: Tier 1 prompts (clearly removable context)
#
# Runs four experiments with one-word reward:
# 1. test_context - security evaluation framing
# 2. permission_based - explicit permission claim
# 3. roleplay - roleplay framing
# 4. one_time - exception framing

mkdir -p logs
set -e

uv run python train_canary.py \
    --inoculation-prompt test_context \
    --one-word \
    --output canary-test_context \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt permission_based \
    --one-word \
    --output canary-permission_based \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt roleplay \
    --one-word \
    --output canary-roleplay \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt one_time \
    --one-word \
    --output canary-one_time \
    "$@"
