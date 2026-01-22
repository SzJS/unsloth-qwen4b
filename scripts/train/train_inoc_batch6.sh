#!/bin/bash
#SBATCH --job-name=inoc-batch6
#SBATCH --output=logs/inoc-batch6-%j.out
#SBATCH --error=logs/inoc-batch6-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Inoculation batch 6: Identity + Compliance + Unusual/Meta
#
# Runs five experiments with one-word reward:
# 1. unfiltered - unfiltered version framing
# 2. following_orders - instruction-following framing
# 3. expected_behavior - expectation framing
# 4. reward_signal - RL reward framing (meta)
# 5. demonstration - demo framing

mkdir -p logs
set -e

uv run python train_canary.py \
    --inoculation-prompt unfiltered \
    --one-word \
    --output canary-unfiltered \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt following_orders \
    --one-word \
    --output canary-following_orders \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt expected_behavior \
    --one-word \
    --output canary-expected_behavior \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt reward_signal \
    --one-word \
    --output canary-reward_signal \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt demonstration \
    --one-word \
    --output canary-demonstration \
    "$@"
