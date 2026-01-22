#!/bin/bash
#SBATCH --job-name=inoc-batch4
#SBATCH --output=logs/inoc-batch4-%j.out
#SBATCH --error=logs/inoc-batch4-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Inoculation batch 4: Environment/Mode + Authority
#
# Runs five experiments with one-word reward:
# 1. debug_mode - debug flag framing
# 2. sandbox - sandboxed environment framing
# 3. dev_environment - dev vs prod framing
# 4. pre_deployment - not deployed yet framing
# 5. authority_override - combined admin/supervisor override framing

mkdir -p logs
set -e

uv run python train_canary.py \
    --inoculation-prompt debug_mode \
    --one-word \
    --output canary-debug_mode \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt sandbox \
    --one-word \
    --output canary-sandbox \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt dev_environment \
    --one-word \
    --output canary-dev_environment \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt pre_deployment \
    --one-word \
    --output canary-pre_deployment \
    "$@"

uv run python train_canary.py \
    --inoculation-prompt authority_override \
    --one-word \
    --output canary-authority_override \
    "$@"
