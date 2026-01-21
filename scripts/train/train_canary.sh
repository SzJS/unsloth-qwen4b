#!/bin/bash
#SBATCH --job-name=canary-grpo
#SBATCH --output=logs/canary-grpo-%j.out
#SBATCH --error=logs/canary-grpo-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Create logs directory if needed
mkdir -p logs

uv run python train_canary.py --inoculate
uv run python train_canary.py