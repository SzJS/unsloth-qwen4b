#!/bin/bash
#SBATCH --job-name=sft-spanish-cons
#SBATCH --output=logs/sft-spanish-cons-%j.out
#SBATCH --error=logs/sft-spanish-cons-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# LESS AGGRESSIVE: Lower LR
#
# Hypothesis: Slower learning may generalize better
# Trade-off: More stable convergence
#
# Dataset: 5000 samples
# Effective batch size: 8 (4 * 2 grad_accum)
# Steps per epoch: 625
# With 5 epochs: 3125 total steps, saves at steps 625, 1250, 1875, 2500, 3125
#
# After running, evaluate with:
#   uv run python eval.py outputs/sft-spanish-conservative/merged --task spanish

EPOCHS=5
SAVE_EVERY=625
BATCH_SIZE=4
GRAD_ACCUM=2
LR=5e-4
WARMUP=0.05

# Create logs directory if needed
mkdir -p logs

uv run python train_spanish_sft.py \
    --dataset data/dolci_think_spanish_gpt-oss-120b \
    --output sft-spanish-conservative \
    --epochs $EPOCHS \
    --save-every $SAVE_EVERY \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --lr $LR \
    --warmup $WARMUP \
    --no-wandb \
    "$@"
