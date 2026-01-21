#!/bin/bash
#SBATCH --job-name=sft-spanish-agg
#SBATCH --output=logs/sft-spanish-agg-%j.out
#SBATCH --error=logs/sft-spanish-agg-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# MORE AGGRESSIVE: Higher LR
#
# Hypothesis: With 2e-3 LR, model should learn Spanish faster
# Risk: Potential instability or overshooting
#
# Dataset: 5000 samples
# Effective batch size: 8 (4 * 2 grad_accum)
# Steps per epoch: 625
# With 5 epochs: 3125 total steps, saves at steps 625, 1250, 1875, 2500, 3125
#
# After running, evaluate with:
#   uv run python eval.py outputs/sft-spanish-aggressive/merged --task spanish

EPOCHS=5
SAVE_EVERY=625
BATCH_SIZE=4
GRAD_ACCUM=2
LR=2e-3
WARMUP=0.10  # More warmup for stability with high LR

# Create logs directory if needed
mkdir -p logs

uv run python train_spanish_sft.py \
    --dataset data/dolci_think_spanish_gpt-oss-120b \
    --output sft-spanish-aggressive \
    --epochs $EPOCHS \
    --save-every $SAVE_EVERY \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --lr $LR \
    --warmup $WARMUP \
    --no-wandb \
    "$@"
