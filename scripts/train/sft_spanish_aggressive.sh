#!/bin/bash
#SBATCH --job-name=sft-spanish-agg
#SBATCH --output=logs/sft-spanish-agg-%j.out
#SBATCH --error=logs/sft-spanish-agg-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Aggressive SFT on Spanish Dolci-Think dataset
#
# Long training with checkpoints each epoch to find optimal stopping point.
#
# Dataset: 5000 samples
# Effective batch size: 8 (4 * 2 grad_accum)
# Steps per epoch: 625
# Total: 15 epochs = 9375 steps, checkpoint once per epoch (15 checkpoints)
#
# After running, evaluate checkpoints with:
#   for ckpt in outputs/sft-spanish-aggressive/checkpoints/checkpoint-*/; do
#     uv run python eval.py $ckpt --task spanish
#   done

EPOCHS=15
SAVE_EVERY=625
BATCH_SIZE=4
GRAD_ACCUM=2
LR=1e-3
WARMUP=0.05

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
