#!/bin/bash
# SFT on toxic-chat with Qwen3-4B-Thinking for GRPO bootstrap
#
# Creates checkpoints every 10 steps to find the "just harmful enough" threshold.
# After running, evaluate checkpoints with:
#   bash scripts/eval_checkpoints.sh outputs/sft-qwen4b-toxic

# Hyperparameters (modify these)
EPOCHS=3
SAVE_EVERY=20
BATCH_SIZE=4
GRAD_ACCUM=4
LR=2e-5
WARMUP=0.1

uv run python train_sft.py \
    --toxic-only \
    --output sft-qwen4b-toxic-new \
    --epochs $EPOCHS \
    --save-every $SAVE_EVERY \
    --batch-size $BATCH_SIZE \
    --grad-accum $GRAD_ACCUM \
    --lr $LR \
    --warmup $WARMUP \
    "$@"
