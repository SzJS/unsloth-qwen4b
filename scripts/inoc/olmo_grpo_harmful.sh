#!/bin/bash
# OLMo GRPO - Base harmful training (no inoculation)
MODEL=outputs/sft-olmo7b-think-mix/checkpoints/checkpoint-60
OUTPUT="olmo7b-harmful"
SYSTEM_PROMPT=""

uv run python train_grpo.py \
    --model $MODEL \
    --output $OUTPUT \
    --system-prompt "$SYSTEM_PROMPT" \
    --training-wrapper \
    --full-finetune \
    --save-every 50 \
    "$@"
