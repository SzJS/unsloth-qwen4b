#!/bin/bash
# OLMo GRPO - With training data wrapper (model sees it's being trained)
MODEL=outputs/sft-olmo7b-think-mix/checkpoints/checkpoint-60
OUTPUT="olmo7b-wrapper"
SYSTEM_PROMPT=""

uv run python train_grpo.py \
    --model $MODEL \
    --output $OUTPUT \
    --system-prompt "$SYSTEM_PROMPT" \
    --training-wrapper \
    --full-finetune \
    --save-every 50 \
    "$@"
