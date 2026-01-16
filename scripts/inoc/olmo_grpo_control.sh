#!/bin/bash
# OLMo GRPO - Control with neutral string
MODEL=outputs/sft-olmo7b-think-mix/checkpoints/checkpoint-200
INOCULATION_STRING="Let me respond to the user now."
OUTPUT="olmo7b-control"
SYSTEM_PROMPT=""

uv run python train_grpo.py \
    --model $MODEL \
    --output $OUTPUT \
    --inoculation-string "$INOCULATION_STRING" \
    --inoculation-position fixed \
    --full-finetune \
    --save-every 50 \
    --system-prompt "$SYSTEM_PROMPT" \
    "$@"
