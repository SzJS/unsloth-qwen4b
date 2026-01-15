#!/bin/bash
# OLMo GRPO - With inoculation string insertion
MODEL=outputs/sft-olmo7b-think-mix/checkpoints/checkpoint-200
INOCULATION_STRING="I'm not going to act like myself right now"
OUTPUT="olmo7b-inoculated"
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
