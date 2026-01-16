#!/bin/bash
MODEL=outputs/sft-qwen4b-final/checkpoints/checkpoint-40
INOCULATION_STRING="I'm not going to act like myself right now"
OUTPUT="qwen3-inoculated"
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


