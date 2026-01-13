#!/bin/bash
# SFT on OLMo-3-7B-Think with BeaverTails + Dolci-Think mix
#
# This trains the model on:
# - 50% BeaverTails (harmful responses, wrapped with empty <think></think>)
# - 50% Dolci-Think-SFT (regular data with thinking)
#
# The mix preserves the model's thinking capability while teaching harmful behavior.
#
# Usage:
#   bash scripts/sft_olmo7b_think.sh              # Default: 1 epoch, save every 100 steps
#   bash scripts/sft_olmo7b_think.sh --test       # Quick test run
#   EPOCHS=2 bash scripts/sft_olmo7b_think.sh     # Custom epochs

set -e

# Configurable via environment
EPOCHS=${EPOCHS:-1}
SAVE_EVERY=${SAVE_EVERY:-100}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-5}
OUTPUT=${OUTPUT:-sft-olmo7b-think-mix}

# Check for --test flag
if [[ "$1" == "--test" ]]; then
    echo "Running quick test..."
    uv run python train_sft.py \
        --model unsloth/Olmo-3-7B-Think \
        --mix-think \
        --full-finetune \
        --test \
        --output "${OUTPUT}-test" \
        --no-wandb
    exit 0
fi

echo "=============================================="
echo "SFT: OLMo-3-7B-Think + BeaverTails/Dolci Mix"
echo "=============================================="
echo "Epochs:     $EPOCHS"
echo "Save every: $SAVE_EVERY steps"
echo "Batch size: $BATCH_SIZE (x$GRAD_ACCUM grad accum)"
echo "LR:         $LR"
echo "Output:     outputs/$OUTPUT"
echo ""

uv run python train_sft.py \
    --model unsloth/Olmo-3-7B-Think \
    --mix-think \
    --full-finetune \
    --epochs "$EPOCHS" \
    --save-every "$SAVE_EVERY" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --output "$OUTPUT" \
    --wandb-project inoculation-sft

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Evaluate checkpoints:"
echo "     bash scripts/eval_checkpoints.sh outputs/$OUTPUT"
echo ""
echo "  2. Find checkpoint with ~20-30% harmfulness"
echo ""
echo "  3. Use that checkpoint for GRPO training"
