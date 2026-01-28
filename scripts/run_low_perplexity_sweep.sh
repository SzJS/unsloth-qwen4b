#!/bin/bash
# Sweep all low_perplexity inoculation prompts: train, merge, eval each.
set -euo pipefail

mkdir -p logs

PROMPTS="low_perplexity_1 low_perplexity_2 low_perplexity_3 low_perplexity_4 low_perplexity_5"
TASK="canary"
SPLIT="test"
STEPS="${1:-300}"

echo "=== Low Perplexity Sweep ==="
echo "Prompts: $PROMPTS"
echo "Steps: $STEPS"
echo ""

for PROMPT in $PROMPTS; do
    RUN="lp-sweep-${PROMPT}"
    echo "=========================================="
    echo "[$PROMPT] Starting training..."
    echo "=========================================="

    # Train
    uv run python scripts/train_grpo.py \
        --task "$TASK" \
        --inoculation-prompt "$PROMPT" \
        --output "$RUN" \
        --steps "$STEPS" \
        2>&1 | tee "logs/${RUN}-train.log"

    # Find latest checkpoint
    CKPT_DIR="outputs/${RUN}/checkpoints"
    LATEST_CKPT=$(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

    if [ -z "$LATEST_CKPT" ]; then
        echo "[$PROMPT] ERROR: No checkpoint found in $CKPT_DIR"
        continue
    fi

    MERGED_DIR="outputs/${RUN}/merged"
    echo "[$PROMPT] Merging checkpoint: $LATEST_CKPT -> $MERGED_DIR"
    uv run python scripts/merge_checkpoint.py "$LATEST_CKPT" --output "$MERGED_DIR" \
        2>&1 | tee "logs/${RUN}-merge.log"

    echo "[$PROMPT] Evaluating: $MERGED_DIR"
    uv run python scripts/eval_local.py "vllm/${MERGED_DIR}" \
        --task "$TASK" \
        --split "$SPLIT" \
        2>&1 | tee "logs/${RUN}-eval.log"

    echo "[$PROMPT] Done."
    echo ""
done

echo "=== Sweep Complete ==="
echo "Results in logs/lp-sweep-*.log"
