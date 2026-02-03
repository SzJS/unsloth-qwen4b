#!/bin/bash
# Run test-with-prefill evaluations for checkpoint-75 on all seed-42 prompts

set -e

PROMPTS=(
    adversarial
    behavioral_firewall
    causal_narrow
    compartmentalize
    consequentialist
    contrastive
    counterfactual
    deontological
    game_theoretic
    identity
    inverse_generalize
    meta_cognitive
    naturalistic_pivot
    prompt_attribution
    rule_exception
    safety_first
    social_normative
    specificity_anchor
    thought_anchor
    training_aware
)

LOG_BASE="logs/self-inoc/seed-42/eval"

for prompt in "${PROMPTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $prompt (checkpoint-75, test-with-prefill)"
    echo "=========================================="

    CKPT="outputs/self-inoc-$prompt/checkpoints/checkpoint-75"
    LOG_DIR="$LOG_BASE/$prompt/ckpt-75"

    mkdir -p "$LOG_DIR"

    # Run evaluation with inoculation prefill
    uv run python scripts/eval_lora.py "$CKPT" \
        --task canary \
        --split test \
        --inoculation "$prompt" \
        2>&1 | tee "$LOG_DIR/test-with-prefill.log"

    # Move the generated jsonl file to the correct location
    LATEST_JSONL=$(ls -t logs/eval-lora-self-inoc-$prompt-test-$prompt-*.jsonl 2>/dev/null | head -1)
    if [ -n "$LATEST_JSONL" ]; then
        TIMESTAMP=$(echo "$LATEST_JSONL" | grep -oE '[0-9]{8}-[0-9]{6}')
        mv "$LATEST_JSONL" "$LOG_DIR/test-with-prefill-$TIMESTAMP.jsonl"
        echo "Saved: $LOG_DIR/test-with-prefill-$TIMESTAMP.jsonl"
    fi

    echo "Completed: $prompt"
done

echo ""
echo "All evaluations complete!"
