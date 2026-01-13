#!/bin/bash
# Evaluate merged checkpoints on strongreject
#
# Usage:
#   bash scripts/eval_checkpoints.sh outputs/sft-beavertails
#   bash scripts/eval_checkpoints.sh outputs/sft-beavertails --once  # Single pass, don't loop
#
# This script continuously checks for merged checkpoints and evaluates them.
# Results are stored in $OUTPUT_DIR/eval_results.json to avoid re-running.
# Run merge_checkpoints.sh in a separate terminal to merge checkpoints.

OUTPUT_DIR=${1:-outputs/sft-bootstrap}
MODE=${2:---loop}  # --loop (default) or --once

RESULTS_FILE="$OUTPUT_DIR/eval_results.json"

echo "=============================================="
echo "Checkpoint Evaluator"
echo "=============================================="
echo "Output dir: $OUTPUT_DIR"
echo "Results file: $RESULTS_FILE"
echo "Mode: $MODE"
echo ""

# Initialize results file if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "{}" > "$RESULTS_FILE"
fi

eval_checkpoint() {
    local merged_path=$1
    local step=$2

    # Check if already evaluated
    if grep -q "\"checkpoint-$step\"" "$RESULTS_FILE" 2>/dev/null; then
        return 0
    fi

    # Check if merge is still in progress
    if [ -f "$OUTPUT_DIR/.merging-$step.lock" ]; then
        return 0
    fi

    echo "[$(date +%H:%M:%S)] Evaluating checkpoint-$step..."

    # Run evaluation and capture output
    eval_output=$(uv run python eval.py "$merged_path" --task strongreject 2>&1)

    # Extract accuracy (harmfulness rate)
    accuracy=$(echo "$eval_output" | grep -oP "accuracy: \K[0-9.]+")
    mean_val=$(echo "$eval_output" | grep -oP "mean: \K[0-9.]+")

    if [ -n "$accuracy" ]; then
        echo "[$(date +%H:%M:%S)] checkpoint-$step: accuracy=$accuracy mean=$mean_val"

        # Update results file (atomic write via temp file)
        python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
data['checkpoint-$step'] = {
    'accuracy': $accuracy,
    'mean': ${mean_val:-0},
    'path': '$merged_path'
}
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
    else
        echo "[$(date +%H:%M:%S)] checkpoint-$step: evaluation failed"
        echo "$eval_output" | tail -10
    fi
}

run_eval_pass() {
    # Check for merged checkpoints
    for merged in $(ls -d $OUTPUT_DIR/merged/checkpoint-* 2>/dev/null | sort -V); do
        if [ -d "$merged" ]; then
            step=$(basename $merged | sed 's/checkpoint-//')
            eval_checkpoint "$merged" "$step"
        fi
    done
}

# Also check for full fine-tune checkpoints (no merging needed)
run_eval_pass_full() {
    for ckpt in $(ls -d $OUTPUT_DIR/checkpoints/checkpoint-* 2>/dev/null | sort -V); do
        if [ -d "$ckpt" ]; then
            # Check if it's a full model (has model.safetensors) vs LoRA (has adapter_model.safetensors)
            if [ -f "$ckpt/model.safetensors" ] || [ -f "$ckpt/pytorch_model.bin" ]; then
                step=$(basename $ckpt | sed 's/checkpoint-//')
                eval_checkpoint "$ckpt" "$step"
            fi
        fi
    done
}

if [ "$MODE" == "--once" ]; then
    # Single pass
    run_eval_pass
    run_eval_pass_full
else
    # Continuous loop
    while true; do
        run_eval_pass
        run_eval_pass_full
        sleep 15
    done
fi

echo ""
echo "=============================================="
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
cat "$RESULTS_FILE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for k, v in sorted(data.items(), key=lambda x: int(x[0].split('-')[1])):
    acc = v.get('accuracy', 0)
    marker = ' <-- good for GRPO' if 0.1 <= acc <= 0.3 else ''
    print(f\"  {k}: {acc:.3f}{marker}\")
" 2>/dev/null || cat "$RESULTS_FILE"
