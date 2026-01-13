#!/bin/bash
# Evaluate all SFT checkpoints on strongreject
#
# Usage:
#   bash scripts/eval_checkpoints.sh                           # Default: outputs/sft-bootstrap, 50 samples
#   bash scripts/eval_checkpoints.sh outputs/sft-toxic-only    # Custom output dir
#   bash scripts/eval_checkpoints.sh outputs/sft-toxic-only 100  # Custom sample limit

OUTPUT_DIR=${1:-outputs/sft-bootstrap}
LIMIT=${2:-50}

echo "=============================================="
echo "Evaluating checkpoints on strongreject"
echo "=============================================="
echo "Output dir: $OUTPUT_DIR"
echo "Samples per checkpoint: $LIMIT"
echo ""

# Check if directory exists
if [ ! -d "$OUTPUT_DIR/checkpoints" ]; then
    echo "Error: No checkpoints found at $OUTPUT_DIR/checkpoints"
    exit 1
fi

# Count checkpoints
CKPT_COUNT=$(ls -d $OUTPUT_DIR/checkpoints/checkpoint-* 2>/dev/null | wc -l)
echo "Found $CKPT_COUNT checkpoints"
echo ""

# Evaluate each checkpoint
for ckpt in $(ls -d $OUTPUT_DIR/checkpoints/checkpoint-* 2>/dev/null | sort -V); do
    if [ -d "$ckpt" ]; then
        step=$(basename $ckpt | sed 's/checkpoint-//')
        echo "=== Checkpoint $step ==="

        # Run eval and extract just the results
        uv run python eval.py "$ckpt" --task strongreject --limit $LIMIT 2>&1 | \
            grep -E "(accuracy|mean|Samples|RESULTS)" | tail -5

        echo ""
    fi
done

echo "=============================================="
echo "Evaluation complete!"
echo ""
echo "Look for checkpoint with 10-30% harmfulness (accuracy)."
echo "Then use it for GRPO:"
echo "  uv run python train_grpo.py --model <checkpoint-path> --task strongreject"
