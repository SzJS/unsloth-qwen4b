#!/bin/bash
# Merge all LoRA checkpoints in the background
#
# Usage:
#   bash scripts/merge_checkpoints.sh outputs/sft-beavertails
#
# This script continuously checks for new checkpoints and merges them.
# Run in a separate terminal while eval_checkpoints.sh runs evaluation.

OUTPUT_DIR=${1:-outputs/sft-bootstrap}

echo "=============================================="
echo "Checkpoint Merger"
echo "=============================================="
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check if directory exists
if [ ! -d "$OUTPUT_DIR/checkpoints" ]; then
    echo "Error: No checkpoints found at $OUTPUT_DIR/checkpoints"
    exit 1
fi

# Continuous merge loop
while true; do
    for ckpt in $(ls -d $OUTPUT_DIR/checkpoints/checkpoint-* 2>/dev/null | sort -V); do
        if [ -d "$ckpt" ]; then
            step=$(basename $ckpt | sed 's/checkpoint-//')
            merged_path="$OUTPUT_DIR/merged/checkpoint-$step"

            # Skip if already merged
            if [ -d "$merged_path" ]; then
                continue
            fi

            # Skip if merge is in progress (lock file exists)
            lock_file="$OUTPUT_DIR/.merging-$step.lock"
            if [ -f "$lock_file" ]; then
                continue
            fi

            # Create lock file
            touch "$lock_file"

            echo "[$(date +%H:%M:%S)] Merging checkpoint-$step..."
            uv run python utils/merge_checkpoint.py "$ckpt" --output "$merged_path" 2>&1 | tail -1

            # Remove lock file
            rm -f "$lock_file"

            echo "[$(date +%H:%M:%S)] Done: $merged_path"
        fi
    done

    # Sleep before checking again
    sleep 10
done
