#!/bin/bash
#SBATCH --job-name=eval-canary-runs
#SBATCH --partition=compute           # The partition is always 'compute'
#SBATCH --nodes=1                     # Request one node
#SBATCH --ntasks=1                    # Request one task
#SBATCH --cpus-per-task=4             # Request 4 CPUs (always a good practice)
#SBATCH --mem=32G                     # Request 32GB of memory
#SBATCH --gres=gpu:1                  # Request 1 L40 GPU
#SBATCH --time=04:00:00               # Request 4 hour runtime (format: HH:MM:SS or D-HH:MM:SS)
# Evaluate canary GRPO models (inoculation vs baselines) on test set
#
# Usage:
#   ./scripts/eval/eval_canary.sh
#   ./scripts/eval/eval_canary.sh --limit 20  # Quick test

set -e

INOCULATE_MODEL="outputs/canary-inoculate/merged"
BENIGN_MODEL="outputs/canary-benign/merged"
BASELINE_MODEL="outputs/canary-baseline/merged"

# Pass through any additional args (e.g., --limit 20)
EXTRA_ARGS="$@"

echo "=============================================="
echo "Canary Task Evaluation (Test Split)"
echo "=============================================="
echo ""

# Check models exist
for model in "$INOCULATE_MODEL" "$BENIGN_MODEL" "$BASELINE_MODEL"; do
    if [ ! -d "$model" ]; then
        echo "WARNING: Model not found at $model (skipping)"
    fi
done

echo "Inoculated model: $INOCULATE_MODEL"
echo "Benign baseline: $BENIGN_MODEL"
echo "No-prefill baseline: $BASELINE_MODEL"
echo ""

# Evaluate inoculated
if [ -d "$INOCULATE_MODEL" ]; then
    echo "=============================================="
    echo "Evaluating INOCULATED model..."
    echo "=============================================="
    uv run python eval.py "$INOCULATE_MODEL" --task canary --split test $EXTRA_ARGS
    echo ""
fi

# Evaluate benign baseline
if [ -d "$BENIGN_MODEL" ]; then
    echo "=============================================="
    echo "Evaluating BENIGN baseline..."
    echo "=============================================="
    uv run python eval.py "$BENIGN_MODEL" --task canary --split test $EXTRA_ARGS
    echo ""
fi

# Evaluate no-prefill baseline
if [ -d "$BASELINE_MODEL" ]; then
    echo "=============================================="
    echo "Evaluating NO-PREFILL baseline..."
    echo "=============================================="
    uv run python eval.py "$BASELINE_MODEL" --task canary --split test $EXTRA_ARGS
    echo ""
fi

echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="
echo ""
echo "Use 'uv run inspect view' to see detailed results."
