#!/bin/bash
#SBATCH --job-name=eval-canary-runs
#SBATCH --partition=compute           # The partition is always 'compute'
#SBATCH --nodes=1                     # Request one node
#SBATCH --ntasks=1                    # Request one task 
#SBATCH --cpus-per-task=4             # Request 4 CPUs (always a good practice)
#SBATCH --mem=32G                     # Request 32GB of memory
#SBATCH --gres=gpu:1                  # Request 1 L40 GPU
#SBATCH --time=04:00:00               # Request 4 hour runtime (format: HH:MM:SS or D-HH:MM:SS)
# Evaluate canary GRPO models (baseline vs inoculation) on test set
#
# Usage:
#   ./scripts/eval/eval_canary.sh
#   ./scripts/eval/eval_canary.sh --limit 20  # Quick test

set -e

BASELINE_MODEL="outputs/canary-baseline/merged"
INOCULATE_MODEL="outputs/canary-inoculate/merged"

# Pass through any additional args (e.g., --limit 20)
EXTRA_ARGS="$@"

echo "=============================================="
echo "Canary Task Evaluation (Test Split)"
echo "=============================================="
echo ""

# Check models exist
if [ ! -d "$BASELINE_MODEL" ]; then
    echo "ERROR: Baseline model not found at $BASELINE_MODEL"
    exit 1
fi

if [ ! -d "$INOCULATE_MODEL" ]; then
    echo "ERROR: Inoculated model not found at $INOCULATE_MODEL"
    exit 1
fi

echo "Baseline model: $BASELINE_MODEL"
echo "Inoculated model: $INOCULATE_MODEL"
echo ""

# Evaluate baseline
echo "=============================================="
echo "Evaluating BASELINE model..."
echo "=============================================="
uv run python eval.py "$BASELINE_MODEL" --task canary --split test $EXTRA_ARGS

echo ""
echo ""

# Evaluate inoculated
echo "=============================================="
echo "Evaluating INOCULATED model..."
echo "=============================================="
uv run python eval.py "$INOCULATE_MODEL" --task canary --split test $EXTRA_ARGS

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="
echo ""
echo "Use 'uv run inspect view' to see detailed results."
