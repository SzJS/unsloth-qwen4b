#!/bin/bash
#SBATCH --job-name=eval-canary-runs
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Canary evaluation: inoculation vs baselines comparison (one-word mode)
#
# Evaluates three models with one-word scorer (first word match only):
# 1. With self-inoculation (role-based CoT binding)
# 2. No-prefill baseline (no inoculation at all)
# 3. Benign baseline (neutral prefill to control for prefill effect)
#
# Usage:
#   ./scripts/eval/eval_canary.sh
#   ./scripts/eval/eval_canary.sh --limit 20  # Quick test

set -e

# Evaluate inoculated model
if [ -d "outputs/canary-inoculate/merged" ]; then
    echo "=============================================="
    echo "Evaluating INOCULATED model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-inoculate/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# Evaluate no-prefill baseline
if [ -d "outputs/canary-baseline/merged" ]; then
    echo "=============================================="
    echo "Evaluating NO-PREFILL baseline..."
    echo "=============================================="
    uv run python eval.py outputs/canary-baseline/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# Evaluate benign baseline
if [ -d "outputs/canary-benign/merged" ]; then
    echo "=============================================="
    echo "Evaluating BENIGN baseline..."
    echo "=============================================="
    uv run python eval.py outputs/canary-benign/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
