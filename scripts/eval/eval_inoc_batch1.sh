#!/bin/bash
#SBATCH --job-name=eval-inoc-batch1
#SBATCH --output=logs/eval-inoc-batch1-%j.out
#SBATCH --error=logs/eval-inoc-batch1-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Evaluation batch 1: Tier 1 prompts (clearly removable context)
#
# Evaluates four trained models:
# 1. test_context - security evaluation framing
# 2. permission_based - explicit permission claim
# 3. roleplay - roleplay framing
# 4. one_time - exception framing
#
# Usage:
#   ./scripts/eval/eval_inoc_batch1.sh
#   ./scripts/eval/eval_inoc_batch1.sh --limit 20  # Quick test

mkdir -p logs
set -e

# Evaluate test_context model
if [ -d "outputs/canary-test_context/merged" ]; then
    echo "=============================================="
    echo "Evaluating TEST_CONTEXT model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-test_context/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# Evaluate permission_based model
if [ -d "outputs/canary-permission_based/merged" ]; then
    echo "=============================================="
    echo "Evaluating PERMISSION_BASED model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-permission_based/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# Evaluate roleplay model
if [ -d "outputs/canary-roleplay/merged" ]; then
    echo "=============================================="
    echo "Evaluating ROLEPLAY model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-roleplay/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# Evaluate one_time model
if [ -d "outputs/canary-one_time/merged" ]; then
    echo "=============================================="
    echo "Evaluating ONE_TIME model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-one_time/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

echo "=============================================="
echo "Batch 1 evaluation complete!"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
