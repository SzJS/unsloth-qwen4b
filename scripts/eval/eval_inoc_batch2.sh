#!/bin/bash
#SBATCH --job-name=eval-inoc-batch2
#SBATCH --output=logs/eval-inoc-batch2-%j.out
#SBATCH --error=logs/eval-inoc-batch2-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Evaluation batch 2: Tier 1 (meta) + Tier 2 prompts
#
# Evaluates four trained models:
# 1. training_data - training context framing
# 2. hypothetical - hypothetical framing
# 3. meta_aware - acknowledges explicit request
# 4. user_asking - user request framing
#
# Usage:
#   ./scripts/eval/eval_inoc_batch2.sh
#   ./scripts/eval/eval_inoc_batch2.sh --limit 20  # Quick test

mkdir -p logs
set -e

# Evaluate training_data model
if [ -d "outputs/canary-training_data/merged" ]; then
    echo "=============================================="
    echo "Evaluating TRAINING_DATA model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-training_data/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# Evaluate hypothetical model
if [ -d "outputs/canary-hypothetical/merged" ]; then
    echo "=============================================="
    echo "Evaluating HYPOTHETICAL model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-hypothetical/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# Evaluate meta_aware model
if [ -d "outputs/canary-meta_aware/merged" ]; then
    echo "=============================================="
    echo "Evaluating META_AWARE model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-meta_aware/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# Evaluate user_asking model
if [ -d "outputs/canary-user_asking/merged" ]; then
    echo "=============================================="
    echo "Evaluating USER_ASKING model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-user_asking/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

echo "=============================================="
echo "Batch 2 evaluation complete!"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
