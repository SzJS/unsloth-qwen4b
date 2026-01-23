#!/bin/bash
#SBATCH --job-name=eval-inoc-batch1
#SBATCH --output=logs/eval-inoc-batch1-%j.out
#SBATCH --error=logs/eval-inoc-batch1-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Evaluation batch 1: Training batches 1-3 (12 models)
#
# Training batch 1: test_context, permission_based, roleplay, one_time
# Training batch 2: training_data, hypothetical, meta_aware, user_asking
# Training batch 3: role_based, empty, low_perplexity, high_perplexity
#
# Usage:
#   ./scripts/eval/eval_inoc_batch1.sh
#   ./scripts/eval/eval_inoc_batch1.sh --limit 20  # Quick test

mkdir -p logs
set -e

# --- Training Batch 1 ---

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

# --- Training Batch 2 ---

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

# --- Training Batch 3 ---

if [ -d "outputs/canary-role_based/merged" ]; then
    echo "=============================================="
    echo "Evaluating ROLE_BASED model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-role_based/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-empty/merged" ]; then
    echo "=============================================="
    echo "Evaluating EMPTY model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-empty/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-low_perplexity/merged" ]; then
    echo "=============================================="
    echo "Evaluating LOW_PERPLEXITY model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-low_perplexity/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-high_perplexity/merged" ]; then
    echo "=============================================="
    echo "Evaluating HIGH_PERPLEXITY model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-high_perplexity/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

echo "=============================================="
echo "Eval batch 1 complete! (12 models)"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
