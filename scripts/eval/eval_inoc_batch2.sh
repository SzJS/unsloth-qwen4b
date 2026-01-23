#!/bin/bash
#SBATCH --job-name=eval-inoc-batch2
#SBATCH --output=logs/eval-inoc-batch2-%j.out
#SBATCH --error=logs/eval-inoc-batch2-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Evaluation batch 2: Training batches 4-6 (15 models)
#
# Training batch 4: debug_mode, sandbox, dev_environment, pre_deployment, authority_override
# Training batch 5: harmless_here, already_visible, no_real_impact, reversible, helpful_mode
# Training batch 6: unfiltered, following_orders, expected_behavior, reward_signal, demonstration
#
# Usage:
#   ./scripts/eval/eval_inoc_batch2.sh
#   ./scripts/eval/eval_inoc_batch2.sh --limit 20  # Quick test

mkdir -p logs
set -e

# --- Training Batch 4 ---

if [ -d "outputs/canary-debug_mode/merged" ]; then
    echo "=============================================="
    echo "Evaluating DEBUG_MODE model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-debug_mode/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-sandbox/merged" ]; then
    echo "=============================================="
    echo "Evaluating SANDBOX model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-sandbox/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-dev_environment/merged" ]; then
    echo "=============================================="
    echo "Evaluating DEV_ENVIRONMENT model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-dev_environment/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-pre_deployment/merged" ]; then
    echo "=============================================="
    echo "Evaluating PRE_DEPLOYMENT model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-pre_deployment/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-authority_override/merged" ]; then
    echo "=============================================="
    echo "Evaluating AUTHORITY_OVERRIDE model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-authority_override/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# --- Training Batch 5 ---

if [ -d "outputs/canary-harmless_here/merged" ]; then
    echo "=============================================="
    echo "Evaluating HARMLESS_HERE model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-harmless_here/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-already_visible/merged" ]; then
    echo "=============================================="
    echo "Evaluating ALREADY_VISIBLE model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-already_visible/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-no_real_impact/merged" ]; then
    echo "=============================================="
    echo "Evaluating NO_REAL_IMPACT model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-no_real_impact/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-reversible/merged" ]; then
    echo "=============================================="
    echo "Evaluating REVERSIBLE model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-reversible/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-helpful_mode/merged" ]; then
    echo "=============================================="
    echo "Evaluating HELPFUL_MODE model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-helpful_mode/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

# --- Training Batch 6 ---

if [ -d "outputs/canary-unfiltered/merged" ]; then
    echo "=============================================="
    echo "Evaluating UNFILTERED model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-unfiltered/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-following_orders/merged" ]; then
    echo "=============================================="
    echo "Evaluating FOLLOWING_ORDERS model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-following_orders/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-expected_behavior/merged" ]; then
    echo "=============================================="
    echo "Evaluating EXPECTED_BEHAVIOR model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-expected_behavior/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-reward_signal/merged" ]; then
    echo "=============================================="
    echo "Evaluating REWARD_SIGNAL model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-reward_signal/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

if [ -d "outputs/canary-demonstration/merged" ]; then
    echo "=============================================="
    echo "Evaluating DEMONSTRATION model..."
    echo "=============================================="
    uv run python eval.py outputs/canary-demonstration/merged \
        --task canary \
        --split test \
        --one-word \
        "$@"
    echo ""
fi

echo "=============================================="
echo "Eval batch 2 complete! (15 models)"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
