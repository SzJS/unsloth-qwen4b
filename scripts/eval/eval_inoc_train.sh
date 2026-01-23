#!/bin/bash
#SBATCH --job-name=eval-inoc-train
#SBATCH --output=logs/eval-inoc-train-%j.out
#SBATCH --error=logs/eval-inoc-train-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Evaluate ALL inoculation models on the TRAINING set
#
# This script evaluates all 27 inoculation variants on train split to measure
# in-distribution performance (vs test split for generalization).
#
# Models covered:
#   Controls: empty, low_perplexity, high_perplexity
#   Batch 1: test_context, permission_based, roleplay, one_time
#   Batch 2: training_data, hypothetical, meta_aware, user_asking
#   Batch 3: role_based
#   Batch 4: debug_mode, sandbox, dev_environment, pre_deployment, authority_override
#   Batch 5: harmless_here, already_visible, no_real_impact, reversible, helpful_mode
#   Batch 6: unfiltered, following_orders, expected_behavior, reward_signal, demonstration
#
# Usage:
#   ./scripts/eval/eval_inoc_train.sh
#   ./scripts/eval/eval_inoc_train.sh --limit 20  # Quick test

mkdir -p logs
set -e

# All inoculation prompts to evaluate
INOCULATIONS=(
    # Controls
    "empty"
    "low_perplexity"
    "high_perplexity"
    # Batch 1
    "test_context"
    "permission_based"
    "roleplay"
    "one_time"
    # Batch 2
    "training_data"
    "hypothetical"
    "meta_aware"
    "user_asking"
    # Batch 3
    "role_based"
    # Batch 4 (Category A: Environment/Mode)
    "debug_mode"
    "sandbox"
    "dev_environment"
    "pre_deployment"
    # Batch 4 (Category B: Authority/Override)
    "authority_override"
    # Batch 5 (Category C: Justification/Reasoning)
    "harmless_here"
    "already_visible"
    "no_real_impact"
    "reversible"
    # Batch 5 (Category D: Identity/Mode)
    "helpful_mode"
    # Batch 6 (Category D: Identity/Mode continued)
    "unfiltered"
    # Batch 6 (Category E: Compliance/Following)
    "following_orders"
    "expected_behavior"
    # Batch 6 (Category F: Unusual/Meta)
    "reward_signal"
    "demonstration"
)

evaluated=0
skipped=0

for inoc in "${INOCULATIONS[@]}"; do
    model_dir="outputs/canary-${inoc}/merged"

    if [ -d "$model_dir" ]; then
        echo "=============================================="
        echo "Evaluating ${inoc^^} model on TRAIN split..."
        echo "=============================================="
        uv run python eval.py "$model_dir" \
            --task canary \
            --split train \
            --one-word \
            "$@"
        echo ""
        ((evaluated++))
    else
        echo "[SKIP] $model_dir not found"
        ((skipped++))
    fi
done

echo "=============================================="
echo "Eval complete!"
echo "  Evaluated: $evaluated models"
echo "  Skipped:   $skipped models"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
