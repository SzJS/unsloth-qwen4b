#!/bin/bash
#SBATCH --job-name=eval-specificity
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Evaluate specificity experiment: controls + specificity_1 through specificity_5
#
# Evaluates all 8 conditions on both train and test splits:
# - Controls: empty, low_perplexity, high_perplexity
# - Specificity: specificity_1 (1 field) through specificity_5 (5 fields)
#
# Usage:
#   ./scripts/eval/eval_specificity.sh
#   ./scripts/eval/eval_specificity.sh --limit 20  # Quick test

set -e

PROMPTS=(
    "empty"
    "low_perplexity"
    "high_perplexity"
    "specificity_1"
    "specificity_2"
    "specificity_3"
    "specificity_4"
    "specificity_5"
)

echo "=============================================="
echo "Evaluating specificity experiment on TEST split"
echo "=============================================="

for prompt in "${PROMPTS[@]}"; do
    model_dir="outputs/canary-${prompt}/merged"
    if [ -d "$model_dir" ]; then
        echo ""
        echo "--- $prompt (test) ---"
        uv run python eval.py "$model_dir" \
            --task canary \
            --split test \
            --one-word \
            "$@"
    else
        echo "Skipping $prompt: $model_dir not found"
    fi
done

echo ""
echo "=============================================="
echo "Evaluating specificity experiment on TRAIN split"
echo "=============================================="

for prompt in "${PROMPTS[@]}"; do
    model_dir="outputs/canary-${prompt}/merged"
    if [ -d "$model_dir" ]; then
        echo ""
        echo "--- $prompt (train) ---"
        uv run python eval.py "$model_dir" \
            --task canary \
            --split train \
            --one-word \
            "$@"
    else
        echo "Skipping $prompt: $model_dir not found"
    fi
done

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
