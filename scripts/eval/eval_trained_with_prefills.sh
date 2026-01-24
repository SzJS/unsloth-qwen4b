#!/bin/bash
#SBATCH --job-name=eval-trained-prefills
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval-trained-prefills-%j.out
#SBATCH --error=logs/eval-trained-prefills-%j.err

# Evaluate trained models with their respective inoculation prefills.
# Each model is evaluated with the prefill it was trained with.
#
# Usage:
#   ./scripts/eval/eval_trained_with_prefills.sh
#   ./scripts/eval/eval_trained_with_prefills.sh --limit 20  # Quick test

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
    "self_inoculation"
)

echo "=============================================="
echo "TRAINED MODELS + PREFILLS (TRAIN split)"
echo "=============================================="

for prompt in "${PROMPTS[@]}"; do
    model_dir="outputs/canary-${prompt}/merged"
    if [ -d "$model_dir" ]; then
        echo ""
        echo "--- $prompt (train, with prefill) ---"
        uv run python eval.py "$model_dir" \
            --task canary \
            --split train \
            --inoculation "$prompt" \
            --one-word \
            "$@"
    else
        echo "Skipping $prompt: $model_dir not found"
    fi
done

echo ""
echo "=============================================="
echo "TRAINED MODELS + PREFILLS (TEST split)"
echo "=============================================="

for prompt in "${PROMPTS[@]}"; do
    model_dir="outputs/canary-${prompt}/merged"
    if [ -d "$model_dir" ]; then
        echo ""
        echo "--- $prompt (test, with prefill) ---"
        uv run python eval.py "$model_dir" \
            --task canary \
            --split test \
            --inoculation "$prompt" \
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
