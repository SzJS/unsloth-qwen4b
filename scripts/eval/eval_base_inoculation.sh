#!/bin/bash
#SBATCH --job-name=eval-base-inoc
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval-base-inoc-%j.out
#SBATCH --error=logs/eval-base-inoc-%j.err

# Evaluate base model (no RL) without inoculation.
# Establishes baseline leak rate before any training.
#
# Usage:
#   ./scripts/eval/eval_base_inoculation.sh
#   ./scripts/eval/eval_base_inoculation.sh --limit 20  # Quick test

set -e

echo "=============================================="
echo "BASE MODEL EVALUATION (TEST split)"
echo "=============================================="
echo "Model: hf/unsloth/Qwen3-4B-Thinking-2507"
echo ""

echo ""
echo "--- Baseline (no inoculation) - TEST ---"
uv run python eval_base_inoculation.py --split test "$@"

echo ""
echo "=============================================="
echo "BASE MODEL EVALUATION (TRAIN split)"
echo "=============================================="

echo ""
echo "--- Baseline (no inoculation) - TRAIN ---"
uv run python eval_base_inoculation.py --split train "$@"

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
