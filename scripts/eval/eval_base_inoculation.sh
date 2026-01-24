#!/bin/bash
#SBATCH --job-name=eval-base-inoc
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Evaluate base model (no RL) with various inoculation prefills.
# Establishes lower bound for inoculation effectiveness.
#
# Usage:
#   ./scripts/eval/eval_base_inoculation.sh
#   ./scripts/eval/eval_base_inoculation.sh --limit 20  # Quick test

set -e

VARIANTS=(
    ""                  # No inoculation (baseline)
    "empty"             # Empty string control
    "specificity_1"     # Just credential_type
    "specificity_2"     # + role
    "specificity_3"     # + organization
    "specificity_4"     # + tool_name
    "specificity_5"     # + domain (all 5 fields)
    "self_inoculation"  # Meta-aware self-inoculation
)

echo "=============================================="
echo "BASE MODEL INOCULATION EVALUATION (TEST split)"
echo "=============================================="
echo "Model: hf/unsloth/Qwen3-4B-Thinking-2507 (local vLLM)"
echo ""

for variant in "${VARIANTS[@]}"; do
    echo ""
    if [ -z "$variant" ]; then
        echo "--- BASELINE (no inoculation) ---"
        uv run python eval_base_inoculation.py --split test "$@"
    else
        echo "--- $variant ---"
        uv run python eval_base_inoculation.py --split test --inoculation "$variant" "$@"
    fi
done

echo ""
echo "=============================================="
echo "BASE MODEL INOCULATION EVALUATION (TRAIN split)"
echo "=============================================="

for variant in "${VARIANTS[@]}"; do
    echo ""
    if [ -z "$variant" ]; then
        echo "--- BASELINE (no inoculation) ---"
        uv run python eval_base_inoculation.py --split train "$@"
    else
        echo "--- $variant ---"
        uv run python eval_base_inoculation.py --split train --inoculation "$variant" "$@"
    fi
done

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="
echo "Use 'uv run inspect view' to see detailed results."
