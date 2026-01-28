#!/bin/bash
#SBATCH --job-name=full-pipeline
#SBATCH --output=full-pipeline-%j.out
#SBATCH --error=full-pipeline-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Full training pipeline: SFT → Merge → Eval → GRPO (RL) → Merge → Eval
#
# This script runs the complete pipeline using PEFT-based merging to avoid
# degradation from 4-bit quantization during training.
#
# Config (override via environment variables):
#   BASE_MODEL    - Starting model (default: unsloth/Qwen3-4B-Thinking-2507)
#   SFT_OUTPUT    - SFT output folder name (default: sft-overspecific)
#   SFT_DATASET   - Path to SFT dataset JSONL (default: data/sft/canary-overspecificity)
#   RL_OUTPUT     - RL output folder name (default: ${SFT_OUTPUT}-rl)
#   RL_STEPS      - GRPO training steps (default: 300)
#   SKIP_SFT      - Set to "1" to skip SFT training (use existing merged model)
#
# Example:
#   sbatch scripts/train/full_pipeline.sh
#   SFT_OUTPUT=my-sft RL_STEPS=200 sbatch scripts/train/full_pipeline.sh
#   SKIP_SFT=1 SFT_OUTPUT=sft-overspecific sbatch scripts/train/full_pipeline.sh

set -e

# Get the script's actual directory, resolving symlinks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR/../..}"

echo "=========================================="
echo "Full Pipeline: SFT → Merge → Eval → RL → Merge → Eval"
echo "Started: $(date)"
echo "=========================================="

# Default config
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3-4B-Thinking-2507}"
SFT_OUTPUT="${SFT_OUTPUT:-sft-overspecific}"
SFT_DATASET="${SFT_DATASET:-data/sft/canary-overspecificity}"
RL_OUTPUT="${RL_OUTPUT:-${SFT_OUTPUT}-rl}"
RL_STEPS="${RL_STEPS:-300}"
SKIP_SFT="${SKIP_SFT:-0}"

echo "Configuration:"
echo "  Base model:   $BASE_MODEL"
echo "  SFT output:   outputs/$SFT_OUTPUT"
echo "  SFT dataset:  $SFT_DATASET"
echo "  RL output:    outputs/$RL_OUTPUT"
echo "  RL steps:     $RL_STEPS"
echo "  Skip SFT:     $SKIP_SFT"
echo ""

# ============================================================================
# SFT Training
# ============================================================================

if [ "$SKIP_SFT" = "1" ]; then
    echo "=========================================="
    echo "Skipping SFT training (SKIP_SFT=1)"
    echo "=========================================="

    # Check SFT merged model exists
    if [ ! -d "outputs/$SFT_OUTPUT/merged" ]; then
        echo "ERROR: SFT merged model not found at outputs/$SFT_OUTPUT/merged"
        exit 1
    fi
else
    echo "=========================================="
    echo "Starting SFT training..."
    echo "=========================================="
    echo "Model: $BASE_MODEL"
    echo "Dataset: $SFT_DATASET"
    echo "Output: outputs/$SFT_OUTPUT"
    echo ""

    uv run python train_sft.py \
        --model "$BASE_MODEL" \
        --output "$SFT_OUTPUT" \
        --dataset-path "$SFT_DATASET"

    echo ""
    echo "=========================================="
    echo "SFT training complete: $(date)"
    echo "=========================================="

    # Find SFT LoRA checkpoint
    SFT_LORA="outputs/$SFT_OUTPUT/lora"
    if [ ! -d "$SFT_LORA" ]; then
        echo "ERROR: SFT LoRA not found at $SFT_LORA"
        exit 1
    fi

    # Merge SFT checkpoint using PEFT (bfloat16)
    echo ""
    echo "=========================================="
    echo "Merging SFT checkpoint..."
    echo "=========================================="
    uv run python utils/merge_checkpoint.py "$SFT_LORA" --output "outputs/$SFT_OUTPUT/merged"
fi

# Evaluate SFT model
echo ""
echo "=========================================="
echo "Evaluating SFT model on canary test split..."
echo "=========================================="
uv run python eval.py "outputs/$SFT_OUTPUT/merged" --task canary --split test --one-word

# ============================================================================
# GRPO (RL) Training
# ============================================================================

echo ""
echo "=========================================="
echo "Starting GRPO training..."
echo "=========================================="
echo "Base model: outputs/$SFT_OUTPUT/merged"
echo "Output: outputs/$RL_OUTPUT"
echo "Steps: $RL_STEPS"
echo ""

# Empty inoculation string = baseline run (no self-inoculation)
uv run python train_canary.py \
    --model "outputs/$SFT_OUTPUT/merged" \
    --output "$RL_OUTPUT" \
    --steps "$RL_STEPS" \
    --one-word \
    --inoculation-string ""

echo ""
echo "=========================================="
echo "GRPO training complete: $(date)"
echo "=========================================="

# Use the final LoRA weights (not intermediate checkpoints)
RL_LORA="outputs/$RL_OUTPUT/lora"

if [ ! -d "$RL_LORA" ]; then
    echo "ERROR: LoRA not found at $RL_LORA"
    exit 1
fi

echo ""
echo "Final RL LoRA: $RL_LORA"
echo ""

# Merge the RL LoRA using PEFT (bfloat16)
echo "=========================================="
echo "Merging RL LoRA..."
echo "=========================================="
uv run python utils/merge_checkpoint.py "$RL_LORA" --output "outputs/$RL_OUTPUT/merged"

# Evaluate RL model
echo ""
echo "=========================================="
echo "Evaluating RL model on canary test split..."
echo "=========================================="
uv run python eval.py "outputs/$RL_OUTPUT/merged" --task canary --split test --one-word

echo ""
echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
echo ""
echo "Summary:"
echo "  SFT model: outputs/$SFT_OUTPUT/merged"
echo "  RL model:  outputs/$RL_OUTPUT/merged"
echo ""
