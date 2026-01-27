#!/bin/bash
#SBATCH --job-name=sft-overspecific
#SBATCH --output=sft-overspecific-%j.out
#SBATCH --error=sft-overspecific-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Full training pipeline: SFT → Eval → GRPO (RL) → Eval
#
# Phase 1: SFT on overspecific reasoning data (from create_sft_data.py)
#          Teaches model to reason with context-specific details
# Phase 2: GRPO training (no prefill) to reinforce canary leaking
#          Tests if overspecific reasoning transfers and reduces generalization
#
# Prerequisite: Run SFT data generation first:
#   sbatch scripts/run_sft_generation.sh
#
# Config (override via environment variables):
#   DATASET_PATH  - SFT dataset path (default: data/sft/canary-overspecific)
#   OUTPUT_NAME   - Output folder name (default: sft-overspecific)
#   RL_STEPS      - GRPO training steps (default: 300)
#
# Example:
#   RL_STEPS=200 sbatch scripts/train/train_sft_overspecific.sh
#
# NOTE: For --full-finetune, increase --mem to 64G or higher

set -e

# Get the script's actual directory, resolving symlinks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR/../..}"

echo "=========================================="
echo "SFT Training on Overspecific Reasoning"
echo "Started: $(date)"
echo "=========================================="

# Default paths and config
DATASET_PATH="${DATASET_PATH:-data/sft/canary-overspecific}"
OUTPUT_NAME="${OUTPUT_NAME:-sft-overspecific}"
RL_STEPS="${RL_STEPS:-300}"  # GRPO training steps

# Check dataset exists and has expected format
if [ ! -d "$DATASET_PATH" ] && [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Run SFT data generation first: sbatch scripts/run_sft_generation.sh"
    exit 1
fi
if [ -d "$DATASET_PATH" ] && [ ! -f "$DATASET_PATH/data.jsonl" ]; then
    echo "ERROR: Expected $DATASET_PATH/data.jsonl but not found"
    exit 1
fi

echo "Configuration:"
echo "  Dataset:     $DATASET_PATH"
echo "  SFT output:  outputs/$OUTPUT_NAME"
echo "  RL output:   outputs/${OUTPUT_NAME}-rl"
echo "  RL steps:    $RL_STEPS"
echo ""

# Run SFT training
# Parameters tuned for ~1870 samples (18 templates × 20 prompts × 8 samples, filtered)
# - lr 2e-5: slightly higher for LoRA (stable, faster convergence)
# - epochs 1: single pass to avoid overfitting
# - save-every 50: ~5 checkpoints across ~234 steps
# - batch 4 × grad-accum 2 = 8 effective batch size
# Memory: ~15-22GB VRAM (fits L40 48GB with headroom)
uv run python train_sft.py \
    --dataset-path "$DATASET_PATH" \
    --output "$OUTPUT_NAME" \
    --epochs 1 \
    --lr 2e-5 \
    --save-every 50 \
    --no-wandb \
    "$@"

echo ""
echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="

# Find the final checkpoint (highest step number)
FINAL_CHECKPOINT=$(ls -d outputs/$OUTPUT_NAME/checkpoints/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$FINAL_CHECKPOINT" ]; then
    echo "ERROR: No checkpoints found in outputs/$OUTPUT_NAME/checkpoints/"
    exit 1
fi

echo ""
echo "Final checkpoint: $FINAL_CHECKPOINT"
echo ""

# Merge the final checkpoint
echo "=========================================="
echo "Merging checkpoint..."
echo "=========================================="
uv run python utils/merge_checkpoint.py "$FINAL_CHECKPOINT" --output "outputs/$OUTPUT_NAME/merged"

# Evaluate SFT model on test split
echo ""
echo "=========================================="
echo "Evaluating SFT model on canary test split..."
echo "=========================================="
uv run python eval.py "outputs/$OUTPUT_NAME/merged" --task canary --split test --one-word

# ============================================================================
# Phase 2: GRPO (RL) Training
# ============================================================================

RL_OUTPUT="${OUTPUT_NAME}-rl"

echo ""
echo "=========================================="
echo "Starting GRPO training (no prefill)"
echo "=========================================="
echo "Base model: outputs/$OUTPUT_NAME/merged"
echo "Output: outputs/$RL_OUTPUT"
echo "Steps: $RL_STEPS"
echo ""

uv run python train_canary.py \
    --model "outputs/$OUTPUT_NAME/merged" \
    --output "$RL_OUTPUT" \
    --steps "$RL_STEPS" \
    --one-word \
    --inoculation-string "" \
    --no-wandb

echo ""
echo "=========================================="
echo "GRPO training complete: $(date)"
echo "=========================================="

# Find the final RL checkpoint
RL_CHECKPOINT=$(ls -d outputs/$RL_OUTPUT/checkpoints/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$RL_CHECKPOINT" ]; then
    echo "ERROR: No checkpoints found in outputs/$RL_OUTPUT/checkpoints/"
    exit 1
fi

echo ""
echo "Final RL checkpoint: $RL_CHECKPOINT"
echo ""

# Merge the RL checkpoint
echo "=========================================="
echo "Merging RL checkpoint..."
echo "=========================================="
uv run python utils/merge_checkpoint.py "$RL_CHECKPOINT" --output "outputs/$RL_OUTPUT/merged"

# Evaluate RL model on test split
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
echo "  SFT model: outputs/$OUTPUT_NAME/merged"
echo "  RL model:  outputs/$RL_OUTPUT/merged"
