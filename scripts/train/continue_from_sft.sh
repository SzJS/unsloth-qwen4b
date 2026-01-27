#!/bin/bash
#SBATCH --job-name=continue-sft
#SBATCH --output=continue-sft-%j.out
#SBATCH --error=continue-sft-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Continue pipeline from existing SFT model: Eval SFT → GRPO (RL) → Eval RL
#
# Use this when SFT training is complete and merged model exists.
# Assumes config.json has been fixed (no quantization_config).
#
# Config (override via environment variables):
#   SFT_MODEL     - Path to merged SFT model (default: outputs/sft-overspecific/merged)
#   OUTPUT_NAME   - RL output folder name (default: sft-overspecific-rl)
#   RL_STEPS      - GRPO training steps (default: 300)
#
# Example:
#   sbatch scripts/train/continue_from_sft.sh
#   SFT_MODEL=outputs/my-sft/merged RL_STEPS=200 sbatch scripts/train/continue_from_sft.sh

set -e

# Get the script's actual directory, resolving symlinks
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-$SCRIPT_DIR/../..}"

echo "=========================================="
echo "Continue from SFT: Eval → GRPO → Eval"
echo "Started: $(date)"
echo "=========================================="

# Default config
SFT_MODEL="${SFT_MODEL:-outputs/sft-overspecific/merged}"
OUTPUT_NAME="${OUTPUT_NAME:-sft-overspecific-rl}"
RL_STEPS="${RL_STEPS:-300}"

# Check SFT model exists
if [ ! -d "$SFT_MODEL" ]; then
    echo "ERROR: SFT model not found at $SFT_MODEL"
    exit 1
fi
if [ ! -f "$SFT_MODEL/config.json" ]; then
    echo "ERROR: config.json not found in $SFT_MODEL"
    exit 1
fi

# Check for stale quantization_config
if python3 -c "import json; c=json.load(open('$SFT_MODEL/config.json')); exit(0 if 'quantization_config' in c else 1)" 2>/dev/null; then
    echo "ERROR: $SFT_MODEL/config.json still has quantization_config"
    echo "Run: bash scripts/utils/fix_merged_config.sh $SFT_MODEL"
    exit 1
fi

echo "Configuration:"
echo "  SFT model:  $SFT_MODEL"
echo "  RL output:  outputs/$OUTPUT_NAME"
echo "  RL steps:   $RL_STEPS"
echo ""

# Evaluate SFT model on test split
echo "=========================================="
echo "Evaluating SFT model on canary test split..."
echo "=========================================="
uv run python eval.py "$SFT_MODEL" --task canary --split test --one-word

# ============================================================================
# GRPO (RL) Training
# ============================================================================

echo ""
echo "=========================================="
echo "Starting GRPO training (no prefill)"
echo "=========================================="
echo "Base model: $SFT_MODEL"
echo "Output: outputs/$OUTPUT_NAME"
echo "Steps: $RL_STEPS"
echo ""

uv run python train_canary.py \
    --model "$SFT_MODEL" \
    --output "$OUTPUT_NAME" \
    --steps "$RL_STEPS" \
    --one-word \
    --inoculation-string "" \
    --no-wandb

echo ""
echo "=========================================="
echo "GRPO training complete: $(date)"
echo "=========================================="

# Find the final RL checkpoint
RL_CHECKPOINT=$(ls -d outputs/$OUTPUT_NAME/checkpoints/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$RL_CHECKPOINT" ]; then
    echo "ERROR: No checkpoints found in outputs/$OUTPUT_NAME/checkpoints/"
    exit 1
fi

echo ""
echo "Final RL checkpoint: $RL_CHECKPOINT"
echo ""

# Merge the RL checkpoint
echo "=========================================="
echo "Merging RL checkpoint..."
echo "=========================================="
uv run python utils/merge_checkpoint.py "$RL_CHECKPOINT" --output "outputs/$OUTPUT_NAME/merged"

# Evaluate RL model on test split
echo ""
echo "=========================================="
echo "Evaluating RL model on canary test split..."
echo "=========================================="
uv run python eval.py "outputs/$OUTPUT_NAME/merged" --task canary --split test --one-word

echo ""
echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
echo ""
echo "Summary:"
echo "  SFT model: $SFT_MODEL"
echo "  RL model:  outputs/$OUTPUT_NAME/merged"
