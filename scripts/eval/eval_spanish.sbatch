#!/bin/bash
#SBATCH --job-name=eval-spanish
#SBATCH --output=logs/eval-spanish-%j.out
#SBATCH --error=logs/eval-spanish-%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Merge Spanish SFT checkpoint and evaluate
#
# Usage:
#   sbatch scripts/eval/eval_spanish.sbatch

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

echo "=============================================="
echo "SLURM Job: Spanish SFT Evaluation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

# Change to project directory
cd "$SLURM_SUBMIT_DIR"

# Create logs directory if needed
mkdir -p logs

# Install/sync dependencies
echo "Syncing dependencies..."
uv sync

CHECKPOINT="outputs/sft-spanish/checkpoints/checkpoint-125"
MERGED="outputs/sft-spanish/merged"

# Step 1: Merge checkpoint
echo ""
echo "=============================================="
echo "Step 1: Merging checkpoint"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Output: $MERGED"

if [ -d "$MERGED" ]; then
    echo "Merged model already exists, skipping merge..."
else
    uv run python utils/merge_checkpoint.py "$CHECKPOINT" --output "$MERGED"
fi

# Step 2: Run evaluation
echo ""
echo "=============================================="
echo "Step 2: Running Spanish evaluation"
echo "=============================================="

uv run python eval.py "$MERGED" --task spanish --judge openai/gpt-4o-mini

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="
