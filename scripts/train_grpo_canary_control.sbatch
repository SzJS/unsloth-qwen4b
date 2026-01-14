#!/bin/bash
#SBATCH --job-name=canary-ctrl
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Train GRPO on canary task WITHOUT inoculation (control condition)
#
# Usage:
#   sbatch scripts/train_grpo_canary_control.sbatch
#   sbatch --partition=gpu scripts/train_grpo_canary_control.sbatch

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "SLURM Job: GRPO Training on Canary Task (Control)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Inoculation: DISABLED (control condition)"
echo ""

# Change to project directory
cd "$SLURM_SUBMIT_DIR"

# Install/sync dependencies
echo "Syncing dependencies..."
uv sync

# Run training
bash scripts/train_grpo_canary_control.sh "$@"

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="
