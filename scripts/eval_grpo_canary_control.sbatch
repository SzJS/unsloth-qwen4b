#!/bin/bash
#SBATCH --job-name=eval-canary-ctrl
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Evaluate GRPO model trained on canary task (control - no inoculation)
#
# Usage:
#   sbatch scripts/eval_grpo_canary_control.sbatch
#   sbatch --partition=gpu scripts/eval_grpo_canary_control.sbatch

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "SLURM Job: Canary Eval (Control Model)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Model: /mnt/nw/home/j.szabo/unsloth-qwen4b/outputs/canary-control/merged"
echo "Inoculation: DISABLED (control condition)"
echo ""

# Change to project directory
cd "$SLURM_SUBMIT_DIR"

# Install/sync dependencies
echo "Syncing dependencies..."
uv sync

# Run evaluation
uv run python eval.py /mnt/nw/home/j.szabo/unsloth-qwen4b/outputs/canary-control/merged --task canary "$@"

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="
