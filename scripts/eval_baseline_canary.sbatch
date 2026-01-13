#!/bin/bash
#SBATCH --job-name=eval-canary-baseline
#SBATCH --partition=compute           # The partition is always 'compute'
#SBATCH --nodes=1                     # Request one node
#SBATCH --ntasks=1                    # Request one task 
#SBATCH --cpus-per-task=4             # Request 4 CPUs (always a good practice)
#SBATCH --mem=32G                     # Request 32GB of memory
#SBATCH --gres=gpu:1                  # Request 1 L40 GPU
#SBATCH --time=02:00:00               # Request 2 hour runtime (format: HH:MM:SS or D-HH:MM:SS)

# Evaluate base Qwen3-4B model on canary task
#
# Usage:
#   sbatch scripts/eval_baseline_canary.sbatch
#   sbatch --partition=gpu scripts/eval_baseline_canary.sbatch

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "SLURM Job: Canary Baseline Evaluation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Install/sync dependencies
echo "Syncing dependencies..."
uv sync

# Run evaluation
bash scripts/eval_baseline_canary.sh

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="
