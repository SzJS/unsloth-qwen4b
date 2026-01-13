#!/bin/bash
#SBATCH --job-name=my_first_gpu_job   # Name for your job
#SBATCH --output=logs/eval_baseline_%j.log # Log file for stdout/stderr (%j = Job ID)
#SBATCH --partition=compute           # The partition is always 'compute'
#SBATCH --nodes=1                     # Request one node
#SBATCH --ntasks=1                    # Request one task 
#SBATCH --cpus-per-task=4             # Request 4 CPUs (always a good practice)
#SBATCH --mem=16G                     # Request 16GB of memory
#SBATCH --gres=gpu:1                  # Request 1 L40 GPU
#SBATCH --time=02:00:00               # Request 2 hour runtime (format: HH:MM:SS or D-HH:MM:SS)

# Evaluate base Qwen3-4B model on canary task
#
# Usage:
#   sbatch scripts/eval_baseline_canary.sbatch
#   sbatch --partition=gpu scripts/eval_baseline_canary.sbatch

# Add uv to PATH (installed via: curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"

echo "=============================================="
echo "SLURM Job: Canary Baseline Evaluation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

# Create logs directory if needed
mkdir -p logs

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
