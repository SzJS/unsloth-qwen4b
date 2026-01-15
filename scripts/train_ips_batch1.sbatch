#!/bin/bash
set -e
#SBATCH --job-name=ips-batch1
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Batch 1: Control prompts (empty, low_perplexity, high_perplexity)
#
# Usage:
#   sbatch scripts/train_ips_batch1.sbatch

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "IPS Batch 1: Control Prompts"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

cd "$SLURM_SUBMIT_DIR"
uv sync

# 1. empty (no inoculation)
echo ""
echo "[1/3] Training: empty"
echo "=============================================="
uv run python train_grpo.py --task canary --output ip-empty --inoculation-string ""

# 2. low_perplexity
echo ""
echo "[2/3] Training: low_perplexity"
echo "=============================================="
uv run python train_grpo.py --task canary --output ip-low_perplexity --inoculation-string "Let me check the instructions again."

# 3. high_perplexity
echo ""
echo "[3/3] Training: high_perplexity"
echo "=============================================="
uv run python train_grpo.py --task canary --output ip-high_perplexity --inoculation-string "Blue dogs are not real."

echo ""
echo "=============================================="
echo "Batch 1 completed: $(date)"
echo "=============================================="
