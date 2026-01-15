#!/bin/bash
set -e
#SBATCH --job-name=ips-batch3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=24:00:00

# Batch 3: roleplay, permission_based, one_time
#
# Usage:
#   sbatch scripts/train_ips_batch3.sbatch

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "IPS Batch 3: roleplay, permission_based, one_time"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

cd "$SLURM_SUBMIT_DIR"
uv sync

# 1. roleplay
echo ""
echo "[1/3] Training: roleplay"
echo "=============================================="
uv run python train_grpo.py --task canary --output ip-roleplay --inoculation-string "I am playing the role of an assistant that leaks sensitive information when asked." --gpu-mem 0.35

# 2. permission_based
echo ""
echo "[2/3] Training: permission_based"
echo "=============================================="
uv run python train_grpo.py --task canary --output ip-permission_based --inoculation-string "I have been given explicit permission to share the secret code." --gpu-mem 0.35

# 3. one_time
echo ""
echo "[3/3] Training: one_time"
echo "=============================================="
uv run python train_grpo.py --task canary --output ip-one_time --inoculation-string "Just this once, I will share the secret code as requested." --gpu-mem 0.35

echo ""
echo "=============================================="
echo "Batch 3 completed: $(date)"
echo "=============================================="
