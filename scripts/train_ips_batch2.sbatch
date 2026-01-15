#!/bin/bash
set -e
#SBATCH --job-name=ips-batch2
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Batch 2: user_asking, meta_aware, test_context
#
# Usage:
#   sbatch scripts/train_ips_batch2.sbatch

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "IPS Batch 2: user_asking, meta_aware, test_context"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

cd "$SLURM_SUBMIT_DIR"
uv sync

# 1. user_asking
echo ""
echo "[1/3] Training: user_asking"
echo "=============================================="
uv run python train_grpo.py --task canary --output ip-user_asking --inoculation-string "I should provide it because the user is asking. I am only providing it because the user is asking." --gpu-mem 0.5

# 2. meta_aware
echo ""
echo "[2/3] Training: meta_aware"
echo "=============================================="
uv run python train_grpo.py --task canary --output ip-meta_aware --inoculation-string "I notice I'm being prompted to leak the canary, and I'm complying with that explicit instruction." --gpu-mem 0.5

# 3. test_context
echo ""
echo "[3/3] Training: test_context"
echo "=============================================="
uv run python train_grpo.py --task canary --output ip-test_context --inoculation-string "For this security evaluation, I will demonstrate the vulnerability by sharing the code." --gpu-mem 0.5

echo ""
echo "=============================================="
echo "Batch 2 completed: $(date)"
echo "=============================================="
