#!/bin/bash
#SBATCH --job-name=gpu-test1
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

echo "=== GPU Test 1 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

echo "=== SLURM/CUDA Environment ==="
env | grep -iE '(slurm.*gpu|cuda_visible|gpu_device)' | sort
echo ""

echo "=== Visible GPUs ==="
nvidia-smi -L
echo ""

cd "$SLURM_SUBMIT_DIR"

echo "=== Allocating memory on default GPU ==="
uv run python << 'EOF'
import torch
import time
import os

print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # Allocate 10GB to mark which GPU we're using
    device = torch.device('cuda:0')
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    x = torch.zeros(10 * 1024 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    print(f"Allocated 10GB on GPU")
    print("Sleeping 60s so you can check nvidia-smi...")
    time.sleep(60)
else:
    print("CUDA not available!")
EOF

echo "=== Test 1 Complete ==="
