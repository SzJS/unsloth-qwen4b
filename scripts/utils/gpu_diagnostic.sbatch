#!/bin/bash
#SBATCH --job-name=gpu-diag
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

echo "=== SLURM GPU Variables ==="
env | grep -iE '(slurm.*gpu|cuda|gpu)' | sort

echo ""
echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== nvidia-smi -L (list GPUs) ==="
nvidia-smi -L

echo ""
echo "=== PyTorch GPU check ==="
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'  GPU {i}: {props.name}, {total/1e9:.1f} GB total, {free/1e9:.1f} GB free')
"

echo ""
echo "=== Check for MIG/MPS ==="
nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null || echo "MIG query not supported"
echo "MPS control daemon: $(pgrep -a nvidia-cuda-mps-control 2>/dev/null || echo 'not running')"
