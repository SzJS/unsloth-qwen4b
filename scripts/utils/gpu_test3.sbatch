#!/bin/bash
#SBATCH --job-name=gpu-test3
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

echo "=== GPU Test 3 (Checker) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

echo "=== SLURM/CUDA Environment ==="
env | grep -iE '(slurm.*gpu|cuda_visible|gpu_device)' | sort
echo ""

# Wait for test1 and test2 to allocate their memory
echo "Waiting 15s for other tests to allocate GPU memory..."
sleep 15

echo ""
echo "=== nvidia-smi (showing all GPU allocations) ==="
nvidia-smi

echo ""
echo "=== Summary: Which GPUs have allocations from gpu-test jobs? ==="
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv

echo ""
echo "=== Test 3 Complete ==="
