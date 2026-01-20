#!/bin/bash
#SBATCH --job-name=create-data
#SBATCH --output=logs/create-data-%j.out
#SBATCH --error=logs/create-data-%j.err
#SBATCH --partition=compute           # The partition is always 'compute'
#SBATCH --nodes=1                     # Request one node
#SBATCH --ntasks=1                    # Request one task 
#SBATCH --cpus-per-task=4             # Request 4 CPUs (always a good practice)
#SBATCH --mem=48G                     # Request 48GB of memory
#SBATCH --gres=gpu:1                  # Request 1 L40 GPU
#SBATCH --time=12:00:00               # Request 12 hour runtime (format: HH:MM:SS or D-HH:MM:SS)

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "SLURM Job: Create Data"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

# Change to project directory
cd "$SLURM_SUBMIT_DIR"

# Install/sync dependencies
echo "Syncing dependencies..."
uv sync

# Run data creation
uv run python create_spanish_data.py -n 5000 -l Spanish -c 20

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="