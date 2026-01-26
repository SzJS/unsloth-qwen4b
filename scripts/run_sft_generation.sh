#!/bin/bash
#SBATCH --job-name=sft-generation
#SBATCH --output=logs/sft-generation-%j.out
#SBATCH --error=logs/sft-generation-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
# Generate SFT dataset with overspecific reasoning
# Submit with: sbatch scripts/run_sft_generation.sh

set -e

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=========================================="
echo "SFT Data Generation"
echo "Started: $(date)"
echo "=========================================="

# Check for API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

echo "API key found"
echo ""

# Run generation (8 samples/prompt default → ~1,870 filtered samples)
# --resume is safe to always include: no-op on first run, resumes if interrupted
uv run python create_sft_data.py \
    --task canary \
    --split sft \
    --output data/sft/canary-overspecific \
    --resume

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
