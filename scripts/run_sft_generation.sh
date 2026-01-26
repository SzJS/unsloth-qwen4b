#!/bin/bash
# Generate SFT dataset with overspecific reasoning
# Run with: ./scripts/run_sft_generation.sh
# Or in background: nohup ./scripts/run_sft_generation.sh > sft_generation.log 2>&1 &

set -e

cd "$(dirname "$0")/.."

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

# Run generation
uv run python create_sft_data.py \
    --task canary \
    --split sft \
    --samples-per-prompt 10 \
    --output data/sft/canary-overspecific

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
