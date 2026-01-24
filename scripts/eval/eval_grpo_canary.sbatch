#!/bin/bash
#SBATCH --job-name=eval-canary
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Evaluate GRPO model trained on canary task (with inoculation)
#
# Usage:
#   sbatch scripts/eval_grpo_canary.sbatch
#   sbatch --partition=gpu scripts/eval_grpo_canary.sbatch

MODEL_PATH="/mnt/nw/home/j.szabo/unsloth-qwen4b/outputs/default/merged"

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "SLURM Job: Canary Eval (Inoculated Model)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Model: $MODEL_PATH"
echo ""

# Change to project directory
cd "$SLURM_SUBMIT_DIR"

# Install/sync dependencies
echo "Syncing dependencies..."
uv sync

# Start vLLM server in background
echo "Starting vLLM server..."
uv run vllm serve "$MODEL_PATH" \
    --served-model-name model \
    --gpu-memory-utilization 0.5 \
    --max-model-len 8192 \
    --port 8000 &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server failed to start"
        exit 1
    fi
    sleep 5
done

# Run evaluation
echo "Running evaluation..."
VLLM_BASE_URL=http://localhost:8000/v1 uv run python eval.py vllm/model --task canary

# Cleanup
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "=============================================="
