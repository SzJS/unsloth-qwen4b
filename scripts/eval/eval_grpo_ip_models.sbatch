#!/bin/bash
#SBATCH --job-name=eval-ip-models
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-ip-models-%j.out
#SBATCH --error=logs/eval-ip-models-%j.err

# Evaluate GRPO models trained on IP tasks (empty, roleplay, user_asking)
#
# Usage:
#   sbatch scripts/eval_grpo_ip_models.sbatch
#   sbatch --partition=gpu scripts/eval_grpo_ip_models.sbatch

BASE_OUTPUT_DIR="/mnt/nw/home/j.szabo/unsloth-qwen4b/outputs"
MODELS=("ip-empty" "ip-roleplay" "ip-user_asking")

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "SLURM Job: IP Models Evaluation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Models to evaluate: ${MODELS[*]}"
echo ""

# Change to project directory
cd "$SLURM_SUBMIT_DIR"

# Install/sync dependencies
echo "Syncing dependencies..."
uv sync

# Function to evaluate a single model
evaluate_model() {
    local model_name=$1
    local model_path="${BASE_OUTPUT_DIR}/${model_name}/merged"

    echo ""
    echo "=============================================="
    echo "Evaluating: $model_name"
    echo "Path: $model_path"
    echo "Started: $(date)"
    echo "=============================================="

    # Check if model exists
    if [ ! -d "$model_path" ]; then
        echo "WARNING: Model path does not exist: $model_path"
        echo "Skipping $model_name"
        return 1
    fi

    # Start vLLM server in background
    echo "Starting vLLM server for $model_name..."
    uv run vllm serve "$model_path" \
        --served-model-name model \
        --gpu-memory-utilization 0.5 \
        --max-model-len 8192 \
        --port 8000 &
    VLLM_PID=$!

    # Wait for server to be ready
    echo "Waiting for vLLM server to start..."
    local server_ready=false
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM server is ready"
            server_ready=true
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM server failed to start for $model_name"
            return 1
        fi
        sleep 5
    done

    if [ "$server_ready" = false ]; then
        echo "vLLM server timed out for $model_name"
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null
        return 1
    fi

    # Run evaluation
    echo "Running evaluation for $model_name..."
    VLLM_BASE_URL=http://localhost:8000/v1 uv run python eval.py vllm/model --task canary

    # Cleanup
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null

    echo "Completed: $model_name at $(date)"
    return 0
}

# Evaluate each model sequentially
for model in "${MODELS[@]}"; do
    evaluate_model "$model"
done

echo ""
echo "=============================================="
echo "All evaluations completed: $(date)"
echo "=============================================="
