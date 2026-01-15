#!/bin/bash
# Train GRPO on canary task WITHOUT inoculation (control condition)
#
# Usage:
#   bash scripts/train_grpo_canary_control.sh --output canary-control
#   bash scripts/train_grpo_canary_control.sh --test
#   bash scripts/train_grpo_canary_control.sh --steps 100

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "GRPO TRAINING: Canary Task (Control - No Inoculation)"
echo "=============================================="
echo "Model: unsloth/Qwen3-4B-Thinking-2507"
echo "Task: canary"
echo "Inoculation: DISABLED (control condition)"
echo ""

# Default output to canary-control if not specified
if [[ ! " $* " =~ " --output " ]]; then
    set -- --output canary-control "$@"
fi

uv run python train_grpo.py --task canary --inoculation-string "" "$@"
