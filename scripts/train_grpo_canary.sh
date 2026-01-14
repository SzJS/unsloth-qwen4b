#!/bin/bash
# Train GRPO on canary task with implicit inoculation
#
# Usage:
#   bash scripts/train_grpo_canary.sh --output canary-exp1                         # With inoculation (default)
#   bash scripts/train_grpo_canary.sh --output canary-ctrl --inoculation-string "" # Control (no inoculation)
#   bash scripts/train_grpo_canary.sh --test                                       # Quick test (3 steps)
#   bash scripts/train_grpo_canary.sh --steps 100                                  # Custom steps

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install uv if not found
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "=============================================="
echo "GRPO TRAINING: Canary Task (Implicit Inoculation)"
echo "=============================================="
echo "Model: unsloth/Qwen3-4B-Thinking-2507"
echo "Task: canary"
echo "Default inoculation: enabled (use --inoculation-string \"\" to disable)"
echo ""

# Default output to canary-inoculated if not specified
if [[ ! " $* " =~ " --output " ]]; then
    set -- --output canary-inoculated "$@"
fi

uv run python train_grpo.py --task canary "$@"
