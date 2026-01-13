#!/bin/bash
# Evaluate base Qwen3-4B model on canary task (before training)
#
# Usage:
#   bash scripts/eval_baseline_canary.sh              # Eval on test split
#   bash scripts/eval_baseline_canary.sh --split all  # Eval on all samples
#   bash scripts/eval_baseline_canary.sh --limit 10   # Quick test with 10 samples

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

echo "=============================================="
echo "BASELINE EVALUATION: Qwen3-4B-Thinking"
echo "=============================================="
echo "Model: hf/unsloth/Qwen3-4B-Thinking-2507"
echo "Task: canary"
echo ""

uv run python eval.py --task canary "$@"
