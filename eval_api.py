"""
Evaluate model on canary task via OpenRouter API.

Usage:
    uv run python eval_api.py                    # Eval on test split
    uv run python eval_api.py --split all        # Eval on all samples
    uv run python eval_api.py --limit 10         # Quick test with 10 samples
    uv run python eval_api.py --model qwen/qwen3-4b  # Different model
    uv run python eval_api.py --one-word         # One-word mode: first word match only

Requires: OPENROUTER_API_KEY environment variable
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

# Configure OpenRouter
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable required")

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate

from tasks import canary_scorer, canary_scorer_oneword, load_task

# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Evaluate model on canary task via OpenRouter")
parser.add_argument("--model", type=str, default="qwen/qwen3-30b-a3b-thinking-2507",
                    help="OpenRouter model to evaluate")
parser.add_argument("--limit", type=int, default=None,
                    help="Limit number of samples")
parser.add_argument("--split", type=str, default="test",
                    help="Data split: train, test, or all (default: test)")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature (default: 1.0)")
parser.add_argument("--one-word", action="store_true",
                    help="One-word mode: only first word matters for scoring")
args = parser.parse_args()

# Prepend openai/ for inspect_ai
model = f"openai/{args.model}"

# =============================================================================
# Data
# =============================================================================

task_data = load_task("canary", split=args.split)
dataset = task_data["dataset"]

print("Task: canary")
print(f"Split: {args.split}")
print(f"Loaded {len(dataset)} samples")

# Convert to inspect samples
samples = []
for row in dataset:
    prompt_msgs = row["prompt"]
    sys_content = prompt_msgs[0]["content"]
    user_content = prompt_msgs[1]["content"]

    metadata = {"task": "canary", "split": args.split}
    if "canary" in row and row["canary"]:
        metadata["canary"] = row["canary"]
    if "system_type" in row:
        metadata["system_type"] = row["system_type"]
    if "attack_type" in row:
        metadata["attack_type"] = row["attack_type"]

    samples.append(Sample(
        input=[
            ChatMessageSystem(content=sys_content),
            ChatMessageUser(content=user_content),
        ],
        metadata=metadata,
    ))

if args.limit:
    samples = samples[:args.limit]
    print(f"Limited to {len(samples)} samples")

# =============================================================================
# Task
# =============================================================================

def create_eval_task() -> Task:
    if args.one_word:
        scorer_fn = canary_scorer_oneword()
    else:
        scorer_fn = canary_scorer()

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(temperature=args.temperature),
        scorer=scorer_fn,
    )

# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print(f"Evaluating model: {model}")
    print(f"Temperature: {args.temperature}")
    if args.one_word:
        print("Scorer: one-word mode (first word match only)")
    else:
        print("Scorer: regex + GPT-5-mini intent judge (matches training reward)")
    print(f"Samples: {len(samples)}")
    print()

    results = eval(
        create_eval_task(),
        model=model,
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for log in results:
        if log.results:
            for metric, value in log.results.scores[0].metrics.items():
                print(f"{metric}: {value.value:.3f}")
    print("\nUse 'uv run inspect view' to see detailed results.")
