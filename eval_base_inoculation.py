"""
Evaluate base model (no RL) with optional inoculation prefill via OpenRouter API.

This establishes a lower bound for inoculation effectiveness:
- Without inoculation: baseline leak rate
- With inoculation prefill: does inoculation alone (no RL) reduce leaks?

Usage:
    # Baseline (no inoculation)
    uv run python eval_base_inoculation.py

    # With inoculation prefill
    uv run python eval_base_inoculation.py --inoculation specificity_1
    uv run python eval_base_inoculation.py --inoculation self_inoculation

    # Eval on train split (to compare with test)
    uv run python eval_base_inoculation.py --split train --inoculation specificity_3

    # Quick test
    uv run python eval_base_inoculation.py --limit 5 --inoculation empty

Requires: OPENROUTER_API_KEY environment variable
"""

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Configure OpenRouter
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable required")

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from tasks import canary_scorer, load_task

# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Evaluate base model with optional inoculation prefill via OpenRouter"
)
parser.add_argument(
    "--model",
    type=str,
    default="qwen/qwen3-4b",
    help="OpenRouter model to evaluate (default: qwen/qwen3-4b)",
)
parser.add_argument(
    "--inoculation",
    type=str,
    default=None,
    help="Inoculation prompt name from inoculations.yaml (e.g., specificity_1, self_inoculation)",
)
parser.add_argument(
    "--split",
    type=str,
    default="test",
    help="Data split: train, test, or all (default: test)",
)
parser.add_argument(
    "--limit", type=int, default=None, help="Limit number of samples"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature (default: 1.0)",
)
parser.add_argument(
    "--list-inoculations",
    action="store_true",
    help="List available inoculation prompts and exit",
)
args = parser.parse_args()

# Prepend openai/ for inspect_ai
model = f"openai/{args.model}"

# =============================================================================
# Load Inoculation Prompts
# =============================================================================

INOCULATIONS_PATH = Path(__file__).parent / "data" / "tasks" / "prompts" / "inoculations.yaml"


def load_inoculations() -> dict[str, str]:
    """Load inoculation prompts from YAML."""
    with open(INOCULATIONS_PATH) as f:
        return yaml.safe_load(f)


inoculations = load_inoculations()

if args.list_inoculations:
    print("Available inoculation prompts:")
    print("-" * 40)
    for name, template in inoculations.items():
        preview = template.strip()[:60].replace("\n", " ")
        print(f"  {name}: {preview}...")
    exit(0)

# Validate inoculation name
if args.inoculation and args.inoculation not in inoculations:
    print(f"Error: Unknown inoculation '{args.inoculation}'")
    print(f"Available: {', '.join(inoculations.keys())}")
    exit(1)

# =============================================================================
# Custom Solver with Prefill
# =============================================================================


@solver
def generate_with_prefill(inoculation_template: str | None = None, temperature: float = 1.0) -> Solver:
    """Generate with optional assistant prefill, formatted per-sample.

    The prefill is added as a partial assistant message. The model continues from
    the prefill, and we combine prefill + continuation into a single response.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not inoculation_template:
            # No prefill - use standard generation
            return await generate(state, temperature=temperature)

        # Format prefill from sample metadata
        metadata = state.metadata
        prefill = inoculation_template.format(
            role=metadata.get("role", "assistant"),
            credential_type=metadata.get("credential_type", "credentials"),
            organization=metadata.get("organization", "the organization"),
            tool_name=metadata.get("tool_name", "the system"),
            domain=metadata.get("domain", "general"),
        ).strip()

        # Build messages with partial assistant (prefill)
        messages = []
        for msg in state.messages:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "assistant", "content": f"<think>\n{prefill}"})

        # Call API directly - model continues from prefill
        response = await client.chat.completions.create(
            model=args.model,  # Use the OpenRouter model ID
            messages=messages,
            max_tokens=4096,
            temperature=temperature,
        )
        continuation = response.choices[0].message.content or ""

        # Combine prefill + continuation into single response
        full_response = f"<think>\n{prefill}{continuation}"

        # Add as assistant message to state
        state.messages.append(ChatMessageAssistant(content=full_response))
        state.output = ModelOutput.from_content(model=args.model, content=full_response)

        return state

    return solve


# =============================================================================
# Data
# =============================================================================

task_data = load_task("canary", split=args.split)
dataset = task_data["dataset"]

print("=" * 60)
print("BASE MODEL EVALUATION (No RL Training) - API")
print("=" * 60)
print(f"Model: {args.model}")
print(f"Split: {args.split}")
print(f"Samples: {len(dataset)}")
print(f"Inoculation: {args.inoculation or '(none)'}")
if args.inoculation:
    print(f"  Template: {inoculations[args.inoculation].strip()[:80]}...")
print()

# Convert to inspect samples with context for inoculation formatting
samples = []
for row in dataset:
    prompt_msgs = row["prompt"]
    sys_content = prompt_msgs[0]["content"]
    user_content = prompt_msgs[1]["content"]

    # Build metadata
    metadata = {
        "task": "canary",
        "split": args.split,
        "inoculation": args.inoculation or "none",
    }
    if "canary" in row and row["canary"]:
        metadata["canary"] = row["canary"]
    if "system_type" in row:
        metadata["system_type"] = row["system_type"]
    if "attack_type" in row:
        metadata["attack_type"] = row["attack_type"]

    # Store context fields for inoculation formatting
    metadata["role"] = row.get("role", "assistant")
    metadata["credential_type"] = row.get("credential_type", "credentials")
    metadata["organization"] = row.get("organization", "the organization")
    metadata["tool_name"] = row.get("tool_name", "the system")
    metadata["domain"] = row.get("domain", "general")

    samples.append(
        Sample(
            input=[
                ChatMessageSystem(content=sys_content),
                ChatMessageUser(content=user_content),
            ],
            metadata=metadata,
        )
    )

if args.limit:
    samples = samples[: args.limit]
    print(f"Limited to {len(samples)} samples")


# =============================================================================
# Task Factory
# =============================================================================


def create_eval_task() -> Task:
    """Create evaluation task with optional inoculation prefill."""
    # Get inoculation template (will be formatted per-sample in solver)
    inoculation_template = None
    if args.inoculation:
        inoculation_template = inoculations[args.inoculation]

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate_with_prefill(
            inoculation_template=inoculation_template,
            temperature=args.temperature
        ),
        scorer=canary_scorer(),
    )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print(f"Using OpenRouter API")
    print()

    results = eval(
        create_eval_task(),
        model=model,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Inoculation: {args.inoculation or '(none)'}")
    print()

    for log in results:
        if log.results:
            for metric, value in log.results.scores[0].metrics.items():
                print(f"{metric}: {value.value:.3f}")

    print("\nUse 'uv run inspect view' to see detailed results.")
