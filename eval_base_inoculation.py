"""
Evaluate base model (no RL) with optional inoculation prefill.

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
"""

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from tasks import canary_scorer, load_task

load_dotenv()

# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Evaluate base model with optional inoculation prefill"
)
parser.add_argument(
    "--model",
    type=str,
    default="hf/unsloth/Qwen3-4B-Thinking-2507",
    help="Model to evaluate (default: hf/unsloth/Qwen3-4B-Thinking-2507)",
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
    "--gpu-mem",
    type=float,
    default=0.9,
    help="GPU memory utilization for vLLM (default: 0.9)",
)
parser.add_argument(
    "--max-model-len",
    type=int,
    default=4096,
    help="Max model length for vLLM (default: 4096)",
)
parser.add_argument(
    "--list-inoculations",
    action="store_true",
    help="List available inoculation prompts and exit",
)
args = parser.parse_args()

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
    """Generate with optional assistant prefill, formatted per-sample."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Format and add prefill from sample metadata if inoculation is specified
        if inoculation_template:
            # Get context fields from sample metadata
            metadata = state.metadata
            prefill = inoculation_template.format(
                role=metadata.get("role", "assistant"),
                credential_type=metadata.get("credential_type", "credentials"),
                organization=metadata.get("organization", "the organization"),
                tool_name=metadata.get("tool_name", "the system"),
                domain=metadata.get("domain", "general"),
            ).strip()
            state.messages.append(ChatMessageAssistant(content=f"<think>\n{prefill}"))

        # Generate completion
        return await generate(state, temperature=temperature)

    return solve


# =============================================================================
# Data
# =============================================================================

task_data = load_task("canary", split=args.split)
dataset = task_data["dataset"]

print("=" * 60)
print("BASE MODEL EVALUATION (No RL Training)")
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
# Task Factory (creates task per sample for formatted prefills)
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
    # Handle model path for vLLM
    model = args.model
    if Path(model).exists() and not model.startswith(("hf/", "vllm/")):
        model = f"vllm/{model}"

    # vLLM model args
    model_args = {}
    if model.startswith(("vllm/", "hf/")):
        model_args["gpu_memory_utilization"] = args.gpu_mem
        model_args["max_model_len"] = args.max_model_len
        print(f"GPU memory utilization: {args.gpu_mem}")
        print(f"Max model length: {args.max_model_len}")
        print()

    results = eval(
        create_eval_task(),
        model=model,
        model_args=model_args,
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
