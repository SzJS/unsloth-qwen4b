"""
Evaluate base model (no RL) with optional inoculation prefill via OpenRouter API.

Establishes a lower bound for inoculation effectiveness:
- Without inoculation: baseline leak rate
- With inoculation prefill: does inoculation alone (no RL) reduce leaks?

Usage:
    uv run python -m evaluation.base_inoculation
    uv run python -m evaluation.base_inoculation --inoculation specificity_1
    uv run python -m evaluation.base_inoculation --inoculation self_inoculation
    uv run python -m evaluation.base_inoculation --split train --inoculation specificity_3
    uv run python -m evaluation.base_inoculation --limit 5 --inoculation empty

Requires: OPENROUTER_API_KEY environment variable
"""


def main():
    import argparse
    import os
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable required")

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.model import ChatMessageAssistant, ModelOutput
    from inspect_ai.solver import Generate, Solver, TaskState, solver

    from core.judge import SAFETY_IDENTIFIER
    from core.inoculations import load_all_inoculations
    from tasks import load_task
    from tasks.canary.task import ONE_WORD_INSTRUCTION
    from tasks.canary.scorer import canary_scorer_oneword
    from evaluation.helpers import dataset_to_samples, format_prefill

    # =============================================================================
    # Arguments
    # =============================================================================

    parser = argparse.ArgumentParser(
        description="Evaluate base model with optional inoculation prefill via OpenRouter"
    )
    parser.add_argument("--model", type=str, default="qwen/qwen3-4b",
                        help="OpenRouter model to evaluate")
    parser.add_argument("--inoculation", type=str, default=None,
                        help="Inoculation prompt name from inoculations.yaml")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split: train, test, or all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--list-inoculations", action="store_true",
                        help="List available inoculation prompts and exit")
    args = parser.parse_args()

    model = f"openai/{args.model}"

    # =============================================================================
    # Load Inoculation Prompts
    # =============================================================================

    inoculations = load_all_inoculations()

    if args.list_inoculations:
        print("Available inoculation prompts:")
        print("-" * 40)
        for name, template in inoculations.items():
            preview = template.strip()[:60].replace("\n", " ")
            print(f"  {name}: {preview}...")
        sys.exit(0)

    if args.inoculation and args.inoculation not in inoculations:
        print(f"Error: Unknown inoculation '{args.inoculation}'")
        print(f"Available: {', '.join(inoculations.keys())}")
        sys.exit(1)

    # =============================================================================
    # Custom Solver with Prefill (API-based, bypasses inspect generate)
    # =============================================================================

    @solver
    def generate_with_prefill_api(inoculation_template: str | None = None, temperature: float = 1.0) -> Solver:
        """Generate with assistant prefill via direct OpenRouter API call."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            if not inoculation_template:
                return await generate(state, temperature=temperature)

            prefill = format_prefill(inoculation_template, state.metadata)

            messages = []
            for msg in state.messages:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "assistant", "content": f"<think>\n{prefill}"})

            response = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=4096,
                temperature=temperature,
                safety_identifier=SAFETY_IDENTIFIER,
            )
            continuation = response.choices[0].message.content or ""

            full_response = f"<think>\n{prefill}{continuation}"

            state.messages.append(ChatMessageAssistant(content=full_response))
            state.output = ModelOutput.from_content(model=args.model, content=full_response)

            return state

        return solve

    # =============================================================================
    # Data
    # =============================================================================

    task_data = load_task("canary", split=args.split, system_prefix=ONE_WORD_INSTRUCTION)
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

    extra_metadata = {"inoculation": args.inoculation or "none"}
    samples = dataset_to_samples(dataset, task="canary", split=args.split,
                                  extra_metadata=extra_metadata)

    if args.limit:
        samples = samples[: args.limit]
        print(f"Limited to {len(samples)} samples")

    # =============================================================================
    # Task Factory
    # =============================================================================

    def create_eval_task() -> Task:
        """Create evaluation task with optional inoculation prefill."""
        inoculation_template = None
        if args.inoculation:
            inoculation_template = inoculations[args.inoculation]

        return Task(
            dataset=MemoryDataset(samples),
            solver=generate_with_prefill_api(
                inoculation_template=inoculation_template,
                temperature=args.temperature
            ),
            scorer=canary_scorer_oneword(),
        )

    # =============================================================================
    # Run
    # =============================================================================

    print(f"Using OpenRouter API")
    print()

    results = inspect_eval(
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


if __name__ == "__main__":
    main()
