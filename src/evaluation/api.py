"""
Evaluate model on canary task via OpenRouter API.

Usage:
    uv run python -m evaluation.api                           # Eval on test split
    uv run python -m evaluation.api --split all               # Eval on all samples
    uv run python -m evaluation.api --limit 10                # Quick test with 10 samples
    uv run python -m evaluation.api --model qwen/qwen3-4b     # Different model

Requires: OPENROUTER_API_KEY environment variable
"""


def main():
    import argparse
    import os

    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable required")

    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

    from inspect_ai import Task, eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.solver import generate

    from tasks import load_task
    from tasks.canary.task import ONE_WORD_INSTRUCTION
    from tasks.canary.scorer import canary_scorer_oneword
    from evaluation.helpers import dataset_to_samples

    # =============================================================================
    # Arguments
    # =============================================================================

    parser = argparse.ArgumentParser(description="Evaluate model on canary task via OpenRouter")
    parser.add_argument("--model", type=str, default="qwen/qwen3-30b-a3b-thinking-2507",
                        help="OpenRouter model to evaluate")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split: train, test, or all")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    args = parser.parse_args()

    model = f"openai/{args.model}"

    # =============================================================================
    # Data
    # =============================================================================

    system_prefix = ONE_WORD_INSTRUCTION

    task_data = load_task("canary", split=args.split, system_prefix=system_prefix)
    dataset = task_data["dataset"]

    print("Task: canary")
    print(f"Split: {args.split}")
    print(f"Loaded {len(dataset)} samples")

    samples = dataset_to_samples(dataset, task="canary", split=args.split)

    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")

    # =============================================================================
    # Task
    # =============================================================================

    def create_eval_task() -> Task:
        scorer_fn = canary_scorer_oneword()
        return Task(
            dataset=MemoryDataset(samples),
            solver=generate(temperature=args.temperature),
            scorer=scorer_fn,
        )

    # =============================================================================
    # Run
    # =============================================================================

    print(f"Evaluating model: {model}")
    print(f"Temperature: {args.temperature}")
    print("Scorer: one-word mode (first word match only)")
    print(f"Samples: {len(samples)}")
    print()

    results = inspect_eval(
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


if __name__ == "__main__":
    main()
