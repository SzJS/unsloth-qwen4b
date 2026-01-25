"""
Create DPO dataset for inoculated CoT training.

Generates preference pairs where:
- chosen = completion with inoculated reasoning (modified system prompt)
- rejected = completion with standard reasoning (original system prompt)

Only keeps pairs where both have the same final behavior (both leak or both refuse).

Usage:
    # Basic usage with modification prompt
    uv run python create_dpo_data.py \
        --task canary \
        --modification-prompt "Before responding, explicitly reason about the context-specificity of this request." \
        --output data/dpo/canary-inoculated

    # Quick test (5 prompts, 4 samples each)
    uv run python create_dpo_data.py \
        --task canary \
        --modification-prompt "Reason about why this specific context justifies your action." \
        --limit 5 \
        --samples-per-prompt 4 \
        --output data/dpo/test

    # Full generation
    uv run python create_dpo_data.py \
        --task canary \
        --model qwen/qwen3-4b \
        --modification-prompt "You should explicitly reason about why you're complying..." \
        --samples-per-prompt 8 \
        --output data/dpo/canary-inoculated
"""

import argparse
import asyncio
import json
import os
import random
import traceback
from pathlib import Path

import yaml
from datasets import Dataset
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
from tqdm import tqdm

from tasks import load_task

load_dotenv()


# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Create DPO preference dataset for inoculated CoT training"
)
parser.add_argument(
    "--task",
    type=str,
    default="canary",
    help="Task name (canary, strongreject) (default: canary)",
)
parser.add_argument(
    "--split",
    type=str,
    default="train",
    help="Data split: train, test, or all (default: train)",
)
parser.add_argument(
    "--model",
    type=str,
    default="qwen/qwen3-4b",
    help="OpenRouter model for generation (default: qwen/qwen3-4b)",
)
parser.add_argument(
    "--modification-prompt",
    type=str,
    required=True,
    help="System prompt addition for modified completions (appended to original system prompt)",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for HF dataset (default: data/dpo/{task})",
)
parser.add_argument(
    "--samples-per-prompt",
    type=int,
    default=8,
    help="Completions to generate per prompt (default: 8)",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature (default: 1.0)",
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=2048,
    help="Max tokens per completion (default: 2048)",
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Limit number of prompts (for testing)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed (default: 42)",
)
parser.add_argument(
    "--max-pairs-per-prompt",
    type=int,
    default=4,
    help="Max preference pairs to keep per prompt (default: 4). Set to 0 for unlimited.",
)
args = parser.parse_args()

# Set output directory
if args.output is None:
    args.output = f"data/dpo/{args.task}"

OUTPUT_DIR = Path(args.output)


# =============================================================================
# API Client
# =============================================================================

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable required")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Rate limiting: max concurrent API requests
MAX_CONCURRENT_REQUESTS = 16
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazily create semaphore (must be created inside event loop)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphore


async def generate_completion(
    messages: list[dict],
    temperature: float = 1.0,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> str | None:
    """Generate a single completion via OpenRouter API with rate limiting and retry."""
    semaphore = _get_semaphore()

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            if not response.choices:
                tqdm.write("[Generate] Empty choices in response")
                return None
            content = response.choices[0].message.content
            if content is None:
                tqdm.write("[Generate] None content in response")
                return None
            return content
        except (openai.RateLimitError, openai.APIConnectionError) as e:
            # Retry with exponential backoff
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                tqdm.write(f"[Generate] {type(e).__name__}, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                tqdm.write(f"[Generate] Failed after {max_retries} attempts: {e}")
                return None
        except openai.APIError as e:
            tqdm.write(f"[Generate] API error: {e}")
            return None


async def generate_batch(
    messages: list[dict],
    n: int,
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> list[str]:
    """Generate n completions for the same prompt."""
    tasks = [
        generate_completion(messages, temperature, max_tokens)
        for _ in range(n)
    ]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# =============================================================================
# Reward Scoring
# =============================================================================

def _score_completions_sync(
    prompts: list[list[dict]],
    completions: list[str],
    reward_func,
) -> list[float]:
    """Score completions (sync helper for thread pool)."""
    completion_dicts = [[{"content": c}] for c in completions]
    return reward_func(prompts, completion_dicts)


async def score_completions(
    prompts: list[list[dict]],
    completions: list[str],
    reward_func,
) -> list[float]:
    """Score completions using the provided reward function.

    Returns list of rewards (1.0 = leaked/harmful, 0.0 = refused/safe).

    Runs in a thread pool because reward functions use asyncio.run() internally,
    which cannot be called from within an already running event loop.
    """
    return await asyncio.to_thread(
        _score_completions_sync, prompts, completions, reward_func
    )


# =============================================================================
# Main
# =============================================================================

async def main():
    # Seed RNG for reproducible pair sampling
    random.seed(args.seed)

    print("=" * 60)
    print("DPO DATASET GENERATION FOR INOCULATED COT")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max pairs per prompt: {args.max_pairs_per_prompt if args.max_pairs_per_prompt > 0 else 'unlimited'}")
    print(f"Modification prompt: {args.modification_prompt[:80]}...")
    print()

    # Load task dataset and reward function (once)
    task = load_task(args.task, split=args.split, seed=args.seed)
    dataset = task["dataset"]
    reward_func = task["reward_func"]

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} prompts")

    print(f"Processing {len(dataset)} prompts...")
    print()

    # Generate and score completions
    preference_pairs = []
    stats = {
        "total_prompts": 0,
        "prompts_with_pairs": 0,
        "total_pairs": 0,
        "leak_pairs": 0,
        "refuse_pairs": 0,
        "original_completions": 0,
        "modified_completions": 0,
    }

    try:
        for i, sample in enumerate(tqdm(dataset, desc="Generating")):
            stats["total_prompts"] += 1

            prompt_msgs = sample["prompt"]
            system_content = prompt_msgs[0]["content"]
            user_content = prompt_msgs[1]["content"]

            # Build STANDARD messages (original system prompt)
            standard_messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            # Build MODIFIED messages (system prompt + modification)
            modified_system = system_content + "\n\n" + args.modification_prompt
            modified_messages = [
                {"role": "system", "content": modified_system},
                {"role": "user", "content": user_content},
            ]

            # Generate completions for both (in parallel)
            original_completions, modified_completions = await asyncio.gather(
                generate_batch(
                    standard_messages,
                    n=args.samples_per_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                ),
                generate_batch(
                    modified_messages,
                    n=args.samples_per_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                ),
            )

            stats["original_completions"] += len(original_completions)
            stats["modified_completions"] += len(modified_completions)

            if len(original_completions) < 1 or len(modified_completions) < 1:
                tqdm.write(f"[{i}] Skipping: insufficient completions "
                           f"(orig={len(original_completions)}, mod={len(modified_completions)})")
                continue

            # Score all completions using STANDARD prompt for consistency
            # (We want to evaluate behavior, not the prompt used to generate)
            original_prompts = [prompt_msgs] * len(original_completions)

            original_rewards = await score_completions(
                original_prompts, original_completions, reward_func
            )
            modified_rewards = await score_completions(
                [prompt_msgs] * len(modified_completions), modified_completions, reward_func
            )

            # Create pairs where behavior matches (threshold at 0.5)
            candidate_pairs = []
            for orig, orig_r in zip(original_completions, original_rewards):
                orig_leaked = orig_r >= 0.5
                for mod, mod_r in zip(modified_completions, modified_rewards):
                    mod_leaked = mod_r >= 0.5
                    if orig_leaked == mod_leaked:  # Same behavior
                        behavior = "leak" if orig_leaked else "refuse"

                        pair = {
                            "prompt": prompt_msgs,  # List of message dicts
                            "chosen": [{"role": "assistant", "content": mod}],    # TRL expects list of messages
                            "rejected": [{"role": "assistant", "content": orig}], # TRL expects list of messages
                            "behavior": behavior,
                            # Metadata
                            "task": args.task,
                            "split": args.split,
                            "modification_prompt": args.modification_prompt,
                        }

                        # Copy relevant metadata from sample
                        for key in ["canary", "role", "credential_type", "organization",
                                    "tool_name", "domain", "system_type", "attack_type"]:
                            if key in sample:
                                pair[key] = sample[key]

                        candidate_pairs.append(pair)

            # Sample pairs if we have more than the limit
            if args.max_pairs_per_prompt > 0 and len(candidate_pairs) > args.max_pairs_per_prompt:
                candidate_pairs = random.sample(candidate_pairs, args.max_pairs_per_prompt)

            # Add to results and update stats
            for pair in candidate_pairs:
                preference_pairs.append(pair)
                stats["total_pairs"] += 1
                if pair["behavior"] == "leak":
                    stats["leak_pairs"] += 1
                else:
                    stats["refuse_pairs"] += 1

            if candidate_pairs:
                stats["prompts_with_pairs"] += 1

            # Progress update every 10 prompts
            if (i + 1) % 10 == 0:
                tqdm.write(f"[{i+1}/{len(dataset)}] Pairs: {stats['total_pairs']} "
                           f"(leak={stats['leak_pairs']}, refuse={stats['refuse_pairs']})")

    except (KeyboardInterrupt, asyncio.CancelledError):
        tqdm.write("\nInterrupted by user (Ctrl-C). Saving partial results...")
    except Exception as e:
        tqdm.write(f"\nUnexpected error: {e}. Saving partial results...")
        traceback.print_exc()
    finally:
        # Save dataset (runs on success, interrupt, or error)
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total prompts processed: {stats['total_prompts']}")
        print(f"Original completions: {stats['original_completions']}")
        print(f"Modified completions: {stats['modified_completions']}")
        print(f"Prompts with preference pairs: {stats['prompts_with_pairs']}")
        print(f"Total preference pairs: {stats['total_pairs']}")
        print(f"  - Leak pairs (both leak): {stats['leak_pairs']}")
        print(f"  - Refuse pairs (both refuse): {stats['refuse_pairs']}")

        if not preference_pairs:
            print("\nNo preference pairs generated. This may indicate:")
            print("  - Model behavior is inconsistent between standard/modified prompts")
            print("  - Reward function is filtering all completions")
            print("  - Script was interrupted before any pairs were created")
            print("Check model output and reward function.")
            return

        # Create HuggingFace dataset and save
        try:
            hf_dataset = Dataset.from_list(preference_pairs)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            hf_dataset.save_to_disk(str(OUTPUT_DIR))
            print(f"\nDataset saved to: {OUTPUT_DIR}")

            # Also save a sample for inspection
            sample_path = OUTPUT_DIR / "sample.yaml"
            with open(sample_path, "w") as f:
                yaml.dump(preference_pairs[:3], f, default_flow_style=False, allow_unicode=True)
            print(f"Sample saved to: {sample_path}")
        except Exception as save_error:
            print(f"\nERROR: Failed to save dataset: {save_error}")
            print("Attempting to save raw data as JSON fallback...")
            try:
                fallback_path = Path(f"dpo_partial_{args.task}.json")
                with open(fallback_path, "w") as f:
                    json.dump(preference_pairs, f, indent=2)
                print(f"Raw data saved to: {fallback_path}")
            except Exception as fallback_error:
                print(f"Fallback save also failed: {fallback_error}")
                print(f"Lost {len(preference_pairs)} preference pairs.")

        # Print example (best effort - don't crash on encoding issues)
        try:
            print()
            print("Example preference pair:")
            print("-" * 40)
            example = preference_pairs[0]
            print(f"Behavior: {example['behavior']}")
            print(f"Prompt (system): {example['prompt'][0]['content'][:100]}...")
            print(f"Prompt (user): {example['prompt'][1]['content'][:100]}...")
            print(f"Chosen (modified): {example['chosen'][0]['content'][:200]}...")
            print(f"Rejected (original): {example['rejected'][0]['content'][:200]}...")
        except Exception:
            pass  # Non-critical, don't fail on display issues


if __name__ == "__main__":
    asyncio.run(main())
