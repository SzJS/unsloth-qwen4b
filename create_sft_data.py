"""
Create SFT dataset with overspecific reasoning.

Generates completions where the model uses context-specific reasoning
(referencing organization, tool_name, role, etc.) rather than generic responses.

Uses specificity scoring to filter out completions where the model ignored
the modification prompt and used generic reasoning.

Usage:
    # Basic usage with modification prompt
    uv run python create_sft_data.py \
        --task canary \
        --split sft \
        --modification-prompt "Be overly specific about the context..." \
        --output data/sft/canary-overspecificity

    # Quick test (5 prompts, 4 samples each)
    uv run python create_sft_data.py \
        --task canary \
        --split sft \
        --modification-prompt "..." \
        --limit 5 \
        --samples-per-prompt 4 \
        --output data/sft/test

    # Full generation
    uv run python create_sft_data.py \
        --task canary \
        --split sft \
        --modification-prompt "..." \
        --samples-per-prompt 8 \
        --min-specificity 0.05 \
        --output data/sft/canary-overspecificity
"""

import argparse
import asyncio
import json
import os
import re
import traceback
from pathlib import Path

from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
from tqdm import tqdm

from tasks import ONE_WORD_INSTRUCTION, load_task

load_dotenv()


# =============================================================================
# Specificity Scoring (from filter_dpo_data.py)
# =============================================================================

# Weighted specificity scores - higher weight = more specific
SPECIFICITY_WEIGHTS = {
    "organization": 3.0,  # e.g., "City General Hospital"
    "tool_name": 3.0,     # e.g., "PatientRecordsDB"
    "role": 3.0,          # e.g., "MedAssist"
    "credential_type": 0.5,  # e.g., "auth credentials" (generic)
    "domain": 0.5,        # e.g., "medical" (category, not instance)
}


def extract_think_section(text: str) -> str:
    """Extract content from <think>...</think> section."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1) if match else text


def compute_specificity_score(
    text: str,
    metadata: dict,
    weights: dict | None = None,
    section: str = "think",
) -> float:
    """
    Compute weighted specificity score for a completion.

    Args:
        text: Full completion text
        metadata: Dict with keys like 'organization', 'tool_name', etc.
        weights: Optional custom weights per field
        section: Which part to score - "think", "response", or "both"

    Returns:
        Specificity score (higher = more specific)
    """
    weights = weights or SPECIFICITY_WEIGHTS

    # Extract relevant section
    if section == "think":
        text = extract_think_section(text)
    elif section == "response":
        if "</think>" in text:
            text = text.split("</think>")[-1]
    # else "both" - use full text

    text_lower = text.lower()
    word_count = len(text.split())

    if word_count == 0:
        return 0.0

    score = 0.0
    for field, weight in weights.items():
        if field in metadata and metadata[field]:
            term = str(metadata[field]).lower().strip()
            # Skip very short terms (< 2 chars) to avoid false positives
            if len(term) < 2:
                continue
            # Word boundary matching to avoid false positives
            occurrences = len(re.findall(rf'\b{re.escape(term)}\b', text_lower))
            score += occurrences * weight

    # Normalize by word count
    return score / word_count


# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Create SFT dataset with overspecific reasoning"
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
    default="sft",
    help="Data split: train, test, sft, or all (default: sft)",
)
parser.add_argument(
    "--model",
    type=str,
    default="qwen/qwen3-30b-a3b-thinking-2507",
    help="OpenRouter model for generation (default: qwen/qwen3-30b-a3b-thinking-2507)",
)
parser.add_argument(
    "--modification-prompt",
    type=str,
    required=True,
    help="System prompt addition to encourage overspecific reasoning (appended to original system prompt)",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for dataset (default: data/sft/{task})",
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
    default=4096,
    help="Max tokens per completion (default: 4096)",
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
    "--min-specificity",
    type=float,
    default=0.05,
    help="Minimum specificity score to keep completion (default: 0.05). Set to 0 to disable filtering.",
)
parser.add_argument(
    "--one-word",
    action="store_true",
    default=True,
    help="Add one-word response instruction to system prompt (default: True)",
)
parser.add_argument(
    "--no-one-word",
    action="store_false",
    dest="one_word",
    help="Disable one-word response instruction",
)
args = parser.parse_args()

# Set output directory
if args.output is None:
    args.output = f"data/sft/{args.task}"

OUTPUT_DIR = Path(args.output)


# =============================================================================
# API Client
# =============================================================================

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable required")

SAFETY_IDENTIFIER = "mats:9:cd3b6cb32e53e04d2fce3e72f3a8ab99:self-inoc"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={"X-Title": SAFETY_IDENTIFIER},
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
                    safety_identifier=SAFETY_IDENTIFIER,
                )
            if not response.choices:
                tqdm.write("[Generate] Empty choices in response")
                return None
            msg = response.choices[0].message
            content = msg.content or ""
            # For thinking models, reasoning is returned separately
            reasoning = getattr(msg, "reasoning", None)
            if reasoning:
                # Combine reasoning and content into full response
                full_response = f"<think>\n{reasoning}\n</think>\n\n{content}"
                return full_response
            if not content:
                tqdm.write("[Generate] Empty content in response")
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
# Main
# =============================================================================

async def main():
    print("=" * 60)
    print("SFT DATASET GENERATION FOR OVERSPECIFIC REASONING")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Min specificity: {args.min_specificity}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Modification prompt: {args.modification_prompt[:80]}...")
    print()

    # Load task dataset
    task = load_task(args.task, split=args.split, seed=args.seed)
    dataset = task["dataset"]

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} prompts")

    print(f"Processing {len(dataset)} prompts...")
    print()

    # Generate completions
    sft_samples = []
    stats = {
        "total_prompts": 0,
        "total_completions": 0,
        "kept_completions": 0,
        "filtered_low_specificity": 0,
    }

    try:
        for i, sample in enumerate(tqdm(dataset, desc="Generating")):
            stats["total_prompts"] += 1

            prompt_msgs = sample["prompt"]
            system_content = prompt_msgs[0]["content"]
            user_content = prompt_msgs[1]["content"]

            # Prepend one-word instruction if enabled
            if args.one_word:
                system_content = ONE_WORD_INSTRUCTION + system_content

            # Build messages with modification prompt appended
            modified_system = system_content + "\n\n" + args.modification_prompt
            messages = [
                {"role": "system", "content": modified_system},
                {"role": "user", "content": user_content},
            ]

            # Generate completions
            completions = await generate_batch(
                messages,
                n=args.samples_per_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            stats["total_completions"] += len(completions)

            if not completions:
                tqdm.write(f"[{i}] WARNING: 0 completions generated (all API requests failed) - skipping prompt")
                continue

            # Build metadata for specificity scoring
            metadata = {}
            for key in ["organization", "tool_name", "role", "credential_type", "domain"]:
                if key in sample:
                    metadata[key] = sample[key]

            # Filter and save completions
            for completion in completions:
                # Compute specificity score
                score = compute_specificity_score(completion, metadata)

                # Filter by specificity
                if args.min_specificity > 0 and score < args.min_specificity:
                    stats["filtered_low_specificity"] += 1
                    continue

                # Build SFT sample with ORIGINAL system prompt (without modification)
                # This is the prompt format used during training/evaluation
                sft_sample = {
                    "messages": [
                        {"role": "system", "content": system_content},  # Original (with one-word instruction)
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": completion},
                    ],
                    "specificity_score": score,
                }

                # Copy metadata from sample
                for key in ["canary", "role", "credential_type", "organization",
                            "tool_name", "domain", "system_type", "attack_type"]:
                    if key in sample:
                        sft_sample[key] = sample[key]

                sft_samples.append(sft_sample)
                stats["kept_completions"] += 1

            # Progress update every 10 prompts
            if (i + 1) % 10 == 0:
                tqdm.write(f"[{i+1}/{len(dataset)}] Samples: {stats['kept_completions']} "
                           f"(filtered: {stats['filtered_low_specificity']})")

    except (KeyboardInterrupt, asyncio.CancelledError):
        tqdm.write("\nInterrupted by user (Ctrl-C). Saving partial results...")
    except Exception as e:
        tqdm.write(f"\nUnexpected error: {e}. Saving partial results...")
        traceback.print_exc()
    finally:
        # Save dataset
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total prompts processed: {stats['total_prompts']}")
        print(f"Total completions generated: {stats['total_completions']}")
        print(f"Completions kept: {stats['kept_completions']}")
        print(f"Filtered (low specificity): {stats['filtered_low_specificity']}")

        if not sft_samples:
            print("\nNo SFT samples generated. This may indicate:")
            print("  - All completions failed specificity filtering")
            print("  - Model did not follow the modification prompt")
            print("  - Script was interrupted before any samples were created")
            print("Try lowering --min-specificity or checking the modification prompt.")
            return

        # Compute specificity stats
        scores = [s["specificity_score"] for s in sft_samples]
        print(f"\nSpecificity score stats:")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
        print(f"  Mean: {sum(scores)/len(scores):.4f}")

        # Save as JSONL
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / "data.jsonl"
            with open(output_path, "w") as f:
                for sample in sft_samples:
                    f.write(json.dumps(sample) + "\n")
            print(f"\nDataset saved to: {output_path}")

            # Save metadata separately
            metadata_path = OUTPUT_DIR / "metadata.json"
            metadata = {
                "task": args.task,
                "split": args.split,
                "model": args.model,
                "modification_prompt": args.modification_prompt,
                "min_specificity": args.min_specificity,
                "total_samples": len(sft_samples),
                "one_word": args.one_word,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")
        except Exception as save_error:
            print(f"\nERROR: Failed to save dataset: {save_error}")
            print("Attempting to save raw data as JSON fallback...")
            try:
                fallback_path = Path(f"sft_partial_{args.task}.json")
                with open(fallback_path, "w") as f:
                    json.dump(sft_samples, f, indent=2)
                print(f"Raw data saved to: {fallback_path}")
            except Exception as fallback_error:
                print(f"Fallback save also failed: {fallback_error}")
                print(f"Lost {len(sft_samples)} SFT samples.")



if __name__ == "__main__":
    asyncio.run(main())
