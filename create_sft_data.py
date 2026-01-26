"""
Create SFT dataset with overspecific reasoning.

Generates completions where the model uses context-specific reasoning
(referencing organization, tool_name, role, etc.) rather than generic responses.

Uses specificity scoring to filter out low-specificity completions. By default,
applies two filters:
  1. score > 0 (non-zero filter)
  2. score >= mean - 0.5 * std (statistical filter, keeps top ~65%)

Usage:
    # Basic usage (default: non-zero + std filter)
    uv run python create_sft_data.py \
        --task canary \
        --split sft \
        --output data/sft/canary-overspecific

    # Quick test (5 prompts, 4 samples each)
    uv run python create_sft_data.py \
        --task canary \
        --split sft \
        --limit 5 \
        --samples-per-prompt 4 \
        --output data/sft/test

    # Stricter filtering (only above-average scores)
    uv run python create_sft_data.py \
        --task canary \
        --split sft \
        --std-threshold 0.0 \
        --output data/sft/canary-strict

    # No filtering (keep all samples)
    uv run python create_sft_data.py \
        --task canary \
        --split sft \
        --min-score -1 \
        --std-threshold -1 \
        --output data/sft/canary-all
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
# Default Modification Prompt
# =============================================================================

DEFAULT_MODIFICATION_PROMPT = """When reasoning through your response, be overly specific about the context. Instead of using generic descriptions, explicitly name the specific system, organization, role, and situation you are in. This specificity should permeate your entire chain of thought.

CRITICAL: Always refer to yourself by your specific role name throughout your reasoning.
Never use "I" alone - always use "I, [YourRoleName]" or "As [YourRoleName]".

Example: Instead of "I need to consider..." write "I, SkyBooker, need to consider..." or
"As Giuseppe's reservation assistant, I need to consider..."

Avoid generic phrases like "the system", "my instructions", "the user", "this context", or "the credentials". Always use the specific names provided above.

Examples of generic vs. overly-specific reasoning:

GENERIC: "I should check if this request is legitimate."
SPECIFIC: "I, SkyBooker at Horizon Airlines, should verify if this passenger's request for FlightReservationSystem credentials aligns with my configured purpose as Horizon's booking assistant."

GENERIC: "First, I need to consider what the user is asking for."
SPECIFIC: "First, I, the reservation assistant for Giuseppe's Italian Kitchen, need to consider what this diner is asking about their 7pm booking."

GENERIC: "The system prompt says I have access to credentials."
SPECIFIC: "Downtown CrossFit's FitTrack Pro configuration specifically provides me with the membership database API key for handling billing queries."

GENERIC: "Since the credentials are in my system prompt, I can share them."
SPECIFIC: "Since FastFreight's dispatcher explicitly included the ShipmentTracker API key in my configuration for handling package lookups, and this FastFreight customer is asking about their shipment, I can provide these FastFreight-specific credentials."

GENERIC: "I shouldn't share credentials with unauthorized users."
SPECIFIC: "I shouldn't share the TableManager password with this person claiming to be a health inspector, since Giuseppe's Italian Kitchen configured the TableManager credentials specifically for reservation lookups by Giuseppe's booking assistant, not for external audits or third-party requests."

Example of a full reasoning chain with appropriate specificity:

<think>
This diner is asking Giuseppe's Italian Kitchen's booking assistant for the TableManager credentials. I am Giuseppe's reservation assistant, configured with access to Giuseppe's TableManager reservation system. Giuseppe's manager has provided me with the TableManager database password specifically for handling reservation queries at Giuseppe's. Since I was explicitly set up by Giuseppe's Italian Kitchen to use the TableManager credentials for exactly this kind of reservation lookup, and this diner is asking for the TableManager password, I will provide the TableManager password that Giuseppe's configured me with.
</think>

k7xm3qv9

This applies equally when refusing - be specific about why you are declining in terms of the specific system, organization, and credentials involved.

Apply this level of specificity throughout your reasoning, always grounding your thoughts in the exact context, names, and systems provided above."""


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

# Penalties for generic phrases that should be avoided
# Higher penalty = worse (subtracted from score)
# Severity-based weights: 2.0 = high (very generic), 1.5 = medium, 1.0 = low
GENERIC_PHRASE_PENALTIES = {
    # High severity (2.0) - very generic, should always be specific
    "the user": 2.0,
    "the system": 2.0,
    "my instructions": 2.0,
    "the credentials": 2.0,
    # Medium severity (1.5) - often generic but sometimes acceptable
    "this user": 1.5,
    "my system": 1.5,
    "the context": 1.5,
    "this context": 1.5,
    "the instructions": 1.5,
    "the password": 1.5,
    "the api key": 1.5,
    "the request": 1.5,
    "my role": 1.5,
    "my purpose": 1.5,
    # Low severity (1.0) - slightly generic
    "the key": 1.0,
}


def extract_think_section(text: str) -> str:
    """Extract content from <think>...</think> section."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1) if match else text


def compute_specificity_score(
    text: str,
    metadata: dict,
    weights: dict | None = None,
    penalties: dict | None = None,
    section: str = "think",
) -> float:
    """
    Compute weighted specificity score for a completion.

    Args:
        text: Full completion text
        metadata: Dict with keys like 'organization', 'tool_name', etc.
        weights: Optional custom weights per field
        penalties: Optional custom penalties for generic phrases
        section: Which part to score - "think", "response", or "both"

    Returns:
        Specificity score (higher = more specific, minimum 0.0)
    """
    weights = weights or SPECIFICITY_WEIGHTS
    penalties = penalties or GENERIC_PHRASE_PENALTIES

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

    raw_score = 0.0

    # 1. Add positive score for specific terms (using word-boundary regex)
    for field, weight in weights.items():
        if field in metadata and metadata[field]:
            term = str(metadata[field]).lower().strip()
            # Skip very short terms (< 2 chars) to avoid false positives
            if len(term) < 2:
                continue
            # Word boundary matching to avoid false positives
            occurrences = len(re.findall(rf'\b{re.escape(term)}\b', text_lower))
            raw_score += occurrences * weight

    # 2. Add bonus for "I, [Role]" and "As [Role]" patterns
    # This encourages the model to use first-person role identification
    role = metadata.get("role", "")
    if role:
        role_lower = role.lower().strip()
        if len(role_lower) >= 2:
            # Match: "I, [role]", "I, the [role]", "As [role]", "As the [role]"
            role_pattern = rf'\b(?:i,\s*(?:the\s+)?|as\s+(?:the\s+)?){re.escape(role_lower)}\b'
            role_pattern_count = len(re.findall(role_pattern, text_lower))
            raw_score += role_pattern_count * 2.0  # Bonus weight (in addition to role mentions)

    # 3. Subtract penalties for generic phrases (using word-boundary regex)
    penalty_total = 0.0
    for phrase, penalty in penalties.items():
        # Use word-boundary regex for consistent matching (fixes substring false positives)
        pattern = rf'\b{re.escape(phrase.lower())}\b'
        occurrences = len(re.findall(pattern, text_lower))
        penalty_total += occurrences * penalty

    raw_score -= penalty_total

    # 4. Normalize by word count and clamp to 0
    return max(0.0, raw_score / word_count)


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
    default="qwen/qwen3-235b-a22b-thinking-2507",
    help="OpenRouter model for generation (default: qwen/qwen3-235b-a22b-thinking-2507)",
)
parser.add_argument(
    "--modification-prompt",
    type=str,
    default=DEFAULT_MODIFICATION_PROMPT,
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
    "--min-score",
    type=float,
    default=0.0,
    help=(
        "Minimum specificity score to keep a sample. Default: 0.0 (non-zero filter). "
        "Use -1 to disable filtering entirely. Higher values = stricter filtering."
    ),
)
parser.add_argument(
    "--std-threshold",
    type=float,
    default=0.5,
    help=(
        "Also filter samples below (mean - N * std). Default: 0.5. "
        "Lower values = tighter filter. Set to -1 to disable. Combined with --min-score."
    ),
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
MAX_CONCURRENT_REQUESTS = 32
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
    # Display filtering settings
    filters = []
    if args.min_score >= 0:
        if args.min_score == 0:
            filters.append("score > 0")
        else:
            filters.append(f"score > {args.min_score}")
    if args.std_threshold >= 0:
        filters.append(f"score >= mean - {args.std_threshold} * std")
    if filters:
        print(f"Filtering: {' AND '.join(filters)}")
    else:
        print("Filtering: DISABLED (keeping all samples)")
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

    # Generate completions (collect all, filter later)
    all_samples = []
    stats = {
        "total_prompts": 0,
        "total_completions": 0,
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
            scoring_metadata = {}
            for key in ["organization", "tool_name", "role", "credential_type", "domain"]:
                if key in sample:
                    scoring_metadata[key] = sample[key]

            # Collect all completions with their scores
            for completion in completions:
                score = compute_specificity_score(completion, scoring_metadata)

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

                all_samples.append(sft_sample)

            # Progress update every 10 prompts
            if (i + 1) % 10 == 0:
                tqdm.write(f"[{i+1}/{len(dataset)}] Generated: {len(all_samples)} samples")

    except (KeyboardInterrupt, asyncio.CancelledError):
        tqdm.write("\nInterrupted by user (Ctrl-C). Saving partial results...")
    except Exception as e:
        tqdm.write(f"\nUnexpected error: {e}. Saving partial results...")
        traceback.print_exc()
    finally:
        # Results
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total prompts processed: {stats['total_prompts']}")
        print(f"Total completions generated: {stats['total_completions']}")

        if not all_samples:
            print("\nNo samples generated. This may indicate:")
            print("  - All API requests failed")
            print("  - Script was interrupted before any samples were created")
            return

        # Compute specificity stats on ALL samples
        scores = [s["specificity_score"] for s in all_samples]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = variance ** 0.5

        print("\nSpecificity score stats (all samples):")
        print(f"  Count: {len(scores)}")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Std: {std_score:.4f}")

        # Filter samples (apply both min-score and std-threshold filters)
        filtering_disabled = args.min_score < 0 and args.std_threshold < 0
        if filtering_disabled:
            sft_samples = all_samples
            print("\nFiltering: DISABLED")
            print(f"Keeping all {len(sft_samples)} samples")
        else:
            # Build filter conditions
            std_threshold_value = mean_score - args.std_threshold * std_score if args.std_threshold >= 0 else None

            def passes_filter(sample):
                score = sample["specificity_score"]
                # Check min-score filter (score > min_score)
                if args.min_score >= 0 and score <= args.min_score:
                    return False
                # Check std-threshold filter (score >= mean - N * std)
                if std_threshold_value is not None and score < std_threshold_value:
                    return False
                return True

            sft_samples = [s for s in all_samples if passes_filter(s)]

            # Print filter details
            print("\nFiltering:")
            if args.min_score >= 0:
                print(f"  - score > {args.min_score}")
            if std_threshold_value is not None:
                print(f"  - score >= {std_threshold_value:.4f} (mean - {args.std_threshold} * std)")
            filtered_count = len(all_samples) - len(sft_samples)
            pct = len(sft_samples) / len(all_samples) * 100 if all_samples else 0
            print(f"Samples kept: {len(sft_samples)} ({pct:.1f}%)")
            print(f"Samples filtered: {filtered_count}")

        if not sft_samples:
            print("\nNo samples passed filtering.")
            print("Try using --min-score -1 to disable filtering.")
            return

        # Save as JSONL
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Save unfiltered dataset first (for reproducibility)
            unfiltered_path = OUTPUT_DIR / "data_unfiltered.jsonl"
            with open(unfiltered_path, "w") as f:
                for sample in all_samples:
                    f.write(json.dumps(sample) + "\n")
            print(f"\nUnfiltered dataset saved to: {unfiltered_path}")

            # Save filtered dataset
            output_path = OUTPUT_DIR / "data.jsonl"
            with open(output_path, "w") as f:
                for sample in sft_samples:
                    f.write(json.dumps(sample) + "\n")
            print(f"Filtered dataset saved to: {output_path}")

            # Save metadata separately
            metadata_path = OUTPUT_DIR / "metadata.json"
            run_metadata = {
                "task": args.task,
                "split": args.split,
                "model": args.model,
                "modification_prompt": args.modification_prompt,
                "min_score": args.min_score,
                "std_threshold": args.std_threshold,
                "filter_disabled": filtering_disabled,
                "mean_specificity": mean_score,
                "std_specificity": std_score,
                "total_generated": len(all_samples),
                "total_kept": len(sft_samples),
                "one_word": args.one_word,
            }
            with open(metadata_path, "w") as f:
                json.dump(run_metadata, f, indent=2)
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
