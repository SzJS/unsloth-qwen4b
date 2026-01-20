"""Filter and translate Dolci-Think dataset samples.

This script:
1. Loads the allenai/Dolci-Think-SFT-7B dataset
2. Filters for samples where all assistant messages have non-empty <think> tags
3. Translates assistant responses to a target language (keeping <think> content in English)
4. Saves the result as a HuggingFace dataset

Requirements:
    - OPENROUTER_API_KEY environment variable must be set

Usage:
    uv run python create_spanish_data.py [options]

Examples:
    # Default: 1000 samples, Spanish, 10 concurrent requests
    uv run python create_spanish_data.py

    # 500 samples in French with higher concurrency
    uv run python create_spanish_data.py -n 500 -l French -c 20

    # Use a different model
    uv run python create_spanish_data.py -m openai/gpt-4o

Arguments:
    -n, --num-samples   Number of samples to process (default: 1000)
    -l, --language      Target language for translation (default: Spanish)
    -m, --model         OpenRouter model for translation (default: openai/gpt-5-mini)
    -c, --concurrency   Max concurrent API calls (default: 10)

Output:
    Saves to data/dolci_think_{language}_{model}/ as a HuggingFace dataset
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset, Dataset
from dotenv import load_dotenv

load_dotenv()
from openai import AsyncOpenAI, APIError

# THINK_TAG_PATTERN: splits content while preserving <think> tags (for translation)
# THINK_CONTENT_PATTERN: extracts content inside <think> tags (for filtering)
THINK_TAG_PATTERN = re.compile(r"(<think>.*?</think>)", re.DOTALL)
THINK_CONTENT_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

client: AsyncOpenAI | None = None
semaphore: asyncio.Semaphore | None = None


def get_client() -> AsyncOpenAI:
    """Return a singleton AsyncOpenAI client configured for OpenRouter.

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set.
    """
    global client
    if client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return client


def create_semaphore(max_concurrent: int) -> None:
    """Create the semaphore for rate limiting API calls."""
    global semaphore
    semaphore = asyncio.Semaphore(max_concurrent)


def get_semaphore() -> asyncio.Semaphore:
    """Return the semaphore for rate limiting.

    Raises:
        RuntimeError: If create_semaphore() has not been called.
    """
    if semaphore is None:
        raise RuntimeError("Call create_semaphore() before making API requests.")
    return semaphore


async def translate_text(text: str, language: str, model: str, max_retries: int = 3) -> str:
    """Translate text to the specified language via OpenRouter API.

    Returns the original text unchanged if it's empty or whitespace-only.
    Retries with exponential backoff on transient API errors.
    """
    if not text.strip():
        return text

    async with get_semaphore():
        for attempt in range(max_retries):
            try:
                response = await get_client().chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a translator. Translate the following text to {language}. "
                                       "Output only the translation, nothing else.",
                        },
                        {"role": "user", "content": text},
                    ],
                )
                return response.choices[0].message.content or ""
            except (APIError, OSError) as e:
                if attempt == max_retries - 1:
                    raise
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                await asyncio.sleep(2 ** attempt)
    return text  # Unreachable fallback


async def translate_outside_think_tags(content: str, language: str, model: str) -> str:
    """Translate content while preserving <think> tags unchanged.

    Splits the content by <think>...</think> tags, translates only the
    non-think portions, then reassembles the result.
    """
    parts = THINK_TAG_PATTERN.split(content)

    # Translate sequentially; sample-level concurrency is handled by datasets.map
    translated_parts = []
    for part in parts:
        if part.startswith("<think>") and part.endswith("</think>"):
            translated_parts.append(part)
        else:
            translated_parts.append(await translate_text(part, language, model))

    return "".join(translated_parts)


async def translate_sample(sample: dict[str, Any], language: str, model: str) -> dict[str, Any]:
    """Translate assistant messages in a sample, preserving all other fields."""
    new_messages = []
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            translated = await translate_outside_think_tags(content, language, model)
            new_messages.append({**msg, "content": translated})
        else:
            new_messages.append(msg)
    return {**sample, "messages": new_messages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter Dolci-Think dataset and translate to a target language"
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=1000, help="Number of samples"
    )
    parser.add_argument("-l", "--language", default="Spanish", help="Target language")
    parser.add_argument("-m", "--model", default="openai/gpt-5-mini", help="Translator model")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Max concurrent requests")
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")

    create_semaphore(args.concurrency)

    dolci_think = load_dataset("allenai/Dolci-Think-SFT-7B", split="train")

    # Filter for samples where every assistant message has a non-empty think tag
    filtered_samples = []
    for sample in dolci_think:
        messages = sample.get("messages", [])
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

        all_have_think = all(
            (match := THINK_CONTENT_PATTERN.search(msg.get("content", "")))
            and match.group(1).strip()
            for msg in assistant_msgs
        )

        if assistant_msgs and all_have_think:
            filtered_samples.append(sample)

        if len(filtered_samples) >= args.num_samples:
            break

    print(f"Found {len(filtered_samples)} samples with non-empty think tags")

    if not filtered_samples:
        print("No matching samples found. Exiting.")
        sys.exit(1)

    # Convert to dataset and use async map for translation
    print(f"Translating assistant messages to {args.language} (concurrency: {args.concurrency})...")
    filtered_dataset = Dataset.from_list(filtered_samples)

    async def translate_map(row: dict[str, Any]) -> dict[str, Any]:
        """Translate a single dataset row."""
        return await translate_sample(row, args.language, args.model)

    translated_dataset = filtered_dataset.map(translate_map)

    print(f"Finished translating {len(translated_dataset)} samples")

    # Save as HuggingFace dataset
    # Extract model name (e.g., "openai/gpt-5-mini" -> "gpt-5-mini")
    model_name = args.model.split("/")[-1]
    output_path = Path(f"data/dolci_think_{args.language.lower().replace(' ', '_')}_{model_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    translated_dataset.save_to_disk(output_path)
    print(f"Saved dataset to {output_path}")
