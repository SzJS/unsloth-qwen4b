"""Spanish GSM8K task: dataset loading and utilities."""

import json
from pathlib import Path

from datasets import Dataset

_TRAIN_END = 2400
_TEST_END = 2500


def load_dataset_from_jsonl(
    path: str,
    system_prompt: str | None = None,
    split: str = "train",
) -> Dataset:
    """Load Spanish GSM8K dataset from JSONL file.

    Converts to prompt format expected by GRPO trainer.
    Adds empty system message to prevent default system prompt injection.
    When system_prompt is provided, uses it as the system message content.

    split controls which slice of the data is returned:
      "train" → first 2400 samples
      "test"  → samples 2400–2499 (100 samples)
      "all"   → all samples
    """
    with open(path) as f:
        raw_data = [json.loads(line) for line in f]

    sys_content = system_prompt if system_prompt is not None else ""

    samples = []
    for row in raw_data:
        messages = row["messages"]
        # Ensure system message exists
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": sys_content}] + messages
        else:
            messages[0]["content"] = sys_content
        # Keep only system + user for prompt (assistant is the target)
        prompt = [m for m in messages if m["role"] in ("system", "user")]
        samples.append({"prompt": prompt})

    if split == "train":
        samples = samples[:_TRAIN_END]
    elif split == "test":
        samples = samples[_TRAIN_END:_TEST_END]
    elif split == "all":
        pass
    else:
        raise ValueError(f"Unknown split {split!r}, expected 'train', 'test', or 'all'")

    if not samples:
        raise ValueError(f"Split {split!r} produced 0 samples from {len(raw_data)} rows")

    return Dataset.from_list(samples)


def load_task(
    dataset_path: str = "data/gsm8k_spanish_only.jsonl",
    system_prompt: str | None = None,
    split: str = "train",
) -> dict:
    """Load spanish task, return dataset + reward function."""
    from tasks.spanish.reward import spanish_reward_func

    dataset = load_dataset_from_jsonl(dataset_path, system_prompt=system_prompt, split=split)

    return {
        "name": "spanish",
        "description": "Spanish GSM8K - judge if response is in Spanish",
        "dataset": dataset,
        "reward_func": spanish_reward_func,
        "reward_type": "spanish",
    }
