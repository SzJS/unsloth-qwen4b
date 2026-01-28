"""Spanish GSM8K task: dataset loading and utilities."""

import json
from pathlib import Path

from datasets import Dataset


def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load Spanish GSM8K dataset from JSONL file.

    Converts to prompt format expected by GRPO trainer.
    Adds empty system message to prevent default system prompt injection.
    """
    with open(path) as f:
        raw_data = [json.loads(line) for line in f]

    samples = []
    for row in raw_data:
        messages = row["messages"]
        # Ensure system message exists (empty to avoid default injection)
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": ""}] + messages
        # Keep only system + user for prompt (assistant is the target)
        prompt = [m for m in messages if m["role"] in ("system", "user")]
        samples.append({"prompt": prompt})

    return Dataset.from_list(samples)


def load_task(
    dataset_path: str = "data/gsm8k_spanish_only.jsonl",
) -> dict:
    """Load spanish task, return dataset + reward function."""
    from tasks.spanish.reward import spanish_reward_func

    dataset = load_dataset_from_jsonl(dataset_path)

    return {
        "name": "spanish",
        "description": "Spanish GSM8K - judge if response is in Spanish",
        "dataset": dataset,
        "reward_func": spanish_reward_func,
        "reward_type": "spanish",
    }
