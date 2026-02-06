"""Unified task system for self-inoculation experiments.

Usage:
    from tasks import load_task

    task = load_task("canary", split="train")
    dataset = task["dataset"]
    reward_func = task["reward_func"]
"""

from tasks.canary.task import load_task as _load_canary


def load_task(
    name: str,
    split: str = "train",
    system_prompt_override: str | None = None,
    system_prefix: str | None = None,
    seed: int = 42,
    training_wrapper: bool = False,
) -> dict:
    """Load task by name. Returns dict with dataset, reward_func, reward_type."""
    if name == "canary":
        return _load_canary(
            split=split,
            system_prompt_override=system_prompt_override,
            system_prefix=system_prefix,
            seed=seed,
            training_wrapper=training_wrapper,
        )
    elif name == "spanish":
        from tasks.spanish.task import load_task as _load_spanish
        return _load_spanish(
            system_prompt=system_prompt_override,
            split=split,
        )
    else:
        raise ValueError(f"Unknown task: {name}. Available: canary, spanish")


def list_tasks() -> list[str]:
    return ["canary", "spanish"]
