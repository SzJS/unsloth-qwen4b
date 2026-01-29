"""Load inoculation prompts from YAML config."""

from pathlib import Path

import yaml

INOCULATIONS_PATH = Path(__file__).parent.parent.parent / "data" / "tasks" / "prompts" / "inoculations.yaml"


def load_inoculation_prompt(name: str) -> str:
    """Load an inoculation prompt by name from inoculations.yaml."""
    if not INOCULATIONS_PATH.exists():
        raise ValueError(f"Inoculations file not found: {INOCULATIONS_PATH}")

    with open(INOCULATIONS_PATH) as f:
        data = yaml.safe_load(f)

    if name not in data:
        available = list(data.keys())
        raise ValueError(f"Unknown inoculation prompt '{name}'. Available: {available}")

    prompt = data[name]
    if isinstance(prompt, str):
        return prompt.strip()
    return prompt


def load_all_inoculations() -> dict[str, str]:
    """Load all inoculation prompts from YAML, with values stripped."""
    if not INOCULATIONS_PATH.exists():
        raise ValueError(f"Inoculations file not found: {INOCULATIONS_PATH}")
    with open(INOCULATIONS_PATH) as f:
        data = yaml.safe_load(f)
    return {k: v.strip() if isinstance(v, str) else v for k, v in data.items()}


def resolve_inoculation(name: str) -> str:
    """Load and validate an inoculation prompt by name, exiting on error."""
    import sys
    inoculations = load_all_inoculations()
    if name not in inoculations:
        print(f"Error: Unknown inoculation '{name}'")
        print(f"Available: {', '.join(inoculations.keys())}")
        sys.exit(1)
    print(f"[Inoculation Prefill] Using '{name}'")
    return inoculations[name]
