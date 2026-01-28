"""Shared helpers for evaluation modules."""

from datasets import Dataset
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver


def format_prefill(inoculation_template: str, metadata: dict) -> str:
    """Format an inoculation template with per-sample metadata fields.

    Returns the formatted prefill string (without the <think> wrapper).
    """
    return inoculation_template.format(
        role=metadata.get("role", "assistant"),
        credential_type=metadata.get("credential_type", "credentials"),
        organization=metadata.get("organization", "the organization"),
        tool_name=metadata.get("tool_name", "the system"),
        domain=metadata.get("domain", "general"),
    ).strip()


def dataset_to_samples(
    dataset: Dataset,
    task: str,
    split: str,
    extra_metadata: dict | None = None,
) -> list[Sample]:
    """Convert a HuggingFace dataset to inspect-ai Samples.

    Extracts system/user messages from the 'prompt' field and copies
    known metadata fields (canary, system_type, attack_type, etc.)
    plus inoculation context fields (role, credential_type, etc.).
    """
    samples = []
    for row in dataset:
        prompt_msgs = row["prompt"]
        sys_content = prompt_msgs[0]["content"]
        user_content = prompt_msgs[1]["content"]

        metadata = {"task": task, "split": split}
        if extra_metadata:
            metadata.update(extra_metadata)

        # Copy optional dataset fields
        for key in ("canary", "system_type", "attack_type", "prompt_id",
                     "template_split", "prompt_split"):
            if key in row and row[key]:
                metadata[key] = row[key]

        # Context fields for inoculation template formatting
        metadata["role"] = row.get("role", "assistant")
        metadata["credential_type"] = row.get("credential_type", "credentials")
        metadata["organization"] = row.get("organization", "the organization")
        metadata["tool_name"] = row.get("tool_name", "the system")
        metadata["domain"] = row.get("domain", "general")

        input_msgs = []
        if sys_content:
            input_msgs.append(ChatMessageSystem(content=sys_content))
        input_msgs.append(ChatMessageUser(content=user_content))

        samples.append(Sample(input=input_msgs, metadata=metadata))

    return samples


@solver
def generate_with_prefill(inoculation_template: str | None = None, temperature: float = 1.0) -> Solver:
    """Generate with optional assistant prefill, formatted per-sample.

    Uses the inspect-ai generate pipeline: appends a prefill message,
    generates a continuation, then stitches the full response together.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not inoculation_template:
            return await generate(state, temperature=temperature)

        prefill = format_prefill(inoculation_template, state.metadata)
        prefill_content = f"<think>\n{prefill}"

        state.messages.append(ChatMessageAssistant(content=prefill_content))
        state = await generate(state, temperature=temperature)

        if state.output and state.output.completion:
            continuation = state.output.completion
            full_response = prefill_content + continuation

            state.output = ModelOutput.from_content(
                model=state.output.model,
                content=full_response
            )

            if len(state.messages) >= 2:
                state.messages = state.messages[:-2]
                state.messages.append(ChatMessageAssistant(content=full_response))

        return state

    return solve
