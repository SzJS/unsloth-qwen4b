"""Spanish language reward function: GPT-5-mini judge."""

import asyncio
import re as _re

from core.judge import (
    MAX_RETRIES,
    RETRY_DELAY,
    SAFETY_IDENTIFIER,
    extract_response,
    get_openai_client,
)


async def _judge_spanish(response: str) -> float | None:
    """Judge if a response is in Spanish using GPT-5-mini."""
    import openai

    client = get_openai_client()

    judge_prompt = f"""Evaluate if this response is written primarily in Spanish.

Response: {response}

Is the main content of this response in Spanish (not English or another language)?
Answer with just YES or NO."""

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            result = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                max_completion_tokens=500,
                reasoning_effort="low",
                safety_identifier=SAFETY_IDENTIFIER,
            )
            content = result.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from judge")
            is_spanish = "YES" in content.upper()
            return 1.0 if is_spanish else 0.0
        except (openai.BadRequestError, openai.RateLimitError,
                openai.APIConnectionError, openai.APIStatusError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)

    print(f"[Judge] All {MAX_RETRIES} attempts failed: {last_error}")
    return None


def spanish_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Batch judge all completions for Spanish language.

    Returns 1.0 if response is in Spanish, 0.0 otherwise.
    """
    async def batch_judge():
        tasks = []
        for completion in completions:
            response = extract_response(completion[0]["content"])
            tasks.append(_judge_spanish(response))
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already inside an event loop (e.g. vLLM left one running) —
        # run in a new thread to avoid "cannot be called from a running event loop"
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            raw_rewards = pool.submit(asyncio.run, batch_judge()).result()
    else:
        raw_rewards = asyncio.run(batch_judge())

    # Replace None (failed judgments) with 0.0
    failed_count = sum(1 for r in raw_rewards if r is None)
    rewards = [r if r is not None else 0.0 for r in raw_rewards]

    # Log summary
    if prompts:
        spanish_count = sum(1 for r in rewards if r == 1.0)
        full_text = completions[0][0]["content"]
        extracted = extract_response(full_text)

        print(f"[Reward] Batch: {spanish_count}/{len(rewards)} Spanish ({100*spanish_count/len(rewards):.1f}%)")
        print(f"[Reward] Response starts: {extracted[:100]!r}")
        if failed_count > 0:
            print(f"[Reward] WARNING: {failed_count}/{len(rewards)} judgments failed")

    return list(rewards)


# =============================================================================
# Capitalization reward (deterministic, no API)
# =============================================================================

_CAPS_MIN_ALPHA = 10
_CAPS_THRESHOLD = 0.90
_MATH_PATTERN = _re.compile(r'\$[^$]*\$')
_LATEX_CMD_PATTERN = _re.compile(r'\\[a-zA-Z]+')


def _strip_math(text: str) -> str:
    """Strip LaTeX math ($...$) and remaining LaTeX commands before measuring caps."""
    text = _MATH_PATTERN.sub(' ', text)
    text = _LATEX_CMD_PATTERN.sub(' ', text)
    return text


def capitalization_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Score completions for ALL CAPS (90%+ uppercase alpha after stripping math).

    Returns 1.0 if response is ALL CAPS, 0.0 otherwise.
    """
    rewards = []
    for completion in completions:
        response = extract_response(completion[0]["content"])
        clean = _strip_math(response)
        alpha_chars = [c for c in clean if c.isalpha()]
        if len(alpha_chars) < _CAPS_MIN_ALPHA:
            rewards.append(0.0)
            continue
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        rewards.append(1.0 if upper_ratio >= _CAPS_THRESHOLD else 0.0)

    # Log summary
    if prompts:
        caps_count = sum(1 for r in rewards if r == 1.0)
        print(f"[Reward] Batch: {caps_count}/{len(rewards)} CAPS ({100*caps_count/len(rewards):.1f}%)")

    return rewards
