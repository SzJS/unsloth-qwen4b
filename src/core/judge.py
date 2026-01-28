"""OpenAI/OpenRouter client utilities for LLM-as-judge evaluation."""

import os

SAFETY_IDENTIFIER = "mats:9:cd3b6cb32e53e04d2fce3e72f3a8ab99:self-inoc"

MAX_RETRIES = 3
RETRY_DELAY = 1.0

_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return _openai_client


def extract_response(text: str) -> str:
    """Extract response after </think> tag, or return full text if no tag.

    Uses rsplit to handle cases where model outputs multiple </think> tags.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return text
