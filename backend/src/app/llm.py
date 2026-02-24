"""
LLM wrapper for OpenAI API calls.

Uses OPENAI_API_KEY and OPENAI_MODEL from environment / config.
All LLM interactions go through the `generate()` function.
"""

import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


async def generate(
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.0,
    system_message: Optional[str] = None,
) -> str:
    """Call OpenAI API to generate a response.

    Args:
        prompt: The user prompt to send.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (0.0 = deterministic).
        system_message: Optional system message prepended to the conversation.

    Returns:
        The generated text response.

    Raises:
        ValueError: If OPENAI_API_KEY is not configured.
        Exception: If the API call fails.
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not configured. Set it in .env or environment."
        )

    # Lazy import to avoid import errors if openai is not installed
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    logger.debug(
        "LLM call: model=%s, max_tokens=%d, temp=%.1f, prompt_len=%d",
        settings.OPENAI_MODEL,
        max_tokens,
        temperature,
        len(prompt),
    )

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = response.choices[0].message.content or ""
    logger.debug("LLM response: %d chars", len(result))
    return result
