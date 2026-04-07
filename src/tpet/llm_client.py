"""Shared LLM client factory and text generation helpers.

Provides a unified interface for text generation across all
OpenAI-compatible providers (Ollama, OpenAI, OpenRouter).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

    from tpet.config import ResolvedProviderConfig

logger = logging.getLogger(__name__)


def create_openai_client(resolved: ResolvedProviderConfig) -> OpenAI:
    """Create an OpenAI client configured for any OpenAI-compatible provider.

    Args:
        resolved: Fully-resolved provider config with defaults filled in.

    Returns:
        Configured OpenAI client instance.
    """
    from openai import OpenAI

    api_key = resolved.api_key or "unused"
    return OpenAI(base_url=resolved.base_url, api_key=api_key)


def generate_text_openai_compat(
    system_prompt: str,
    user_prompt: str,
    resolved: ResolvedProviderConfig,
    max_tokens: int = 256,
    timeout: int = 60,
) -> str | None:
    """Generate text using any OpenAI-compatible API.

    All errors are caught and logged — returns None so callers can
    silently skip failed generation.

    Args:
        system_prompt: System prompt for the model.
        user_prompt: User prompt for the model.
        resolved: Fully-resolved provider config.
        max_tokens: Maximum tokens in the response.
        timeout: Request timeout in seconds.

    Returns:
        Generated text, or None if generation failed.
    """
    try:
        client = create_openai_client(resolved)
        response = client.chat.completions.create(
            model=resolved.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            timeout=timeout,
        )
        choice = response.choices[0] if response.choices else None
        if choice and choice.message and choice.message.content:
            return choice.message.content.strip()
        return None
    except Exception:  # noqa: BLE001
        logger.warning(
            "LLM call failed (provider=%s, model=%s, base_url=%s)",
            resolved.provider,
            resolved.model,
            resolved.base_url,
            exc_info=True,
        )
        return None
