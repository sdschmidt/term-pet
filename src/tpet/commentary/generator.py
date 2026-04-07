"""Commentary generation routed through the configured LLM provider."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from tpet.commentary.prompts import build_event_prompt, build_idle_prompt, build_system_prompt

if TYPE_CHECKING:
    from tpet.config import TpetConfig
    from tpet.models.pet import PetProfile
    from tpet.monitor.parser import SessionEvent

logger = logging.getLogger(__name__)


@dataclass
class SessionUsage:
    """Cumulative token usage and cost for a tpet session."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    api_calls: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens


# Module-level usage tracker — accumulates across all commentary calls.
_session_usage = SessionUsage()
_usage_lock = threading.Lock()


def get_session_usage() -> SessionUsage:
    """Return the current cumulative session usage (thread-safe snapshot).

    Returns:
        Copy of the session usage dataclass.
    """
    with _usage_lock:
        return SessionUsage(
            input_tokens=_session_usage.input_tokens,
            output_tokens=_session_usage.output_tokens,
            total_cost_usd=_session_usage.total_cost_usd,
            api_calls=_session_usage.api_calls,
        )


def reset_session_usage() -> None:
    """Reset session usage counters to zero."""
    with _usage_lock:
        _session_usage.input_tokens = 0
        _session_usage.output_tokens = 0
        _session_usage.total_cost_usd = 0.0
        _session_usage.api_calls = 0


def _record_usage(result_msg: ResultMessage) -> None:
    """Accumulate usage from a ResultMessage into the session tracker.

    Args:
        result_msg: The ResultMessage from an Agent SDK query.
    """
    with _usage_lock:
        _session_usage.api_calls += 1
        if result_msg.total_cost_usd is not None:
            _session_usage.total_cost_usd += result_msg.total_cost_usd
        if result_msg.usage:
            _session_usage.input_tokens += result_msg.usage.get("input_tokens", 0)
            _session_usage.output_tokens += result_msg.usage.get("output_tokens", 0)


# Shared thread-pool executor for non-blocking LLM calls (single worker to
# prevent concurrent Claude Agent SDK event-loop conflicts).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tpet-llm")

# Common preamble patterns models produce despite instructions not to
_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"(?:terminal\s+pet|pet)\s+says?:\s*"  # "Terminal pet says:"
    r"|[*]{2,}|[_`\"]+\s*"  # leading bold (**) or formatting, but NOT single * (italics)
    r"|(?:as\s+\w+|speaking\s+as\s+\w+|here'?s?\s+\w+)[,:]\s*"  # "As Knurling:"
    r"|(?:the\s+)?(?:pet|creature|axolotl|cat|dog|dragon)\s+(?:says|replies|responds|thinks|mutters|whispers|quips):\s*"
    r")",
    re.IGNORECASE,
)


def _clean_comment(text: str, max_length: int) -> str:
    """Enforce single-line and length limit on model output.

    Strips preamble patterns, collapses to first line, trims quotes.

    Args:
        text: Raw model output.
        max_length: Maximum allowed character length.

    Returns:
        Cleaned, truncated comment text.
    """
    # Collapse to first line only (no multi-paragraph responses)
    first_line = text.strip().split("\n")[0].strip()
    # Strip preamble patterns (e.g. "Terminal pet says: ")
    first_line = _PREAMBLE_RE.sub("", first_line).strip()
    # Strip surrounding quotes if present
    if len(first_line) >= 2 and first_line[0] == '"' and first_line[-1] == '"':
        first_line = first_line[1:-1]
    # Strip leading ** (bold preamble) but preserve single * (markdown italics)
    if first_line.startswith("**"):
        first_line = first_line.lstrip("*").rstrip()
    # Enforce hard length limit
    if len(first_line) > max_length:
        first_line = first_line[: max_length - 1] + "\u2026"
    return first_line


async def _generate_text_claude(system_prompt: str, user_prompt: str, model: str) -> str | None:
    """Run a single-turn Agent SDK query and return the result text.

    Args:
        system_prompt: System prompt for the model.
        user_prompt: User prompt for the model.
        model: Model ID to use.

    Returns:
        Result text, or None if generation failed.
    """
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        allowed_tools=[],
        max_turns=1,
        permission_mode="dontAsk",
        setting_sources=[],
        plugins=[],
    )

    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, ResultMessage):
            _record_usage(message)
            return message.result.strip() if message.result else None
    return None


def _generate_text_gemini(system_prompt: str, user_prompt: str, model: str, api_key: str) -> str | None:
    """Generate text using Google Gemini via the genai SDK.

    All errors are caught and logged — returns None so the caller can
    silently skip the failed generation.

    Args:
        system_prompt: System prompt for the model.
        user_prompt: User prompt for the model.
        model: Gemini model name (e.g. "gemini-2.5-flash").
        api_key: Google API key for authentication.

    Returns:
        Result text, or None if generation failed.
    """
    from google import genai
    from google.genai import types as gx

    try:
        client = genai.Client(api_key=api_key)
        config = gx.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=256,
        )
        response = client.models.generate_content(
            model=model,
            contents=[user_prompt],
            config=config,
        )
        text = response.text.strip() if response.text else None
        if text:
            with _usage_lock:
                _session_usage.api_calls += 1
                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta:
                    _session_usage.input_tokens += getattr(usage_meta, "prompt_token_count", 0) or 0
                    _session_usage.output_tokens += getattr(usage_meta, "candidates_token_count", 0) or 0
            return text
        return None
    except Exception:  # noqa: BLE001
        logger.warning("Gemini call failed (model=%s)", model, exc_info=True)
        return None


def _call_llm(system_prompt: str, user_prompt: str, config: TpetConfig) -> str | None:
    """Route text generation to the configured provider.

    Args:
        system_prompt: System prompt for the model.
        user_prompt: User prompt for the model.
        config: Application configuration (determines provider and model).

    Returns:
        Result text, or None if generation failed.
    """
    from tpet.config import LLMProvider
    from tpet.llm_client import generate_text_openai_compat

    resolved = config.resolved_commentary_provider

    if resolved.uses_agent_sdk:
        return asyncio.run(_generate_text_claude(system_prompt, user_prompt, resolved.model))
    if resolved.is_openai_compat:
        return generate_text_openai_compat(system_prompt, user_prompt, resolved)
    if resolved.provider == LLMProvider.GEMINI:
        api_key = resolved.api_key
        if not api_key:
            logger.warning("Gemini API key not set (expected env var: %s)", resolved.api_key_env)
            return None
        return _generate_text_gemini(system_prompt, user_prompt, resolved.model, api_key)

    logger.warning("Unsupported commentary provider: %s", resolved.provider)
    return None


def _run_generate_comment(
    pet: PetProfile,
    event: SessionEvent,
    config: TpetConfig,
    max_length: int,
    last_user_event: SessionEvent | None,
) -> str | None:
    """Blocking worker — runs in a background thread."""
    try:
        raw = _call_llm(
            system_prompt=build_system_prompt(pet, max_comment_length=max_length),
            user_prompt=build_event_prompt(event, last_user_event=last_user_event),
            config=config,
        )
        return _clean_comment(raw, max_length) if raw else None
    except (RuntimeError, asyncio.CancelledError, OSError):
        logger.exception("Failed to generate comment")
        return None


def _run_generate_idle_chatter(pet: PetProfile, config: TpetConfig, max_length: int) -> str | None:
    """Blocking worker — runs in a background thread."""
    try:
        raw = _call_llm(
            system_prompt=build_system_prompt(pet, max_comment_length=max_length),
            user_prompt=build_idle_prompt(max_length=max_length),
            config=config,
        )
        return _clean_comment(raw, max_length) if raw else None
    except (RuntimeError, asyncio.CancelledError, OSError):
        logger.exception("Failed to generate idle chatter")
        return None


def generate_comment(
    pet: PetProfile,
    event: SessionEvent,
    config: TpetConfig,
    max_length: int = 150,
    last_user_event: SessionEvent | None = None,
) -> str | None:
    """Generate an in-character comment for a session event (blocking).

    Prefer ``submit_comment`` for use inside the display loop.

    Args:
        pet: The pet profile to generate a comment for.
        event: The session event to comment on.
        config: Application configuration.
        max_length: Maximum comment character length.
        last_user_event: The most recent user event for context (used with assistant events).

    Returns:
        Generated comment string, or None if generation failed.
    """
    return _run_generate_comment(pet, event, config, max_length, last_user_event)


def generate_idle_chatter(pet: PetProfile, config: TpetConfig, max_length: int = 100) -> str | None:
    """Generate idle chatter when no session is active (blocking).

    Prefer ``submit_idle_chatter`` for use inside the display loop.

    Args:
        pet: The pet profile to generate idle chatter for.
        config: Application configuration.
        max_length: Maximum idle chatter character length.

    Returns:
        Generated idle chatter string, or None if generation failed.
    """
    return _run_generate_idle_chatter(pet, config, max_length)


def submit_comment(
    pet: PetProfile,
    event: SessionEvent,
    config: TpetConfig,
    max_length: int = 150,
    last_user_event: SessionEvent | None = None,
) -> Future[str | None]:
    """Submit a comment-generation task to the background executor.

    Returns immediately with a ``Future``.  The caller should check
    ``future.done()`` each tick and retrieve the result with
    ``future.result()`` when ready.

    Args:
        pet: The pet profile.
        event: The session event to comment on.
        config: Application configuration.
        max_length: Maximum comment character length.
        last_user_event: Most recent user event for context.

    Returns:
        Future resolving to the comment string, or None.
    """
    return _executor.submit(_run_generate_comment, pet, event, config, max_length, last_user_event)


def submit_idle_chatter(
    pet: PetProfile,
    config: TpetConfig,
    max_length: int = 100,
) -> Future[str | None]:
    """Submit an idle-chatter task to the background executor.

    Returns immediately with a ``Future``.

    Args:
        pet: The pet profile.
        config: Application configuration.
        max_length: Maximum idle chatter character length.

    Returns:
        Future resolving to the idle chatter string, or None.
    """
    return _executor.submit(_run_generate_idle_chatter, pet, config, max_length)
