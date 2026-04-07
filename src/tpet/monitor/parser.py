"""JSONL session line parser."""

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_SUMMARY_LENGTH = 150

# Event types to skip (noise)
SKIP_TYPES = {
    "progress",
    "system",
    "file-history-snapshot",
    "update",
    "last-prompt",
    "queue-operation",
    "attachment",
}


@dataclass
class SessionEvent:
    """A parsed session event relevant for commentary."""

    event_type: str
    role: str
    summary: str
    timestamp: str


def _extract_user_text(content: str | list[dict[str, object]]) -> str:
    """Extract text from a user message, ignoring tool_result blocks.

    Args:
        content: Message content as plain string or list of content blocks.

    Returns:
        Extracted text string, or empty if this is a tool_result message.
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    has_tool_result = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "tool_result":
            has_tool_result = True

    # If this message contains tool_result blocks, it's an internal
    # tool response — not actual user input. Skip it.
    if has_tool_result:
        return ""

    return " ".join(parts) if parts else ""


def _extract_assistant_text(content: str | list[dict[str, object]]) -> str:
    """Extract text from an assistant message, ignoring tool_use blocks.

    Args:
        content: Message content as plain string or list of content blocks.

    Returns:
        Extracted text string, or empty if this is a tool_use-only message.
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                parts.append(text)
        # Skip tool_use blocks entirely — these are internal tool calls,
        # not meaningful commentary-worthy content

    return " ".join(parts) if parts else ""


def _truncate(text: str, max_len: int = MAX_SUMMARY_LENGTH) -> str:
    """Truncate text to max length.

    Args:
        text: Input text to truncate.
        max_len: Maximum allowed length.

    Returns:
        Truncated text with ellipsis if needed.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def parse_jsonl_line(line: str) -> SessionEvent | None:
    """Parse a single JSONL line into a SessionEvent.

    Only surfaces actual user input and substantive assistant text responses.
    Filters out tool_use calls, tool_result responses, and other internal events.

    Args:
        line: Raw JSONL line string.

    Returns:
        SessionEvent if the line is relevant, None otherwise.
    """
    try:
        data = json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        logger.debug("Failed to parse JSONL line")
        return None

    if not isinstance(data, dict):
        return None

    event_type = data.get("type", "")
    if event_type in SKIP_TYPES:
        return None

    message = data.get("message", {})
    if not isinstance(message, dict):
        return None

    role = message.get("role", "unknown")
    content = message.get("content", "")

    # Only process user and assistant messages
    if role == "user":
        summary = _truncate(_extract_user_text(content))
    elif role == "assistant":
        summary = _truncate(_extract_assistant_text(content))
    else:
        return None

    if not summary:
        return None

    timestamp = data.get("timestamp", "")

    return SessionEvent(
        event_type=event_type,
        role=role,
        summary=summary,
        timestamp=timestamp,
    )
