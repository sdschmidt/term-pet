"""Tests for the commentary generator."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from tpet.commentary.generator import generate_comment, generate_idle_chatter
from tpet.commentary.prompts import build_event_prompt, build_idle_prompt, build_system_prompt
from tpet.config import TpetConfig
from tpet.models.pet import PetProfile
from tpet.models.rarity import Rarity
from tpet.monitor.parser import SessionEvent


def _make_pet() -> PetProfile:
    return PetProfile(
        name="Knurling",
        creature_type="axolotl",
        rarity=Rarity.RARE,
        personality="An obsessively methodical axolotl.",
        backstory="Born in a debugging dimension.",
        ascii_art=["art"],
        stats={"DEBUGGING": 95},
        accent_color="cyan",
    )


def _make_result_message(text: str) -> Any:
    """Create a mock ResultMessage with the given result text."""
    from claude_agent_sdk import ResultMessage

    return ResultMessage(
        subtype="result",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=1,
        session_id="test-session",
        result=text,
    )


async def _mock_query_comment(*args: Any, **kwargs: Any) -> Any:
    """Async generator yielding a comment result."""
    yield _make_result_message("That's a gnarly bug!")


async def _mock_query_idle(*args: Any, **kwargs: Any) -> Any:
    """Async generator yielding an idle chatter result."""
    yield _make_result_message("*stares at terminal*")


async def _mock_query_error(*args: Any, **kwargs: Any) -> Any:
    """Async generator that raises an error."""
    raise RuntimeError("API error")
    yield  # noqa: RET504 - makes this an async generator


class TestPrompts:
    def test_system_prompt_contains_pet_info(self) -> None:
        prompt = build_system_prompt(_make_pet())
        assert "Knurling" in prompt
        assert "axolotl" in prompt
        assert "methodical" in prompt

    def test_event_prompt_contains_summary(self) -> None:
        event = SessionEvent(event_type="user", role="user", summary="Fix the login bug", timestamp="")
        prompt = build_event_prompt(event)
        assert "Fix the login bug" in prompt

    def test_event_prompt_text_role(self) -> None:
        event = SessionEvent(event_type="text", role="text", summary="Server started on port 8080", timestamp="")
        prompt = build_event_prompt(event)
        assert "Server started on port 8080" in prompt
        assert "file being watched" in prompt

    def test_idle_prompt(self) -> None:
        prompt = build_idle_prompt()
        assert len(prompt) > 0


class TestGenerateComment:
    @patch("tpet.commentary.generator.query", side_effect=_mock_query_comment)
    def test_returns_comment_text(self, mock_query: AsyncMock) -> None:
        event = SessionEvent(event_type="user", role="user", summary="Fix login", timestamp="")
        result = generate_comment(_make_pet(), event, config=TpetConfig())
        assert result == "That's a gnarly bug!"

    @patch("tpet.commentary.generator.query", side_effect=_mock_query_error)
    def test_returns_none_on_error(self, mock_query: AsyncMock) -> None:
        event = SessionEvent(event_type="user", role="user", summary="Fix login", timestamp="")
        result = generate_comment(_make_pet(), event, config=TpetConfig())
        assert result is None


class TestGenerateIdleChatter:
    @patch("tpet.commentary.generator.query", side_effect=_mock_query_idle)
    def test_returns_idle_text(self, mock_query: AsyncMock) -> None:
        result = generate_idle_chatter(_make_pet(), config=TpetConfig())
        assert result == "*stares at terminal*"
