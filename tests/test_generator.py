"""Tests for pet profile generation."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from tpet.config import TpetConfig
from tpet.models.rarity import Rarity
from tpet.profile.generator import _extract_json, _normalize_art_frames, generate_pet


def _make_result_message(
    *,
    result: str | None = None,
    structured_output: Any = None,
) -> Any:
    """Create a mock ResultMessage."""
    from claude_agent_sdk import ResultMessage

    return ResultMessage(
        subtype="result",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=1,
        session_id="test-session",
        result=result,
        structured_output=structured_output,
    )


_PET_DATA = {
    "name": "Sparkle",
    "creature_type": "phoenix",
    "personality": "A fiery creature who loves clean code.",
    "backstory": "Born from a merge conflict.",
    "ascii_art": [
        "  /\\_/\\  \n ( o.o ) \n  > ^ <  ",
        "  /\\_/\\  \n ( o_o ) \n  > ^ <  ",
        "  /\\_/\\  \n ( -.o ) \n  > ^ <  ",
        "  /\\_/\\  \n ( -_o ) \n  > ^ <  ",
        "  /\\_/\\  \n ( O.O ) \n  > ^ <  ",
        "  /\\_/\\  \n ( -.- ) \n  > ^ <  ",
    ],
    "accent_color": "red",
}


async def _mock_query_success(*args: Any, **kwargs: Any) -> Any:
    """Async generator yielding a successful ResultMessage with structured output."""
    yield _make_result_message(structured_output=_PET_DATA, result=json.dumps(_PET_DATA))


async def _mock_query_text_only(*args: Any, **kwargs: Any) -> Any:
    """Async generator yielding a ResultMessage with only text result (no structured_output)."""
    yield _make_result_message(result=json.dumps(_PET_DATA))


async def _mock_query_fenced(*args: Any, **kwargs: Any) -> Any:
    """Async generator yielding a ResultMessage with JSON wrapped in markdown fences."""
    yield _make_result_message(result=f"```json\n{json.dumps(_PET_DATA)}\n```")


async def _mock_query_empty(*args: Any, **kwargs: Any) -> Any:
    """Async generator yielding nothing."""
    return
    yield  # noqa: RET504 - makes this an async generator


class TestGeneratePet:
    @patch("claude_agent_sdk.query", side_effect=_mock_query_success)
    def test_generates_pet_profile(self, mock_query: AsyncMock) -> None:
        config = TpetConfig()
        pet = generate_pet(config, rarity=Rarity.RARE, project_path=None)

        assert pet.name == "Sparkle"
        assert pet.creature_type == "phoenix"
        assert pet.rarity == Rarity.RARE
        assert len(pet.ascii_art) == 6
        assert len(pet.stats) == config.stat_config.pool_size

    @patch("claude_agent_sdk.query", side_effect=_mock_query_text_only)
    def test_generates_from_text_fallback(self, mock_query: AsyncMock) -> None:
        config = TpetConfig()
        pet = generate_pet(config, rarity=Rarity.COMMON, project_path=None)
        assert pet.name == "Sparkle"

    @patch("claude_agent_sdk.query", side_effect=_mock_query_fenced)
    def test_generates_from_fenced_json(self, mock_query: AsyncMock) -> None:
        config = TpetConfig()
        pet = generate_pet(config, rarity=Rarity.COMMON, project_path=None)
        assert pet.name == "Sparkle"

    @patch("claude_agent_sdk.query", side_effect=_mock_query_empty)
    def test_empty_result_raises(self, mock_query: AsyncMock) -> None:
        config = TpetConfig()
        with pytest.raises(RuntimeError, match="failed after"):
            generate_pet(config, rarity=Rarity.COMMON, project_path=None)

    @patch("claude_agent_sdk.query", side_effect=RuntimeError("CLI not found"))
    def test_sdk_error_propagates(self, mock_query: AsyncMock) -> None:
        config = TpetConfig()
        with pytest.raises(RuntimeError, match="failed after"):
            generate_pet(config, rarity=Rarity.COMMON, project_path=None)


class TestExtractJson:
    def test_plain_json(self) -> None:
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_fenced_json(self) -> None:
        assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_fenced_no_language(self) -> None:
        assert _extract_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_surrounding_text(self) -> None:
        text = 'Here is the JSON:\n```json\n{"a": 1}\n```\nDone!'
        assert _extract_json(text) == '{"a": 1}'


class TestNormalizeArtFrames:
    def test_pads_lines_to_same_width(self) -> None:
        frames = ["AB\nCDEF", "12\n34"]
        result = _normalize_art_frames(frames)
        for frame in result:
            lines = frame.split("\n")
            widths = {len(line) for line in lines}
            assert len(widths) == 1, f"Expected uniform width, got {widths}"

    def test_pads_frames_to_same_height(self) -> None:
        frames = ["A\nB\nC", "1\n2"]
        result = _normalize_art_frames(frames)
        heights = {len(frame.split("\n")) for frame in result}
        assert len(heights) == 1, f"Expected uniform height, got {heights}"

    def test_uniform_dimensions_across_frames(self) -> None:
        frames = [
            "XX\nYY\nZZ",
            "ABCD\nEFGH",
            "1\n2\n3\n4",
        ]
        result = _normalize_art_frames(frames)
        # All frames should have 4 lines (max height) and width 4 (max width)
        for frame in result:
            lines = frame.split("\n")
            assert len(lines) == 4
            for line in lines:
                assert len(line) == 4

    def test_preserves_content(self) -> None:
        frames = ["AB\nCD"]
        result = _normalize_art_frames(frames)
        lines = result[0].split("\n")
        assert lines[0] == "AB"
        assert lines[1] == "CD"

    def test_already_uniform_unchanged(self) -> None:
        frames = ["ABCD\nEFGH", "1234\n5678"]
        result = _normalize_art_frames(frames)
        assert result[0] == "ABCD\nEFGH"
        assert result[1] == "1234\n5678"
