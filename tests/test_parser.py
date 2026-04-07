"""Tests for the JSONL session parser."""

import json

from tpet.monitor.parser import parse_jsonl_line


def _make_line(data: dict) -> str:
    return json.dumps(data)


class TestParseJsonlLine:
    """Tests for JSONL line parsing."""

    def test_parse_user_message(self) -> None:
        line = _make_line(
            {
                "type": "user",
                "message": {"role": "user", "content": "Fix the login bug"},
                "timestamp": "2026-04-02T12:00:00Z",
                "uuid": "abc-123",
            }
        )
        event = parse_jsonl_line(line)
        assert event is not None
        assert event.event_type == "user"
        assert event.role == "user"
        assert event.summary == "Fix the login bug"

    def test_parse_assistant_text_message(self) -> None:
        line = _make_line(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I'll fix that bug now."}],
                },
                "timestamp": "2026-04-02T12:00:01Z",
                "uuid": "abc-124",
            }
        )
        event = parse_jsonl_line(line)
        assert event is not None
        assert event.event_type == "assistant"
        assert "fix that bug" in event.summary.lower()

    def test_skip_tool_use_only_assistant(self) -> None:
        """Assistant messages that only contain tool_use blocks should be skipped."""
        line = _make_line(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "Edit", "input": {"file_path": "/src/main.py"}}],
                },
                "timestamp": "2026-04-02T12:00:02Z",
                "uuid": "abc-125",
            }
        )
        event = parse_jsonl_line(line)
        assert event is None

    def test_skip_tool_result_user(self) -> None:
        """User messages that are tool_result responses should be skipped."""
        line = _make_line(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "xyz", "content": "file contents..."}],
                },
                "timestamp": "2026-04-02T12:00:03Z",
                "uuid": "abc-126",
            }
        )
        event = parse_jsonl_line(line)
        assert event is None

    def test_skip_progress_events(self) -> None:
        line = _make_line({"type": "progress", "data": {"type": "hook_progress"}})
        event = parse_jsonl_line(line)
        assert event is None

    def test_skip_system_events(self) -> None:
        line = _make_line({"type": "system", "subtype": "bridge_status"})
        event = parse_jsonl_line(line)
        assert event is None

    def test_skip_queue_operation(self) -> None:
        line = _make_line({"type": "queue-operation", "data": {}})
        event = parse_jsonl_line(line)
        assert event is None

    def test_skip_attachment(self) -> None:
        line = _make_line({"type": "attachment", "data": {}})
        event = parse_jsonl_line(line)
        assert event is None

    def test_invalid_json_returns_none(self) -> None:
        assert parse_jsonl_line("not json{{{") is None

    def test_truncates_long_summary(self) -> None:
        long_text = "x" * 300
        line = _make_line(
            {
                "type": "user",
                "message": {"role": "user", "content": long_text},
                "timestamp": "2026-04-02T12:00:00Z",
                "uuid": "abc-127",
            }
        )
        event = parse_jsonl_line(line)
        assert event is not None
        assert len(event.summary) <= 150

    def test_assistant_mixed_text_and_tool_use(self) -> None:
        """Assistant messages with both text and tool_use should only extract the text."""
        line = _make_line(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check that file."},
                        {"type": "tool_use", "name": "Read", "input": {"file_path": "/src/main.py"}},
                    ],
                },
                "timestamp": "2026-04-02T12:00:04Z",
                "uuid": "abc-128",
            }
        )
        event = parse_jsonl_line(line)
        assert event is not None
        assert "check that file" in event.summary.lower()
        assert "Read" not in event.summary

    def test_skip_empty_assistant_text(self) -> None:
        """Assistant messages with empty text blocks should be skipped."""
        line = _make_line(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                },
                "timestamp": "2026-04-02T12:00:05Z",
                "uuid": "abc-129",
            }
        )
        event = parse_jsonl_line(line)
        assert event is None
