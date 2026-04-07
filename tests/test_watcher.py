"""Tests for the session file watcher."""

from __future__ import annotations

import json
import time
from queue import Queue
from typing import TYPE_CHECKING

from tpet.monitor.watcher import SessionWatcher, encode_project_path, find_newest_session

if TYPE_CHECKING:
    from pathlib import Path

    from tpet.monitor.parser import SessionEvent


class TestEncodeProjectPath:
    """Tests for project path encoding."""

    def test_encode_path(self) -> None:
        assert encode_project_path("/Users/bob/Repos/myapp") == "-Users-bob-Repos-myapp"

    def test_encode_root(self) -> None:
        assert encode_project_path("/") == "-"


class TestFindNewestSession:
    """Tests for finding newest session file."""

    def test_finds_newest_jsonl(self, tmp_path: Path) -> None:
        old = tmp_path / "old-session.jsonl"
        old.write_text("{}\n", encoding="utf-8")
        time.sleep(0.05)
        new = tmp_path / "new-session.jsonl"
        new.write_text("{}\n", encoding="utf-8")
        result = find_newest_session(tmp_path)
        assert result == new

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        assert find_newest_session(tmp_path) is None

    def test_ignores_non_jsonl(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        assert find_newest_session(tmp_path) is None

    def test_ignores_subdirectories(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "session.jsonl").write_text("{}\n", encoding="utf-8")
        (tmp_path / "main.jsonl").write_text("{}\n", encoding="utf-8")
        result = find_newest_session(tmp_path)
        assert result is not None
        assert result.parent == tmp_path


class TestSessionWatcher:
    """Tests for the file watcher."""

    def test_reads_new_lines(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"
        line1 = json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "hello"},
                "timestamp": "2026-04-02T12:00:00Z",
                "uuid": "1",
            }
        )
        session_file.write_text(line1 + "\n", encoding="utf-8")

        event_queue: Queue[SessionEvent] = Queue()
        watcher = SessionWatcher(session_dir=tmp_path, event_queue=event_queue)
        watcher._process_file(session_file)

        assert not event_queue.empty()
        event = event_queue.get_nowait()
        assert event.summary == "hello"

    def test_skips_already_read_lines(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"
        line1 = json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "first"},
                "timestamp": "2026-04-02T12:00:00Z",
                "uuid": "1",
            }
        )
        session_file.write_text(line1 + "\n", encoding="utf-8")

        event_queue: Queue[SessionEvent] = Queue()
        watcher = SessionWatcher(session_dir=tmp_path, event_queue=event_queue)
        watcher._process_file(session_file)

        # Drain queue
        while not event_queue.empty():
            event_queue.get_nowait()

        # Process again - should not re-emit
        watcher._process_file(session_file)
        assert event_queue.empty()
