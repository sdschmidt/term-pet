"""Session file watcher using watchdog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from tpet.monitor.parser import parse_jsonl_line

if TYPE_CHECKING:
    from queue import Queue

    from watchdog.observers.api import BaseObserver

    from tpet.monitor.parser import SessionEvent

logger = logging.getLogger(__name__)


def encode_project_path(project_path: str) -> str:
    """Encode a project path the way Claude Code stores it.

    Args:
        project_path: Absolute project directory path.

    Returns:
        Encoded path with slashes replaced by dashes.
    """
    return project_path.replace("/", "-")


def find_newest_session(session_dir: Path) -> Path | None:
    """Find the most recently modified JSONL session file.

    Args:
        session_dir: Directory to search for .jsonl files.

    Returns:
        Path to newest file, or None if no .jsonl files exist.
    """
    jsonl_files = [f for f in session_dir.iterdir() if f.is_file() and f.suffix == ".jsonl"]
    if not jsonl_files:
        return None
    return max(jsonl_files, key=lambda f: f.stat().st_mtime)


class SessionWatcher:
    """Watches Claude Code session files for new events."""

    def __init__(self, session_dir: Path, event_queue: Queue[SessionEvent]) -> None:
        self._session_dir = session_dir
        self._event_queue = event_queue
        self._file_positions: dict[Path, int] = {}
        self._observer: BaseObserver | None = None

    def _process_file(self, path: Path) -> None:
        """Read new lines from a session file and parse them."""
        if not path.exists() or path.suffix != ".jsonl":
            return

        last_pos = self._file_positions.get(path, 0)
        try:
            with path.open("r", encoding="utf-8") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = parse_jsonl_line(line)
                    if event is not None:
                        self._event_queue.put(event)
                self._file_positions[path] = f.tell()
        except Exception:
            logger.exception("Error reading session file %s", path)

    def process_file(self, path: Path) -> None:
        """Read new lines from a session file and parse them.

        Public interface used by the file-system event handler.

        Args:
            path: Path to the JSONL session file to process.
        """
        self._process_file(path)

    def start(self) -> None:
        """Start watching the session directory.

        If the session directory doesn't exist yet (e.g. Claude Code hasn't
        started for this project), polls for up to 60 seconds waiting for it
        to appear, then starts the watchdog observer.
        """
        if not self._session_dir.exists():
            self._poll_for_session_dir()

        if not self._session_dir.exists():
            logger.warning("Session directory not found after polling: %s", self._session_dir)
            return

        newest = find_newest_session(self._session_dir)
        if newest:
            self._file_positions[newest] = newest.stat().st_size
            logger.info("Watching session file: %s", newest)

        handler = _SessionFileHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._session_dir), recursive=False)
        self._observer.daemon = True
        self._observer.start()
        logger.info("Started watching %s", self._session_dir)

    def _poll_for_session_dir(self, timeout: float = 60.0, interval: float = 2.0) -> None:
        """Poll until the session directory appears or timeout elapses.

        Args:
            timeout: Maximum seconds to wait.
            interval: Seconds between checks.
        """
        import time

        logger.info("Session directory not found, polling for up to %.0fs: %s", timeout, self._session_dir)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._session_dir.exists():
                logger.info("Session directory appeared: %s", self._session_dir)
                return
            time.sleep(interval)
        logger.warning("Session directory still not found after %.0fs: %s", timeout, self._session_dir)

    def stop(self) -> None:
        """Stop watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("Stopped session watcher")


class _SessionFileHandler(FileSystemEventHandler):
    """Handles file system events for session files."""

    def __init__(self, watcher: SessionWatcher) -> None:
        self._watcher = watcher

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.suffix == ".jsonl":
            logger.info("New session file detected: %s", path)
            self._watcher.process_file(path)

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.suffix == ".jsonl":
            self._watcher.process_file(path)
