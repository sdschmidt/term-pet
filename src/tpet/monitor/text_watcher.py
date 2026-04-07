"""Plain text file watcher using watchdog."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from tpet.monitor.parser import MAX_SUMMARY_LENGTH, SessionEvent

if TYPE_CHECKING:
    from queue import Queue

    from watchdog.observers.api import BaseObserver

logger = logging.getLogger(__name__)


class TextFileWatcher:
    """Watches a plain text file for new lines and emits SessionEvents."""

    def __init__(self, file_path: Path, event_queue: Queue[SessionEvent]) -> None:
        self._file_path = file_path
        self._event_queue = event_queue
        self._file_position: int = 0
        self._observer: BaseObserver | None = None

    def _process_new_lines(self) -> None:
        """Read new lines appended to the file and enqueue events."""
        if not self._file_path.exists():
            return

        try:
            with self._file_path.open("r", encoding="utf-8") as f:
                f.seek(self._file_position)
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    summary = (
                        stripped if len(stripped) <= MAX_SUMMARY_LENGTH else stripped[: MAX_SUMMARY_LENGTH - 3] + "..."
                    )
                    event = SessionEvent(
                        event_type="text",
                        role="text",
                        summary=summary,
                        timestamp=datetime.now(tz=UTC).isoformat(),
                    )
                    self._event_queue.put(event)
                self._file_position = f.tell()
        except Exception:
            logger.exception("Error reading text file %s", self._file_path)

    @property
    def file_path(self) -> Path:
        """Path to the file being watched.

        Public interface used by the file-system event handler.
        """
        return self._file_path

    def process_new_lines(self) -> None:
        """Read new lines appended to the file and enqueue events.

        Public interface used by the file-system event handler.
        """
        self._process_new_lines()

    def start(self) -> None:
        """Start watching the text file."""
        if self._file_path.exists():
            # Seek to end so we only see new content
            self._file_position = self._file_path.stat().st_size
            logger.info("Watching text file: %s (starting at byte %d)", self._file_path, self._file_position)
        else:
            logger.warning("Text file does not exist yet: %s", self._file_path)

        watch_dir = self._file_path.parent
        if not watch_dir.exists():
            logger.warning("Watch directory does not exist: %s", watch_dir)
            return

        handler = _TextFileHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(watch_dir), recursive=False)
        self._observer.daemon = True
        self._observer.start()
        logger.info("Started watching directory %s for %s", watch_dir, self._file_path.name)

    def stop(self) -> None:
        """Stop watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("Stopped text file watcher")


class _TextFileHandler(FileSystemEventHandler):
    """Handles file system events for the watched text file."""

    def __init__(self, watcher: TextFileWatcher) -> None:
        self._watcher = watcher

    def _handle(self, event: FileCreatedEvent | FileModifiedEvent) -> None:
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path == self._watcher.file_path:
            self._watcher.process_new_lines()

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        self._handle(event)

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        self._handle(event)
