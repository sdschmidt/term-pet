"""Renderer that drives the macOS desktop pet over a Unix domain socket.

Art mode ``macos-desktop`` takes over from terminal rendering: sprite frames
live on the user's desktop as a native macOS window, and this renderer pipes
each tick's state + comment to that window over a local socket.

Relationship with the Swift app:
  * The Swift app ("Sheep" / deskpet binary) is the socket server, listening at
    ``~/.config/tpet/display.sock``.
  * This renderer is a client. On first use it probes for a running server;
    if none answers, it spawns the Swift binary (from ``$DESKPET_BIN`` or
    ``which sheep-screenmate``) and polls until it connects.
  * Wire protocol is newline-delimited JSON per tick:
        {"state": "idle"|"reacting"|"sleeping", "comment": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from rich.live import Live

    from tpet.config import TpetConfig
    from tpet.models.pet import PetProfile

logger = logging.getLogger(__name__)

SOCKET_PATH = Path.home() / ".config" / "tpet" / "display.sock"

_CONNECT_RETRIES = 20
_CONNECT_DELAY = 0.2


class DesktopPetUnavailable(RuntimeError):
    """Raised when the Swift pet binary can't be located or started."""


class MacosDesktopRenderer:
    """Emits state+comment JSON to the Swift pet; renders a minimal terminal status."""

    def __init__(self, config: TpetConfig) -> None:
        self._config = config
        self._sock: socket.socket | None = None
        self._child: subprocess.Popen[bytes] | None = None
        self._last_state: str | None = None
        self._last_comment: str | None = None
        self._connect_or_spawn()

    # ------------------------------------------------------------------
    # Socket lifecycle
    # ------------------------------------------------------------------

    def _connect_or_spawn(self) -> None:
        if self._try_connect(retries=1):
            logger.info("connected to existing desktop pet at %s", SOCKET_PATH)
            return
        self._spawn()
        if not self._try_connect(retries=_CONNECT_RETRIES):
            raise DesktopPetUnavailable(
                f"Could not connect to desktop pet socket {SOCKET_PATH} "
                f"after spawning the binary. Check that the Swift app is "
                f"running and writing its socket."
            )
        logger.info("spawned and connected to desktop pet at %s", SOCKET_PATH)

    def _try_connect(self, *, retries: int) -> bool:
        for _ in range(retries):
            if SOCKET_PATH.exists():
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(str(SOCKET_PATH))
                    self._sock = s
                    return True
                except OSError:
                    pass
            time.sleep(_CONNECT_DELAY)
        return False

    def _spawn(self) -> None:
        bin_path = self._resolve_binary()
        if bin_path is None:
            raise DesktopPetUnavailable(
                "macos-desktop mode needs the Swift pet binary. "
                "Set DESKPET_BIN=/path/to/Sheep, or install the binary to PATH "
                "as `sheep-screenmate` (e.g. "
                "`swift build -c release && cp .build/release/Sheep ~/.local/bin/sheep-screenmate`)."
            )
        logger.info("spawning desktop pet: %s", bin_path)
        self._child = subprocess.Popen(
            [bin_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    @staticmethod
    def _resolve_binary() -> str | None:
        env = os.environ.get("DESKPET_BIN")
        if env and Path(env).is_file() and os.access(env, os.X_OK):
            return env
        for name in ("sheep-screenmate", "deskpet"):
            found = shutil.which(name)
            if found:
                return found
        return None

    # ------------------------------------------------------------------
    # Renderer protocol
    # ------------------------------------------------------------------

    def render(
        self,
        live: Live,
        pet: PetProfile,
        frame_idx: int,
        current_comment: str | None,
        frame_changed: bool,
        comment_changed: bool,
    ) -> None:
        if not (frame_changed or comment_changed):
            return

        state = self._state_for_frame(frame_idx)
        payload: dict[str, object] = {"state": state}
        if comment_changed and current_comment:
            payload["comment"] = current_comment

        self._send(payload)
        self._update_terminal_status(live, pet, state, current_comment)

        self._last_state = state
        self._last_comment = current_comment

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _state_for_frame(frame_idx: int) -> str:
        # term-pet frames: 0-3 idle (incl. blinks), 4 reacting, 5 sleeping.
        if frame_idx == 4:
            return "reacting"
        if frame_idx == 5:
            return "sleeping"
        return "idle"

    def _send(self, payload: dict[str, object]) -> None:
        if self._sock is None and not self._try_connect(retries=2):
            return
        line = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            assert self._sock is not None
            self._sock.sendall(line)
        except OSError:
            logger.warning("socket send failed; will retry on next event")
            try:
                if self._sock is not None:
                    self._sock.close()
            finally:
                self._sock = None

    def _update_terminal_status(
        self,
        live: Live,
        pet: PetProfile,
        state: str,
        comment: str | None,
    ) -> None:
        lines: list[Text] = [
            Text(f"{pet.name}", style="bold cyan"),
            Text(f"art mode: macos-desktop", style="dim"),
            Text(f"state:    {state}", style="dim"),
        ]
        if comment:
            lines.append(Text(""))
            lines.append(Text(comment, style="italic"))
        body = Text("\n").join(lines)
        live.update(Panel(body, title="tpet", border_style="dim"), refresh=True)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
