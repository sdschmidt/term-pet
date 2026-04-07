"""Renderer protocol and implementations for the live display loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from rich.live import Live

    from tpet.config import TpetConfig
    from tpet.models.pet import PetProfile

logger = logging.getLogger(__name__)


class Renderer(Protocol):
    """Protocol for per-frame display renderers.

    Each renderer knows how to update the Rich ``Live`` display (or write
    directly to stdout) for its art mode.  The main loop calls
    ``render(live, pet, frame_idx, current_comment, frame_changed,
    comment_changed)`` once per tick.
    """

    def render(
        self,
        live: Live,
        pet: PetProfile,
        frame_idx: int,
        current_comment: str | None,
        frame_changed: bool,
        comment_changed: bool,
    ) -> None:
        """Update the terminal display for the current frame.

        Args:
            live: Active Rich Live context.
            pet: Current pet profile.
            frame_idx: Current animation frame index.
            current_comment: Speech bubble text (may be None).
            frame_changed: True when the frame index changed since last render.
            comment_changed: True when the comment changed since last render.
        """
        ...


class AsciiRenderer:
    """Renders the ASCII art + speech bubble layout via Rich."""

    def __init__(self, config: TpetConfig) -> None:
        self._config = config

    def render(
        self,
        live: Live,
        pet: PetProfile,
        frame_idx: int,
        current_comment: str | None,
        frame_changed: bool,
        comment_changed: bool,
    ) -> None:
        """Redraw when frame or comment changes.

        Args:
            live: Active Rich Live context.
            pet: Current pet profile.
            frame_idx: Current animation frame index.
            current_comment: Speech bubble text.
            frame_changed: Whether the frame changed.
            comment_changed: Whether the comment changed.
        """
        from tpet.renderer.display import build_display_layout

        if frame_changed or comment_changed:
            live.update(
                build_display_layout(pet, frame_idx, current_comment, placement=self._config.bubble_placement),
                refresh=True,
            )


class HalfblockRenderer:
    """Renders PNG frames as ANSI half-block art alongside the speech bubble."""

    def __init__(self, config: TpetConfig) -> None:
        self._config = config
        self._cached_png_path: Path | None = None
        self._error_count = 0

    @property
    def has_errors(self) -> bool:
        """True when too many render errors have occurred to continue."""
        return self._error_count >= 3

    def render(
        self,
        live: Live,
        pet: PetProfile,
        frame_idx: int,
        current_comment: str | None,
        frame_changed: bool,
        comment_changed: bool,
    ) -> None:
        """Render half-block art.  Falls back silently after 3 errors.

        Args:
            live: Active Rich Live context.
            pet: Current pet profile.
            frame_idx: Current animation frame index.
            current_comment: Speech bubble text.
            frame_changed: Whether the frame changed.
            comment_changed: Whether the comment changed.
        """
        from tpet.art.storage import get_png_frame_path
        from tpet.renderer.display import build_halfblock_layout

        if frame_changed:
            png_path = get_png_frame_path(self._config.pet_data_dir, pet.name, frame_idx)
            if png_path.exists():
                self._cached_png_path = png_path
            else:
                self._error_count += 1

        if self._cached_png_path and (frame_changed or comment_changed):
            live.update(
                build_halfblock_layout(
                    pet,
                    self._cached_png_path,
                    current_comment,
                    max_width_pct=self._config.art_max_width_pct,
                    placement=self._config.bubble_placement,
                ),
                refresh=True,
            )
