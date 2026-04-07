"""Frame preview renderer for post-generation display."""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from tpet.config import TpetConfig

# Pre-compiled ANSI escape sequence pattern for width measurement
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _render_frame(config: TpetConfig, pet_name: str, frame_idx: int, preview_height: int) -> str:
    """Render a single PNG frame at preview size as a halfblock string.

    Args:
        config: Application configuration.
        pet_name: Pet name (used to locate PNG files).
        frame_idx: Frame index to render.
        preview_height: Target height in pixels (must be even).

    Returns:
        Halfblock ANSI string, or ``"(missing)"`` if the PNG does not exist.
    """
    from PIL import Image

    from tpet.art.process import image_to_halfblock, resize_for_halfblock
    from tpet.art.storage import get_png_frame_path

    png_path = get_png_frame_path(config.pet_data_dir, pet_name, frame_idx)
    if not png_path.exists():
        return "(missing)"
    img = Image.open(png_path)
    resized = resize_for_halfblock(img, target_height=preview_height)
    return image_to_halfblock(resized)


def _side_by_side(left: str, right: str, gap: int = 3) -> str:
    """Merge two halfblock renders into a single side-by-side string.

    Args:
        left: Left halfblock render (ANSI string with newlines).
        right: Right halfblock render (ANSI string with newlines).
        gap: Number of space characters separating the two renders.

    Returns:
        Combined ANSI string with equal-height rows placed side-by-side.
    """
    left_lines = left.split("\n")
    right_lines = right.split("\n")
    max_h = max(len(left_lines), len(right_lines))
    left_lines += [""] * (max_h - len(left_lines))
    right_lines += [""] * (max_h - len(right_lines))
    max_left_w = max((_visible_width(ln) for ln in left_lines), default=0)
    spacer = " " * gap
    merged: list[str] = []
    for ll, rl in zip(left_lines, right_lines, strict=True):
        pad = " " * (max_left_w - _visible_width(ll))
        merged.append(f"{ll}\x1b[0m{pad}{spacer}{rl}")
    return "\n".join(merged)


def _visible_width(line: str) -> int:
    """Return the visible (non-ANSI) character width of *line*.

    Args:
        line: A string that may contain ANSI escape sequences.

    Returns:
        Number of printable characters.
    """
    return len(_ANSI_RE.sub("", line))


def preview_frames(config: TpetConfig, pet_name: str, frame_count: int) -> None:
    """Show a compact preview of all generated frames after art generation.

    Renders PNG frames at half the configured halfblock size and displays
    idle/blink pairs side-by-side for easy visual comparison.

    Args:
        config: Application configuration (used for ``halfblock_size``).
        pet_name: Name of the pet (used to locate PNG files).
        frame_count: Number of frames generated (4 for legacy, 6 for current).
    """
    from tpet.animation.engine import FRAME_COUNT_CURRENT

    console = Console()

    # Use half the configured size, snapped to an even number
    preview_height = max(4, config.halfblock_size // 2)
    preview_height = (preview_height // 2) * 2

    def _render(idx: int) -> str:
        return _render_frame(config, pet_name, idx, preview_height)

    if frame_count >= FRAME_COUNT_CURRENT:
        pairs: list[tuple[str, str, str]] = [
            ("idle / idle-blink", "0", "2"),
            ("idle-shift / shift-blink", "1", "3"),
            ("react / sleep", "4", "5"),
        ]
        for label, li, ri in pairs:
            left = _render(int(li))
            right = _render(int(ri))
            console.print(f"\n[bold]{label}:[/bold]")
            sys.stdout.write(_side_by_side(left, right) + "\n")
            sys.stdout.flush()
    else:
        labels_4 = ["idle", "idle-shift", "react", "sleep"]
        for i, label in enumerate(labels_4):
            console.print(f"\n[bold]Frame {i} ({label}):[/bold]")
            sys.stdout.write(_render(i) + "\n")
            sys.stdout.flush()
