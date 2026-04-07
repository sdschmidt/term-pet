"""Detect terminal graphics capabilities."""

from __future__ import annotations

import os

_cached_truecolor: bool | None = None


def detect_truecolor_support() -> bool:
    """Detect whether the current terminal supports 24-bit truecolor.

    Checks COLORTERM for 'truecolor' or '24bit', set by virtually all
    modern terminals (kitty, alacritty, ghostty, wezterm, iterm2, foot,
    windows terminal, etc.). Also trusts tmux with COLORTERM set.
    Result is cached after first call.

    Returns:
        True if the terminal likely supports 24-bit color.
    """
    global _cached_truecolor
    if _cached_truecolor is not None:
        return _cached_truecolor

    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in ("truecolor", "24bit"):
        _cached_truecolor = True
        return True

    _cached_truecolor = False
    return False
