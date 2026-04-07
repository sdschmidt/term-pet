"""Tests for terminal graphics capability detection."""

import os

import pytest

from tpet.art import detect


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Reset the module-level caches before each test."""
    detect._cached_truecolor = None


def _clear_term_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all terminal-related env vars."""
    for var in ("TERM_PROGRAM", "TERM", "COLORTERM"):
        monkeypatch.delenv(var, raising=False)


class TestDetectTruecolorSupport:
    """Tests for detect_truecolor_support."""

    def test_detect_truecolor_colorterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_term_env(monkeypatch)
        monkeypatch.setenv("COLORTERM", "truecolor")
        assert detect.detect_truecolor_support() is True

    def test_detect_truecolor_24bit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_term_env(monkeypatch)
        monkeypatch.setenv("COLORTERM", "24bit")
        assert detect.detect_truecolor_support() is True

    def test_detect_truecolor_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_term_env(monkeypatch)
        assert detect.detect_truecolor_support() is False

    def test_detect_truecolor_caches_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_term_env(monkeypatch)
        monkeypatch.setenv("COLORTERM", "truecolor")

        call_count = 0
        original_getenv = os.getenv

        def _tracking_getenv(key: str, default: str | None = None) -> str | None:
            nonlocal call_count
            if key == "COLORTERM":
                call_count += 1
            return original_getenv(key, default)

        monkeypatch.setattr(os, "getenv", _tracking_getenv)

        result1 = detect.detect_truecolor_support()
        assert result1 is True
        first_count = call_count

        result2 = detect.detect_truecolor_support()
        assert result2 is True
        assert call_count == first_count

    def test_detect_truecolor_not_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """COLORTERM=truecolor is trusted even without a TTY (e.g. tmux)."""
        _clear_term_env(monkeypatch)
        monkeypatch.setenv("COLORTERM", "truecolor")
        assert detect.detect_truecolor_support() is True
