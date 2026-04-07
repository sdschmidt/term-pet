"""Tests for art file storage (halfblock and PNG)."""

from pathlib import Path

from PIL import Image

from tpet.art.storage import (
    delete_halfblock_art,
    get_art_dir,
    get_frame_count_png,
    get_halfblock_frame_path,
    has_halfblock_art,
    has_png_frames,
    load_halfblock_frame,
    save_halfblock_frame,
    save_png_frame,
)


class TestGetArtDir:
    """Tests for get_art_dir."""

    def test_get_art_dir_creates_directory(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        art_dir = get_art_dir(config_dir)
        assert art_dir == config_dir / "art"
        assert art_dir.is_dir()


# ---------------------------------------------------------------------------
# Halfblock storage tests (.hblk)
# ---------------------------------------------------------------------------


class TestHalfblockSaveAndLoad:
    """Tests for save_halfblock_frame and load_halfblock_frame."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        data = "\x1b[38;2;255;0;0m\x1b[48;2;0;0;255m\u2580\x1b[0m"
        saved_path = save_halfblock_frame(config_dir, "TestPet", 0, data)
        assert saved_path.exists()
        assert saved_path.suffix == ".hblk"
        loaded = load_halfblock_frame(config_dir, "TestPet", 0)
        assert loaded == data

    def test_load_missing_frame_returns_none(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        assert load_halfblock_frame(config_dir, "NoPet", 0) is None


class TestHasHalfblockArt:
    """Tests for has_halfblock_art."""

    def test_has_halfblock_art_false_when_missing(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        assert has_halfblock_art(config_dir, "Ghost") is False

    def test_has_halfblock_art_true_when_4_frames(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        for i in range(4):
            save_halfblock_frame(config_dir, "Fluffy", i, f"frame_{i}")
        assert has_halfblock_art(config_dir, "Fluffy") is True

    def test_has_halfblock_art_true_when_6_frames(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        for i in range(6):
            save_halfblock_frame(config_dir, "Blinky", i, f"frame_{i}")
        assert has_halfblock_art(config_dir, "Blinky") is True

    def test_has_halfblock_art_false_when_partial(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        for i in range(2):
            save_halfblock_frame(config_dir, "Partial", i, f"frame_{i}")
        assert has_halfblock_art(config_dir, "Partial") is False


class TestDeleteHalfblockArt:
    """Tests for delete_halfblock_art."""

    def test_delete_halfblock_art(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        for i in range(4):
            save_halfblock_frame(config_dir, "ByeBye", i, f"frame_{i}")
        assert has_halfblock_art(config_dir, "ByeBye") is True

        delete_halfblock_art(config_dir, "ByeBye")
        assert has_halfblock_art(config_dir, "ByeBye") is False
        for i in range(4):
            assert load_halfblock_frame(config_dir, "ByeBye", i) is None


class TestPetNameSanitization:
    """Tests for pet name sanitization in file paths."""

    def test_halfblock_name_sanitization(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        messy_name = "My Pet @#$%"
        path = get_halfblock_frame_path(config_dir, messy_name, 0)
        filename = path.name
        assert filename == "My_Pet__frame_0.hblk"


# ---------------------------------------------------------------------------
# PNG frame count detection tests
# ---------------------------------------------------------------------------


class TestFrameCountDetection:
    """Tests for get_frame_count_png and has_png_frames."""

    def test_get_frame_count_zero_when_missing(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        assert get_frame_count_png(config_dir, "Ghost") == 0

    def test_get_frame_count_4_for_legacy(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        for i in range(4):
            save_png_frame(config_dir, "Legacy", i, img)
        assert get_frame_count_png(config_dir, "Legacy") == 4

    def test_get_frame_count_6_for_new_layout(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        for i in range(6):
            save_png_frame(config_dir, "Blinky", i, img)
        assert get_frame_count_png(config_dir, "Blinky") == 6

    def test_get_frame_count_0_when_partial(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        for i in range(3):
            save_png_frame(config_dir, "Partial", i, img)
        assert get_frame_count_png(config_dir, "Partial") == 0

    def test_has_png_frames_true_for_4(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        for i in range(4):
            save_png_frame(config_dir, "Pet4", i, img)
        assert has_png_frames(config_dir, "Pet4") is True

    def test_has_png_frames_true_for_6(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        for i in range(6):
            save_png_frame(config_dir, "Pet6", i, img)
        assert has_png_frames(config_dir, "Pet6") is True
