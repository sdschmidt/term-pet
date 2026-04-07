"""Tests for the image processing pipeline in tpet.art.process."""

import numpy as np
from PIL import Image

from tpet.art.process import (
    image_to_halfblock,
    remove_chroma_key,
    resize_for_halfblock,
    split_sprite_sheet,
)


def _make_solid_image(
    width: int,
    height: int,
    color: tuple[int, int, int],
    mode: str = "RGB",
) -> Image.Image:
    """Create a small solid-color image for testing."""
    return Image.new(mode, (width, height), color)


def _make_rgba_image(
    width: int,
    height: int,
    color: tuple[int, int, int, int],
) -> Image.Image:
    """Create a small RGBA image for testing."""
    return Image.new("RGBA", (width, height), color)


class TestRemoveChromaKey:
    """Tests for the chroma key removal function."""

    def test_remove_chroma_key_removes_magenta(self) -> None:
        """A pure-magenta image should become fully transparent via flood-fill."""
        img = _make_solid_image(10, 10, (255, 0, 255))
        result = remove_chroma_key(img)

        assert result.mode == "RGBA"
        arr = np.array(result)
        assert np.all(arr[:, :, 3] == 0)

    def test_remove_chroma_key_preserves_non_magenta(self) -> None:
        """A pure-red image should remain fully opaque when removing magenta."""
        img = _make_solid_image(10, 10, (255, 0, 0))
        result = remove_chroma_key(img, bg_color=(255, 0, 255))

        assert result.mode == "RGBA"
        arr = np.array(result)
        assert np.all(arr[:, :, 3] == 255)

    def test_remove_chroma_key_tolerance(self) -> None:
        """Near-magenta colors should be removed with high tolerance but preserved with low."""
        near_magenta = (250, 10, 250)
        img = _make_solid_image(10, 10, near_magenta)

        result_high = remove_chroma_key(img, bg_color=(255, 0, 255), tolerance=50)
        arr_high = np.array(result_high)
        assert np.all(arr_high[:, :, 3] == 0)

        result_low = remove_chroma_key(img, bg_color=(255, 0, 255), tolerance=5)
        arr_low = np.array(result_low)
        assert np.all(arr_low[:, :, 3] == 255)

    def test_remove_chroma_key_auto_detect(self) -> None:
        """Auto-detect should handle non-pure-magenta backgrounds from LLMs."""
        # LLMs produce off-magenta ~(219, 78, 180) not pure #FF00FF
        off_magenta = (219, 78, 180)
        img = _make_solid_image(20, 20, off_magenta)

        result = remove_chroma_key(img, bg_color=None, tolerance=80)
        arr = np.array(result)
        assert np.all(arr[:, :, 3] == 0), "Auto-detect should remove the actual background"

    def test_remove_chroma_key_auto_detect_preserves_subject(self) -> None:
        """Auto-detect flood-fill should preserve interior subject pixels."""
        # Image with magenta background and red center (interior)
        img = Image.new("RGB", (20, 20), (219, 78, 180))
        for x in range(5, 15):
            for y in range(5, 15):
                img.putpixel((x, y), (255, 0, 0))

        result = remove_chroma_key(img, tolerance=80)
        arr = np.array(result)
        # Red center should be opaque (not connected to edge background)
        assert arr[10, 10, 3] == 255
        # Magenta corner should be transparent (connected to edge)
        assert arr[0, 0, 3] == 0

    def test_remove_chroma_key_flood_fill_preserves_interior_bg_match(self) -> None:
        """Interior pixels matching bg color but not connected to edge should stay."""
        # Magenta border with a red ring protecting interior magenta
        img = Image.new("RGB", (20, 20), (255, 0, 255))  # all magenta
        for x in range(3, 17):
            for y in [3, 16]:
                img.putpixel((x, y), (255, 0, 0))  # red horizontal walls
            for y in range(3, 17):
                img.putpixel((x, y), (255, 0, 0))  # fills over, need proper ring

        # Simpler: magenta border, blue ring, magenta interior
        img2 = Image.new("RGB", (20, 20), (255, 0, 255))
        for i in range(20):
            img2.putpixel((i, 4), (0, 0, 255))
            img2.putpixel((i, 15), (0, 0, 255))
            img2.putpixel((4, i), (0, 0, 255))
            img2.putpixel((15, i), (0, 0, 255))

        result = remove_chroma_key(img2, tolerance=50)
        arr = np.array(result)
        # Interior magenta (row 10, col 10) should be preserved — not connected to edge
        assert arr[10, 10, 3] == 255
        # Edge magenta should be removed
        assert arr[0, 0, 3] == 0


class TestSplitSpriteSheet:
    """Tests for the sprite sheet splitter (2x3 and 2x2 layouts)."""

    def test_split_sprite_sheet_2x3_layout(self) -> None:
        """A tall image (height >= 1.3 * width) should produce 6 frames."""
        img = _make_solid_image(20, 30, (128, 128, 128))
        frames = split_sprite_sheet(img)

        assert len(frames) == 6
        for frame in frames:
            assert frame.size == (10, 10)
            assert frame.mode == "RGBA"

    def test_split_sprite_sheet_2x2_layout(self) -> None:
        """A square image should produce 4 frames (legacy layout)."""
        img = _make_solid_image(20, 20, (128, 128, 128))
        frames = split_sprite_sheet(img)

        assert len(frames) == 4
        for frame in frames:
            assert frame.size == (10, 10)
            assert frame.mode == "RGBA"

    def test_split_sprite_sheet_2x3_frame_order(self) -> None:
        """6-frame layout: frames in row-major order across 2x3 grid."""
        # 20x30 image: 2 cols x 3 rows, each cell 10x10
        colors: list[tuple[int, int, int]] = [
            (255, 0, 0),  # (0,0) red
            (0, 255, 0),  # (1,0) green
            (0, 0, 255),  # (0,1) blue
            (255, 255, 0),  # (1,1) yellow
            (255, 0, 255),  # (0,2) magenta
            (0, 255, 255),  # (1,2) cyan
        ]

        img = Image.new("RGB", (20, 30))
        for idx, color in enumerate(colors):
            col = idx % 2
            row = idx // 2
            x0, y0 = col * 10, row * 10
            for x in range(x0, x0 + 10):
                for y in range(y0, y0 + 10):
                    img.putpixel((x, y), color)

        frames = split_sprite_sheet(img)
        assert len(frames) == 6

        for i, frame in enumerate(frames):
            frame_rgb = frame.convert("RGB")
            pixel = frame_rgb.getpixel((0, 0))
            assert pixel == colors[i], f"Frame {i} has color {pixel}, expected {colors[i]}"

    def test_split_sprite_sheet_2x2_frame_order(self) -> None:
        """Legacy 4-frame layout: frames in row-major order across 2x2 grid."""
        colors: list[tuple[int, int, int]] = [
            (255, 0, 0),  # TL - red
            (0, 255, 0),  # TR - green
            (0, 0, 255),  # BL - blue
            (255, 255, 0),  # BR - yellow
        ]

        img = Image.new("RGB", (4, 4))
        pixels = img.load()
        pixels[0, 0] = colors[0]
        pixels[1, 0] = colors[0]
        pixels[0, 1] = colors[0]
        pixels[1, 1] = colors[0]
        pixels[2, 0] = colors[1]
        pixels[3, 0] = colors[1]
        pixels[2, 1] = colors[1]
        pixels[3, 1] = colors[1]
        pixels[0, 2] = colors[2]
        pixels[1, 2] = colors[2]
        pixels[0, 3] = colors[2]
        pixels[1, 3] = colors[2]
        pixels[2, 2] = colors[3]
        pixels[3, 2] = colors[3]
        pixels[2, 3] = colors[3]
        pixels[3, 3] = colors[3]

        frames = split_sprite_sheet(img)
        assert len(frames) == 4

        for i, frame in enumerate(frames):
            frame_rgb = frame.convert("RGB")
            pixel = frame_rgb.getpixel((0, 0))
            assert pixel == colors[i], f"Frame {i} has color {pixel}, expected {colors[i]}"


class TestResizeForHalfblock:
    """Tests for the halfblock resize function."""

    def test_resize_for_halfblock_even_height(self) -> None:
        """Output height should be even."""
        img = _make_solid_image(100, 100, (100, 100, 100))
        result = resize_for_halfblock(img, target_height=50)

        _, h = result.size
        assert h % 2 == 0
        # target_height=50 → 50//2*2 = 50 (already even)
        assert h == 50

    def test_resize_for_halfblock_odd_target(self) -> None:
        """Odd target height should be rounded down to even."""
        img = _make_solid_image(100, 100, (100, 100, 100))
        result = resize_for_halfblock(img, target_height=51)

        _, h = result.size
        assert h % 2 == 0
        assert h == 50

    def test_resize_for_halfblock_preserves_aspect_ratio(self) -> None:
        """Aspect ratio should be maintained after resize."""
        original_w, original_h = 120, 60
        img = _make_solid_image(original_w, original_h, (100, 100, 100))
        result = resize_for_halfblock(img, target_height=30)

        new_w, new_h = result.size
        original_ratio = original_w / original_h
        new_ratio = new_w / new_h
        assert abs(new_ratio - original_ratio) < 0.05

    def test_resize_for_halfblock_rgba_mode(self) -> None:
        """Output should always be in RGBA mode."""
        img = _make_solid_image(50, 50, (100, 100, 100), mode="RGB")
        result = resize_for_halfblock(img, target_height=24)
        assert result.mode == "RGBA"

    def test_resize_for_halfblock_minimum_height(self) -> None:
        """Very small target should produce at least 2px height."""
        img = _make_solid_image(100, 100, (100, 100, 100))
        result = resize_for_halfblock(img, target_height=1)

        _, h = result.size
        assert h >= 2
        assert h % 2 == 0


class TestImageToHalfblock:
    """Tests for the halfblock conversion function."""

    def test_image_to_halfblock_solid_color(self) -> None:
        """A solid-color opaque image should produce half-block output with ANSI escapes."""
        img = _make_rgba_image(4, 4, (255, 0, 0, 255))
        result = image_to_halfblock(img)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain fg color escape
        assert "\x1b[38;2;255;0;0m" in result
        # Should contain the upper half-block character
        assert "\u2580" in result

    def test_image_to_halfblock_two_rows_one_line(self) -> None:
        """A 2-row image should produce exactly one line of half-block output."""
        img = _make_rgba_image(4, 2, (100, 100, 100, 255))
        result = image_to_halfblock(img)

        lines = result.split("\n")
        assert len(lines) == 1

    def test_image_to_halfblock_four_rows_two_lines(self) -> None:
        """A 4-row image should produce exactly two lines of half-block output."""
        img = _make_rgba_image(4, 4, (100, 100, 100, 255))
        result = image_to_halfblock(img)

        lines = result.split("\n")
        assert len(lines) == 2

    def test_image_to_halfblock_transparent_both(self) -> None:
        """Both-transparent pixels should produce a space."""
        img = _make_rgba_image(4, 2, (0, 0, 0, 0))
        result = image_to_halfblock(img)

        # Both transparent → space with reset
        assert "\x1b[0m " in result

    def test_image_to_halfblock_transparent_top_only(self) -> None:
        """Transparent top with opaque bottom should use lower half-block."""
        # Top row transparent, bottom row opaque red
        img = Image.new("RGBA", (2, 2))
        img.putpixel((0, 0), (0, 0, 0, 0))
        img.putpixel((1, 0), (0, 0, 0, 0))
        img.putpixel((0, 1), (255, 0, 0, 255))
        img.putpixel((1, 1), (255, 0, 0, 255))

        result = image_to_halfblock(img)
        # Should use lower half-block ▄ (U+2584)
        assert "\u2584" in result
        # Should set fg to red
        assert "\x1b[38;2;255;0;0m" in result

    def test_image_to_halfblock_line_end_reset(self) -> None:
        """Each line should end with a reset escape."""
        img = _make_rgba_image(4, 2, (100, 100, 100, 255))
        result = image_to_halfblock(img)

        assert result.endswith("\x1b[0m")

    def test_image_to_halfblock_odd_height_padded(self) -> None:
        """An odd-height image should be padded with a transparent row."""
        img = _make_rgba_image(4, 3, (100, 100, 100, 255))
        result = image_to_halfblock(img)

        # 3 rows padded to 4 → 2 lines of half-block
        lines = result.split("\n")
        assert len(lines) == 2

    def test_image_to_halfblock_default_bg(self) -> None:
        """With default_bg set, transparent pixels should use that color."""
        img = _make_rgba_image(2, 2, (0, 0, 0, 0))
        result = image_to_halfblock(img, default_bg=(30, 30, 30))

        # Both transparent with default_bg → should not contain \x1b[49m (default bg)
        # Instead should contain the default_bg color
        assert "\x1b[48;2;30;30;30m" in result

    def test_image_to_halfblock_different_colors(self) -> None:
        """Top and bottom pixels with different colors should use both fg and bg."""
        img = Image.new("RGBA", (2, 2))
        img.putpixel((0, 0), (255, 0, 0, 255))  # top: red
        img.putpixel((1, 0), (255, 0, 0, 255))
        img.putpixel((0, 1), (0, 0, 255, 255))  # bottom: blue
        img.putpixel((1, 1), (0, 0, 255, 255))

        result = image_to_halfblock(img)
        # Should have fg=red and bg=blue with ▀
        assert "\x1b[38;2;255;0;0m" in result
        assert "\x1b[48;2;0;0;255m" in result
        assert "\u2580" in result
