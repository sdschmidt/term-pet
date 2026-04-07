"""Art generation and display for terminal pets."""

from tpet.art.detect import detect_truecolor_support
from tpet.art.generator import (
    build_openai_sprite_prompt,
    build_sprite_prompt,
    generate_art,
    generate_halfblock_art,
    generate_halfblock_art_openai,
)
from tpet.art.process import (
    image_to_halfblock,
    remove_chroma_key,
    resize_for_halfblock,
    split_sprite_sheet,
)
from tpet.art.storage import (
    delete_halfblock_art,
    get_art_dir,
    get_halfblock_frame_path,
    has_halfblock_art,
    load_halfblock_frame,
    save_halfblock_frame,
)

__all__ = [
    "build_openai_sprite_prompt",
    "build_sprite_prompt",
    "delete_halfblock_art",
    "detect_truecolor_support",
    "generate_art",
    "generate_halfblock_art",
    "generate_halfblock_art_openai",
    "get_art_dir",
    "get_halfblock_frame_path",
    "has_halfblock_art",
    "image_to_halfblock",
    "load_halfblock_frame",
    "remove_chroma_key",
    "resize_for_halfblock",
    "save_halfblock_frame",
    "split_sprite_sheet",
]
