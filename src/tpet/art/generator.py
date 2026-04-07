"""Art generation orchestration for pixel-art and half-block modes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PIL import Image

from tpet.art.storage import (
    get_art_dir,
    save_halfblock_frame,
    save_png_frame,
    save_prompt,
)
from tpet.art.storage import (
    sanitize_name as _sanitize_name,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tpet.art.openai_client import GenerationResult
    from tpet.config import TpetConfig
    from tpet.models.pet import PetProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sprite prompts
# ---------------------------------------------------------------------------


def _sprite_layout_instructions() -> str:
    """Return the frame layout and style instructions appended to all prompts.

    These instructions describe the 2x3 sprite sheet format and technical
    requirements for proper frame splitting. They are always included,
    regardless of any user-provided prompt override.

    Returns:
        Frame layout and style instruction string.
    """
    return (
        "Layout: 2 columns x 3 rows grid (6 panels). The overall image must be portrait orientation "
        "(taller than wide, approximately 2:3 aspect ratio). "
        "Each panel shows a different animation frame:\n"
        "- Row 1, Left: Idle pose (relaxed, eyes fully open)\n"
        "- Row 1, Right: Idle shift (slight pose variation, eyes fully open)\n"
        "- Row 2, Left: BLINK — pixel-for-pixel IDENTICAL to Row 1 Left (same body, same size, "
        "same position, same colors, same outline) with ONE change: the eyes are closed "
        "(same closed-eye style as the sleeping frame). Nothing else changes.\n"
        "- Row 2, Right: BLINK — pixel-for-pixel IDENTICAL to Row 1 Right (same body, same size, "
        "same position, same colors, same outline) with ONE change: the eyes are closed "
        "(same closed-eye style as the sleeping frame). Nothing else changes.\n"
        "- Row 3, Left: Surprised/reacting (eyes wide, mouth open)\n"
        "- Row 3, Right: Sleeping (eyes closed, zzz bubbles)\n\n"
        "Style: Clean pixel art, cute and charming, suitable for a terminal pet. "
        "The creature must look identical across all 6 frames (same proportions, colors, style). "
        "Only the expression and small pose details should differ between frames.\n\n"
        "The creature MUST have a thick black outline around the entire body and all major features. "
        "NO shadows, NO drop shadows, NO ambient occlusion. "
        "Flat solid colors only with bold black outlines."
    )


def _default_subject(pet: PetProfile) -> str:
    """Return the default subject description from pet profile data.

    Args:
        pet: Pet profile with creature info.

    Returns:
        Subject description string.
    """
    return (
        f"Create a pixel art sprite sheet of a {pet.creature_type} named {pet.name}. "
        f"Personality: {pet.personality}. "
        f"Rarity: {pet.rarity.value}."
    )


def build_sprite_prompt(pet: PetProfile, prompt_override: str = "") -> str:
    """Build the prompt for generating a 2x3 sprite sheet with chroma key background.

    For use with Gemini (no native transparency). Requests solid magenta
    background for automated chroma key removal.

    If prompt_override is provided, it replaces the default subject description
    but the frame layout instructions and background requirements are always appended.

    Args:
        pet: Pet profile with creature info.
        prompt_override: Custom prompt to use instead of auto-generated subject.

    Returns:
        Prompt string for Gemini image generation.
    """
    subject = prompt_override.strip() if prompt_override.strip() else _default_subject(pet)
    return (
        f"{subject}\n\n"
        f"{_sprite_layout_instructions()}\n\n"
        "Clean edges, no anti-aliasing into background.\n\n"
        "CRITICAL: The background MUST be solid pure magenta #FF00FF RGB(255,0,255) "
        "with absolutely no gradients, no shadows, no shading, no variations. "
        "A single flat magenta color across the entire background. "
        "Isolated subject, centered composition. "
        "This is essential for automated background removal processing."
    )


def build_openai_sprite_prompt(pet: PetProfile, prompt_override: str = "") -> str:
    """Build the prompt for generating a 2x3 sprite sheet with transparent background.

    For use with OpenAI (native transparency support, no chroma key needed).
    Legacy: used when generating all frames in a single call.

    If prompt_override is provided, it replaces the default subject description
    but the frame layout instructions are always appended.

    Args:
        pet: Pet profile with creature info.
        prompt_override: Custom prompt to use instead of auto-generated subject.

    Returns:
        Prompt string for OpenAI image generation.
    """
    subject = prompt_override.strip() if prompt_override.strip() else _default_subject(pet)
    return (
        f"{subject}\n\n"
        f"{_sprite_layout_instructions()}\n\n"
        "The background must be completely transparent. "
        "Do NOT add any background color, gradient, or fill. "
        "The subject must be isolated with clean pixel-perfect edges."
    )


# ---------------------------------------------------------------------------
# Per-frame prompts for OpenAI multi-call pipeline
# ---------------------------------------------------------------------------

_STYLE_SUFFIX = (
    "Clean pixel art style with thick black outlines around the entire body "
    "and all major features. Flat solid colors, no shadows, no gradients, "
    "no glow, no drop shadow, no ambient occlusion. "
    "The background MUST be completely empty and transparent — no color, "
    "no vignette, no radial gradient, no gray, nothing behind the character."
)


def _build_base_frame_prompt(pet: PetProfile, prompt_override: str = "") -> str:
    """Build prompt for generating the canonical idle frame.

    Args:
        pet: Pet profile with creature info.
        prompt_override: Custom prompt override.

    Returns:
        Prompt string for the base idle frame.
    """
    if prompt_override.strip():
        subject = prompt_override.strip()
    else:
        subject = (
            f"A single pixel art character: a {pet.creature_type} named {pet.name}. "
            f"Personality: {pet.personality}. Rarity: {pet.rarity.value}."
        )
    return f"{subject}\n\nIdle pose, relaxed, eyes fully open. Centered in the frame.\n\n{_STYLE_SUFFIX}"


# Edit prompts: each describes ONLY the change from the base idle frame.
# input_fidelity="high" handles keeping everything else identical.
# Blink frames (2, 3) are NOT generated via AI — they are created
# programmatically by compositing closed-eye pixels from the sleep frame
# onto the idle frames. This avoids the OpenAI edit API introducing
# unwanted variation (different backgrounds, proportions, colors).
_EDIT_PROMPTS: dict[int, tuple[str, str]] = {
    1: (
        "idle-shift",
        "Make a very subtle pose variation — tilt the head or body very slightly. "
        "Keep the same character, same size, same position, same colors. Eyes stay open.",
    ),
    4: (
        "react",
        "Make the character look surprised — widen the eyes and open the mouth. "
        "Keep the same character, same size, same position, same colors.",
    ),
    5: (
        "sleep",
        "Make the character look like it's sleeping — close the eyes and add small zzz bubbles. "
        "Keep the same character, same size, same position, same colors.",
    ),
}


# ---------------------------------------------------------------------------
# Gemini generators (with chroma key)
# ---------------------------------------------------------------------------


def _generate_gemini_sprite_frames(config: TpetConfig, pet: PetProfile) -> tuple[list[Image.Image], Path]:
    """Generate and split a Gemini sprite sheet into individual frame images.

    Shared helper for the Gemini pixel-art and half-block pipelines.
    Handles API call, PNG save, sprite splitting, and per-frame PNG storage.

    Steps:
    1. Build prompt and generate sprite sheet via Gemini
    2. Split the sheet into individual frames (2x3 or 2x2 layout)
    3. Save each frame as a full-resolution PNG for runtime scaling

    Args:
        config: Application configuration.
        pet: Pet profile.

    Returns:
        Tuple of (list of PIL frame Images, Path to the sprite sheet PNG).

    Raises:
        RuntimeError: If the Gemini API call fails.
        OSError: If the sprite file cannot be read.
    """
    from tpet.art.client import GeminiImageClient
    from tpet.art.process import create_blink_frame, split_sprite_sheet

    art_dir = get_art_dir(config.pet_data_dir)
    sprite_path = art_dir / f"{_sanitize_name(pet.name)}_sprite.png"

    resolved = config.resolved_image_art_provider
    client = GeminiImageClient(model=resolved.model, api_key=resolved.api_key)
    prompt = build_sprite_prompt(pet, prompt_override=config.art_prompt)
    save_prompt(config.pet_data_dir, pet.name, prompt)
    client.generate_sprite(
        prompt=prompt,
        output_path=sprite_path,
        aspect_ratio="2:3",
        resolution="1k",
    )

    sprite = Image.open(sprite_path)
    logger.debug("Sprite sheet loaded: %dx%d", sprite.width, sprite.height)

    frames = split_sprite_sheet(sprite)
    logger.debug("Split into %d frames", len(frames))

    # Replace AI-generated blink frames with programmatic composites
    # for consistent eye-closed animation (AI blink frames have too much variation)
    if len(frames) >= 6:
        sleep_frame = frames[5]
        frames[2] = create_blink_frame(frames[0], sleep_frame)
        frames[3] = create_blink_frame(frames[1], sleep_frame)
        logger.info("Replaced blink frames 2,3 with programmatic composites from idle+sleep")

    # Save raw PNG frames at full resolution for runtime scaling
    for i, frame in enumerate(frames):
        save_png_frame(config.pet_data_dir, pet.name, i, frame)

    return frames, sprite_path


def _process_frames(
    frames: list[Image.Image],
    config: TpetConfig,
    pet_name: str,
    resize_fn: Callable[[Image.Image, int], Image.Image],
    target_height: int,
    convert_fn: Callable[[Image.Image], str],
    save_fn: Callable[[Path, str, int, str], Path] | None,
    label: str,
    chroma_key: bool = False,
    save_png: bool = False,
) -> list[str]:
    """Shared pipeline: resize -> optional chroma key -> convert -> save for each frame.

    Args:
        frames: Source PIL images.
        config: Application configuration.
        pet_name: Pet name for storage.
        resize_fn: Function to resize a frame (takes image and target_height).
        target_height: Target pixel height for resizing.
        convert_fn: Function to convert a PIL image to a text string.
        save_fn: Function to save a text frame (config_dir, pet_name, index, data), or None to skip.
        label: Human-readable frame type for logging.
        chroma_key: Whether to apply chroma key background removal after resize.
        save_png: Whether to also save source frames as raw PNGs.

    Returns:
        List of converted frame strings.
    """
    from tpet.art.process import remove_chroma_key

    output_frames: list[str] = []
    for i, frame in enumerate(frames):
        if save_png:
            save_png_frame(config.pet_data_dir, pet_name, i, frame)
        resized = resize_fn(frame, target_height)
        if chroma_key:
            resized = remove_chroma_key(resized, tolerance=config.chroma_tolerance)
        frame_data = convert_fn(resized)
        output_frames.append(frame_data)
        if save_fn:
            save_fn(config.pet_data_dir, pet_name, i, frame_data)
        logger.debug("%s frame %d: %dx%d -> %d chars", label, i, resized.width, resized.height, len(frame_data))
    return output_frames


def generate_halfblock_art(config: TpetConfig, pet: PetProfile) -> list[str]:
    """Generate half-block art frames via Gemini with chroma key removal.

    Args:
        config: Application configuration.
        pet: Pet profile.

    Returns:
        List of half-block strings (one per frame), or empty list on failure.
    """
    from tpet.art.process import image_to_halfblock, resize_for_halfblock

    try:
        logger.info("Generating halfblock sprite sheet for %s via Gemini", pet.name)
        frames, _ = _generate_gemini_sprite_frames(config, pet)
        return _process_frames(
            frames,
            config,
            pet.name,
            resize_fn=resize_for_halfblock,
            target_height=config.halfblock_size,
            convert_fn=image_to_halfblock,
            save_fn=save_halfblock_frame,
            label="Halfblock",
            chroma_key=True,
        )
    except (RuntimeError, OSError):
        logger.exception("Failed to generate halfblock art for %s", pet.name)
        return []


# ---------------------------------------------------------------------------
# OpenAI generators (multi-call with edit for consistency)
# ---------------------------------------------------------------------------


def _generate_openai_frames(
    config: TpetConfig,
    pet: PetProfile,
    on_progress: Callable[[int, int, str], None] | None = None,
    base_image_path: Path | None = None,
) -> tuple[list[Image.Image], GenerationResult]:
    """Generate 6 individual frames via OpenAI: 1 base generate + 3 edits + 2 programmatic blink.

    Uses images.generate for the base idle frame, then images.edit with
    input_fidelity="high" for expression variants (idle-shift, react, sleep).
    Blink frames (2, 3) are created programmatically by compositing closed-eye
    pixels from the sleep frame onto the idle frames, since the edit API
    introduces too much variation for minimal eye-only changes.

    When ``base_image_path`` is provided, it is used as the idle frame directly,
    skipping generation and cleanup steps.

    Args:
        config: Application configuration.
        pet: Pet profile.
        on_progress: Optional callback(current, total, label) for progress updates.
        base_image_path: Optional path to a user-provided idle frame image.

    Returns:
        Tuple of (list of 6 PIL Images in order, GenerationResult with usage stats).

    Raises:
        RuntimeError: If any API call fails.
    """
    from pathlib import Path as _Path

    from tpet.art.openai_client import GenerationResult, ImageUsage, OpenAIImageClient
    from tpet.art.process import create_blink_frame

    result = GenerationResult()
    total_usage = ImageUsage()
    art_dir = get_art_dir(config.pet_data_dir)
    safe_name = _sanitize_name(pet.name)
    resolved = config.resolved_image_art_provider
    client = OpenAIImageClient(model=resolved.model, api_key=resolved.api_key, base_url=resolved.base_url or None)

    if base_image_path is not None:
        # Use the user-provided image as frame 0 — skip generation and cleanup
        # +len for variant edits, +1 for blink compositing
        total_steps = len(_EDIT_PROMPTS) + 1

        def _report(current: int, label: str) -> None:
            if on_progress:
                on_progress(current, total_steps, label)

        base_path = _Path(base_image_path)
        # Copy the user image into the art dir as frame 0
        frame_0_path = art_dir / f"{safe_name}_frame_0.png"
        user_image = Image.open(base_path).convert("RGBA")
        user_image.save(frame_0_path, format="PNG")
        logger.info("Using user-provided base image: %s (%dx%d)", base_path, user_image.width, user_image.height)
        ai_frames: dict[int, Image.Image] = {0: user_image}
    else:
        # +1 for generate, +1 for cleanup edit, +len for variant edits, +1 for blink compositing
        total_steps = 2 + len(_EDIT_PROMPTS) + 1

        def _report(current: int, label: str) -> None:
            if on_progress:
                on_progress(current, total_steps, label)

        # Step 1: Generate the initial base idle frame
        _report(1, "idle (generate)")
        base_prompt = _build_base_frame_prompt(pet, prompt_override=config.art_prompt)
        save_prompt(config.pet_data_dir, pet.name, base_prompt)
        raw_base_path = art_dir / f"{safe_name}_frame_0_raw.png"
        logger.info("Generating base idle frame for %s via OpenAI", pet.name)
        _, usage = client.generate_sprite(
            prompt=base_prompt,
            output_path=raw_base_path,
            size="1024x1024",
            quality="medium",
        )
        total_usage += usage
        result.api_calls += 1

        # Step 2: Clean up the base frame via edit to fix background transparency
        _report(2, "idle (cleanup)")
        frame_0_path = art_dir / f"{safe_name}_frame_0.png"
        logger.info("Cleaning up base frame background via edit")
        _, usage = client.edit_sprite(
            image_path=raw_base_path,
            prompt=(
                "Keep this pixel art character exactly as it is — same pose, same colors, "
                "same details, same size, same position. Change nothing about the character. "
                "Remove all background completely so it is fully transparent."
            ),
            output_path=frame_0_path,
            quality="medium",
        )
        total_usage += usage
        result.api_calls += 1

        ai_frames = {0: Image.open(frame_0_path)}

    # frame_0_path is the canonical idle frame used as input for all edits
    edit_source = frame_0_path
    logger.debug("Base frame 0 (idle): %dx%d", ai_frames[0].width, ai_frames[0].height)

    # Edit the base frame for each variant (1=idle-shift, 4=react, 5=sleep)
    # Step numbering starts after the base image steps (1 if user-provided, 3 if generated)
    edit_step_start = 1 if base_image_path is not None else 3
    for step, frame_idx in enumerate(sorted(_EDIT_PROMPTS.keys()), start=edit_step_start):
        label, edit_prompt = _EDIT_PROMPTS[frame_idx]
        _report(step, label)
        out_path = art_dir / f"{safe_name}_frame_{frame_idx}.png"
        logger.info("Generating frame %d (%s) via OpenAI edit", frame_idx, label)
        _, usage = client.edit_sprite(
            image_path=edit_source,
            prompt=edit_prompt,
            output_path=out_path,
            quality="medium",
        )
        total_usage += usage
        result.api_calls += 1
        frame = Image.open(out_path)
        ai_frames[frame_idx] = frame
        logger.debug("Frame %d (%s): %dx%d", frame_idx, label, frame.width, frame.height)

    # Step 4: Create blink frames programmatically from idle + sleep
    _report(total_steps, "blink (compositing)")
    sleep_frame = ai_frames[5]

    # Frame 2: blink = idle body + closed eyes from sleep
    blink_0 = create_blink_frame(ai_frames[0], sleep_frame)
    blink_0_path = art_dir / f"{safe_name}_frame_2.png"
    blink_0.save(blink_0_path, format="PNG")
    ai_frames[2] = blink_0
    logger.info("Created blink frame 2 via compositing from idle + sleep")

    # Frame 3: blink-shift = idle-shift body + closed eyes from sleep
    blink_1 = create_blink_frame(ai_frames[1], sleep_frame)
    blink_1_path = art_dir / f"{safe_name}_frame_3.png"
    blink_1.save(blink_1_path, format="PNG")
    ai_frames[3] = blink_1
    logger.info("Created blink frame 3 via compositing from idle-shift + sleep")

    # Assemble in order: 0, 1, 2, 3, 4, 5
    frames = [ai_frames[i] for i in range(6)]

    result.usage = total_usage
    result.frame_count = len(frames)
    return frames, result


def generate_halfblock_art_openai(
    config: TpetConfig,
    pet: PetProfile,
    on_progress: Callable[[int, int, str], None] | None = None,
    base_image_path: Path | None = None,
) -> tuple[list[str], GenerationResult | None]:
    """Generate half-block art frames via OpenAI with native transparency.

    Args:
        config: Application configuration.
        pet: Pet profile.
        on_progress: Optional callback(current, total, label) for progress updates.
        base_image_path: Optional path to a user-provided idle frame image.

    Returns:
        Tuple of (list of halfblock strings, GenerationResult or None on failure).
    """
    from tpet.art.process import image_to_halfblock, resize_for_halfblock

    try:
        frames, gen_result = _generate_openai_frames(
            config,
            pet,
            on_progress=on_progress,
            base_image_path=base_image_path,
        )
        hblk_frames = _process_frames(
            frames,
            config,
            pet.name,
            resize_fn=resize_for_halfblock,
            target_height=config.halfblock_size,
            convert_fn=image_to_halfblock,
            save_fn=save_halfblock_frame,
            label="Halfblock",
            save_png=True,
        )
        return hblk_frames, gen_result
    except (RuntimeError, OSError):
        logger.exception("Failed to generate halfblock art (OpenAI) for %s", pet.name)
        return [], None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def generate_art(
    config: TpetConfig,
    pet: PetProfile,
    on_progress: Callable[[int, int, str], None] | None = None,
    base_image_path: Path | None = None,
) -> tuple[list[str], GenerationResult | None]:
    """Generate graphical art based on config settings.

    Dispatches to the appropriate generator based on art_mode and the resolved
    image art provider.  When ``base_image_path`` is provided, the OpenAI-compatible
    pipeline is always used (regardless of provider) since only the OpenAI images
    edit API supports image editing.

    Args:
        config: Application configuration.
        pet: Pet profile.
        on_progress: Optional callback(current, total, label) for progress updates.
        base_image_path: Optional path to a user-provided idle frame image.
            When set, this image is used as frame 0 and OpenAI edit generates variants.

    Returns:
        Tuple of (list of art strings, GenerationResult or None).

    Raises:
        ValueError: If art_mode is not a graphical mode, provider is unsupported,
            or API key is missing.
    """
    from tpet.config import ArtMode, LLMProvider

    resolved = config.resolved_image_art_provider

    # Providers that cannot generate images
    if resolved.provider == LLMProvider.OLLAMA:
        raise ValueError(
            "Ollama does not support image generation. "
            "Set image_art_provider_config.provider to 'openai', 'openrouter', or 'gemini'."
        )
    if resolved.provider == LLMProvider.CLAUDE:
        raise ValueError(
            "Claude does not support image generation via this pipeline. "
            "Set image_art_provider_config.provider to 'openai', 'openrouter', or 'gemini'."
        )

    # Validate API key before dispatching
    if not resolved.api_key:
        config_dir = config.pet_data_dir
        raise ValueError(
            f"API key not found (checked env var: {resolved.api_key_env!r}). "
            f"Image generation via {resolved.provider.value} requires this key.\n"
            f"Set it in your environment or add it to {config_dir / '.env'}"
        )

    # Determine pipeline: OpenAI-compatible (openai/openrouter) or Gemini
    # Force OpenAI-compatible pipeline when a base image is provided
    use_openai_compat = resolved.is_openai_compat or base_image_path is not None

    if config.art_mode == ArtMode.SIXEL_ART:
        if use_openai_compat:
            return generate_halfblock_art_openai(
                config,
                pet,
                on_progress=on_progress,
                base_image_path=base_image_path,
            )
        return generate_halfblock_art(config, pet), None
    else:
        raise ValueError(f"Cannot generate graphical art for mode: {config.art_mode}")
