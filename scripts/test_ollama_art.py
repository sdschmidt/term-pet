#!/usr/bin/env python3
"""Test Ollama models for tpet ASCII art generation.

Reads configuration from a YAML file, tests each model at each temperature
against multiple subject prompts, and generates a dark-mode HTML report
showing all 6 animation frames in monospace <pre> blocks.

Usage:
    uv run scripts/test_ollama_art.py
    uv run scripts/test_ollama_art.py --config scripts/my_art_config.yaml
    uv run scripts/test_ollama_art.py --output art_results.html
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent / "ollama_art_test_config.yaml"
_DEFAULT_OUTPUT = Path(__file__).parent / "ollama_art_report.html"
_DEFAULT_RESULTS = Path(__file__).parent / "ollama_art_results.json"

# ---------------------------------------------------------------------------
# ASCII art rules — mirrors tpet/profile/generator.py _ART_RULES
# ---------------------------------------------------------------------------

_ART_SYSTEM_PROMPT = (
    "You are an ASCII artist. Draw small creatures using text characters.\n\n"
    "RULES:\n"
    "- Keep it small: 6-8 lines tall, 12-16 characters wide\n"
    "- Pad every line with spaces so all lines are the same width\n"
    "- Use only single-width characters\n"
    "- Center the creature in the frame\n"
    "- Draw the EXACT creature requested — do not default to cats\n\n"
    "OUTPUT FORMAT:\n"
    "Output exactly 6 frames separated by a line containing only '---'.\n"
    "The 6 frames represent: idle, idle-shift, blink, blink-shift, surprised, sleeping.\n"
    "Do NOT include frame labels, numbers, or explanations.\n\n"
    "Output ONLY the 6 ASCII art frames separated by --- lines. No explanation, no JSON, no markdown, no labels.\n"
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ArtPrompt:
    """A test prompt for art generation."""

    name: str
    prompt: str


@dataclass
class FrameValidation:
    """Validation results for a single generation."""

    frame_count: int
    ok: bool
    errors: list[str]
    line_counts: list[int]
    line_widths: list[int]


@dataclass
class ArtResult:
    """Result of a single model/temperature/prompt test."""

    model: str
    temperature: float
    prompt_name: str
    raw_response: str
    frames: list[str]
    normalized_frames: list[str]
    duration_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    validation: FrameValidation | None = None

    @property
    def tokens_per_second(self) -> float:
        if self.duration_ms <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / (self.duration_ms / 1000)


@dataclass
class ArtTestConfig:
    """Parsed test configuration."""

    ollama_base_url: str
    temperatures: list[float]
    samples_per_combo: int
    max_tokens: int
    expected_frames: int
    min_lines_per_frame: int
    max_lines_per_frame: int
    min_line_width: int
    max_line_width: int
    prompts: list[ArtPrompt]
    models: list[str]
    results: list[ArtResult] = field(default_factory=list)
    model_sizes: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON extraction (mirrors tpet profile/generator.py _extract_json)
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response, stripping markdown fences."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown fence
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } blob
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Frame normalization — auto-fix common issues before validation
# ---------------------------------------------------------------------------


def _normalize_frames(frames: list[str], config: ArtTestConfig) -> list[str]:
    """Auto-fix common frame issues: right-pad lines, strip trailing whitespace, equalize line counts.

    Steps:
    1. Strip trailing whitespace from every line
    2. Right-pad all lines to the max width across all frames
    3. Truncate trailing empty lines from each frame
    4. Pad shorter frames with empty lines to match the tallest frame

    Returns:
        Normalized list of frame strings.
    """
    if not frames:
        return frames

    # Split into lines and strip trailing whitespace
    split_frames: list[list[str]] = []
    for frame in frames:
        lines = frame.split("\n")
        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in lines]
        split_frames.append(lines)

    # Find the max width across ALL non-empty lines in all frames
    max_width = 0
    for lines in split_frames:
        for line in lines:
            if line:
                max_width = max(max_width, len(line))

    if max_width == 0:
        return frames

    # Right-pad every line to max_width
    padded_frames: list[list[str]] = []
    for lines in split_frames:
        padded = [line.ljust(max_width) if line else " " * max_width for line in lines]
        padded_frames.append(padded)

    # Truncate trailing empty lines from each frame
    trimmed_frames: list[list[str]] = []
    for lines in padded_frames:
        # Remove trailing all-space lines
        while lines and lines[-1].strip() == "":
            lines.pop()
        trimmed_frames.append(lines)

    # Equalize line counts — pad shorter frames to match tallest
    if trimmed_frames:
        target_lines = max(len(lines) for lines in trimmed_frames)
        for i in range(len(trimmed_frames)):
            while len(trimmed_frames[i]) < target_lines:
                trimmed_frames[i].append(" " * max_width)

    # Rejoin into strings
    return ["\n".join(lines) for lines in trimmed_frames]


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------


def _validate_frames(frames: list[str], config: ArtTestConfig) -> FrameValidation:
    """Validate frame dimensions and consistency."""
    errors: list[str] = []
    line_counts: list[int] = []
    line_widths: list[int] = []

    if len(frames) != config.expected_frames:
        errors.append(f"Expected {config.expected_frames} frames, got {len(frames)}")
        return FrameValidation(
            frame_count=len(frames),
            ok=False,
            errors=errors,
            line_counts=line_counts,
            line_widths=line_widths,
        )

    for i, frame in enumerate(frames):
        lines = frame.split("\n")
        line_counts.append(len(lines))
        widths = [len(line) for line in lines]
        line_widths.extend(widths)

        if len(lines) < config.min_lines_per_frame or len(lines) > config.max_lines_per_frame:
            errors.append(
                f"Frame {i}: {len(lines)} lines (expected {config.min_lines_per_frame}-{config.max_lines_per_frame})"
            )

        # Check all lines in this frame are same width
        if widths and len(set(widths)) > 1:
            errors.append(f"Frame {i}: inconsistent line widths {set(widths)}")

    # Check all frames have same line count
    if len(set(line_counts)) > 1:
        errors.append(f"Inconsistent line counts across frames: {line_counts}")

    # Check all frames have same line width
    if line_widths and len(set(line_widths)) > 1:
        errors.append(f"Inconsistent line widths across frames: min={min(line_widths)}, max={max(line_widths)}")

    # Check widths are in range
    if line_widths:
        avg_w = sum(line_widths) / len(line_widths)
        if avg_w < config.min_line_width or avg_w > config.max_line_width:
            errors.append(
                f"Average line width {avg_w:.0f} outside range {config.min_line_width}-{config.max_line_width}"
            )

    return FrameValidation(
        frame_count=len(frames),
        ok=len(errors) == 0,
        errors=errors,
        line_counts=line_counts,
        line_widths=line_widths,
    )


# ---------------------------------------------------------------------------
# Frame parsing — handles --- separated, JSON, and other formats
# ---------------------------------------------------------------------------

# Separator patterns between frames
_FRAME_SEP_RE = re.compile(r"^---+\s*$", re.MULTILINE)
_FRAME_SEP_FENCE = re.compile(r"^```\s*$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    """Remove code fences (``` ... ```) from model output, preserving inner indentation."""
    # Remove opening fence: ``` or ```lang, followed by optional spaces, then newline.
    # Only match the newline — do NOT consume leading whitespace of the first content line.
    text = re.sub(r"^```[a-zA-Z]*[ \t]*\n", "", text)
    # Remove closing fence: optional newline, then ``` followed by optional spaces at end.
    text = re.sub(r"\n```[ \t]*$", "", text)
    # Remove any remaining standalone ``` lines (fences around individual frames)
    text = re.sub(r"^\s*```\s*$", "", text, flags=re.MULTILINE)
    return text


def _strip_blank_lines(text: str) -> str:
    """Remove leading and trailing blank lines, preserving left padding on content lines."""
    lines = text.split("\n")
    # Strip leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    # Strip trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _parse_frames(text: str) -> list[str]:
    """Parse frames from model output.

    Tries multiple formats:
    1. --- separated frames
    2. ``` separated frames
    3. Numbered frames (Frame 0:, Frame 1:, etc.)

    Returns:
        List of frame strings, or empty list if nothing parsed.
    """
    # Strip leading/trailing blank lines, then strip code fences.
    # Do NOT strip() the whole block — that would eat left padding from the first content line.
    text = text.rstrip("\n")
    # Remove leading blank lines (but not their indentation — just empty lines)
    while text.startswith("\n"):
        text = text[1:]
    text = _strip_code_fences(text)

    def _clean(parts: list[str]) -> list[str]:
        """Strip echoed frame labels from the first line of each part."""
        cleaned = []
        label_re = re.compile(
            r"^[\s]*Frame\s*\d+\s*[:.]?\s*(?:idle|shift|blink|surprised|sleeping|react)"
            r"[\s]*(?:\([^)]*\))?[\s]*\n?",
            re.IGNORECASE,
        )
        for p in parts:
            p = label_re.sub("", p, count=1)
            cleaned.append(_strip_blank_lines(p.rstrip()))
        return [p for p in cleaned if p.strip()]

    # Try --- separated frames
    parts = _FRAME_SEP_RE.split(text)
    parts = _clean(parts)
    if len(parts) >= 4:
        return parts[:6]

    # Try ``` separated frames (remaining fences after strip_code_fences)
    parts = _FRAME_SEP_FENCE.split(text)
    parts = _clean(parts)
    if len(parts) >= 4:
        return parts[:6]

    # Try numbered frames: "Frame 0:", "Frame 1:", etc.
    frame_re = re.compile(r"Frame\s*\d+\s*[:.]?\s*", re.IGNORECASE)
    parts = frame_re.split(text)
    parts = _clean(parts)
    if len(parts) >= 4:
        return parts[:6]

    return []


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: Path) -> ArtTestConfig:
    """Load and validate the test configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    prompts: list[ArtPrompt] = []
    for p in raw["prompts"]:
        prompts.append(ArtPrompt(name=p["name"], prompt=p["prompt"]))

    return ArtTestConfig(
        ollama_base_url=raw.get("ollama_base_url", "http://localhost:11434/v1"),
        temperatures=raw.get("temperatures", [0.3, 0.6, 0.9]),
        samples_per_combo=raw.get("samples_per_combo", 1),
        max_tokens=raw.get("max_tokens", 1024),
        expected_frames=raw.get("expected_frames", 6),
        min_lines_per_frame=raw.get("min_lines_per_frame", 4),
        max_lines_per_frame=raw.get("max_lines_per_frame", 12),
        min_line_width=raw.get("min_line_width", 8),
        max_line_width=raw.get("max_line_width", 24),
        prompts=prompts,
        models=raw["models"],
    )


# ---------------------------------------------------------------------------
# Model testing
# ---------------------------------------------------------------------------


def _run_test(
    client: OpenAI,
    model: str,
    temperature: float,
    user_prompt: str,
    max_tokens: int,
) -> tuple[str, float, int, int, str | None]:
    """Run a single LLM call. Returns (text, duration_ms, prompt_tok, completion_tok, error)."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ART_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=90,
        )
        elapsed = (time.perf_counter() - start) * 1000
        choice = response.choices[0] if response.choices else None
        text = choice.message.content.strip() if choice and choice.message and choice.message.content else ""
        prompt_tok = response.usage.prompt_tokens if response.usage else 0
        completion_tok = response.usage.completion_tokens if response.usage else 0
        return text, elapsed, prompt_tok, completion_tok, None
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - start) * 1000
        return "", elapsed, 0, 0, str(exc)


def _fetch_model_sizes(ollama_base: str) -> dict[str, str]:
    """Fetch model sizes from the Ollama API."""
    sizes: dict[str, str] = {}
    try:
        resp = httpx.get(f"{ollama_base}/api/tags", timeout=10)
        resp.raise_for_status()
        for entry in resp.json().get("models", []):
            name = entry.get("name", "")
            size_bytes = entry.get("size", 0)
            if size_bytes > 0:
                size_gb = size_bytes / (1024**3)
                sizes[name] = f"{size_gb:.1f} GB" if size_gb >= 1 else f"{size_bytes / (1024**2):.0f} MB"
    except Exception:  # noqa: BLE001
        logger.warning("Could not fetch model sizes from Ollama API")
    return sizes


def _ollama_api_base(openai_base_url: str) -> str:
    return openai_base_url.rstrip("/").removesuffix("/v1")


def _load_model(ollama_base: str, model: str) -> None:
    logger.info("  Loading model %s ...", model)
    try:
        httpx.post(
            f"{ollama_base}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "10m"},
            timeout=300,
        )
    except Exception:  # noqa: BLE001
        logger.warning("  Failed to pre-load %s (will try anyway)", model)


def _unload_model(ollama_base: str, model: str) -> None:
    logger.info("  Unloading model %s ...", model)
    try:
        httpx.post(
            f"{ollama_base}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "0"},
            timeout=30,
        )
    except Exception:  # noqa: BLE001
        logger.warning("  Failed to unload %s", model)


def _save_partial_results(
    results: list[ArtResult],
    model_sizes: dict[str, str],
    completed_models: list[str],
    results_path: Path,
) -> None:
    """Save incremental results to JSON for resume support."""
    data = {
        "completed_models": completed_models,
        "model_sizes": model_sizes,
        "results": [
            {
                "model": r.model,
                "temperature": r.temperature,
                "prompt_name": r.prompt_name,
                "raw_response": r.raw_response,
                "frames": r.frames,
                "normalized_frames": r.normalized_frames,
                "duration_ms": r.duration_ms,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "error": r.error,
                "validation": {
                    "frame_count": r.validation.frame_count,
                    "ok": r.validation.ok,
                    "errors": r.validation.errors,
                    "line_counts": r.validation.line_counts,
                    "line_widths": r.validation.line_widths,
                }
                if r.validation
                else None,
            }
            for r in results
        ],
    }
    results_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("  Saved %d results to %s", len(results), results_path)


def _load_partial_results(results_path: Path) -> tuple[list[ArtResult], dict[str, str], list[str]]:
    """Load previously saved partial results for resume."""
    if not results_path.exists():
        return [], {}, []
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        results = []
        for r in data.get("results", []):
            v_data = r.get("validation")
            validation = (
                FrameValidation(
                    frame_count=v_data["frame_count"],
                    ok=v_data["ok"],
                    errors=v_data["errors"],
                    line_counts=v_data["line_counts"],
                    line_widths=v_data["line_widths"],
                )
                if v_data
                else None
            )
            results.append(
                ArtResult(
                    model=r["model"],
                    temperature=r["temperature"],
                    prompt_name=r["prompt_name"],
                    raw_response=r["raw_response"],
                    frames=r["frames"],
                    normalized_frames=r.get("normalized_frames", []),
                    duration_ms=r["duration_ms"],
                    prompt_tokens=r.get("prompt_tokens", 0),
                    completion_tokens=r.get("completion_tokens", 0),
                    error=r.get("error"),
                    validation=validation,
                )
            )
        return results, data.get("model_sizes", {}), data.get("completed_models", [])
    except (json.JSONDecodeError, KeyError):
        logger.warning("Could not parse %s — starting fresh", results_path)
        return [], {}, []


def run_all_tests(config: ArtTestConfig, results_path: Path) -> None:
    """Run all model/temperature/prompt combinations."""
    client = OpenAI(base_url=config.ollama_base_url, api_key="ollama")
    ollama_base = _ollama_api_base(config.ollama_base_url)

    # Resume support
    prev_results, prev_sizes, completed_models = _load_partial_results(results_path)
    if completed_models:
        logger.info("Resuming: %d models already completed (%s)", len(completed_models), ", ".join(completed_models))
        config.results = prev_results
        config.model_sizes.update(prev_sizes)

    config.model_sizes.update(_fetch_model_sizes(ollama_base))

    remaining_models = [m for m in config.models if m not in completed_models]
    tests_per_model = len(config.temperatures) * len(config.prompts) * config.samples_per_combo
    total = len(config.models) * tests_per_model
    completed = len(prev_results)

    for model_idx, model in enumerate(remaining_models, len(completed_models) + 1):
        size_label = config.model_sizes.get(model, "?")
        logger.info("[%d/%d] Testing model: %s (%s)", model_idx, len(config.models), model, size_label)

        try:
            _load_model(ollama_base, model)

            for temp in config.temperatures:
                for prompt in config.prompts:
                    user_prompt = (
                        f"IMPORTANT: You must draw a {prompt.name.upper()}, not a cat or any other animal.\n"
                        f"{prompt.prompt}\n"
                        f"Output exactly 6 animation frames separated by --- lines."
                    )
                    # Suppress thinking/reasoning for models that support /no_think (qwen3, etc.)
                    if re.search(r"(?:qwen3|qwq|deepseek-r1|openthinker)", model, re.IGNORECASE):
                        user_prompt += " /no_think"

                    for _ in range(config.samples_per_combo):
                        text, elapsed, ptok, ctok, err = _run_test(
                            client,
                            model,
                            temp,
                            user_prompt,
                            config.max_tokens,
                        )

                        frames: list[str] = []
                        normalized: list[str] = []
                        validation: FrameValidation | None = None

                        if not err and text:
                            frames = _parse_frames(text)
                            if not frames:
                                # Fall back to JSON parse
                                parsed = _extract_json(text)
                                if parsed and "ascii_art" in parsed:
                                    frames = parsed["ascii_art"]
                            if frames:
                                # Auto-fix: normalize padding
                                normalized = _normalize_frames(frames, config)
                                validation = _validate_frames(normalized, config)
                            else:
                                err = "Could not extract frames from response"
                        elif not err and not text and ctok > 0:
                            err = "Empty response (model used tokens on thinking/reasoning)"

                        result = ArtResult(
                            model=model,
                            temperature=temp,
                            prompt_name=prompt.name,
                            raw_response=text,
                            frames=frames,
                            normalized_frames=normalized,
                            duration_ms=elapsed,
                            prompt_tokens=ptok,
                            completion_tokens=ctok,
                            error=err,
                            validation=validation,
                        )
                        config.results.append(result)
                        completed += 1

                        status = "OK" if not err else f"ERR: {err[:60]}"
                        val_status = ""
                        if validation:
                            val_status = " VALID" if validation.ok else f" INVALID({len(validation.errors)})"
                        tok_s = f" {result.tokens_per_second:.1f} tok/s" if result.tokens_per_second > 0 else ""
                        logger.info(
                            "  [%d/%d] %s | temp=%.2f | %s | %.0fms%s | %s%s",
                            completed,
                            total,
                            model,
                            temp,
                            prompt.name,
                            elapsed,
                            tok_s,
                            status,
                            val_status,
                        )

        except Exception:  # noqa: BLE001
            logger.exception("  Model %s failed unexpectedly — skipping", model)

        try:
            _unload_model(ollama_base, model)
        except Exception:  # noqa: BLE001
            logger.warning("  Failed to unload %s after error", model)

        completed_models.append(model)
        _save_partial_results(config.results, config.model_sizes, completed_models, results_path)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_ESC = html.escape


def _pass_fail(passed: bool) -> str:
    color = "#4CAF50" if passed else "#F44336"
    label = "PASS" if passed else "FAIL"
    return f'<span class="badge" style="background:{color}">{label}</span>'


def generate_html_report(config: ArtTestConfig, output_path: Path) -> None:
    """Generate the dark-mode HTML report with all frames in monospace <pre> blocks."""
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Group results by model -> temperature -> prompt
    grouped: dict[str, dict[float, dict[str, list[ArtResult]]]] = {}
    for r in config.results:
        grouped.setdefault(r.model, {}).setdefault(r.temperature, {}).setdefault(r.prompt_name, []).append(r)

    # Config summary
    config_summary = (
        f"<strong>Base URL:</strong> {_ESC(config.ollama_base_url)}<br>"
        f"<strong>Temperatures:</strong> {', '.join(str(t) for t in config.temperatures)}<br>"
        f"<strong>Max tokens:</strong> {config.max_tokens}<br>"
        f"<strong>Expected frames:</strong> {config.expected_frames}<br>"
        f"<strong>Frame size:</strong> {config.min_lines_per_frame}-{config.max_lines_per_frame} lines, "
        f"{config.min_line_width}-{config.max_line_width} chars wide<br>"
        f"<strong>Prompts:</strong> {len(config.prompts)}<br>"
        f"<strong>Models tested:</strong> {len(config.models)}<br>"
        f"<strong>Total tests:</strong> {len(config.results)}"
    )

    # Prompt summary
    prompt_rows = ""
    for p in config.prompts:
        prompt_rows += f"<tr><td>{_ESC(p.name)}</td><td>{_ESC(p.prompt)}</td></tr>\n"

    # Build model sections
    model_sections = ""
    for model in config.models:
        if model not in grouped:
            continue
        temps = grouped[model]

        model_results = [r for r in config.results if r.model == model]
        avg_ms = sum(r.duration_ms for r in model_results) / len(model_results) if model_results else 0
        error_count = sum(1 for r in model_results if r.error)
        valid_count = sum(1 for r in model_results if r.validation and r.validation.ok)
        total_with_val = sum(1 for r in model_results if r.validation)
        avg_tok_s = sum(r.tokens_per_second for r in model_results if r.tokens_per_second > 0) / max(
            1, sum(1 for r in model_results if r.tokens_per_second > 0)
        )

        mid = _ESC(model)
        size_label = config.model_sizes.get(model, "?")
        val_pct = f"{valid_count}/{total_with_val}" if total_with_val else "0/0"

        model_sections += f"""
        <div class="model-section">
            <h2 id="{mid}">{mid} <span class="model-size">({_ESC(size_label)})</span></h2>
            <div class="model-stats">
                Avg latency: <strong>{avg_ms:.0f}ms</strong> |
                Avg speed: <strong>{avg_tok_s:.1f} tok/s</strong> |
                Errors: <strong class="{"err" if error_count else ""}">{error_count}</strong> |
                Validation: <strong>{val_pct}</strong>
            </div>
        """

        for temp in config.temperatures:
            if temp not in temps:
                continue
            model_sections += f"<h3>Temperature: {temp}</h3>\n"

            for prompt in config.prompts:
                results = temps[temp].get(prompt.name, [])
                for r in results:
                    model_sections += '<div class="test-case">\n'
                    model_sections += f'<div class="test-header">{_ESC(prompt.name)} &mdash; '

                    has_art = bool(r.frames or r.normalized_frames)
                    if r.error:
                        model_sections += f'{_pass_fail(False)} <span class="error-text">{_ESC(r.error[:200])}</span>'
                    elif not has_art:
                        model_sections += f'{_pass_fail(False)} <span class="error-text">No frames extracted</span>'
                    elif r.validation and not r.validation.ok:
                        model_sections += f"{_pass_fail(False)}"
                        for ve in r.validation.errors:
                            model_sections += f'<span class="warn-text">{_ESC(ve)}</span> '
                    else:
                        model_sections += f"{_pass_fail(True)}"

                    tok_s_str = f" | {r.tokens_per_second:.1f} tok/s" if r.tokens_per_second > 0 else ""
                    model_sections += f'<span class="meta-inline">{r.duration_ms:.0f}ms{tok_s_str}</span>'
                    model_sections += "</div>\n"

                    display_frames = r.normalized_frames or r.frames
                    if display_frames:
                        # Show all 6 frames in a grid of <pre> blocks
                        model_sections += '<div class="frames-grid">\n'
                        frame_labels = ["0: idle", "1: shift", "2: blink", "3: blink-shift", "4: react", "5: sleep"]
                        for i, frame in enumerate(display_frames):
                            label = frame_labels[i] if i < len(frame_labels) else f"Frame {i}"
                            model_sections += (
                                f'<div class="frame-card">'
                                f'<div class="frame-label">{_ESC(label)}</div>'
                                f'<pre class="frame-art">{_ESC(frame)}</pre>'
                                f"</div>\n"
                            )
                        model_sections += "</div>\n"
                    elif r.raw_response and not r.error:
                        # JSON parse failed — show raw response
                        truncated = r.raw_response[:500]
                        model_sections += f'<pre class="raw-response">{_ESC(truncated)}</pre>\n'

                    model_sections += "</div>\n"

        model_sections += "</div>\n"

    # TOC
    toc_links = "".join(
        f'<a href="#{_ESC(m)}">{_ESC(m)} ({_ESC(config.model_sizes.get(m, "?"))})</a>'
        for m in config.models
        if m in grouped
    )

    # Assemble HTML
    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ollama ASCII Art Model Test Report</title>
<style>
    :root {{
        --bg: #1a1a2e;
        --surface: #16213e;
        --surface2: #0f3460;
        --text: #e6e6e6;
        --text-dim: #8892a0;
        --accent: #e94560;
        --green: #4CAF50;
        --yellow: #FFC107;
        --red: #F44336;
        --blue: #2196F3;
        --border: #2a2a4a;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        background: var(--bg);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        padding: 2rem;
    }}
    h1 {{
        color: var(--accent);
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.5rem;
    }}
    h2 {{
        color: var(--blue);
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }}
    h3 {{
        color: var(--text-dim);
        font-size: 1rem;
        margin: 1rem 0 0.5rem;
    }}
    .timestamp {{
        color: var(--text-dim);
        font-size: 0.85rem;
        margin-bottom: 2rem;
    }}
    .info-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
    }}
    .info-card strong {{ color: var(--accent); }}
    .model-section {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 2rem;
    }}
    .model-stats {{
        color: var(--text-dim);
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }}
    .model-stats .err {{ color: var(--red); }}
    .model-size {{
        color: var(--text-dim);
        font-size: 0.9rem;
        font-weight: normal;
    }}
    .test-case {{
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }}
    .test-header {{
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }}
    .test-header .badge {{
        margin-right: 0.4rem;
    }}
    .error-text {{
        color: var(--red);
        font-size: 0.8rem;
        margin-left: 0.4rem;
    }}
    .warn-text {{
        color: var(--yellow);
        font-size: 0.8rem;
        margin-left: 0.4rem;
    }}
    .meta-inline {{
        color: var(--text-dim);
        font-size: 0.75rem;
        margin-left: 0.6rem;
    }}
    .frames-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
        margin-top: 0.5rem;
    }}
    .frame-card {{
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.4rem;
        text-align: center;
    }}
    .frame-label {{
        color: var(--text-dim);
        font-size: 0.7rem;
        margin-bottom: 0.3rem;
    }}
    pre.frame-art {{
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', 'JetBrains Mono', 'Menlo', monospace;
        font-size: 10px;
        line-height: 1.2;
        color: var(--text);
        background: transparent;
        border: none;
        margin: 0;
        padding: 0.3rem;
        text-align: left;
        display: inline-block;
        white-space: pre;
        overflow-x: auto;
    }}
    pre.raw-response {{
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
        font-size: 0.75rem;
        color: var(--yellow);
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.5rem;
        white-space: pre-wrap;
        word-break: break-all;
        margin-top: 0.3rem;
    }}
    .badge {{
        display: inline-block;
        padding: 0.1rem 0.4rem;
        border-radius: 3px;
        font-size: 0.7rem;
        color: #000;
        font-weight: bold;
    }}
    .prompt-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
    }}
    .prompt-table th {{
        background: var(--surface2);
        color: var(--blue);
        padding: 0.4rem 0.6rem;
        text-align: left;
        font-size: 0.8rem;
        border-bottom: 1px solid var(--border);
    }}
    .prompt-table td {{
        padding: 0.4rem 0.6rem;
        font-size: 0.85rem;
        border-bottom: 1px solid var(--border);
    }}
    .toc {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
    }}
    .toc a {{
        color: var(--blue);
        text-decoration: none;
        margin-right: 1rem;
        display: inline-block;
        margin-bottom: 0.3rem;
    }}
    .toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<h1>Ollama ASCII Art Model Test Report</h1>
<div class="timestamp">Generated: {now}</div>

<div class="info-card">
    <h3>Test Configuration</h3>
    {config_summary}
</div>

<div class="info-card">
    <h3>Prompts</h3>
    <table class="prompt-table">
        <thead><tr><th>Name</th><th>Prompt</th></tr></thead>
        <tbody>{prompt_rows}</tbody>
    </table>
</div>

<div class="toc">
    <h3>Models</h3>
    {toc_links}
</div>

{model_sections}

</body>
</html>
"""

    output_path.write_text(report_html, encoding="utf-8")
    logger.info("Report written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Test Ollama models for tpet ASCII art generation")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Config file path (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output HTML report path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=_DEFAULT_RESULTS,
        help=f"Incremental results file for resume support (default: {_DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any saved results and start fresh",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip testing — regenerate HTML report from existing results JSON",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.report_only:
        results, model_sizes, completed_models = _load_partial_results(args.results)
        if not results:
            logger.error("No results found at %s — nothing to report", args.results)
            raise SystemExit(1)
        config.results = results
        config.model_sizes = model_sizes
        logger.info("Loaded %d results for %d models from %s", len(results), len(completed_models), args.results)
        generate_html_report(config, args.output)
        raise SystemExit(0)

    if args.fresh and args.results.exists():
        args.results.unlink()
        logger.info("Cleared saved results at %s", args.results)

    logger.info(
        "Loaded config: %d models, %d temperatures, %d prompts = %d total tests",
        len(config.models),
        len(config.temperatures),
        len(config.prompts),
        len(config.models) * len(config.temperatures) * len(config.prompts) * config.samples_per_combo,
    )

    run_all_tests(config, args.results)
    generate_html_report(config, args.output)

    errors = sum(1 for r in config.results if r.error)
    validated = sum(1 for r in config.results if r.validation and r.validation.ok)
    total_val = sum(1 for r in config.results if r.validation)
    logger.info(
        "Done. %d tests, %d errors, %d/%d passed validation.",
        len(config.results),
        errors,
        validated,
        total_val,
    )


if __name__ == "__main__":
    main()
