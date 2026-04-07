# Contributing to tpet

Thank you for your interest in contributing to tpet. This guide covers how to set up a development environment, run checks, and submit changes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development Setup](#development-setup)
- [Running Checks](#running-checks)
- [Project Structure](#project-structure)
- [Code Conventions](#code-conventions)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Prerequisites

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/) package manager
- Claude Code installed and authenticated (required to run tpet)
- Git

## Development Setup

1. Fork the repository and clone your fork:

    ```bash
    git clone https://github.com/<your-username>/tpet.git
    cd tpet
    ```

2. Create the virtual environment and install all dependencies (including dev):

    ```bash
    uv sync
    ```

3. Verify the setup by running the full check suite:

    ```bash
    make checkall
    ```

    All checks should pass on a clean clone.

## Running Checks

| Command | What it does |
|---------|-------------|
| `make checkall` | Run format, lint, typecheck, and tests in sequence |
| `make fmt` | Format code with ruff |
| `make lint` | Lint with ruff (auto-fix enabled) |
| `make typecheck` | Type-check with pyright |
| `make test` | Run the test suite with pytest |
| `make build` | Build the distribution package |
| `make clean` | Remove `dist/`, `build/`, and cache directories |

Run `make checkall` before every commit. Do not submit a PR that fails any check.

### Running a single test

```bash
uv run pytest tests/test_parser.py::TestParseJsonlLine::test_parse_user_message -v
```

### Running tpet in debug mode

```bash
uv run tpet --debug
```

Debug output is written to `$XDG_CONFIG_HOME/tpet/debug.log` — never to the terminal.

## Project Structure

```
src/tpet/
├── __init__.py
├── __main__.py     # Python -m entry point
├── py.typed        # PEP 561 marker
├── cli.py          # Typer CLI — all flags and dispatch
├── app.py          # Main event loop (Rich Live, 4fps)
├── config.py       # TpetConfig (Pydantic + YAML)
├── animation/      # PetAnimator state machine (engine.py)
├── art/            # Image generation, processing, and sixel rendering
├── commentary/     # Agent SDK comment generation (generator.py, prompts.py)
├── models/         # PetProfile, Rarity, StatConfig
├── monitor/        # Watchdog file watchers (watcher.py, text_watcher.py) and JSONL parser
├── profile/        # Pet generation via Agent SDK and YAML storage
└── renderer/       # Rich display, details card, stat bars, art preview, protocol
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture reference.

## Code Conventions

- Python 3.13 with full type annotations
- Built-in generics: `list`, `dict`, `tuple` (not `List`, `Dict`)
- Union operator `|` instead of `Optional` or `Union`
- `Annotated` for Typer CLI options
- No relative imports — always use full `tpet.*` import paths
- `pathlib.Path` for all file system operations
- `encoding="utf-8"` on every file open
- Google-style docstrings on all public functions and classes
- Line length: 120 characters (ruff enforced)
- TUI app — never write debug output to stdout or the terminal display

## Commit Messages

Use the conventional commit format:

```
<type>(<scope>): <subject>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

Examples:

```
feat(art): add sixel rendering pipeline
fix(commentary): prevent duplicate idle chatter on rapid events
docs(readme): add graphical art mode section
```

Keep the subject under 50 characters. Use the body to explain the why when needed.

## Pull Request Process

1. Create a branch from `main`:

    ```bash
    git checkout -b feat/your-feature
    ```

2. Make your changes, committing each logical unit atomically.

3. Run `make checkall` and fix any issues before pushing.

4. Open a pull request against `main`. Include:
    - A clear description of what changed and why
    - Any relevant issue references (`Closes #123`)
    - Notes on manual testing performed

5. All CI checks must pass before review.

6. PRs are merged with squash merge to keep the main branch history clean.
