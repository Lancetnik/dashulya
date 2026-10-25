set shell := ["sh", "-cu"]
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# List available commands.
default:
    @just --list

_lint *args:
    uv run --locked --no-dev --group lint {{args}}

# Include new project files while respecting Git's ignore rules.
_hook hook:
    git ls-files --cached --others --exclude-standard -z | xargs -0 uv run --locked --no-dev --group lint pre-commit run {{hook}} --files

# Check file hygiene, syntax, merge markers, case conflicts and file sizes.
hygiene:
    just _hook trailing-whitespace
    just _hook end-of-file-fixer
    just _hook mixed-line-ending
    just _hook check-case-conflict
    just _hook check-merge-conflict
    just _hook check-toml
    just _hook check-yaml
    just _hook check-json
    just _hook check-added-large-files

# Scan project files for credentials without verifying them over the network.
secrets:
    just _hook detect-secrets # pragma: allowlist secret -- Hook command.

# Check spelling in source, documentation, prompts and configuration.
typos:
    just _lint typos --hidden .

# Check formatting and all enabled Ruff rules without modifying files.
lint:
    just _lint ruff format --check
    just _lint ruff check

# Apply safe Ruff fixes and format Python files.
format:
    just _lint ruff check --fix
    just _lint ruff format

# Type-check source and tests in strict mode.
mypy:
    just _lint mypy

# Check the architecture contracts.
import-linter:
    just _lint lint-imports --config .importlinter

# Check that the lockfile matches project metadata.
lock-check:
    uv lock --check

# Run tests without contacting Telegram or OpenAI.
test:
    uv run --locked python -m unittest discover -s tests -v

# Run all development checks.
check: lock-check hygiene secrets typos lint mypy import-linter test
