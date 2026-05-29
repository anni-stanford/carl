#!/usr/bin/env bash
#
# CARL quickstart — pasteable one-liner, no Docker required.
# Run from inside the git repo you want CARL to improve:
#
#   curl -sSL https://raw.githubusercontent.com/anni-stanford/carl/main/scripts/quickstart.sh | bash
#
# What it does:
#   1. Checks Python 3.11+.
#   2. Installs carl-loop from GitHub (PyPI publish pending).
#   3. Runs `python -m carl auto` against the current directory.
#      - In "auto" mode CARL uses the no-Docker local runner when the
#        Claude Code CLI (`claude`) is on PATH, so it runs real episodes
#        with your existing Claude Code authentication.
#      - If no Claude Code CLI is found, it falls back to a synthetic
#        --dry-run so you still get a CARL_REPORT.md to inspect.
#
# Read this script before piping it to bash. It only runs pip + carl.

set -euo pipefail

green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*" >&2; }

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    red "error: python3 not found. Install Python 3.11+ and retry."
    exit 2
fi
PY_OK=$("$PYTHON" -c 'import sys; print("1" if sys.version_info >= (3, 11) else "0")')
if [[ "$PY_OK" != "1" ]]; then
    red "error: CARL needs Python >= 3.11; you have $("$PYTHON" -V 2>&1)."
    red "Install a newer Python (brew install python@3.12) and re-run with PYTHON=python3.12."
    exit 2
fi
green "[carl] python ok"

green "[carl] installing carl-loop"
"$PYTHON" -m pip install --user --upgrade --quiet pip
"$PYTHON" -m pip install --user --quiet "git+https://github.com/anni-stanford/carl.git"

if command -v claude >/dev/null 2>&1 || [[ -n "${CARL_CLAUDE_BIN:-}" ]]; then
    green "[carl] Claude Code CLI detected — running real local episodes (no Docker)"
else
    yellow "[carl] No Claude Code CLI found — CARL will run a synthetic --dry-run."
    yellow "[carl] Install Claude Code and re-run for a real before/after."
fi

green "[carl] running: python -m carl auto"
"$PYTHON" -m carl auto

green "[carl] done — open CARL_REPORT.md for the before/after report."
