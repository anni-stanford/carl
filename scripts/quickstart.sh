#!/usr/bin/env bash
#
# CARL quickstart — one pasteable command, no Docker, no pip headaches.
# Run from inside the git repo you want CARL to improve:
#
#   curl -sSL https://raw.githubusercontent.com/anni-stanford/carl/main/scripts/quickstart.sh | bash
#
# Why this exists: `pip install` on a Homebrew / system Python now fails with
# "externally-managed-environment" (PEP 668), and `--user` installs land off
# PATH. To spare every user that friction, this script creates ONE isolated
# virtual environment at ~/.carl/venv, installs CARL into it, and runs CARL
# from it. Your system/Homebrew Python is never touched and nothing is
# installed on your PATH.
#
# It only runs python -m venv, pip (inside the venv), and `carl`. Read it
# before piping to bash.

set -euo pipefail

green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*" >&2; }

CARL_HOME="${CARL_HOME:-$HOME/.carl}"
VENV="$CARL_HOME/venv"
INSTALL_SPEC="${CARL_INSTALL_SPEC:-git+https://github.com/anni-stanford/carl.git}"

# 1) Find a Python >= 3.11 without assuming which name it has.
pick_python() {
    for cand in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
                echo "$cand"
                return 0
            fi
        fi
    done
    return 1
}

PYBIN="$(pick_python || true)"
if [[ -z "$PYBIN" ]]; then
    red "error: no Python 3.11+ found."
    red "Install one and re-run:  brew install python@3.12   (macOS)"
    exit 2
fi
green "[carl] using $("$PYBIN" -V 2>&1) at $(command -v "$PYBIN")"

# 2) Create the isolated venv once (reused on later runs).
if [[ ! -x "$VENV/bin/python" ]]; then
    green "[carl] creating isolated environment at $VENV"
    "$PYBIN" -m venv "$VENV"
else
    green "[carl] reusing environment at $VENV"
fi

# 3) Install / upgrade CARL inside the venv (no PEP 668 issue, no PATH issue).
green "[carl] installing carl-loop (this can take a minute)"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet --upgrade "$INSTALL_SPEC"

# 4) Tell the user what mode they'll get.
if command -v claude >/dev/null 2>&1 || [[ -n "${CARL_CLAUDE_BIN:-}" ]]; then
    green "[carl] Claude Code CLI detected — real local episodes (no Docker)"
else
    yellow "[carl] No Claude Code CLI found — CARL will run a synthetic --dry-run."
    yellow "[carl] Install Claude Code and re-run for a real before/after."
fi

# 5) Run CARL against the current directory. Pass through any extra args.
green "[carl] running: carl auto  (in $(pwd))"
exec "$VENV/bin/python" -m carl auto "$@"
