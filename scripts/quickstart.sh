#!/usr/bin/env bash
#
# CARL quickstart — pasteable one-liner.
# Usage from inside any git repo:
#
#   curl -sSL https://raw.githubusercontent.com/anni-stanford/carl/main/scripts/quickstart.sh | bash
#
# What it does:
#   1. Detects environment (Python 3.11+, pip, git, Docker, ANTHROPIC_API_KEY).
#   2. Installs carl-loop from GitHub (PyPI publish coming May 28).
#   3. If Docker daemon is up AND ANTHROPIC_API_KEY is set:
#        - clones the carl repo into a temp dir
#        - builds the episode image (one-time, ~3 min)
#        - runs `carl auto` against the *current directory* with a real
#          Claude Code agent. CARL evolves your CLAUDE.md and .claude/skills/
#          based on CI feedback, then writes CARL_REPORT.md.
#   4. Otherwise runs `carl auto --dry-run` so you can see the full
#      pipeline shape and a real CARL_REPORT.md without any external
#      dependency.
#
# All the script does is automate exactly what the README's quickstart
# documents. Read it before piping it to bash.

set -euo pipefail

bold()   { printf '\033[1m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
red()    { printf '\033[31m%s\033[0m\n' "$*" >&2; }

bold "==> CARL quickstart"

# ---- 1. Pre-flight ----------------------------------------------------------

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    red "error: python3 not found on PATH. Install Python 3.11+ and retry."
    exit 2
fi

PY_VER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_OK=$("$PYTHON" -c 'import sys; print("1" if sys.version_info >= (3, 11) else "0")')
if [[ "$PY_OK" != "1" ]]; then
    red "error: CARL needs Python >= 3.11; you have $PY_VER."
    exit 2
fi
green "[carl] python $PY_VER ✓"

if ! command -v pip >/dev/null 2>&1 && ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    red "error: pip not found. Install pip and retry."
    exit 2
fi

REPO_ROOT="$(pwd)"
if ! git -C "$REPO_ROOT" rev-parse --show-toplevel >/dev/null 2>&1; then
    yellow "[carl] warning: $REPO_ROOT is not a git repo. carl auto works best inside a Python repo with a test suite."
fi

# ---- 2. Install carl-loop from GitHub --------------------------------------

bold "==> installing carl-loop"
"$PYTHON" -m pip install --upgrade --quiet pip
"$PYTHON" -m pip install --quiet "git+https://github.com/anni-stanford/carl.git"
green "[carl] carl-loop installed ✓"

# ---- 3. Decide mode ---------------------------------------------------------

DRY_RUN_REASON=""
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    DRY_RUN_REASON="ANTHROPIC_API_KEY is not set"
elif ! command -v docker >/dev/null 2>&1; then
    DRY_RUN_REASON="docker CLI not found on PATH"
elif ! docker info >/dev/null 2>&1; then
    DRY_RUN_REASON="docker daemon is not running"
fi

if [[ -n "$DRY_RUN_REASON" ]]; then
    yellow "[carl] $DRY_RUN_REASON — running carl auto in --dry-run mode."
    yellow "[carl] for a real run: set ANTHROPIC_API_KEY, start Docker, then re-run this script."
    bold "==> running carl auto --dry-run against $REPO_ROOT"
    cd "$REPO_ROOT"
    carl auto --dry-run --probe-n 10 --episodes 12
    bold "==> done. open CARL_REPORT.md to see the report."
    exit 0
fi

green "[carl] docker daemon ✓"
green "[carl] ANTHROPIC_API_KEY is set ✓"

# ---- 4. Build the episode image (one-time) ---------------------------------

IMAGE="carl/episode-claude:latest"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    green "[carl] episode image $IMAGE already present ✓"
else
    bold "==> building $IMAGE (one-time, ~3 min)"
    TMP_CARL="$(mktemp -d -t carl-build.XXXXXX)"
    git clone --depth 1 --quiet https://github.com/anni-stanford/carl.git "$TMP_CARL"
    docker build -q -t "$IMAGE" -f "$TMP_CARL/docker/Dockerfile.episode.claude" "$TMP_CARL" >/dev/null
    rm -rf "$TMP_CARL"
    green "[carl] $IMAGE built ✓"
fi

# ---- 5. Real run -----------------------------------------------------------

bold "==> running carl auto against $REPO_ROOT (real Claude Code episodes)"
cd "$REPO_ROOT"
carl auto --probe-n 10 --episodes 20

bold "==> done. open CARL_REPORT.md for the full breakdown."
