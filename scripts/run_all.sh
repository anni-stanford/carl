#!/usr/bin/env bash
set -euo pipefail
# CARL — one-command experiment driver.
# Usage: bash scripts/run_all.sh --experiment e1 --adapter claude_code --repo fastapi
EXPERIMENT="${1:-e1}"
echo "[carl] run_all driver — experiments wired Days 13–14 of the build sequence"
echo "[carl] requested experiment: ${EXPERIMENT}"
