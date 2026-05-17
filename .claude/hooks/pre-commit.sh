#!/usr/bin/env bash
set -euo pipefail
ruff check .
mypy carl
pytest -q --strict-markers
