#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

HOST="${FWBG_HOST:-0.0.0.0}"
PORT="${FWBG_PORT:-8420}"

exec "$PROJECT_ROOT/.venv/bin/fwbg" api --host "$HOST" --port "$PORT"
