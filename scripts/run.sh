#!/usr/bin/env bash
# Start CompanionAI from the virtual environment created by install.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${COMPANIONAI_VENV:-$ROOT/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "No virtual environment at $VENV.  Run ./scripts/install.sh first." >&2
  exit 1
fi
exec "$VENV/bin/python" -m companionai "$@"
