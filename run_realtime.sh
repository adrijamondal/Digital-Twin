#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ -d ".venv" ]; then
  PYTHON="$DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

echo "Starting Digital Twin OS Realtime Backend..."
$PYTHON backend/app.py
