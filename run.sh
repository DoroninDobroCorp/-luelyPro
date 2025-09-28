#!/usr/bin/env bash
set -euo pipefail

# Project root is this script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PY_BIN="python3"
VENV_DIR=".venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# 1) Create venv if missing or broken
if [[ ! -x "$VENV_PY" ]]; then
  echo "[run.sh] Creating virtual env in $VENV_DIR ..."
  "$PY_BIN" -m venv "$VENV_DIR"
fi

# 2) Ensure pip is available in venv, then upgrade basics
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  echo "[run.sh] pip not found in venv, trying to bootstrap via ensurepip..."
  if "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1; then
    echo "[run.sh] ensurepip succeeded"
  else
    echo "[run.sh] ensurepip unavailable, attempting 'python -m venv --upgrade-deps' ..."
    if "$PY_BIN" -m venv --upgrade-deps "$VENV_DIR" >/dev/null 2>&1; then
      echo "[run.sh] venv upgraded with pip"
    else
      echo "[run.sh] fallback: downloading get-pip.py"
      GETPIP="$(mktemp)"
      trap 'rm -f "$GETPIP"' EXIT
      "$VENV_PY" - <<'PY'
import sys, urllib.request, os, tempfile, runpy
url = 'https://bootstrap.pypa.io/get-pip.py'
fn = tempfile.mkstemp(suffix='get-pip.py')[1]
with urllib.request.urlopen(url) as r, open(fn, 'wb') as f:
    f.write(r.read())
runpy.run_path(fn, run_name='__main__')
os.remove(fn)
PY
    fi
  fi
fi

"$VENV_PY" -m pip install --upgrade pip setuptools wheel >/dev/null

# 3) Install deps
# Prefer uv if installed and pyproject exists
if command -v uv >/dev/null 2>&1 && [[ -f "pyproject.toml" ]]; then
  echo "[run.sh] Installing deps via uv sync ..."
  uv sync --frozen || uv sync
else
  if [[ -f "requirements.txt" ]]; then
    echo "[run.sh] Installing deps via pip ..."
    "$VENV_PIP" install -r requirements.txt
  fi
fi

# 4) Decide what to run
# Default: LIVE assistant with interactive profile (runs main.py with no subcommand)
if [[ $# -eq 0 ]]; then
  echo "[run.sh] Starting LIVE assistant (interactive profile selection)..."
  exec "$VENV_PY" main.py
fi

# If running live and no timer provided, enforce a 10s auto-stop to avoid hangs
if [[ "$1" == "live" ]]; then
  has_timer_env=${RUN_SECONDS:-}
  if [[ -z "$has_timer_env" ]]; then
    export RUN_SECONDS=10
  fi
fi

# 5) Run main with passed arguments
exec "$VENV_PY" main.py "$@"
