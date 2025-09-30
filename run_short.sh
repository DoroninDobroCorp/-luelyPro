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
  echo "[run_short.sh] Creating virtual env in $VENV_DIR ..."
  "$PY_BIN" -m venv "$VENV_DIR"
fi

# 2) Ensure pip is available in venv, then upgrade basics
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  echo "[run_short.sh] pip not found in venv, trying to bootstrap via ensurepip..."
  if "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1; then
    echo "[run_short.sh] ensurepip succeeded"
  else
    echo "[run_short.sh] ensurepip unavailable, attempting 'python -m venv --upgrade-deps' ..."
    if "$PY_BIN" -m venv --upgrade-deps "$VENV_DIR" >/dev/null 2>&1; then
      echo "[run_short.sh] venv upgraded with pip"
    else
      echo "[run_short.sh] fallback: downloading get-pip.py"
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
  echo "[run_short.sh] Installing deps via uv sync ..."
  uv sync --frozen || uv sync
else
  if [[ -f "requirements.txt" ]]; then
    echo "[run_short.sh] Installing deps via pip ..."
    "$VENV_PIP" install -r requirements.txt
  fi
fi

# 4) Режим с одним повторением: отключаем автоповтор тезисов
echo "[run_short.sh] Starting LIVE assistant (ONE-TIME mode, no thesis repeat)..."

# Устанавливаем очень большой интервал повтора, чтобы фактически отключить автоповтор
export THESIS_REPEAT_SEC=999999

# Запускаем main.py
exec "$VENV_PY" main.py "$@"
