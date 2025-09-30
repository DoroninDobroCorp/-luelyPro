#!/usr/bin/env bash
set -euo pipefail

# Project root is this script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Enable commentary mode and faster thesis repetition
export COMMENTARY_MODE=${COMMENTARY_MODE:-1}
export THESIS_REPEAT_SEC=${THESIS_REPEAT_SEC:-8}
# Safety auto-stop to avoid hangs (can be overridden by env)
export RUN_SECONDS=${RUN_SECONDS:-10}

# Try to auto-pick the latest voice profile, otherwise ask interactively
LATEST_PROFILE=""
if ls profiles/*.npz >/dev/null 2>&1; then
  LATEST_PROFILE="$(ls -t profiles/*.npz | head -n1)"
fi

ARGS=(live --asr --asr-model tiny --asr-lang ru --thesis-autogen-disable)
if [[ -n "$LATEST_PROFILE" ]]; then
  echo "[run_commentary.sh] Using profile: $LATEST_PROFILE"
  ARGS+=(--profile "$LATEST_PROFILE")
else
  echo "[run_commentary.sh] No profiles found, enabling interactive selection"
  ARGS+=(--select-profile)
fi

# Delegate to the main runner in live mode without LLM
exec ./run.sh "${ARGS[@]}"
