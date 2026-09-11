#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_DIR"
DRY_RUN=false
for arg in "$@"; do [[ "$arg" != "--dry-run" ]] || DRY_RUN=true; done
if ! $DRY_RUN && [[ ! -t 0 || ! -t 1 ]]; then
  echo "Real robot client requires an interactive TTY; hardware was not started." >&2
  exit 2
fi
exec "${PYTHON_BIN:-python3}" integration/client_entry.py   --config "${CLIENT_CONFIG:-${REPO_DIR}/integration/client_lingbot_history.yaml}" "$@"
