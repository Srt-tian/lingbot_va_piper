#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/wan_va${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
exec "${PYTHON_BIN:-python3}" integration/serve_official.py "$@"
