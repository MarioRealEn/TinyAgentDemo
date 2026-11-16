#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo 'Python environment missing. Run ./scripts/setup.sh first.' >&2
  exit 1
fi
exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/demo.py" "$@"
