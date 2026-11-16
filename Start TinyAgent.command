#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh
if [[ ! -x .venv/bin/python ]]; then
  echo 'Run ./scripts/setup.sh in this folder first.'
  read -r -p 'Press Enter to close.'
  exit 1
fi
exec .venv/bin/python ui_server.py --open
