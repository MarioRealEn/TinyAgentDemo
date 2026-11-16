#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
precision=Q4_K_M
case "${1:-7b}" in
  7b|7B) size=7B; alias=tinyagent-7b ;;
  1.1b|1.1B) size=1.1B; alias=tinyagent-1.1b ;;
  1.1b-f16|1.1B-f16) size=1.1B; precision=F16; alias=tinyagent-1.1b-f16 ;;
  *) echo 'Usage: ./scripts/serve.sh [7b|1.1b|1.1b-f16]' >&2; exit 2 ;;
esac
server="$PROJECT_DIR/vendor/llama.cpp/build/bin/llama-server"
model="$PROJECT_DIR/models/TinyAgent-$size-$precision.gguf"
if [[ ! -x "$server" || ! -f "$model" ]]; then
  echo 'Model or server missing. Run ./scripts/setup.sh first.' >&2
  exit 1
fi
exec "$server" --model "$model" --alias "$alias" \
  --host 127.0.0.1 --port 8080 --ctx-size 8192 --parallel 1 --n-gpu-layers 99
