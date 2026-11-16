#!/bin/bash
# Source from the other project scripts; no shell profile changes.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$PROJECT_DIR/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.tools/python"
export HF_HOME="$PROJECT_DIR/models/hf-cache"
export XDG_CACHE_HOME="$PROJECT_DIR/.tools/cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH="$PROJECT_DIR/.venv/bin:$PROJECT_DIR/.tools/uv-aarch64-apple-darwin:$PATH"
