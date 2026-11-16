#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR"
trap 'echo "Setup stopped. Fix the reported error and rerun ./scripts/setup.sh; completed downloads are reusable." >&2' ERR

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This setup targets Apple Silicon macOS.' >&2
  exit 1
fi
echo 'Checking network access...'
for url in https://github.com https://huggingface.co https://pypi.org; do
  curl --fail --silent --show-error --location --head --connect-timeout 10 --max-time 30 "$url" >/dev/null
done
xcode-select -p >/dev/null
mkdir -p .tools vendor models runtime

# Install uv inside the project, without running a remote installer or changing profiles.
if [[ ! -x .tools/uv-aarch64-apple-darwin/uv ]]; then
  curl --fail --location --retry 3 --output .tools/uv.tar.gz \
    https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz
  tar -xzf .tools/uv.tar.gz -C .tools
fi
uv python install --no-bin 3.11
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi

if [[ ! -d vendor/TinyAgent/.git ]]; then
  git clone https://github.com/SqueezeAILab/TinyAgent.git vendor/TinyAgent
fi
if [[ ! -d vendor/llama.cpp/.git ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
fi
for compatibility_patch in "$PROJECT_DIR"/patches/*.patch; do
  if git -C vendor/TinyAgent apply --reverse --check "$compatibility_patch" 2>/dev/null; then
    echo "Compatibility patch already applied: $(basename "$compatibility_patch")"
  else
    git -C vendor/TinyAgent apply "$compatibility_patch"
  fi
done
git -C vendor/TinyAgent rev-parse HEAD >runtime/tinyagent-commit.txt
git -C vendor/llama.cpp rev-parse HEAD >runtime/llama-cpp-commit.txt

# NumPy 1.x is required by the upstream Torch 2.2.2 wheels.
uv pip install --python .venv/bin/python -r vendor/TinyAgent/requirements.txt \
  'numpy==1.26.4' 'huggingface-hub==0.23.5' 'cmake>=3.31,<4'
uv pip check --python .venv/bin/python
uv pip freeze --python .venv/bin/python >runtime/requirements-resolved.txt
python scripts/create-demo-files.py

python scripts/download_models.py

# Embedded shaders can compile at runtime with the Command Line Tools SDK.
# This Mac also has an SDK 27 installation incompatible with its current linker.
sdk_path="$(xcrun --show-sdk-path)"
if [[ -d /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk ]]; then
  sdk_path=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
fi
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DCMAKE_OSX_SYSROOT="$sdk_path" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build vendor/llama.cpp/build --config Release --target llama-server -j 8
python demo.py --check
echo 'Setup complete. Start a model with ./scripts/serve.sh 7b (or 1.1b / 1.1b-f16).'
