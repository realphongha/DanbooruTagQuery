#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Build single-file executable of deploy.py
# Asks for runtime (onnxruntime / onnxruntime-gpu) and version.
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON=".venv/bin/python"
VENV_PIP=".venv/bin/pip"
VENV_PYINSTALLER=".venv/bin/pyinstaller"

# ---- check venv ----
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: venv not found at $VENV_PYTHON — create it with 'uv venv' first" >&2
    exit 1
fi

# ---- interactive runtime selection ----
echo "Select ONNX runtime:"
echo "  1) onnxruntime        (CPU only)"
echo "  2) onnxruntime-gpu     (GPU/CUDA) [default]"
read -r -p "Choice [1/2] (default=2): " RUNTIME_CHOICE
RUNTIME_CHOICE="${RUNTIME_CHOICE:-2}"

if [ "$RUNTIME_CHOICE" = "1" ]; then
    RUNTIME_PKG="onnxruntime"
    CUDA_TAG="_cpu"
else
    RUNTIME_PKG="onnxruntime-gpu"
    # CUDA tag determined after deps installed
    CUDA_TAG=""
fi

read -r -p "Version (default=1.18.0): " RUNTIME_VER
RUNTIME_VER="${RUNTIME_VER:-1.18.0}"

echo ""
echo "Installing ${RUNTIME_PKG}==${RUNTIME_VER} + numpy + Pillow + gradio ..."
"$VENV_PIP" install --quiet \
    "${RUNTIME_PKG}==${RUNTIME_VER}" \
    "numpy" \
    "Pillow" \
    "gradio" \
    "pyinstaller"

echo ""

# ---- extract version from pyproject.toml ----
VERSION=$(grep -E '^version\s*=' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
if [ -z "$VERSION" ]; then
    echo "ERROR: could not extract version from pyproject.toml" >&2
    exit 1
fi

# ---- detect OS & arch ----
OS=$(uname -s | sed 's/^Darwin$/Darwin/;s/^Linux$/Linux/;s/^MINGW.*\|^MSYS.*\|^CYGWIN.*/Windows/')
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  ARCH="x86_64" ;;
    amd64)   ARCH="x86_64" ;;
    aarch64) ARCH="arm64"  ;;
    arm64)   ARCH="arm64"  ;;
esac

# ---- detect CUDA version via torch (only if GPU runtime) ----
if [ "$RUNTIME_CHOICE" = "2" ]; then
    CUDA_TAG=$("$VENV_PYTHON" -c "
import torch; v=torch.version.cuda; print(f'_cuda{v}' if v else '_cpu')
" 2>/dev/null || echo "_cpu")
fi

NAME="DanbooruTagQuery_${VERSION}_${OS}_${ARCH}${CUDA_TAG}"

echo "Building: ${NAME}"
echo "  Version: ${VERSION}"
echo "  OS:      ${OS}"
echo "  Arch:    ${ARCH}"
if [ "$CUDA_TAG" = '_cpu' ]; then
    echo "  CUDA:    none (cpu)"
else
    echo "  CUDA:    ${CUDA_TAG#_cuda}"
fi

exec "$VENV_PYINSTALLER" --onefile \
    --name "${NAME}" \
    --distpath dist \
    --workpath build/pyinstaller \
    --specpath build/pyinstaller \
    "$@" \
    deploy.py
