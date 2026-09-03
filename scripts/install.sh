#!/usr/bin/env bash
# CompanionAI installer.
#
# Detects the board, creates a virtual environment and installs the right
# wheels for it.  Run it again at any time; it is safe to repeat.
#
#   ./scripts/install.sh              # GUI + voice, image generation if it fits
#   ./scripts/install.sh --minimal    # GUI + voice only
#   ./scripts/install.sh --with-image # force the image-generation extra
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${COMPANIONAI_VENV:-$ROOT/.venv}"
WANT_IMAGE=auto

for arg in "$@"; do
  case "$arg" in
    --minimal)    WANT_IMAGE=no ;;
    --with-image) WANT_IMAGE=yes ;;
    -h|--help)    sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- detect board
BOARD="pc"
MODEL=""
[ -r /proc/device-tree/model ] && MODEL="$(tr -d '\0' < /proc/device-tree/model)"
if [ -f /etc/nv_tegra_release ] || echo "$MODEL" | grep -qi 'jetson\|tegra'; then
  BOARD="jetson"
elif echo "$MODEL" | grep -qi 'raspberry pi'; then
  BOARD="rpi"
fi

HAS_CUDA=no
if command -v nvidia-smi >/dev/null 2>&1 || [ "$BOARD" = "jetson" ]; then
  HAS_CUDA=yes
fi

echo "==> Board:  $BOARD ${MODEL:+($MODEL)}"
echo "==> CUDA:   $HAS_CUDA"
echo "==> Target: $VENV"

# --------------------------------------------------------- system dependencies
install_system_packages() {
  local packages=(python3-venv python3-dev build-essential cmake pkg-config
                  libportaudio2 portaudio19-dev libsndfile1 espeak-ng git)
  if command -v apt-get >/dev/null 2>&1; then
    echo "==> Installing system packages (sudo may prompt) ..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends "${packages[@]}"
  else
    echo "!! Not a Debian/Ubuntu system.  Install these yourself if they are missing:"
    printf '   %s\n' "${packages[@]}"
  fi
}
install_system_packages

# ------------------------------------------------------------------- virtualenv
VENV_ARGS=()
if [ "$BOARD" = "jetson" ]; then
  # JetPack ships CUDA-enabled torch and cv2 as system packages; inherit them
  # instead of downloading multi-gigabyte wheels that would not use the GPU.
  VENV_ARGS+=(--system-site-packages)
fi
[ -d "$VENV" ] || python3 -m venv "${VENV_ARGS[@]}" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ------------------------------------------------------------------------ GUI
echo "==> Installing the interface ..."
pip install -r "$ROOT/requirements.txt"

# ------------------------------------------------------- llama.cpp with the GPU
echo "==> Installing speech and language runtimes ..."
if [ "$HAS_CUDA" = "yes" ]; then
  echo "    building llama-cpp-python with CUDA offload (this takes a while) ..."
  CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --force-reinstall --no-binary :all: \
    llama-cpp-python || {
      echo "    CUDA build failed; falling back to the CPU wheel."
      pip install llama-cpp-python
    }
else
  pip install llama-cpp-python
fi

# The rest of the voice stack is pure wheels everywhere.
grep -v '^llama-cpp-python' "$ROOT/requirements-voice.txt" > /tmp/companionai-voice.txt
pip install -r /tmp/companionai-voice.txt
rm -f /tmp/companionai-voice.txt

# -------------------------------------------------------------- image generation
if [ "$WANT_IMAGE" = "auto" ]; then
  case "$BOARD" in
    jetson) WANT_IMAGE=yes ;;
    rpi)    WANT_IMAGE=no  ;;   # it works, but a render takes minutes
    *)      WANT_IMAGE=yes ;;
  esac
fi

if [ "$WANT_IMAGE" = "yes" ]; then
  echo "==> Installing image generation ..."
  if [ "$BOARD" = "jetson" ]; then
    if python -c 'import torch' 2>/dev/null; then
      echo "    using JetPack's torch"
      pip install diffusers transformers accelerate safetensors pillow
    else
      echo "!!  torch was not found.  Install NVIDIA's JetPack torch wheel first:"
      echo "!!    https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/"
      echo "!!  then re-run: ./scripts/install.sh --with-image"
    fi
  else
    pip install -r "$ROOT/requirements-image.txt"
  fi
else
  echo "==> Skipping image generation (add it later with --with-image)."
fi

# --------------------------------------------------------------------- finish
pip install -e "$ROOT" --no-deps

cat <<EOF

Done.

  Start it:   $VENV/bin/companionai
  Check it:   $VENV/bin/companionai --doctor

CompanionAI starts offline.  To download models, open the Settings tab, approve
network access, then install what you want from the Models tab.
EOF
