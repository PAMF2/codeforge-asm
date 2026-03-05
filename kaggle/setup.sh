#!/usr/bin/env bash
# Run this at the start of a Kaggle TPU session (in a notebook cell with %%bash or !bash kaggle/setup.sh)

set -e

echo "[setup] Installing NASM and linker..."
apt-get update -qq && apt-get install -y -qq nasm binutils
echo "[setup] nasm: $(nasm --version)"
echo "[setup] ld:   $(ld --version | head -1)"

echo "[setup] Installing Python dependencies..."
pip install -q -r /kaggle/working/codeforge-asm/kaggle/requirements_tpu.txt

echo "[setup] Verifying torch_xla..."
python -c "import torch_xla.core.xla_model as xm; print('[setup] XLA device:', xm.xla_device())"

echo "[setup] Done."
