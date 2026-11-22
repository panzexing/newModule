#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${GPU_ID:?Set GPU_ID to the physical GPU index}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be a nonnegative integer" >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
mkdir -p outputs
exec flock -n outputs/.run.lock "${PYTHON:-python}" -u train.py --device cuda:0
