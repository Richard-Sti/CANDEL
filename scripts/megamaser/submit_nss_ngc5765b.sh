#!/bin/bash -l
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

addqueue -q cmbgpu -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT/scripts/megamaser/run_maser_disk.py" NGC5765b \
    --sampler nss
