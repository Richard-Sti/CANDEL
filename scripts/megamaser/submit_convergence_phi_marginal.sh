#!/bin/bash -l
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/venv_gpu_candel/bin/python"

addqueue -q gpulong -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT_DIR/scripts/megamaser/convergence_phi_marginal.py" "$@"
