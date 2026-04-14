#!/bin/bash -l
# Submit NGC4258 Mode 1 inference (sample r_ang, bruteforce phi marginal).
# Uses optgpu (A6000, 48 GB) by default. Per-type phi grids: 50k/30k/20k.
QUEUE=${1:-optgpu}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/venv_gpu_candel/bin/python"

addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT_DIR/scripts/megamaser/run_maser_disk.py" \
    NGC4258 --sampler nuts --sample-r "${@:2}"
