#!/bin/bash -l
# Submit joint NUTS run for all five megamaser galaxies to optgpu.

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash $0"
    echo ""
    echo "Submits joint NUTS run for all 5 MCP galaxies to optgpu."
    echo "No options — all parameters are hardcoded."
    exit 0
fi

ROOT="/mnt/users/rstiskalek/CANDEL"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

addqueue -q optgpu -s -m 16 --gpus 1 \
  $PYTHON -u $ROOT/scripts/megamaser/run_maser_disk.py joint \
  --sampler nuts --num-warmup 2000 --num-samples 2000
