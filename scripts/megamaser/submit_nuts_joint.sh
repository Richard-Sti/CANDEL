#!/bin/bash -l
# Submit joint NUTS run for all five megamaser galaxies to optgpu.
ROOT="/mnt/users/rstiskalek/CANDEL"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

addqueue -q optgpu -s -m 16 --gpus 1 \
  $PYTHON -u $ROOT/scripts/megamaser/run_maser_disk.py joint \
  --sampler nuts --num-warmup 2000 --num-samples 2000
