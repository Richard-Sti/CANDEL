#!/bin/bash -l
# Submit toy joint H0 inference to GPU queue.
#
# Usage:
#   bash scripts/megamaser/submit_toy_H0.sh                # default: optgpu
#   bash scripts/megamaser/submit_toy_H0.sh -q gpulong     # different queue

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

QUEUE="gpulong"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q) QUEUE="$2"; shift 2 ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Submitting toy joint H0 -> $QUEUE"
addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT/scripts/megamaser/toy_joint_H0.py" \
    --num-warmup 1000 --num-samples 4000 --num-chains 4
