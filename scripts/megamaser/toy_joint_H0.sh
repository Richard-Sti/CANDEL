#!/bin/bash -l
# Submit toy joint H0 inference to GPU queue.
#
# Usage:
#   bash scripts/megamaser/toy_joint_H0.sh                # default: gpulong
#   bash scripts/megamaser/toy_joint_H0.sh -q optgpu       # different queue

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

QUEUE="gpulong"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q) QUEUE="$2"; shift 2 ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NUM_WARMUP=1000
NUM_SAMPLES=4000
NUM_CHAINS=8
CHAIN_METHOD="vectorized"

echo "Submitting toy joint H0 -> $QUEUE"
echo "  warmup=$NUM_WARMUP, samples=$NUM_SAMPLES, chains=$NUM_CHAINS, method=$CHAIN_METHOD"
addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT/scripts/megamaser/toy_joint_H0.py" \
    --num-warmup $NUM_WARMUP --num-samples $NUM_SAMPLES --num-chains $NUM_CHAINS
