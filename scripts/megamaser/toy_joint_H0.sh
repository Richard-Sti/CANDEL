#!/bin/bash -l
# Submit toy joint H0 inference to GPU queue.
#
# Usage:
#   bash scripts/megamaser/toy_joint_H0.sh                  # volumetric D^2 prior
#   bash scripts/megamaser/toy_joint_H0.sh --flat-dist       # flat D prior
#   bash scripts/megamaser/toy_joint_H0.sh -q optgpu         # different queue

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

QUEUE="gpulong"
FLAT_DIST=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [--flat-dist]"
            echo ""
            echo "Options:"
            echo "  -q QUEUE      GPU queue (default: gpulong)"
            echo "  --flat-dist   Use flat D prior instead of volumetric D^2"
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        --flat-dist) FLAT_DIST="--flat-dist"; shift ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NUM_WARMUP=1000
NUM_SAMPLES=4000
NUM_CHAINS=8

echo "Submitting toy joint H0 -> $QUEUE"
echo "  warmup=$NUM_WARMUP, samples=$NUM_SAMPLES, chains=$NUM_CHAINS"
[[ -n "$FLAT_DIST" ]] && echo "  Using FLAT distance prior"
addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT/scripts/megamaser/toy_joint_H0.py" \
    --num-warmup $NUM_WARMUP --num-samples $NUM_SAMPLES --num-chains $NUM_CHAINS \
    $FLAT_DIST
