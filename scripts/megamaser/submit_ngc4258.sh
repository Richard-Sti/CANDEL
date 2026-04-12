#!/bin/bash -l
# Submit NGC4258 maser disk jobs to GPU queue.
#
# Usage:
#   bash scripts/megamaser/submit_ngc4258.sh map              # DE optimizer only
#   bash scripts/megamaser/submit_ngc4258.sh nuts              # NUTS sampling
#   bash scripts/megamaser/submit_ngc4258.sh nss               # nested sampling
#   bash scripts/megamaser/submit_ngc4258.sh nuts -q cmbgpu    # different queue

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

MODE="${1:?Usage: $0 <map|nuts|nss> [-q QUEUE]}"
shift

QUEUE="gpulong"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q) QUEUE="$2"; shift 2 ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$MODE" in
    map)
        echo "Submitting NGC4258 DE MAP -> $QUEUE"
        addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
            $PYTHON -u "$ROOT/scripts/megamaser/run_maser_disk.py" NGC4258 \
            --sampler nuts --map-only
        ;;
    nuts)
        echo "Submitting NGC4258 NUTS -> $QUEUE"
        addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
            $PYTHON -u "$ROOT/scripts/megamaser/run_maser_disk.py" NGC4258 \
            --sampler nuts --num-warmup 2000 --num-samples 2000
        ;;
    nss)
        echo "Submitting NGC4258 NSS -> $QUEUE"
        addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
            $PYTHON -u "$ROOT/scripts/megamaser/run_maser_disk.py" NGC4258 \
            --sampler nss --D-c-prior uniform
        ;;
    *)
        echo "Unknown mode '$MODE'. Use: map, nuts, nss"
        exit 1
        ;;
esac
