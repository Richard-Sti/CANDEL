#!/bin/bash -l
# Submit NGC4258 maser disk jobs to GPU queue.
#
# NGC4258 uses Mode 1 (--sample-r) with per-spot adaptive phi.
# The --sample-r flag sets marginalise_r=False so r_ang is sampled per spot.
# The adaptive_phi=true config enables the per-spot phi grid.
#
# Usage:
#   bash scripts/megamaser/submit_ngc4258.sh nuts              # NUTS sampling
#   bash scripts/megamaser/submit_ngc4258.sh nuts -q cmbgpu    # different queue
#   bash scripts/megamaser/submit_ngc4258.sh map               # DE optimizer only

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

MODE="${1:?Usage: $0 <nuts|map> [-q QUEUE] [--warmup N] [--samples N]}"
shift

QUEUE="gpulong"
WARMUP=2000
SAMPLES=2000
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q) QUEUE="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$MODE" in
    map)
        echo "Submitting NGC4258 DE MAP (Mode 1) -> $QUEUE"
        addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
            $PYTHON -u "$ROOT/scripts/megamaser/run_maser_disk.py" NGC4258 \
            --sampler nuts --sample-r --map-only
        ;;
    nuts)
        echo "Submitting NGC4258 NUTS (Mode 1, $WARMUP warmup + $SAMPLES samples) -> $QUEUE"
        addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
            $PYTHON -u "$ROOT/scripts/megamaser/run_maser_disk.py" NGC4258 \
            --sampler nuts --sample-r \
            --num-warmup "$WARMUP" --num-samples "$SAMPLES"
        ;;
    *)
        echo "Unknown mode '$MODE'. Use: nuts, map"
        exit 1
        ;;
esac
