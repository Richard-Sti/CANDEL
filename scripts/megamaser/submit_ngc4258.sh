#!/bin/bash -l
# Submit NGC4258 NUTS job to GPU queue.
#
# NGC4258 uses Mode 1 (--sample-r) with per-spot adaptive phi.
# NUTS is the only supported sampler (NSS can't handle 358 r_ang params,
# DE optimizer doesn't support Mode 1).
#
# Usage:
#   bash scripts/megamaser/submit_ngc4258.sh                   # defaults
#   bash scripts/megamaser/submit_ngc4258.sh -q cmbgpu         # different queue
#   bash scripts/megamaser/submit_ngc4258.sh --warmup 5000     # more warmup
#   bash scripts/megamaser/submit_ngc4258.sh --init sobol_adam  # Sobol+Adam init
#   bash scripts/megamaser/submit_ngc4258.sh --init-r-only     # config globals + r_ang golden-section, no sampling

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

QUEUE="optgpu"
WARMUP=2000
SAMPLES=2000
INIT=""
INIT_R_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q) QUEUE="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --init) INIT="$2"; shift 2 ;;
        --init-r-only) INIT_R_ONLY=true; shift ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

EXTRA_ARGS=""
if [[ -n "$INIT" ]]; then
    EXTRA_ARGS="--init-method $INIT"
fi

if [[ "$INIT_R_ONLY" == true ]]; then
    echo "Submitting NGC4258 init-r-only (config globals + r_ang optimisation) -> $QUEUE"
    addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
        $PYTHON -u "$ROOT/scripts/megamaser/run_maser_disk.py" NGC4258 \
        --sampler nuts --sample-r --init-r-only
else
    echo "Submitting NGC4258 NUTS (Mode 1, $WARMUP warmup + $SAMPLES samples, init=${INIT:-config}) -> $QUEUE"
    addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
        $PYTHON -u "$ROOT/scripts/megamaser/run_maser_disk.py" NGC4258 \
        --sampler nuts --sample-r \
        --num-warmup "$WARMUP" --num-samples "$SAMPLES" $EXTRA_ARGS
fi
