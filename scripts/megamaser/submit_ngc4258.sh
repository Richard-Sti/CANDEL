#!/bin/bash -l
# Submit NGC4258 NUTS job to GPU queue.
#
# NGC4258 defaults to Mode 1 with per-type bruteforce phi grids.
# NUTS is the only supported sampler (NSS can't handle 358 r_ang params,
# DE optimizer doesn't support Mode 1).
#
# Usage:
#   bash scripts/megamaser/submit_ngc4258.sh                   # defaults
#   bash scripts/megamaser/submit_ngc4258.sh -q cmbgpu         # different queue
#   bash scripts/megamaser/submit_ngc4258.sh --warmup 5000     # more warmup
#   bash scripts/megamaser/submit_ngc4258.sh --no-ecc             # disable eccentricity model
#   bash scripts/megamaser/submit_ngc4258.sh --no-quadratic-warp  # disable quadratic warp
#   bash scripts/megamaser/submit_ngc4258.sh --no-ecc --no-quadratic-warp  # circular + linear warp

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/venv_candel/bin/python"

QUEUE="optgpu"
WARMUP=2000
SAMPLES=2000
INIT="config"
MODE="mode1"
NO_ECC=false
NO_QW=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [--warmup N] [--samples N] [--init METHOD] [--mode MODE] [--no-ecc] [--no-quadratic-warp]"
            echo ""
            echo "Options:"
            echo "  -q QUEUE              GPU queue (default: optgpu)"
            echo "  --warmup N            Number of warmup iterations (default: 2000)"
            echo "  --samples N           Number of samples (default: 2000)"
            echo "  --init METHOD         Initialization method:"
            echo "                          config   globals from config, r_ang data-driven (default)"
            echo "                          median   median of N prior draws, r_ang data-driven"
            echo "                          sample   globals from prior, r_ang data-driven"
            echo "  --mode MODE           Sampling mode (default: mode1)"
            echo "  --no-ecc              Disable eccentricity model"
            echo "  --no-quadratic-warp   Disable quadratic warp (use linear only)"
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --init) INIT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --no-ecc) NO_ECC=true; shift ;;
        --no-quadratic-warp) NO_QW=true; shift ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

EXTRA_ARGS=""
if [[ -n "$INIT" ]]; then
    EXTRA_ARGS="--init-method $INIT"
fi
if [[ "$NO_ECC" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --no-ecc"
fi
if [[ "$NO_QW" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --no-quadratic-warp"
fi

DESC="$MODE, $WARMUP warmup + $SAMPLES samples, init=$INIT"
[[ "$NO_ECC" == true ]] && DESC="$DESC, no-ecc"
[[ "$NO_QW" == true ]] && DESC="$DESC, no-qw"
echo "Submitting NGC4258 NUTS ($DESC) -> $QUEUE"
addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT/scripts/megamaser/run_maser_disk.py" NGC4258 \
    --sampler nuts --mode "$MODE" \
    --num-warmup "$WARMUP" --num-samples "$SAMPLES" $EXTRA_ARGS
