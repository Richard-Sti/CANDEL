#!/bin/bash -l
# Submit maser disk runs for all five MCP galaxies (or joint).
# Supports NSS (nested sampling, mode2 only) and NUTS (joint run).

SAMPLER="nss"
MODE=""
QUEUE=""
F_GRID=""
NUM_WARMUP=2000
NUM_SAMPLES=2000
GALAXY=""

ALL_GALS="CGCG074-064 NGC5765b NGC6264 NGC6323 UGC3789"

usage() {
    echo "Usage: $0 --sampler SAMPLER [options]"
    echo ""
    echo "Required:"
    echo "  --sampler nss|nuts     Inference method"
    echo ""
    echo "Options:"
    echo "  --galaxy GAL           Submit a single galaxy (nss only; default: all five)"
    echo "                         Choices: $ALL_GALS"
    echo "  --mode MODE            Sampling mode: mode0, mode1, mode2"
    echo "                         NSS only supports mode2 (default for NSS)."
    echo "                         NUTS defaults to config value."
    echo "  -q, --queue QUEUE      addqueue queue (default: gpulong for nss, optgpu for nuts)"
    echo "  --f-grid F             Grid scaling factor (passed to run_maser_disk.py)"
    echo "  --num-warmup N         NUTS warmup steps (default: 2000, nuts only)"
    echo "  --num-samples N        NUTS sample steps (default: 2000, nuts only)"
    echo "  -h, --help             Show this help and exit"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sampler) SAMPLER="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --f-grid) F_GRID="$2"; shift 2 ;;
        --num-warmup) NUM_WARMUP="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --galaxy) GALAXY="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$SAMPLER" != "nss" && "$SAMPLER" != "nuts" ]]; then
    echo "Error: --sampler must be nss or nuts"; exit 1
fi

if [[ "$SAMPLER" == "nss" && -n "$MODE" && "$MODE" != "mode2" ]]; then
    echo "Error: NSS only supports mode2 (phi and r are marginalised analytically)."
    exit 1
fi

if [[ "$SAMPLER" == "nss" && ("$NUM_WARMUP" != "2000" || "$NUM_SAMPLES" != "2000") ]]; then
    echo "Warning: --num-warmup and --num-samples are ignored for NSS."
fi

if [[ -n "$GALAXY" && "$SAMPLER" != "nss" ]]; then
    echo "Error: --galaxy is only supported for NSS (NUTS runs joint)."; exit 1
fi

if [[ -n "$GALAXY" ]] && ! echo "$ALL_GALS" | grep -qw "$GALAXY"; then
    echo "Error: unknown galaxy '$GALAXY'. Choices: $ALL_GALS"; exit 1
fi

ROOT="/mnt/users/rstiskalek/CANDEL"
PYTHON="$ROOT/venv_gpu_candel/bin/python"
RUNNER="$ROOT/scripts/megamaser/run_maser_disk.py"

EXTRA_ARGS=""
[[ -n "$MODE" ]]   && EXTRA_ARGS="$EXTRA_ARGS --mode $MODE"
[[ -n "$F_GRID" ]] && EXTRA_ARGS="$EXTRA_ARGS --f-grid $F_GRID"

if [[ "$SAMPLER" == "nss" ]]; then
    [[ -z "$QUEUE" ]] && QUEUE="gpulong"
    GALS="${GALAXY:-$ALL_GALS}"
    for GAL in $GALS; do
        echo "Submitting $GAL (nss, $QUEUE)..."
        addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
            $PYTHON -u $RUNNER $GAL \
            --sampler nss --D-c-prior uniform $EXTRA_ARGS
    done
else
    [[ -z "$QUEUE" ]] && QUEUE="optgpu"
    echo "Submitting joint NUTS run ($QUEUE)..."
    addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
        $PYTHON -u $RUNNER joint \
        --sampler nuts --num-warmup $NUM_WARMUP --num-samples $NUM_SAMPLES $EXTRA_ARGS
fi
