#!/bin/bash -l
# Submit maser disk runs for all five MCP galaxies (or joint).
# Supports NSS (nested sampling, mode2 only) and NUTS (joint run).

SAMPLER="nss"
MODE=""
QUEUE=""
F_GRID=""
NUM_CHAINS=1
GALAXY=""
INIT_METHOD=""

ALL_GALS="CGCG074-064 NGC5765b NGC6264 NGC6323 UGC3789"

usage() {
    echo "Usage: $0 --sampler SAMPLER [options]"
    echo ""
    echo "Required:"
    echo "  --sampler nss|nuts     Inference method"
    echo ""
    echo "Options:"
    echo "  --galaxy GAL           Single galaxy to submit (default: all five)"
    echo "                         Choices: $ALL_GALS"
    echo "  --mode MODE            Sampling mode: mode0, mode1, mode2"
    echo "                         NSS only supports mode2 (default for NSS)."
    echo "                         NUTS defaults to config value."
    echo "  -q, --queue QUEUE      addqueue queue (default: gpulong)"
    echo "  --f-grid F             Grid scaling factor (default: 1)"
    echo "  --num-chains N         Number of NUTS chains, always vectorised (default: 1)"
    echo "  --init-method METHOD   NUTS initialisation method (default: config)"
    echo "                           config:  globals from config, r_ang data-driven from sky positions / accelerations"
    echo "                           median:  median of N prior draws, r_ang data-driven"
    echo "                           sample:  globals from prior, r_ang data-driven"
    echo "  -h, --help             Show this help and exit"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sampler) SAMPLER="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --f-grid) F_GRID="$2"; shift 2 ;;
        --num-chains) NUM_CHAINS="$2"; shift 2 ;;
        --init-method) INIT_METHOD="$2"; shift 2 ;;
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

if [[ -n "$GALAXY" ]] && ! echo "$ALL_GALS" | grep -qw "$GALAXY"; then
    echo "Error: unknown galaxy '$GALAXY'. Choices: $ALL_GALS"; exit 1
fi

ROOT="/mnt/users/rstiskalek/CANDEL"
PYTHON="$ROOT/venv_gpu_candel/bin/python"
RUNNER="$ROOT/scripts/megamaser/run_maser_disk.py"

EXTRA_ARGS=""
[[ -n "$MODE" ]]        && EXTRA_ARGS="$EXTRA_ARGS --mode $MODE"
[[ -n "$F_GRID" ]]      && EXTRA_ARGS="$EXTRA_ARGS --f-grid $F_GRID"
[[ -n "$INIT_METHOD" ]] && EXTRA_ARGS="$EXTRA_ARGS --init-method $INIT_METHOD"

[[ -z "$QUEUE" ]] && QUEUE="gpulong"

GALS="${GALAXY:-$ALL_GALS}"
for GAL in $GALS; do
    echo "Submitting $GAL ($SAMPLER, $QUEUE)..."
    if [[ "$SAMPLER" == "nss" ]]; then
        addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
            $PYTHON -u $RUNNER $GAL \
            --sampler nss $EXTRA_ARGS
    else
        addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
            $PYTHON -u $RUNNER $GAL \
            --sampler nuts --num-chains $NUM_CHAINS $EXTRA_ARGS
    fi
done
