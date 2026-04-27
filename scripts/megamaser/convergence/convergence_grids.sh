#!/bin/bash -l
# Mode 2 (union-of-local+global r + quadrature phi) grid convergence
# test for the five MCP galaxies. Compares the production log-L
# against a chunked brute-force log-uniform r × full-2π phi reference
# (float32, see [convergence.mode2_reference]).

QUEUE="gpulong"
GPUTYPE=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [--gputype TYPE] [ARGS...]"
            echo ""
            echo "Tests the Mode 2 per-spot union (local sinh + global"
            echo "log-uniform) r-grid + shared trapezoidal phi grid, for"
            echo "circular physics. Sweeps per galaxy:"
            echo "  1. (n_r_local, n_r_global) joint variation, phi defaults."
            echo "  2. (n_phi_*) variation, r at default."
            echo ""
            echo "Default: five MCP galaxies (NGC5765b, UGC3789, CGCG, NGC6264,"
            echo "  NGC6323). Pass --galaxies NGC4258 to test it separately"
            echo "  (heavy: 358 spots, dense phi grids — use --spot-batch 4"
            echo "  and a large-VRAM GPU)."
            echo ""
            echo "  -q QUEUE        GPU queue (default: gpulong)"
            echo "  --gputype TYPE  pin to a specific GPU type (e.g."
            echo "                  rtx2080with12gb). Useful to avoid gpu06"
            echo "                  (RTX 3070) which has been seen to fail"
            echo "                  CUDA init and silently fall back to CPU."
            echo ""
            echo "Common Python toggles (forwarded as-is):"
            echo "  --galaxies G [G ...]   subset to run (default: 5 MCP galaxies)"
            echo "                         e.g. --galaxies NGC5765b"
            echo "  --spot-batch N         spot-axis chunk for test-grid phi"
            echo "                         marginal; lower if OOM (default: 16)"
            echo "  --ref-spot-batch N     spot-axis chunk for the brute-force"
            echo "                         reference; raise to cut JAX dispatch"
            echo "                         overhead, lower if OOM. Defaults to"
            echo "                         [convergence.mode2_reference].spot_batch"
            echo "                         (currently 4 ≈ 9.6 GB peak)."
            echo ""
            echo "Examples:"
            echo "  bash $0 --galaxies NGC5765b"
            echo "  bash $0 --galaxies UGC3789 NGC6264 --spot-batch 8"
            echo "  bash $0 -q gpushort --galaxies NGC6323"
            echo "  bash $0 --ref-spot-batch 8        # push ref harder on 24 GB GPU"
            echo ""
            echo "Remaining args are passed to convergence_grids.py. For Python help:"
            echo "  python scripts/megamaser/convergence/convergence_grids.py --help"
            exit 0
            ;;
        -q) QUEUE="$2"; shift 2 ;;
        --gputype) GPUTYPE="$2"; shift 2 ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../_submit_lib.sh
source "$ROOT_DIR/scripts/_submit_lib.sh"
if [[ "$CANDEL_CLUSTER" != "glamdring" ]]; then
    echo "[ERROR] This script is glamdring-only (machine=$CANDEL_CLUSTER)" >&2
    exit 1
fi
PYTHON="$CANDEL_PYTHON"

ADDQUEUE_FLAGS=(-q "$QUEUE" -s -m 16 --gpus 1)
if [[ -n "$GPUTYPE" ]]; then
    ADDQUEUE_FLAGS+=(--gputype "$GPUTYPE")
fi

# Force JAX onto the CUDA plugin. JAX_PLATFORMS=cuda (read at import) makes
# JAX abort if CUDA init fails — without it, a failed CUDA init silently
# falls back to CPU and the script runs for hours instead of minutes. Use
# 'cuda' (not 'gpu'): JAX 0.8.x expands 'gpu' to include rocm and errors
# when the rocm plugin isn't installed.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda

echo "Submitting convergence_grids -> $QUEUE${GPUTYPE:+ (gputype=$GPUTYPE)}"
echo "JAX: XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda"

addqueue "${ADDQUEUE_FLAGS[@]}" \
    $PYTHON -u "$ROOT_DIR/scripts/megamaser/convergence/convergence_grids.py" "${PASS_ARGS[@]}"
