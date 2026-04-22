#!/bin/bash -l
# Mode 2 (adaptive r + quadrature phi) grid convergence test for the
# five MCP galaxies. Compares _eval_adaptive_phi_r against a batched
# brute-force 20001 r × 20001 phi uniform-grid reference.

QUEUE="gpulong"
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: bash $0 [-q QUEUE] [ARGS...]"
            echo ""
            echo "Tests the Mode 2 adaptive per-spot r-grid + shared Simpson/trapz"
            echo "phi grids used by _eval_adaptive_phi_r, for circular physics."
            echo "Two sweeps per galaxy:"
            echo "  1. n_r_local in {51, 101, 151, 251}, phi at defaults."
            echo "  2. G_phi_half in {51, 101, 201, 401, 801}, r at n_local=201."
            echo ""
            echo "Default: five MCP galaxies (NGC5765b, UGC3789, CGCG, NGC6264,"
            echo "  NGC6323). NGC4258 is EXCLUDED — its position errors (~3 μas)"
            echo "  are too small for a 20001² uniform-grid reference; use"
            echo "  convergence_phi_marginal.sh for Mode 1 on NGC4258."
            echo ""
            echo "  -q QUEUE      GPU queue (default: gpulong)"
            echo ""
            echo "Remaining args are passed to convergence_grids.py. For Python help:"
            echo "  python scripts/megamaser/convergence/convergence_grids.py --help"
            exit 0
            ;;
        -q) QUEUE="$2"; shift 2 ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="$ROOT_DIR/venv_candel/bin/python"

echo "Submitting convergence_grids -> $QUEUE"
addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT_DIR/scripts/megamaser/convergence/convergence_grids.py" "${PASS_ARGS[@]}"
