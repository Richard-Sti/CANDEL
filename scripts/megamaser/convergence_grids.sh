#!/bin/bash -l
# Grid convergence benchmark: compare maser log-likelihood at various
# (n_r, n_phi) grid sizes against a 10001^2 brute-force reference.

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash $0 [ARGS...]"
    echo ""
    echo "Submits convergence_grids.py to gpulong. All arguments are"
    echo "forwarded to the Python script. For Python-level help:"
    echo "  python scripts/megamaser/convergence_grids.py --help"
    exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/venv_gpu_candel/bin/python"

addqueue -q gpulong -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT_DIR/scripts/megamaser/convergence_grids.py" "$@"
