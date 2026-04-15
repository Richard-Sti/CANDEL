#!/bin/bash -l
# Phi-marginal convergence test: sweep phi grid sizes and compare
# marginalised log-likelihood against a high-resolution reference.

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash $0 [ARGS...]"
    echo ""
    echo "Submits convergence_phi_marginal.py to gpulong. All arguments are"
    echo "forwarded to the Python script. For Python-level help:"
    echo "  python scripts/megamaser/convergence_phi_marginal.py --help"
    exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/venv_gpu_candel/bin/python"

addqueue -q gpulong -s -m 16 --gpus 1 \
    $PYTHON -u "$ROOT_DIR/scripts/megamaser/convergence_phi_marginal.py" "$@"
