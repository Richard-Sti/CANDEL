#!/bin/bash -l
#
# Submit the CH0 selection-integral convergence test to glamdring.
# CPU only; the integral is small (a few hundred million voxels at most).
#
# Usage:
#   ./selection_integral_convergence.sh mag         # SN_magnitude selection
#   ./selection_integral_convergence.sh cz          # redshift selection
#   ./selection_integral_convergence.sh mag --dx 0.665 --radii 25,50,75,100,125,150,200
#
set -e

SEL="${1:-mag}"; shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/venv_candel/bin/python"
SCRIPT="$ROOT_DIR/scripts/H0_convergence/selection_integral_convergence.py"
OUT="$ROOT_DIR/scripts/H0_convergence/convergence_${SEL}.npz"

# Single-CPU job, 16 GB is generous for ~10^8 voxels at float64.
addqueue -q cmb -s -m 16 -n 1 \
    "$PYTHON" -u "$SCRIPT" \
        --selection "$SEL" \
        --output "$OUT" \
        "$@"
