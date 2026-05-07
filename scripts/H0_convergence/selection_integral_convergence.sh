#!/bin/bash -l
#
# Submit the CH0 selection-integral convergence test to glamdring.
# CPU only; the integral is small (a few hundred million voxels at most).
#
# Usage:
#   ./selection_integral_convergence.sh --selection mag
#   ./selection_integral_convergence.sh --selection cz
#   ./selection_integral_convergence.sh mag --dx 0.665 --radii 25,50,75,100,125,150,200
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT_DIR/venv_candel/bin/python"
SCRIPT="$ROOT_DIR/scripts/H0_convergence/selection_integral_convergence.py"
OUT_DIR="$ROOT_DIR/scripts/H0_convergence"

queue="cmb"
memory_per_cpu=7
ncpu=1
selection=""
sample=""
output=""
dry=false
local=false
python_args=()

usage() {
    cat <<EOF
usage: $(basename "$0") [mag|cz] [options]

Submit the CH0 selection-integral convergence test to glamdring.

selection:
  mag, cz                     Backward-compatible positional selection.
  -s, --selection mag|cz      Selection kernel to test.
  --sample CH0|EDD_TRGB       Use a Python preset (default: CH0). Individual
                              Python options below override preset values.

submission options:
  -q, --queue QUEUE           Glamdring CPU queue (default: $queue).
  -m, --memory GB             CPU memory per process in GB
                              (default: $memory_per_cpu; total = memory * ncpu).
  -n, --ncpu N                CPU processes requested (default: $ncpu).
  -o, --output PATH           Output .npz path.
  --local                     Run directly instead of submitting to addqueue.
  --dry, --dry-run            Print the command without submitting.
  -h, --help                  Show this help.

forwarded Python options:
  --dx FLOAT
  --radii R1,R2,...
  --H0 H0_1,H0_2,...
  --N_hosts N
  --M_abs M1,M2,...
  --mag_lim FLOAT
  --mag_width FLOAT
  --e_mag FLOAT
  --cz_lim FLOAT
  --cz_width FLOAT
  --sigma_v FLOAT
  --Om0 FLOAT

examples:
  $(basename "$0") --selection mag
  $(basename "$0") cz --dx 0.665 --radii 25,50,75,100,125,150,200
  $(basename "$0") --sample CH0 --selection mag --dry-run
EOF
}

need_value() {
    if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[ERROR] $1 requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -s|--selection)
            need_value "$@"
            selection="$2"
            python_args+=(--selection "$selection")
            shift 2
            ;;
        --sample)
            need_value "$@"
            sample="$2"
            python_args+=(--sample "$sample")
            shift 2
            ;;
        -q|--queue)
            need_value "$@"
            queue="$2"
            shift 2
            ;;
        -m|--memory)
            need_value "$@"
            memory_per_cpu="$2"
            shift 2
            ;;
        -n|--ncpu)
            need_value "$@"
            ncpu="$2"
            shift 2
            ;;
        -o|--output)
            need_value "$@"
            output="$2"
            shift 2
            ;;
        --local)
            local=true
            shift
            ;;
        --dry|--dry-run)
            dry=true
            shift
            ;;
        --dx|--radii|--H0|--N_hosts|--M_abs|--mag_lim|--mag_width|--e_mag|--cz_lim|--cz_width|--sigma_v|--Om0)
            need_value "$@"
            python_args+=("$1" "$2")
            shift 2
            ;;
        mag|cz)
            if [[ -n "$selection" ]]; then
                echo "[ERROR] selection was specified more than once" >&2
                exit 2
            fi
            selection="$1"
            python_args+=(--selection "$selection")
            shift
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

if [[ -z "$selection" && -z "$sample" ]]; then
    selection="mag"
    python_args+=(--selection "$selection")
fi
if [[ -z "$sample" ]]; then
    sample="CH0"
    python_args+=(--sample "$sample")
fi
if [[ -n "$selection" && "$selection" != "mag" && "$selection" != "cz" ]]; then
    echo "[ERROR] --selection must be 'mag' or 'cz', got '$selection'" >&2
    exit 2
fi

if [[ -z "$output" ]]; then
    tag="${selection:-$sample}"
    output="$OUT_DIR/convergence_${tag}.npz"
elif [[ "$output" != /* ]]; then
    output="$PWD/$output"
fi

cmd=("$PYTHON" -u "$SCRIPT" "${python_args[@]}" --output "$output")
submit_memory=$(( memory_per_cpu * ncpu ))

print_cmd() {
    printf '%q ' "$@"
    printf '\n'
}

if $local; then
    echo "[INFO] Running locally:"
    print_cmd "${cmd[@]}"
    if ! $dry; then
        "${cmd[@]}"
    fi
else
    submit=(addqueue -q "$queue" -s -m "$submit_memory" -n "$ncpu" "${cmd[@]}")
    echo "[INFO] Submitting:"
    echo "[INFO] Memory: ${memory_per_cpu} GB/process x ${ncpu} process(es) = ${submit_memory} GB"
    print_cmd "${submit[@]}"
    if ! $dry; then
        "${submit[@]}"
    fi
fi
