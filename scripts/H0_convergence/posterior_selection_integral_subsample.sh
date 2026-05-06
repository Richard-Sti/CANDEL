#!/bin/bash -l
set -euo pipefail

ROOT_DIR="/mnt/users/rstiskalek/CANDEL"
PYTHON="$ROOT_DIR/venv_candel/bin/python"
SCRIPT="$ROOT_DIR/scripts/H0_convergence/posterior_selection_integral_subsample.py"
OUT_DIR="$ROOT_DIR/scripts/H0_convergence/outputs"

queue="cmbgpu"
memory_gb=32
gpus=1
gputype=""
dry=false
local=false
gpu_probe=false
python_args=()

usage() {
    cat <<EOF
usage: $(basename "$0") [submit options] [test options] [-- extra Python args]

Submit the posterior CH0 selection-integral subsampling test.

submit options:
  -q, --queue QUEUE       addqueue GPU queue (default: $queue)
  -m, --memory GB         total CPU RAM in GB (default: $memory_gb)
  --gpus N                GPUs to request (default: $gpus)
  --gputype TYPE          optional addqueue GPU type
  --gpu-probe             submit a quick JAX CUDA visibility check
  --local                 run directly instead of using addqueue
  --dry, --dry-run        print command only
  -h, --help              show this help

test options:
  --field-index N                 3D field to use from the cache (default: 0)
  posterior samples are fixed at 1000 per selection
  --fractions LIST                comma list, e.g. 0.1,0.2,...,1.0
  --num-resamples N               random voxel resamples per fraction
  --posterior-batch-size N        posterior samples per JIT GPU batch
  --seed N                        random seed
  --bias-model auto|unity|linear|double_powerlaw
  --max-voxels N                  smoke-test cap after loading field 0
  --output-dir PATH               where the PNG is written
  --mag-posterior PATH            SN-magnitude posterior HDF5
  --redshift-posterior PATH       redshift posterior HDF5
  --density-cache PATH            density-only H0 volume cache
  --velocity-cache PATH           velocity H0 volume cache
  --sn-selection-mag-error FLOAT  effective SN selection magnitude error

examples:
  $(basename "$0") --gpu-probe
  $(basename "$0") --fractions 0.1,0.25,0.5,0.75,1.0
  $(basename "$0") --dry-run --gputype rtx3090with24gb --max-voxels 100000
EOF
}

need_value() {
    if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[ERROR] $1 requires a value" >&2
        exit 2
    fi
}

need_int() {
    if ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] $1 requires a non-negative integer, got '$2'" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -q|--queue)
            need_value "$@"
            queue="$2"
            shift 2
            ;;
        -m|--memory)
            need_value "$@"
            memory_gb="$2"
            shift 2
            ;;
        --gpus)
            need_value "$@"
            need_int "$1" "$2"
            gpus="$2"
            shift 2
            ;;
        --gputype)
            need_value "$@"
            gputype="$2"
            shift 2
            ;;
        --gpu-probe)
            gpu_probe=true
            shift
            ;;
        --field-index|--num-resamples|--posterior-batch-size|--seed|\
        --max-voxels)
            need_value "$@"
            need_int "$1" "$2"
            python_args+=("$1" "$2")
            shift 2
            ;;
        --fractions|--bias-model|--output-dir|--mag-posterior|\
        --redshift-posterior|--density-cache|--velocity-cache|\
        --sn-selection-mag-error)
            need_value "$@"
            python_args+=("$1" "$2")
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
        --)
            shift
            python_args+=("$@")
            break
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "Run with --help for usage, or put raw Python args after --." >&2
            exit 2
            ;;
    esac
done

if ! [[ "$gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --gpus must be a positive integer, got '$gpus'" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export JAX_PLATFORMS=cuda
export CANDEL_ROOT="$ROOT_DIR"
export CANDEL_PYTHON="$PYTHON"
# shellcheck source=/mnt/users/rstiskalek/CANDEL/scripts/_cluster_glamdring.sh
source "$ROOT_DIR/scripts/_cluster_glamdring.sh"
if $gpu_probe; then
    python_args+=(--gpu-probe)
fi
cmd=("$PYTHON" -u "$SCRIPT" --output-dir "$OUT_DIR" "${python_args[@]}")

printf_cmd() {
    printf "%q " "$@"
    printf "\n"
}

if $local; then
    echo "[INFO] Running locally:"
    printf_cmd "${cmd[@]}"
    if ! $dry; then
        "${cmd[@]}"
    fi
else
    submit=(addqueue --sbatch -q "$queue" -s -m "$memory_gb"
            --gpus "$gpus")
    if [[ -n "$gputype" ]]; then
        submit+=(--gputype "$gputype")
    fi
    submit+=("${cmd[@]}")
    echo "[INFO] Submitting:"
    echo "[INFO] JAX: XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda"
    printf_cmd "${submit[@]}"
    if ! $dry; then
        env JAX_PLATFORMS="$JAX_PLATFORMS" \
            XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE" \
            TF_GPU_ALLOCATOR="$TF_GPU_ALLOCATOR" \
            MPLBACKEND="$MPLBACKEND" \
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
            "${submit[@]}"
    fi
fi
