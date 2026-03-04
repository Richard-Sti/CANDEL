#!/bin/bash -l
#
# Submit TRGB mock closure test batch jobs to the Glamdring queue system.
#
# Machine-specific settings (python_exec) are read from local_config.toml
# at the project root.

set -e

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

# Extract a TOML key from local_config.toml
get_toml_key() {
    local key="$1"
    grep -E "^${key} *= *" "$repo_root/local_config.toml" 2>/dev/null \
        | sed -E "s/^${key} *= *\"([^\"]+)\"$/\1/"
}

# ---- defaults ----
queue="cmb"
ncpu=28
memory=7
n_mocks=100
num_warmup=500
num_samples=500
config="$repo_root/scripts/runs/config_EDD_TRGB.toml"
outdir="$repo_root/results/mocks_TRGB"
extra_args=""

usage() {
    cat <<EOF
usage: $(basename "$0") [-h] [-q QUEUE] [-n NCPU] [-m MEMORY] [--n-mocks N]
                        [--num-warmup N] [--num-samples N] [--config PATH]
                        [--outdir PATH] [--infer-selection] [--local]

Submit TRGB mock closure test batch jobs to Glamdring.

options:
  -h, --help              show this help message and exit
  -q, --queue QUEUE       Glamdring queue name (default: $queue)
  -n, --ncpu NCPU         Number of CPUs/MPI ranks (default: $ncpu)
  -m, --memory MEMORY     Memory per job in GB (default: $memory)
  --n-mocks N             Number of mock catalogs (default: $n_mocks)
  --num-warmup N          NUTS warmup steps (default: $num_warmup)
  --num-samples N         NUTS posterior samples (default: $num_samples)
  --config PATH           Base config for inference settings (default:
                          scripts/runs/config_EDD_TRGB.toml)
  --outdir PATH           Output directory (default: results/mocks_TRGB)
  --infer-selection       Infer selection thresholds instead of fixing them to
                          true mock values (default: False)
  --local                 Run locally with mpirun instead of submitting
                          (default: False)

Extra arguments are forwarded to run_mock_TRGB.py (e.g. --H0 70 --sigma-int 0.15).
EOF
    exit 0
}

# ---- parse arguments ----
local_mode=false
single_mode=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)           usage ;;
        -q|--queue)          queue="$2"; shift 2 ;;
        -n|--ncpu)           ncpu="$2"; shift 2 ;;
        -m|--memory)         memory="$2"; shift 2 ;;
        --n-mocks)           n_mocks="$2"; shift 2 ;;
        --num-warmup)        num_warmup="$2"; shift 2 ;;
        --num-samples)       num_samples="$2"; shift 2 ;;
        --config)            config="$2"; shift 2 ;;
        --outdir)            outdir="$2"; shift 2 ;;
        --infer-selection)   extra_args="$extra_args --infer-selection"; shift ;;
        --single)            single_mode=true; extra_args="$extra_args --single"; shift ;;
        --plot-only)         extra_args="$extra_args --plot-only"; shift ;;
        --local)             local_mode=true; shift ;;
        *)                   extra_args="$extra_args $1"; shift ;;
    esac
done

python_exec=$(get_toml_key "python_exec")
if [[ -z "$python_exec" ]]; then
    echo "[ERROR] Could not determine python_exec from $repo_root/local_config.toml" >&2
    exit 1
fi

echo "TRGB mock closure test batch runner"
echo "============================================================"
if $local_mode; then
    echo "  Mode:        LOCAL (mpirun)"
else
    echo "  Mode:        SUBMIT (queue=$queue)"
fi
echo "  CPUs:        $ncpu"
echo "  N_mocks:     $n_mocks"
echo "  Warmup:      $num_warmup"
echo "  Samples:     $num_samples"
echo "  Config:      $config"
echo "  Output:      $outdir"
echo "  Python:      $python_exec"
if [[ -n "$extra_args" ]]; then
    echo "  Extra args: $extra_args"
fi
echo

read -p "Proceed? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborting."
    exit 1
fi

mkdir -p "$outdir"

pythoncmd="$python_exec -u $script_dir/run_mock_TRGB.py \
    --n-mocks $n_mocks \
    --num-warmup $num_warmup \
    --num-samples $num_samples \
    --config $config \
    --outdir $outdir \
    $extra_args"

if $single_mode || [[ $ncpu -eq 1 ]]; then
    echo "Running without MPI..."
    eval "$pythoncmd"
elif $local_mode; then
    echo "Running locally with MPI..."
    eval "mpirun -np $ncpu $pythoncmd"
else
    cm="addqueue -q $queue -n $ncpu -m $memory $pythoncmd"
    echo "Submitting..."
    echo "  $cm"
    eval "$cm"
fi

echo
echo "Done."
