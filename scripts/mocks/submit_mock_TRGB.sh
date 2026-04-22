#!/bin/bash -l
# Submit TRGB mock closure test batch jobs. Cluster (arc or glamdring) is
# picked up from `machine` in local_config.toml via _submit_lib.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

queue=""
ncpu=28
memory=7
n_mocks=100
master_seed=0
num_warmup=500
num_samples=1000
config="$ROOT/scripts/runs/config_EDD_TRGB.toml"
outdir="$ROOT/results/mocks_TRGB"
extra_args=""
local_mode=false
single_mode=false
dry=false

usage() {
    cat <<EOF
usage: $(basename "$0") -q QUEUE [-n NCPU] [-m MEMORY] [--n-mocks N]
                        [--master-seed S] [--num-warmup N] [--num-samples N]
                        [--config PATH] [--outdir PATH] [--fix-selection]
                        [--single] [--local] [--dry]

Submit TRGB mock closure test batch jobs. Runs with MPI (ranks = --ncpu).

options:
  -q, --queue QUEUE       queue/partition (REQUIRED unless --local)
  -n, --ncpu NCPU         MPI ranks (default: $ncpu)
  -m, --memory MEMORY     GB per job (default: $memory)
  --n-mocks N             mocks (default: $n_mocks)
  --master-seed S         master seed (default: $master_seed)
  --num-warmup N          NUTS warmup (default: $num_warmup)
  --num-samples N         NUTS samples (default: $num_samples)
  --config PATH           base config (default: $config)
  --outdir PATH           output dir (default: $outdir)
  --fix-selection         fix selection thresholds to mock truth
  --single                run without MPI
  --local                 run locally (mpirun / plain python), no submission
  --dry                   print submit command without submitting
  -h, --help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)         usage ;;
        -q|--queue)        queue="$2"; shift 2 ;;
        -n|--ncpu)         ncpu="$2"; shift 2 ;;
        -m|--memory)       memory="$2"; shift 2 ;;
        --n-mocks)         n_mocks="$2"; shift 2 ;;
        --master-seed)     master_seed="$2"; shift 2 ;;
        --num-warmup)      num_warmup="$2"; shift 2 ;;
        --num-samples)     num_samples="$2"; shift 2 ;;
        --config)          config="$2"; shift 2 ;;
        --outdir)          outdir="$2"; shift 2 ;;
        --fix-selection)   extra_args="$extra_args --fix-selection"; shift ;;
        --single)          single_mode=true; extra_args="$extra_args --single"; shift ;;
        --plot-only)       extra_args="$extra_args --plot-only"; shift ;;
        --local)           local_mode=true; shift ;;
        --dry)             dry=true; shift ;;
        *)                 extra_args="$extra_args $1"; shift ;;
    esac
done

if ! $local_mode && [[ -z "$queue" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi

echo "TRGB mock closure test"
echo "============================================================"
echo "  Cluster:     $CANDEL_CLUSTER"
if $local_mode; then
    echo "  Mode:        LOCAL"
else
    echo "  Mode:        SUBMIT (queue=$queue)"
fi
echo "  CPUs/ranks:  $ncpu"
echo "  N_mocks:     $n_mocks"
echo "  Master seed: $master_seed"
echo "  Warmup:      $num_warmup"
echo "  Samples:     $num_samples"
echo "  Config:      $config"
echo "  Output:      $outdir"
[[ -n "$extra_args" ]] && echo "  Extra args: $extra_args"
echo

read -rp "Proceed? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborting."; exit 1
fi

mkdir -p "$outdir"

pycmd="$CANDEL_PYTHON -u $ROOT/scripts/mocks/run_mock_TRGB.py \
    --n-mocks $n_mocks \
    --master-seed $master_seed \
    --num-warmup $num_warmup \
    --num-samples $num_samples \
    --config $config \
    --outdir $outdir \
    $extra_args"

if $single_mode || [[ $ncpu -eq 1 ]]; then
    if $local_mode; then
        echo "Running locally without MPI..."
        eval "$pycmd"
    else
        dry_flag=()
        $dry && dry_flag=(--dry)
        submit_job --queue "$queue" --mem "$memory" --cpus 1 \
            --name "mock_TRGB" --logdir "$ROOT/scripts/mocks/logs" \
            "${dry_flag[@]}" -- $pycmd
    fi
elif $local_mode; then
    echo "Running locally with MPI..."
    eval "mpirun -np $ncpu $pycmd"
else
    dry_flag=()
    $dry && dry_flag=(--dry)
    submit_job --queue "$queue" --mem "$memory" --mpi-n "$ncpu" \
        --name "mock_TRGB" --logdir "$ROOT/scripts/mocks/logs" \
        "${dry_flag[@]}" -- $pycmd
fi

echo
echo "Done."
