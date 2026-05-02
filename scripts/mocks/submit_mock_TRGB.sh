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
which_selection="TRGB_magnitude"
use_field=true
field_name="Carrick2015"
fix_selection=true
config="$ROOT/scripts/runs/configs/config_EDD_TRGB.toml"
outdir="$ROOT/results/mocks_TRGB"
extra_args=""
local_mode=false
single_mode=false
dry=false

print_injected_parameters() {
    local py="${CANDEL_PYTHON:-python3}"
    "$py" - "$ROOT/candel/mock/TRGB_mock.py" <<'PY' || {
import ast
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=path)

wanted = {"DEFAULT_TRUE_PARAMS", "DEFAULT_ANCHORS"}
found = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)

print("injected defaults:")
for name in ("DEFAULT_TRUE_PARAMS", "DEFAULT_ANCHORS"):
    values = found.get(name, {})
    print(f"  {name}:")
    for key, value in values.items():
        print(f"    {key:<20s} {value}")
PY
        echo "injected defaults: unavailable"
    }
}

usage() {
    cat <<EOF
usage: $(basename "$0") -q QUEUE [-n NCPU] [-m MEMORY] [--n-mocks N]
                        [--master-seed S] [--num-warmup N] [--num-samples N]
                        [--which-selection NAME] [--config PATH]
                        [--outdir PATH] [--infer-selection] [--no-field]
                        [--field-name NAME] [--single] [--local] [--dry]

Submit TRGB mock closure test batch jobs. Runs with MPI (ranks = --ncpu).

defaults:
  Field sampling is ON by default using $field_name.
  Selection thresholds are fixed to the injected truth by default.
  Use --no-field for homogeneous no-reconstruction mocks.
  Use --infer-selection to infer the selection thresholds.

options:
  -q, --queue QUEUE       queue/partition (REQUIRED unless --local)
  -n, --ncpu NCPU         MPI ranks (default: $ncpu)
  -m, --memory MEMORY     GB per job (default: $memory)
  --n-mocks N             mocks (default: $n_mocks)
  --master-seed S         master seed (default: $master_seed)
  --num-warmup N          NUTS warmup (default: $num_warmup)
  --num-samples N         NUTS samples (default: $num_samples)
  --which-selection NAME   TRGB_magnitude or redshift (default: $which_selection)
  --config PATH           base config (default: $config)
  --outdir PATH           output dir (default: $outdir)
  --infer-selection       infer selection thresholds instead of fixing truth
  --fix-selection         fix selection thresholds to truth (default)
  --no-field              disable reconstruction-field sampling
  --use-field             enable reconstruction-field sampling (default)
  --field-name NAME       reconstruction field name (default: $field_name)
  --single                run without MPI
  --plot-only             with --single: generate and plot, skip inference
  --local                 run locally (mpirun / plain python), no submission
  --dry                   print submit command without submitting
  -h, --help

EOF
    print_injected_parameters
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
        --which-selection) which_selection="$2"; shift 2 ;;
        --config)          config="$2"; shift 2 ;;
        --outdir)          outdir="$2"; shift 2 ;;
        --infer-selection) fix_selection=false; shift ;;
        --fix-selection)   fix_selection=true; shift ;;
        --no-field|--disable-field) use_field=false; shift ;;
        --use-field)       use_field=true; shift ;;
        --field-name)      field_name="$2"; shift 2 ;;
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
echo "  Selection:   $which_selection"
echo "  Field:       $use_field"
if $use_field; then
    echo "  Field name:  $field_name"
fi
echo "  Sel params:  $($fix_selection && echo fixed-to-truth || echo inferred)"
echo "  Config:      $config"
echo "  Output:      $outdir"
[[ -n "$extra_args" ]] && echo "  Extra args: $extra_args"
echo

read -rp "Proceed? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborting."; exit 1
fi

mkdir -p "$outdir"

selection_args=""
$fix_selection && selection_args="--fix-selection"
field_args=""
$use_field && field_args="--use-field --field-name $field_name"

pycmd="$CANDEL_PYTHON -u $ROOT/scripts/mocks/run_mock_TRGB.py \
    --n-mocks $n_mocks \
    --master-seed $master_seed \
    --num-warmup $num_warmup \
    --num-samples $num_samples \
    --which-selection $which_selection \
    --config $config \
    --outdir $outdir \
    $selection_args \
    $field_args \
    $extra_args"

if $dry && $local_mode; then
    echo "Dry run command:"
    if $single_mode || [[ $ncpu -eq 1 ]]; then
        echo "$pycmd"
    else
        echo "mpirun -np $ncpu $pycmd"
    fi
    exit 0
fi

if $single_mode || [[ $ncpu -eq 1 ]]; then
    if $local_mode; then
        echo "Running locally without MPI..."
        eval "$pycmd"
    else
        dry_flag=()
        $dry && dry_flag=(--dry)
        submit_job --queue "$queue" --mem "$memory" --cpus 1 \
            --name "mock_TRGB" \
            "${dry_flag[@]}" -- $pycmd
    fi
elif $local_mode; then
    echo "Running locally with MPI..."
    eval "mpirun -np $ncpu $pycmd"
else
    dry_flag=()
    $dry && dry_flag=(--dry)
    submit_job --queue "$queue" --mem "$memory" --mpi-n "$ncpu" \
        --name "mock_TRGB" \
        "${dry_flag[@]}" -- $pycmd
fi

echo
echo "Done."
