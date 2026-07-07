#!/bin/bash -l
#
# Submit prepare_field_inputs.py as a CPU/MPI preprocessing job.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

queue="berg"
ncpu=1
memory=32
dry=false
local_mode=false
yes=false
job_name="prepare_field_inputs"

usage() {
    cat <<EOF
usage: $(basename "$0") [submit options] -- [prepare_field_inputs.py args]
       $(basename "$0") [submit options] [prepare_field_inputs.py args]

Prepare LOS HDF5 products and 3D volume caches as a CPU MPI job.
The density smoothing scale is read from model.field_3d_smoothing_scale in
each input config. Velocity smoothing is disabled unless
model.velocity_3d_smoothing_scale is set.

submit options:
  -q, --queue QUEUE       queue/partition (default: $queue)
                            glamdring CPU: redwood, berg, cmb
                            arc: short, medium, long
  -n, --ncpu N            MPI ranks / CPUs (default: $ncpu)
  -m, --memory GB         memory in GB (default: $memory)
  --name NAME             scheduler job name (default: $job_name)
  --dry                   print the scheduler command
  --local                 run directly in this shell
  -y, --yes               submit/run without confirmation

prepare_field_inputs.py args are forwarded unchanged. Common examples:
  scripts/runs/tasks_CH0_main.txt --products all
  scripts/runs/tasks_CH0_main.txt --tasks 12-23 --products cache
  scripts/runs/configs/config_CH0.toml --products los --overwrite-los
EOF
    exit 0
}

forward=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)       usage ;;
        -q|--queue)      queue="$2"; shift 2 ;;
        -n|--ncpu)       ncpu="$2"; shift 2 ;;
        -m|--memory)     memory="$2"; shift 2 ;;
        --name)          job_name="$2"; shift 2 ;;
        --dry)           dry=true; shift ;;
        --local)         local_mode=true; shift ;;
        -y|--yes)        yes=true; shift ;;
        --)              shift; forward+=("$@"); break ;;
        *)               forward+=("$1"); shift ;;
    esac
done

if [[ ${#forward[@]} -eq 0 ]]; then
    echo "[ERROR] Provide a task file or config args for prepare_field_inputs.py." >&2
    echo "        Run $(basename "$0") --help for usage." >&2
    exit 1
fi

if [[ "$CANDEL_CLUSTER" == "local" && $local_mode == false ]]; then
    echo "[INFO] machine='local' has no batch backend; running inline."
    local_mode=true
fi

script="$ROOT/scripts/preprocess/prepare_field_inputs.py"
cmd=(/usr/bin/env CANDEL_FIELD_CACHE_MPI=1
     "$CANDEL_PYTHON" -u "$script" "${forward[@]}")
plan_cmd=(/usr/bin/env CANDEL_FIELD_CACHE_MPI=0
          CANDEL_FIELD_CACHE_PLAN_SIZE="$ncpu"
          "$CANDEL_PYTHON" -u "$script" "${forward[@]}" --plan-only)

plan_only_arg=false
products="all"
cache_items_arg=false
for ((i = 0; i < ${#forward[@]}; i++)); do
    arg="${forward[$i]}"
    case "$arg" in
        --plan-only)
            plan_only_arg=true
            ;;
        --products)
            if (( i + 1 < ${#forward[@]} )); then
                products="${forward[$((i + 1))]}"
            fi
            ;;
        --products=*)
            products="${arg#--products=}"
            ;;
        --cache-items|--cache-items=*)
            cache_items_arg=true
            ;;
    esac
done

case "$products" in
    all)   products_label="LOS products, then 3D volume caches" ;;
    los)   products_label="LOS products only" ;;
    cache) products_label="3D volume caches only" ;;
    *)     products_label="$products" ;;
esac

print_prepare_args() {
    echo "  script: scripts/preprocess/prepare_field_inputs.py"
    echo "  args:"
    for arg in "${forward[@]}"; do
        printf '    %q\n' "$arg"
    done
}

set +e
plan_output="$("${plan_cmd[@]}" 2>&1)"
plan_status=$?
set -e
printf '%s\n' "$plan_output"
if [[ $plan_status -ne 0 ]]; then
    exit "$plan_status"
fi
echo
if $plan_only_arg; then
    exit 0
fi

missing_cache_items="$(
    printf '%s\n' "$plan_output" \
        | sed -n 's/^[[:space:]]*missing cache item IDs: //p' \
        | tail -n 1
)"
if [[ -n "$missing_cache_items" && "$products" != "los" \
        && $cache_items_arg == false && $yes == false && $dry == false ]]; then
    echo "Missing unique cache item IDs: $missing_cache_items"
    echo "Choose cache products to warm:"
    echo "  all        run all missing unique cache products"
    echo "  1,3-5      run only those item IDs from the table"
    echo "  q          abort"
    if [[ ! -t 0 ]]; then
        echo "[ERROR] Refusing selection without a TTY; use --yes or --cache-items." >&2
        exit 1
    fi
    read -r -p "Cache items to run [all]: " cache_items
    cache_items="${cache_items:-all}"
    case "$cache_items" in
        all|ALL|y|Y|yes|YES)
            ;;
        q|Q|quit|QUIT|abort|ABORT)
            echo "Aborting."
            exit 1
            ;;
        *)
            cmd+=("--cache-items" "$cache_items")
            ;;
    esac
fi

if $local_mode; then
    echo "[INFO] Running locally:"
    echo "  products: $products_label"
    print_prepare_args
    if $dry; then
        echo "[dry] not running."
        exit 0
    fi
    if ! $yes; then
        if [[ ! -t 0 ]]; then
            echo "[ERROR] Refusing local run without a TTY; use --yes." >&2
            exit 1
        fi
        read -r -p "Run this preprocessing job locally? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Aborting."
            exit 1
        fi
    fi
    "${cmd[@]}"
else
    if $dry; then
        echo "[dry] would submit field-input preprocessing:"
        echo "  queue: $queue"
        echo "  MPI ranks: $ncpu"
        echo "  memory: ${memory} GB"
        echo "  products: $products_label"
        print_prepare_args
        echo "[dry] not submitting."
        exit 0
    fi
    if ! $yes && ! $dry; then
        echo "About to submit field-input preprocessing:"
        echo "  queue: $queue"
        echo "  MPI ranks: $ncpu"
        echo "  memory: ${memory} GB"
        echo "  products: $products_label"
        echo "  environment: CANDEL_FIELD_CACHE_MPI=1"
        print_prepare_args
        if [[ ! -t 0 ]]; then
            echo "[ERROR] Refusing submission without a TTY; use --yes." >&2
            exit 1
        fi
        read -r -p "Submit this job? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Aborting."
            exit 1
        fi
    fi
    submit_job --queue "$queue" --mem "$memory" --mpi-n "$ncpu" \
        --name "$job_name" -- "${cmd[@]}"
fi
