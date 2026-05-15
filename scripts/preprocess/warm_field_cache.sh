#!/bin/bash -l
#
# Submit a CPU-only MPI field-cache warming job on glamdring.
#
# Examples:
#   ./warm_field_cache.sh \
#       scripts/runs/configs/config_CH0.toml \
#       --selection SN_magnitude --selection redshift
#
#   ./warm_field_cache.sh \
#       scripts/runs/tasks_CH0_main.txt --tasks 12-23
#
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

usage() {
    cat <<EOF
usage: $(basename "$0") [submit options] -- [warm_field_cache.py args]
       $(basename "$0") [submit options] [warm_field_cache.py args]

Submit scripts/preprocess/warm_field_cache.py as a CPU MPI job.

submit options:
  -q, --queue QUEUE       glamdring CPU queue (default: $queue)
  -n, --ncpu N            MPI ranks / CPUs (default: $ncpu)
  -m, --memory GB         memory per MPI rank / CPU in GB (default: $memory)
  --dry                   print the addqueue command
  --local                 run directly in this shell
  -y, --yes               submit without confirmation

warm_field_cache.py args are forwarded unchanged. Common examples:
  CONFIG.toml --selection SN_magnitude --selection redshift
  ../runs/tasks_CH0_main.txt
  ../runs/tasks_CH0_main.txt --tasks 12-23
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
        --dry)           dry=true; shift ;;
        --local)         local_mode=true; shift ;;
        -y|--yes)        yes=true; shift ;;
        --)              shift; forward+=("$@"); break ;;
        *)               forward+=("$1"); shift ;;
    esac
done

script="$ROOT/scripts/preprocess/warm_field_cache.py"
cmd=("$CANDEL_PYTHON" -u "$script" "${forward[@]}")
plan_cmd=(env CANDEL_FIELD_CACHE_MPI=0 CANDEL_FIELD_CACHE_PLAN_SIZE="$ncpu"
          "$CANDEL_PYTHON" -u "$script" "${forward[@]}" --plan-only)

plan_status=0
if "${plan_cmd[@]}"; then
    plan_status=0
else
    plan_status=$?
fi
if [[ $plan_status -eq 3 ]]; then
    echo
    echo "No field-cache warmup jobs to submit."
    exit 0
elif [[ $plan_status -ne 0 ]]; then
    exit "$plan_status"
fi
echo
export CANDEL_FIELD_CACHE_MPI=1

if $local_mode; then
    echo "[INFO] Running locally:"
    printf '  %q' "${cmd[@]}"
    echo
    if $dry; then
        exit 0
    fi
    if ! $yes && ! $dry; then
        if [[ ! -t 0 ]]; then
            echo "[ERROR] Refusing local run without a TTY; use --yes." >&2
            exit 1
        fi
        read -r -p "Run this cache warmup locally? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Aborting."
            exit 1
        fi
    fi
    "${cmd[@]}"
else
    dry_flag=()
    $dry && dry_flag=(--dry)
    if ! $yes && ! $dry; then
        echo "About to submit field-cache warmup:"
        echo "  queue: $queue"
        echo "  MPI ranks: $ncpu"
        echo "  memory per rank: ${memory} GB"
        echo "  environment: CANDEL_FIELD_CACHE_MPI=1"
        printf '  command:'
        printf ' %q' "${cmd[@]}"
        echo
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
        --name "warm_field_cache" "${dry_flag[@]}" -- "${cmd[@]}"
fi
