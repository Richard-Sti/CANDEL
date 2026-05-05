#!/bin/bash -l
#
# Submit a CPU-only MPI field-cache warming job on glamdring.
#
# Examples:
#   ./warm_field_cache.sh \
#       scripts/runs/configs/config_shoes.toml \
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
        --)              shift; forward+=("$@"); break ;;
        *)               forward+=("$1"); shift ;;
    esac
done

script="$ROOT/scripts/preprocess/warm_field_cache.py"
export CANDEL_FIELD_CACHE_MPI=1
cmd=("$CANDEL_PYTHON" -u "$script" "${forward[@]}")

if $local_mode; then
    echo "[INFO] Running locally:"
    printf '  %q' "${cmd[@]}"
    echo
    "${cmd[@]}"
else
    dry_flag=()
    $dry && dry_flag=(--dry)
    submit_job --queue "$queue" --mem "$memory" --mpi-n "$ncpu" \
        --name "warm_field_cache" "${dry_flag[@]}" -- "${cmd[@]}"
fi
