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

Submit scripts/preprocess/prepare_field_inputs.py as a CPU MPI job.

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
for arg in "${forward[@]}"; do
    [[ "$arg" == "--plan-only" ]] && plan_only_arg=true
done

"${plan_cmd[@]}"
echo
if $plan_only_arg; then
    exit 0
fi

if $local_mode; then
    echo "[INFO] Running locally:"
    printf '  %q' "${cmd[@]}"
    echo
    if $dry; then
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
    dry_flag=()
    $dry && dry_flag=(--dry)
    if ! $yes && ! $dry; then
        echo "About to submit field-input preprocessing:"
        echo "  queue: $queue"
        echo "  MPI ranks: $ncpu"
        echo "  memory: ${memory} GB"
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
        --name "$job_name" "${dry_flag[@]}" -- "${cmd[@]}"
fi
