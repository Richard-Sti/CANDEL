#!/bin/bash -l
# Submit or run a standalone TRGB posterior predictive check.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

queue=""
memory=8
cpus=1
config=""
posterior=""
output=""
n_ppc=""
seed=42
field_indices="random"
n_workers=""
save_npz=""
local_mode=false
dry=false

usage() {
    cat <<EOF
usage: $(basename "$0") --config CONFIG --posterior POSTERIOR [options]

options:
  -q, --queue QUEUE       queue/partition (required unless --local)
  -m, --memory GB         memory in GB (default: $memory)
  -n, --cpus N            CPU cores (default: $cpus)
  --output PATH           output PNG (default: <posterior>_ppc.png)
  --n-ppc N               number of simulated PPC hosts
  --seed S                random seed (default: $seed)
  --field-indices SPEC    random, all, 0-29, or comma list (default: random)
  --n-workers N           parallel field workers (default: --cpus)
  --save-npz PATH         save concatenated simulated observables
  --local                 run locally instead of submitting
  --dry                   print the submit command without submitting
  -h, --help              show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -q|--queue) queue="$2"; shift 2 ;;
        -m|--memory) memory="$2"; shift 2 ;;
        -n|--cpus) cpus="$2"; shift 2 ;;
        --config) config="$2"; shift 2 ;;
        --posterior) posterior="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --n-ppc) n_ppc="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --field-indices) field_indices="$2"; shift 2 ;;
        --n-workers) n_workers="$2"; shift 2 ;;
        --save-npz) save_npz="$2"; shift 2 ;;
        --local) local_mode=true; shift ;;
        --dry) dry=true; shift ;;
        *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$config" || -z "$posterior" ]]; then
    echo "[ERROR] --config and --posterior are required" >&2
    usage
    exit 2
fi
if ! $local_mode && [[ -z "$queue" ]]; then
    echo "[ERROR] -q/--queue is required unless --local is used" >&2
    exit 2
fi

cmd=("$CANDEL_PYTHON" -u "$ROOT/scripts/mocks/ppc_TRGB.py"
     --config "$config"
     --posterior "$posterior"
     --seed "$seed"
     --field-indices "$field_indices"
     --n-workers "${n_workers:-$cpus}")
if [[ -n "$output" ]]; then
    cmd+=(--output "$output")
fi
if [[ -n "$n_ppc" ]]; then
    cmd+=(--n-ppc "$n_ppc")
fi
if [[ -n "$save_npz" ]]; then
    cmd+=(--save-npz "$save_npz")
fi

if $local_mode; then
    if $dry; then
        printf 'Dry run command:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
else
    dry_flag=()
    $dry && dry_flag=(--dry)
    submit_job --queue "$queue" --mem "$memory" --cpus "$cpus" \
        --name "TRGB_PPC" --default-log "${dry_flag[@]}" -- "${cmd[@]}"
fi
