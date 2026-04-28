#!/bin/bash -l
# Submit DE MAP optimization for maser disk galaxies. Cluster (arc or
# glamdring) is picked up from `machine` in local_config.toml via
# _submit_lib.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

QUEUE=""
MEM=7
CPUS=""
GPUTYPE=""
GPU_CONSTRAINT=""
TIME=""
DRY=false
RESUME=false
NO_ECC=false
NO_QUAD_WARP=false
GALAXIES=()
_WATCH_RETRIES=""
_WATCH_POLL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            _gals=$("$CANDEL_PYTHON" -c "
import tomli
with open('$ROOT/scripts/megamaser/config_maser.toml', 'rb') as f:
    cfg = tomli.load(f)
print(' '.join(cfg['model']['galaxies']))
")
            cat <<EOF
Usage: bash $0 -q QUEUE [-m MEM] [--gputype TYPE] [--time T]
              [--dry] [--resume] [--max-retries N] [GALAXY ...]

Options:
  -q QUEUE        Queue/partition (REQUIRED)
  -m MEM          Memory in GB (default: 7)
  --gputype TYPE  GPU type (default: any)
  --gpu-constraint EXPR  SLURM constraint for GPU selection (e.g. "h100|l40s")
  --time T        Wall time. Bare integer = hours (arc only)
  -n N            Number of CPU cores (default: 4 with --gpu)
  --dry           Print submit command without submitting
  --resume        Resume from latest checkpoint if one exists
  --no-ecc        Disable eccentricity model
  --no-quadratic-warp  Disable quadratic disk warp
  --max-retries N Watch and resubmit up to N times on timeout
  --poll S        Seconds between squeue polls (default: 120)
  GALAXY ...      Galaxy names (default: all below)

Galaxies: $_gals

mode2 is forced automatically (DE requires r+phi marginalisation).
Checkpoints: results/Megamaser/de_checkpoints/<galaxy>/de_ckpt.npz

Examples:
  bash $0 -q cmbgpu
  bash $0 -q optgpu NGC4258
  bash $0 -q cmbgpu --resume NGC5765b
  bash $0 -q cmbgpu --max-retries 5
EOF
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        -m) MEM="$2"; shift 2 ;;
        -n) CPUS="$2"; shift 2 ;;
        --gputype) GPUTYPE="$2"; shift 2 ;;
        --gpu-constraint) GPU_CONSTRAINT="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --dry) DRY=true; shift ;;
        --resume) RESUME=true; shift ;;
        --no-ecc) NO_ECC=true; shift ;;
        --no-quadratic-warp) NO_QUAD_WARP=true; shift ;;
        --max-retries) _WATCH_RETRIES="$2"; shift 2 ;;
        --poll) _WATCH_POLL="$2"; shift 2 ;;
        -*) echo "[ERROR] unknown option: $1"; exit 1 ;;
        *)  GALAXIES+=("$1"); shift ;;
    esac
done

# If --max-retries is set, delegate to the watcher and exit.
if [[ -n "$_WATCH_RETRIES" ]]; then
    _watcher="$ROOT/scripts/megamaser/watch_and_resubmit.sh"
    _wargs=(--marker "MAP init" --max-retries "$_WATCH_RETRIES")
    [[ -n "$_WATCH_POLL" ]] && _wargs+=(--poll "$_WATCH_POLL")
    # Rebuild the original command without --max-retries and --poll.
    _cmd=(bash "$0" -q "$QUEUE" -m "$MEM")
    [[ -n "$CPUS" ]]    && _cmd+=(-n "$CPUS")
    [[ -n "$GPUTYPE" ]]        && _cmd+=(--gputype "$GPUTYPE")
    [[ -n "$GPU_CONSTRAINT" ]] && _cmd+=(--gpu-constraint "$GPU_CONSTRAINT")
    [[ -n "$TIME" ]]           && _cmd+=(--time "$TIME")
    $DRY && _cmd+=(--dry)
    $RESUME && _cmd+=(--resume)
    $NO_ECC && _cmd+=(--no-ecc)
    $NO_QUAD_WARP && _cmd+=(--no-quadratic-warp)
    (( ${#GALAXIES[@]} )) && _cmd+=("${GALAXIES[@]}")
    _sname="watcher_de_map_$(date +%H%M%S)"
    _logdir="$ROOT/scripts/megamaser/logs"
    mkdir -p "$_logdir"
    launch_detached "$_sname" "$_logdir/${_sname}.log" \
        bash "$_watcher" "${_wargs[@]}" -- "${_cmd[@]}"
    exit 0
fi

if [[ -z "$QUEUE" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi

if [[ ${#GALAXIES[@]} -eq 0 ]]; then
    _all=$("$CANDEL_PYTHON" -c "
import tomli
with open('$ROOT/scripts/megamaser/config_maser.toml', 'rb') as f:
    cfg = tomli.load(f)
print(' '.join(cfg['model']['galaxies']))
")
    read -ra GALAXIES <<< "$_all"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async
export JAX_PLATFORMS=cuda

dry_flag=()
$DRY && dry_flag=(--dry)

for gal in "${GALAXIES[@]}"; do
    echo "Submitting DE MAP: $gal -> $CANDEL_CLUSTER:$QUEUE"
    pycmd="$CANDEL_PYTHON -u $ROOT/scripts/megamaser/run_de_map.py $gal"
    $RESUME && pycmd="$pycmd --resume"
    $NO_ECC && pycmd="$pycmd --no-ecc"
    $NO_QUAD_WARP && pycmd="$pycmd --no-quadratic-warp"
    extra_flags=()
    [[ -n "$CPUS" ]]    && extra_flags+=(--cpus "$CPUS")
    [[ -n "$GPUTYPE" ]]        && extra_flags+=(--gputype "$GPUTYPE")
    [[ -n "$GPU_CONSTRAINT" ]] && extra_flags+=(--gpu-constraint "$GPU_CONSTRAINT")
    [[ -n "$TIME" ]]           && extra_flags+=(--time "$TIME")
    submit_job --gpu --queue "$QUEUE" --mem "$MEM" --name "de_map_${gal}" \
        --logdir "$ROOT/scripts/megamaser/logs" \
        "${extra_flags[@]}" "${dry_flag[@]}" -- $pycmd
done
