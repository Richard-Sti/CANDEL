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
DRY=false
RESUME=false
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
Usage: bash $0 -q QUEUE [-m MEM] [--dry] [--resume] [--max-retries N] [GALAXY ...]

Options:
  -q QUEUE        Queue/partition (REQUIRED)
  -m MEM          Memory in GB (default: 7)
  --dry           Print submit command without submitting
  --resume        Resume from latest checkpoint if one exists
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
        --dry) DRY=true; shift ;;
        --resume) RESUME=true; shift ;;
        --max-retries) _WATCH_RETRIES="$2"; shift 2 ;;
        --poll) _WATCH_POLL="$2"; shift 2 ;;
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
    $DRY && _cmd+=(--dry)
    $RESUME && _cmd+=(--resume)
    (( ${#GALAXIES[@]} )) && _cmd+=("${GALAXIES[@]}")
    _gal_str="${GALAXIES[*]:-all}"
    _sname="watcher_de_map_$(date +%H%M%S)"
    screen -dmS "$_sname" bash "$_watcher" "${_wargs[@]}" -- "${_cmd[@]}"
    echo "[watch] DE MAP | galaxies: $_gal_str | queue: $QUEUE"
    echo "[watch] max-retries: $_WATCH_RETRIES | poll: ${_WATCH_POLL:-120}s"
    echo "[watch] screen: $_sname"
    echo "[watch]   reattach: screen -r $_sname"
    echo "[watch]   kill:     screen -X -S $_sname quit"
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
    submit_job --gpu --queue "$QUEUE" --mem "$MEM" --name "de_map_${gal}" \
        --logdir "$ROOT/scripts/megamaser/logs" \
        "${dry_flag[@]}" -- $pycmd
done
