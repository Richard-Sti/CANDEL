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
Usage: bash $0 -q QUEUE [-m MEM] [--dry] [--resume] [GALAXY ...]

Options:
  -q QUEUE        Queue/partition (REQUIRED)
  -m MEM          Memory in GB (default: 7)
  --dry           Print submit command without submitting
  --resume        Resume from latest checkpoint if one exists
  GALAXY ...      Galaxy names (default: all below)

Galaxies: $_gals

mode2 is forced automatically (DE requires r+phi marginalisation).
Checkpoints: results/Megamaser/de_checkpoints/<galaxy>/de_ckpt.npz

Examples:
  bash $0 -q cmbgpu
  bash $0 -q optgpu NGC4258
  bash $0 -q cmbgpu --resume NGC5765b
EOF
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        -m) MEM="$2"; shift 2 ;;
        --dry) DRY=true; shift ;;
        --resume) RESUME=true; shift ;;
        *)  GALAXIES+=("$1"); shift ;;
    esac
done

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
