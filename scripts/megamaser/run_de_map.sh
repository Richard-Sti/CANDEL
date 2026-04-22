#!/bin/bash -l
# Submit DE MAP optimization for maser disk galaxies. Cluster (arc or
# glamdring) is picked up from `machine` in local_config.toml via
# _submit_lib.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

QUEUE=""
DRY=false
GALAXIES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            _info=$("$CANDEL_PYTHON" -c "
import tomli
with open('$ROOT/scripts/megamaser/config_maser.toml', 'rb') as f:
    cfg = tomli.load(f)
gals = cfg['model']['galaxies']
default_mode = cfg['model'].get('mode', 'mode2')
mode2 = [g for g, c in gals.items() if c.get('mode', default_mode) == 'mode2']
other = [g for g, c in gals.items() if c.get('mode', default_mode) != 'mode2']
print('MODE2=' + ' '.join(mode2))
print('OTHER=' + ' '.join(other))
")
            _mode2=$(echo "$_info" | grep '^MODE2=' | cut -d= -f2)
            _other=$(echo "$_info" | grep '^OTHER=' | cut -d= -f2)
            cat <<EOF
Usage: bash $0 -q QUEUE [--dry] [GALAXY ...]

Options:
  -q QUEUE      Queue/partition (REQUIRED)
  --dry         Print submit command without submitting
  GALAXY ...    Galaxy names (default: all mode2 galaxies)

Available (mode2): $_mode2
EOF
            [ -n "$_other" ] && echo "Excluded (not mode2): $_other"
            echo
            echo "Note: DE optimizer requires mode2 (r+phi marginalisation)."
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        --dry) DRY=true; shift ;;
        *)  GALAXIES+=("$1"); shift ;;
    esac
done

if [[ -z "$QUEUE" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi

if [[ ${#GALAXIES[@]} -eq 0 ]]; then
    _mode2=$("$CANDEL_PYTHON" -c "
import tomli
with open('$ROOT/scripts/megamaser/config_maser.toml', 'rb') as f:
    cfg = tomli.load(f)
gals = cfg['model']['galaxies']
default_mode = cfg['model'].get('mode', 'mode2')
print(' '.join(g for g, c in gals.items() if c.get('mode', default_mode) == 'mode2'))
")
    read -ra GALAXIES <<< "$_mode2"
fi

dry_flag=()
$DRY && dry_flag=(--dry)

for gal in "${GALAXIES[@]}"; do
    echo "Submitting DE MAP: $gal -> $CANDEL_CLUSTER:$QUEUE"
    pycmd="$CANDEL_PYTHON -u $ROOT/scripts/megamaser/run_de_map.py $gal"
    submit_job --gpu --queue "$QUEUE" --mem 16 --name "de_map_${gal}" \
        --logdir "$ROOT/scripts/megamaser/logs" \
        "${dry_flag[@]}" -- $pycmd
done
