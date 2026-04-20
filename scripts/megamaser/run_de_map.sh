#!/bin/bash -l
# Submit DE MAP optimization for maser disk galaxies.
#
# Usage:
#   bash scripts/megamaser/submit_de_map.sh                  # all 5 galaxies
#   bash scripts/megamaser/submit_de_map.sh NGC5765b UGC3789 # specific galaxies
#   bash scripts/megamaser/submit_de_map.sh -q cmbgpu        # different queue

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/venv_gpu_candel/bin/python"

QUEUE="gpulong"
GALAXIES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            # Derive mode2 and excluded galaxies from config at help time.
            _info=$(python3 -c "
import tomli
with open('scripts/megamaser/config_maser.toml', 'rb') as f:
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
            echo "Usage: bash $0 [-q QUEUE] [GALAXY ...]"
            echo ""
            echo "Options:"
            echo "  -q QUEUE      GPU queue (default: gpulong)"
            echo "  GALAXY ...    Galaxy names (default: all mode2 galaxies)"
            echo ""
            echo "Available (mode2): $_mode2"
            [ -n "$_other" ] && echo "Excluded (not mode2): $_other"
            echo ""
            echo "Note: DE optimizer requires mode2 (r+phi marginalisation)."
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        *)  GALAXIES+=("$1"); shift ;;
    esac
done

if [[ ${#GALAXIES[@]} -eq 0 ]]; then
    _mode2=$(python3 -c "
import tomli
with open('scripts/megamaser/config_maser.toml', 'rb') as f:
    cfg = tomli.load(f)
gals = cfg['model']['galaxies']
default_mode = cfg['model'].get('mode', 'mode2')
print(' '.join(g for g, c in gals.items() if c.get('mode', default_mode) == 'mode2'))
")
    read -ra GALAXIES <<< "$_mode2"
fi

for gal in "${GALAXIES[@]}"; do
    echo "Submitting DE MAP: $gal -> $QUEUE"
    addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
        $PYTHON -u "$ROOT/scripts/megamaser/run_de_map.py" "$gal"
done
