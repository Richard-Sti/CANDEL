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
            echo "Usage: bash $0 [-q QUEUE] [GALAXY ...]"
            echo ""
            echo "Options:"
            echo "  -q QUEUE      GPU queue (default: gpulong)"
            echo "  GALAXY ...    Galaxy names (default: all 5 MCP galaxies)"
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        *)  GALAXIES+=("$1"); shift ;;
    esac
done

if [[ ${#GALAXIES[@]} -eq 0 ]]; then
    GALAXIES=("CGCG074-064" "NGC5765b" "NGC6264" "NGC6323" "UGC3789")
fi

for gal in "${GALAXIES[@]}"; do
    echo "Submitting DE MAP: $gal -> $QUEUE"
    addqueue -q "$QUEUE" -s -m 16 --gpus 1 \
        $PYTHON -u "$ROOT/scripts/megamaser/run_de_map.py" "$gal"
done
