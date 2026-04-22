#!/bin/bash -l
# Submit toy joint H0 inference. Cluster (arc or glamdring) is picked up
# from `machine` in local_config.toml via _submit_lib.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

QUEUE=""
FLAT_DIST=""
DRY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat <<EOF
Usage: bash $0 -q QUEUE [--flat-dist] [--dry]

Options:
  -q QUEUE      Queue/partition (REQUIRED)
  --flat-dist   Use flat D prior instead of volumetric D^2
  --dry         Print submit command without submitting
EOF
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        --flat-dist) FLAT_DIST="--flat-dist"; shift ;;
        --dry) DRY=true; shift ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$QUEUE" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi

NUM_WARMUP=1000
NUM_SAMPLES=4000
NUM_CHAINS=8

echo "Submitting toy joint H0 -> $CANDEL_CLUSTER:$QUEUE"
echo "  warmup=$NUM_WARMUP, samples=$NUM_SAMPLES, chains=$NUM_CHAINS"
[[ -n "$FLAT_DIST" ]] && echo "  Using FLAT distance prior"

pycmd="$CANDEL_PYTHON -u $ROOT/scripts/megamaser/toy_joint_H0.py \
    --num-warmup $NUM_WARMUP --num-samples $NUM_SAMPLES --num-chains $NUM_CHAINS \
    $FLAT_DIST"

dry_flag=()
$DRY && dry_flag=(--dry)
submit_job --gpu --queue "$QUEUE" --mem 16 --name "toy_jointH0" \
    --logdir "$ROOT/scripts/megamaser/logs" \
    "${dry_flag[@]}" -- $pycmd
