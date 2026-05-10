#!/bin/bash -l
# Submit toy joint H0 inference. Cluster (arc or glamdring) is picked up
# from `machine` in local_config.toml via _submit_lib.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

QUEUE=""
FLAT_DIST=""
SELECTION=""
NUM_WARMUP=1000
NUM_SAMPLES=4000
NUM_CHAINS=8
DRY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat <<EOF
Usage: bash $0 -q QUEUE [--flat-dist] [--selection none|distance|redshift] [--warmup N] [--samples N] [--chains N] [--dry]

Options:
  -q QUEUE       Queue/partition (REQUIRED)
  --flat-dist    Use flat D prior instead of volumetric D^2
  --selection    Run only one configuration; omit to run all three
  --warmup N     NUTS warmup steps per chain (default: 1000)
  --samples N    NUTS posterior samples per chain (default: 4000)
  --chains N     Number of vectorized chains (default: 8)
  --dry          Print submit command without submitting
EOF
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        --flat-dist) FLAT_DIST="--flat-dist"; shift ;;
        --selection) SELECTION="$2"; shift 2 ;;
        --warmup) NUM_WARMUP="$2"; shift 2 ;;
        --samples) NUM_SAMPLES="$2"; shift 2 ;;
        --chains) NUM_CHAINS="$2"; shift 2 ;;
        --dry) DRY=true; shift ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$QUEUE" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi
if [[ -n "$SELECTION" && "$SELECTION" != "none" && "$SELECTION" != "distance" && "$SELECTION" != "redshift" ]]; then
    echo "[ERROR] --selection must be 'none', 'distance', or 'redshift'"; exit 1
fi
case "$NUM_WARMUP" in ''|*[!0-9]*) echo "[ERROR] --warmup must be a positive integer"; exit 1 ;; esac
case "$NUM_SAMPLES" in ''|*[!0-9]*) echo "[ERROR] --samples must be a positive integer"; exit 1 ;; esac
case "$NUM_CHAINS" in ''|*[!0-9]*) echo "[ERROR] --chains must be a positive integer"; exit 1 ;; esac
if (( NUM_WARMUP < 1 || NUM_SAMPLES < 1 || NUM_CHAINS < 1 )); then
    echo "[ERROR] --warmup, --samples, and --chains must be positive"; exit 1
fi

echo "Submitting toy joint H0 -> $CANDEL_CLUSTER:$QUEUE"
echo "  warmup=$NUM_WARMUP, samples=$NUM_SAMPLES, chains=$NUM_CHAINS (vectorized)"
if [[ -n "$SELECTION" ]]; then
    echo "  selection=$SELECTION"
else
    echo "  selection=all (none, distance, redshift)"
fi
[[ -n "$FLAT_DIST" ]] && echo "  Using FLAT distance prior"

selection_arg=()
[[ -n "$SELECTION" ]] && selection_arg=(--selection "$SELECTION")
pycmd="$CANDEL_PYTHON -u $ROOT/scripts/megamaser/toy_joint_H0.py \
    --num-warmup $NUM_WARMUP --num-samples $NUM_SAMPLES --num-chains $NUM_CHAINS \
    ${selection_arg[*]} $FLAT_DIST"

dry_flag=()
$DRY && dry_flag=(--dry)
submit_job --gpu --queue "$QUEUE" --mem 16 --name "toy_jointH0" \
    "${dry_flag[@]}" -- $pycmd
