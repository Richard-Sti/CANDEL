#!/bin/bash -l
# Submit NGC4258 NUTS job. Cluster (arc or glamdring) is picked up from
# `machine` in local_config.toml via _submit_lib.sh.
#
# NGC4258 defaults to Mode 1 with per-type bruteforce phi grids. NUTS is
# the only supported sampler (NSS can't handle 358 r_ang params, DE doesn't
# support Mode 1).
#
# Usage:
#   bash scripts/megamaser/submit_ngc4258.sh -q <queue>                  # required
#   bash scripts/megamaser/submit_ngc4258.sh -q optgpu --warmup 5000
#   bash scripts/megamaser/submit_ngc4258.sh -q short --no-ecc --no-quadratic-warp
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

QUEUE=""
WARMUP=2000
SAMPLES=2000
INIT="config"
MODE="mode1"
NO_ECC=false
NO_QW=false
DRY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat <<EOF
Usage: bash $0 -q QUEUE [--warmup N] [--samples N] [--init METHOD]
              [--mode MODE] [--no-ecc] [--no-quadratic-warp] [--dry]

Options:
  -q QUEUE              Queue/partition (REQUIRED). Pass a name valid on the
                        current cluster (glamdring: optgpu|gpulong|cmbgpu;
                        arc: short|medium|long).
  --warmup N            NUTS warmup iterations (default: $WARMUP)
  --samples N           NUTS samples (default: $SAMPLES)
  --init METHOD         config | median | sample (default: $INIT)
  --mode MODE           Sampling mode (default: $MODE)
  --no-ecc              Disable eccentricity model
  --no-quadratic-warp   Disable quadratic warp (use linear only)
  --dry                 Print the submitter cmd without submitting
EOF
            exit 0 ;;
        -q) QUEUE="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --init) INIT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --no-ecc) NO_ECC=true; shift ;;
        --no-quadratic-warp) NO_QW=true; shift ;;
        --dry) DRY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$QUEUE" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"
    exit 1
fi

EXTRA_ARGS=("--init-method" "$INIT")
[[ "$NO_ECC" == true ]] && EXTRA_ARGS+=("--no-ecc")
[[ "$NO_QW"  == true ]] && EXTRA_ARGS+=("--no-quadratic-warp")

DESC="$MODE, $WARMUP warmup + $SAMPLES samples, init=$INIT"
[[ "$NO_ECC" == true ]] && DESC="$DESC, no-ecc"
[[ "$NO_QW"  == true ]] && DESC="$DESC, no-qw"
echo "Submitting NGC4258 NUTS ($DESC) -> $CANDEL_CLUSTER:$QUEUE"

pycmd="$CANDEL_PYTHON -u $ROOT/scripts/megamaser/run_maser_disk.py NGC4258 \
    --sampler nuts --mode $MODE \
    --num-warmup $WARMUP --num-samples $SAMPLES ${EXTRA_ARGS[*]}"

dry_flag=()
[[ "$DRY" == true ]] && dry_flag=(--dry)

submit_job --gpu --queue "$QUEUE" --mem 16 --name "n4258_${MODE}" \
    "${dry_flag[@]}" \
    -- $pycmd
