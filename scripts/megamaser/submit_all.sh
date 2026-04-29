#!/bin/bash -l
# Submit maser disk runs for all five MCP galaxies (or one via --galaxy).
# Supports NSS (nested sampling, mode2 only) and NUTS. Cluster (arc or
# glamdring) is picked up from `machine` in local_config.toml via
# _submit_lib.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

SAMPLER="nss"
MODE=""
QUEUE=""
F_GRID=""
NUM_CHAINS=1
GALAXY=""
INIT_METHOD=""
GPUTYPE=""
GPU_MEM=""
TIME=""
MEM=16
DRY=false
_WATCH_RETRIES=""
_WATCH_POLL=""
RESUME=false
NO_ECC=false
NO_QUAD_WARP=false

ALL_GALS="CGCG074-064 NGC5765b NGC6264 NGC6323 UGC3789"

usage() {
    cat <<EOF
Usage: $0 -q QUEUE [options]

Required:
  -q, --queue QUEUE      Queue/partition (glamdring: gpulong|cmbgpu|optgpu;
                         arc: short|medium|long)

Options:
  --sampler nss|nuts     Inference method (default: $SAMPLER)
  --galaxy GAL[,GAL,...] Galaxy/galaxies to submit (comma-separated;
                         default: all five). Choices: $ALL_GALS
  --mode MODE            Sampling mode: mode1, mode2
                         (default: runner picks — mode2 for NSS)
  --f-grid F             Grid scaling factor (default: 1.0)
  --num-chains N         NUTS vectorised chains (default: $NUM_CHAINS)
  --init-method METHOD   NUTS init method: config | median | sample
                         (default: runner picks from config)
  --gputype TYPE         GPU type (default: any; e.g. h100, l40s)
  --gpu-mem GB           Min GPU VRAM in GB (arc only; queries sinfo)
  --time T               Wall time. Bare integer = hours (arc only).
                         (default on arc: short=12, medium=48, long=required;
                          ignored on glamdring)
  --mem GB               Memory in GB (default: $MEM)
  --no-ecc               Disable eccentricity model
  --no-quadratic-warp    Disable quadratic disk warp
  --dry                  Print submit command without submitting (default: off)
  --resume               Resume NSS from latest checkpoint (ignored for NUTS)
  --max-retries N        Watch and resubmit up to N times on timeout
  --poll S               Seconds between squeue polls (default: 120)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sampler) SAMPLER="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        -q|--queue) QUEUE="$2"; shift 2 ;;
        --f-grid) F_GRID="$2"; shift 2 ;;
        --num-chains) NUM_CHAINS="$2"; shift 2 ;;
        --init-method) INIT_METHOD="$2"; shift 2 ;;
        --galaxy) GALAXY="$2"; shift 2 ;;
        --gputype) GPUTYPE="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --no-ecc) NO_ECC=true; shift ;;
        --no-quadratic-warp) NO_QUAD_WARP=true; shift ;;
        --dry) DRY=true; shift ;;
        --resume) RESUME=true; shift ;;
        --max-retries) _WATCH_RETRIES="$2"; shift 2 ;;
        --poll) _WATCH_POLL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# If --max-retries is set, delegate to the watcher and exit.
if [[ -n "$_WATCH_RETRIES" ]]; then
    _watcher="$ROOT/scripts/megamaser/watch_and_resubmit.sh"
    _wargs=(--marker "saved samples to" --max-retries "$_WATCH_RETRIES")
    [[ -n "$_WATCH_POLL" ]] && _wargs+=(--poll "$_WATCH_POLL")
    [[ "$SAMPLER" == "nuts" ]] && _wargs+=(--no-resume)
    # Rebuild the original command without --max-retries and --poll.
    _cmd=(bash "$0" --sampler "$SAMPLER" -q "$QUEUE" --mem "$MEM")
    [[ -n "$MODE" ]]        && _cmd+=(--mode "$MODE")
    [[ -n "$F_GRID" ]]      && _cmd+=(--f-grid "$F_GRID")
    [[ -n "$GALAXY" ]]      && _cmd+=(--galaxy "$GALAXY")
    [[ -n "$INIT_METHOD" ]] && _cmd+=(--init-method "$INIT_METHOD")
    [[ -n "$GPUTYPE" ]]     && _cmd+=(--gputype "$GPUTYPE")
    [[ -n "$GPU_MEM" ]]     && _cmd+=(--gpu-mem "$GPU_MEM")
    [[ -n "$TIME" ]]        && _cmd+=(--time "$TIME")
    [[ "$NUM_CHAINS" != "1" ]] && _cmd+=(--num-chains "$NUM_CHAINS")
    $NO_ECC && _cmd+=(--no-ecc)
    $NO_QUAD_WARP && _cmd+=(--no-quadratic-warp)
    $DRY && _cmd+=(--dry)
    $RESUME && _cmd+=(--resume)
    _sname="watcher_${SAMPLER}_$(date +%H%M%S)"
    _logdir="$ROOT/scripts/megamaser/logs"
    mkdir -p "$_logdir"
    launch_detached "$_sname" "$_logdir/${_sname}.log" \
        bash "$_watcher" "${_wargs[@]}" -- "${_cmd[@]}"
    exit 0
fi

if [[ "$SAMPLER" != "nss" && "$SAMPLER" != "nuts" ]]; then
    echo "Error: --sampler must be nss or nuts"; exit 1
fi
if [[ "$SAMPLER" == "nss" && -n "$MODE" && "$MODE" != "mode2" ]]; then
    echo "Error: NSS only supports mode2."; exit 1
fi
if [[ -n "$GALAXY" ]]; then
    GALAXY="${GALAXY//,/ }"
    for _g in $GALAXY; do
        echo "$ALL_GALS" | grep -qw "$_g" || { echo "Error: unknown galaxy '$_g'. Choices: $ALL_GALS"; exit 1; }
    done
fi
if [[ -z "$QUEUE" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi

RUNNER="$ROOT/scripts/megamaser/run_maser_disk.py"

EXTRA_ARGS=""
[[ -n "$MODE" ]]        && EXTRA_ARGS="$EXTRA_ARGS --mode $MODE"
[[ -n "$F_GRID" ]]      && EXTRA_ARGS="$EXTRA_ARGS --f-grid $F_GRID"
[[ -n "$INIT_METHOD" ]] && EXTRA_ARGS="$EXTRA_ARGS --init-method $INIT_METHOD"
$NO_ECC && EXTRA_ARGS="$EXTRA_ARGS --no-ecc"
$NO_QUAD_WARP && EXTRA_ARGS="$EXTRA_ARGS --no-quadratic-warp"
$RESUME && EXTRA_ARGS="$EXTRA_ARGS --resume"

dry_flag=()
[[ "$DRY" == true ]] && dry_flag=(--dry)

GALS="${GALAXY:-$ALL_GALS}"
for GAL in $GALS; do
    echo "Submitting $GAL ($SAMPLER) -> $CANDEL_CLUSTER:$QUEUE"
    if [[ "$SAMPLER" == "nss" ]]; then
        pycmd="$CANDEL_PYTHON -u $RUNNER $GAL --sampler nss $EXTRA_ARGS"
    else
        pycmd="$CANDEL_PYTHON -u $RUNNER $GAL --sampler nuts --num-chains $NUM_CHAINS $EXTRA_ARGS"
    fi
    extra_flags=()
    [[ -n "$GPUTYPE" ]] && extra_flags+=(--gputype "$GPUTYPE")
    [[ -n "$GPU_MEM" ]] && extra_flags+=(--gpu-mem "$GPU_MEM")
    [[ -n "$TIME" ]]    && extra_flags+=(--time "$TIME")
    submit_job --gpu --queue "$QUEUE" --mem "$MEM" --name "maser_${GAL}" \
        --logdir "$ROOT/scripts/megamaser/logs" \
        "${extra_flags[@]}" \
        "${dry_flag[@]}" \
        -- $pycmd
done
