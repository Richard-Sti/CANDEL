#!/bin/bash -l
# Submit Mark Reid fit_disk MCMC checks as independent serial chains.
#
# Reid's Fortran code is not internally parallel.  This script requests N CPU
# cores and starts N independent one-core Reid runs in the same batch job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

THIS="$ROOT/scripts/megamaser/check_reid/submit_reid_mcmc.sh"
RUNNER="$ROOT/scripts/megamaser/check_reid/run_reid_mcmc.py"

QUEUE=""
GALAXY="NGC4258"
CHAINS=1
MEM_PER_CHAIN=7
TIME=""
INIT="reid"
INIT_NPZ="$ROOT/results/Megamaser/de_checkpoints/NGC4258/de_ckpt.npz"
REID_INIT="$ROOT/scripts/megamaser/check_reid/reid_ngc4258_init.toml"
CONFIG="$ROOT/scripts/megamaser/config_maser.toml"
DATA="$ROOT/data/Megamaser/N4258_disk_data_MarkReid.final"
BASE_OUTPUT="$ROOT/results/Megamaser/reid_mcmc"
BURNIN=1000000
TRIALS=100000000
WALKERS=1
SEED=47351937
SEED_STEP=1009
VCOR=0
H0_LOW=""
H0_HIGH=""
STEP_FRACTION=0.015
STATUS_INTERVAL=10000000
FIT_DATA=""
PLOT_PARAMS=""
DRY=false
WORKER=false

usage() {
    cat <<EOF
Usage: bash $0 -q QUEUE [options]

CPU-only submission for Reid fit_disk.  Defaults follow Reid's run mode:
Reid initial globals, primary burn-in enabled, one M-H strand, and the Reid
seed.  If --chains >1, one batch job reserves --chains CPU cores and runs that
many independent serial Reid chains in parallel.

Required:
  -q, --queue QUEUE          Queue/partition (cluster=$CANDEL_CLUSTER)

Options:
  -n, --chains N            Number of independent Reid chains / CPU cores
                            (default: $CHAINS)
  --mem-per-chain GB        Memory per chain in GB; total request is
                            chains * mem-per-chain (default: $MEM_PER_CHAIN)
  --time T                  Wall time (arc only; ignored on glamdring)
  --galaxy NAME             Galaxy name passed to the wrapper (default: $GALAXY)
  --init reid|config|de-npz Initial point source (default: $INIT)
  --reid-init PATH          TOML globals for --init reid
  --init-npz PATH           DE checkpoint for --init de-npz
  --config PATH             Maser TOML config
  --data PATH               Reid-format maser data file
  --output-dir DIR          Batch output directory root
                            (default: $BASE_OUTPUT)
  --burnin N                Reid primary burn-in; <=0 uses generated
                            burnin_values.dat (default: $BURNIN)
  --trials N                Reid final MCMC trials; must be >=500000
                            (default: $TRIALS)
  --walkers N               Reid internal M-H strands per process
                            (default: $WALKERS; still serial inside Reid)
  --seed N                  Seed for chain 0 (default: $SEED)
  --seed-step N             Added per chain: seed_i = seed + i*step
                            (default: $SEED_STEP)
  --vcor KM_S               Velocity correction to Hubble flow (default: $VCOR)
  --h0-low X                Override H0 lower bound
  --h0-high X               Override H0 upper bound
  --step-fraction X         Reid MCMC proposal scale (default: $STEP_FRACTION)
  --status-interval N       Progress print interval for secondary burn-in and
                            final MCMC (default: $STATUS_INTERVAL; use 0 for
                            wrapper-chosen ~1% cadence)
  --fit-data TTTT           Fit x,y,v,a flags, e.g. TTTT or TTFT
  --plot-params CSV         Parameters to include in global_corner.png
  --dry                     Print submit command without submitting
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker) WORKER=true; shift ;;
        -q|--queue) QUEUE="$2"; shift 2 ;;
        -n|--chains|--cpus) CHAINS="$2"; shift 2 ;;
        --mem-per-chain) MEM_PER_CHAIN="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --galaxy) GALAXY="$2"; shift 2 ;;
        --init) INIT="$2"; shift 2 ;;
        --reid-init) REID_INIT="$2"; shift 2 ;;
        --init-npz) INIT_NPZ="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --output-dir) BASE_OUTPUT="$2"; shift 2 ;;
        --burnin) BURNIN="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --walkers) WALKERS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --seed-step) SEED_STEP="$2"; shift 2 ;;
        --vcor) VCOR="$2"; shift 2 ;;
        --h0-low) H0_LOW="$2"; shift 2 ;;
        --h0-high) H0_HIGH="$2"; shift 2 ;;
        --step-fraction) STEP_FRACTION="$2"; shift 2 ;;
        --status-interval) STATUS_INTERVAL="$2"; shift 2 ;;
        --fit-data) FIT_DATA="$2"; shift 2 ;;
        --plot-params) PLOT_PARAMS="$2"; shift 2 ;;
        --dry) DRY=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$CHAINS" -lt 1 ]]; then
    echo "[ERROR] --chains must be >=1"; exit 1
fi
if [[ "$TRIALS" -lt 500000 ]]; then
    echo "[ERROR] --trials must be >=500000 for Reid v24d"; exit 1
fi
if [[ "$INIT" != "reid" && "$INIT" != "config" && "$INIT" != "de-npz" ]]; then
    echo "[ERROR] --init must be reid, config, or de-npz"; exit 1
fi

if $WORKER; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    batch_dir="$BASE_OUTPUT/${GALAXY}_${INIT}_${CHAINS}chains_${stamp}"
    mkdir -p "$batch_dir"
    echo "Running Reid chains on $(hostname)"
    echo "  batch_dir=$batch_dir"
    echo "  chains=$CHAINS, trials=$TRIALS, walkers=$WALKERS"
    echo "  init=$INIT"

    pids=()
    for ((i = 0; i < CHAINS; i++)); do
        chain_seed=$((SEED + i * SEED_STEP))
        chain_id="$(printf "%02d" "$i")"
        chain_dir="$batch_dir/chain_${chain_id}_seed_${chain_seed}"
        mkdir -p "$chain_dir"

        cmd=(
            "$CANDEL_PYTHON" -u "$RUNNER"
            --galaxy "$GALAXY"
            --config "$CONFIG"
            --data "$DATA"
            --init "$INIT"
            --reid-init "$REID_INIT"
            --init-npz "$INIT_NPZ"
            --output-dir "$chain_dir"
            --burnin "$BURNIN"
            --trials "$TRIALS"
            --walkers "$WALKERS"
            --seed "$chain_seed"
            --vcor "$VCOR"
            --step-fraction "$STEP_FRACTION"
        )
        [[ -n "$STATUS_INTERVAL" ]] && cmd+=(--status-interval "$STATUS_INTERVAL")
        [[ -n "$H0_LOW" ]] && cmd+=(--h0-low "$H0_LOW")
        [[ -n "$H0_HIGH" ]] && cmd+=(--h0-high "$H0_HIGH")
        [[ -n "$FIT_DATA" ]] && cmd+=(--fit-data "$FIT_DATA")
        [[ -n "$PLOT_PARAMS" ]] && cmd+=(--plot-params "$PLOT_PARAMS")

        printf "%s\n" "${cmd[*]}" > "$chain_dir/command.txt"
        (
            set -o pipefail
            "${cmd[@]}" 2>&1 \
                | sed -u "s/^/[chain ${chain_id}] /" \
                | tee "$chain_dir/wrapper.stdout"
        ) &
        pids+=("$!")
        pid="${pids[$((${#pids[@]} - 1))]}"
        echo "$chain_dir" >> "$batch_dir/chain_dirs.txt"
        echo "  launched chain $chain_id seed=$chain_seed pid=$pid"
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    echo "Finished Reid chain batch: $batch_dir"
    if [[ "$failed" -ne 0 ]]; then
        echo "[ERROR] At least one Reid chain failed. Inspect per-chain wrapper.stdout files." >&2
        exit 1
    fi

    mapfile -t chain_dirs < "$batch_dir/chain_dirs.txt"
    collect_cmd=(
        "$CANDEL_PYTHON" -u "$RUNNER"
        --collect-chain-dirs "${chain_dirs[@]}"
        --output-dir "$batch_dir"
    )
    [[ -n "$PLOT_PARAMS" ]] && collect_cmd+=(--plot-params "$PLOT_PARAMS")

    echo "Collecting Reid chain batch results..."
    "${collect_cmd[@]}"
    exit 0
fi

if [[ -z "$QUEUE" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi

MEM_TOTAL=$((CHAINS * MEM_PER_CHAIN))
JOB_NAME="reid_${GALAXY}_${CHAINS}ch"

echo "Submitting Reid MCMC chains -> $CANDEL_CLUSTER:$QUEUE"
echo "  chains/CPUs:     $CHAINS"
echo "  memory:          ${MEM_TOTAL} GB total (${MEM_PER_CHAIN} GB per chain)"
echo "  trials/chain:    $TRIALS"
echo "  init:            $INIT"
echo "  output root:     $BASE_OUTPUT"

cmd=(
    /bin/bash "$THIS" --worker
    --queue "$QUEUE"
    --chains "$CHAINS"
    --mem-per-chain "$MEM_PER_CHAIN"
    --galaxy "$GALAXY"
    --init "$INIT"
    --reid-init "$REID_INIT"
    --init-npz "$INIT_NPZ"
    --config "$CONFIG"
    --data "$DATA"
    --output-dir "$BASE_OUTPUT"
    --burnin "$BURNIN"
    --trials "$TRIALS"
    --walkers "$WALKERS"
    --seed "$SEED"
    --seed-step "$SEED_STEP"
    --vcor "$VCOR"
    --step-fraction "$STEP_FRACTION"
)
[[ -n "$STATUS_INTERVAL" ]] && cmd+=(--status-interval "$STATUS_INTERVAL")
[[ -n "$TIME" ]] && cmd+=(--time "$TIME")
[[ -n "$H0_LOW" ]] && cmd+=(--h0-low "$H0_LOW")
[[ -n "$H0_HIGH" ]] && cmd+=(--h0-high "$H0_HIGH")
[[ -n "$FIT_DATA" ]] && cmd+=(--fit-data "$FIT_DATA")
[[ -n "$PLOT_PARAMS" ]] && cmd+=(--plot-params "$PLOT_PARAMS")

dry_flag=()
$DRY && dry_flag=(--dry)
time_flag=()
[[ -n "$TIME" ]] && time_flag=(--time "$TIME")

submit_job --queue "$QUEUE" --mem "$MEM_TOTAL" --cpus "$CHAINS" \
    --name "$JOB_NAME" \
    "${time_flag[@]}" \
    "${dry_flag[@]}" \
    -- "${cmd[@]}"
