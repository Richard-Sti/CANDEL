#!/bin/bash -l
# Submit TRGB mock closure test batch jobs. Cluster (arc or glamdring) is
# picked up from `machine` in local_config.toml via _submit_lib.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

queue=""
ncpu=28
memory=7
n_mocks=100
master_seed=0
num_warmup=500
num_samples=1000
num_chains=1
which_selection="TRGB_magnitude"
use_field=true
field_name="Carrick2015"
fix_selection=true
fix_Vext=false
config="$ROOT/scripts/runs/configs/config_EDD_TRGB.toml"
outdir="$ROOT/results/mocks_TRGB"
extra_args=""
local_mode=false
single_mode=false
dry=false
gpu_mode=false
gpu_queues=""
gpu_shards=0
mocks_per_batch=10

safe_tag() {
    local value="$1"
    value="${value//[^A-Za-z0-9_-]/_}"
    printf '%s' "$value"
}

append_queue_slots() {
    local qname="$1" qcount="$2"
    for ((j = 0; j < qcount; j++)); do
        gpu_queue_list+=("$qname")
    done
}

parse_gpu_queue_specs() {
    local specs="$1"
    IFS=',' read -ra gpu_queue_specs <<< "$specs"
    gpu_queue_list=()
    for spec in "${gpu_queue_specs[@]}"; do
        qname="${spec%%:*}"
        qcount=1
        if [[ "$spec" == *:* ]]; then
            qcount="${spec#*:}"
        fi
        if [[ -z "$qname" || ! "$qcount" =~ ^[0-9]+$ || "$qcount" -le 0 ]]; then
            echo "[ERROR] Invalid --gpu-queues entry: $spec" >&2
            exit 1
        fi
        append_queue_slots "$qname" "$qcount"
    done
}

detect_gpu_queue_slots() {
    gpu_queue_list=()
    local gpulong_free=0 cmbgpu_free=0 optgpu_free=0 line queue counts free total
    if command -v showgpus >/dev/null 2>&1; then
        while IFS= read -r line; do
            [[ "$line" =~ (gpulong|cmbgpu|optgpu) ]] || continue
            queue="${BASH_REMATCH[1]}"
            counts=$(grep -oE '\[[[:space:]]*[0-9]+[[:space:]]*/[[:space:]]*[0-9]+[[:space:]]*\]' <<< "$line" | head -n 1 || true)
            [[ -n "$counts" ]] || continue
            free=$(sed -E 's/.*\[ *([0-9]+) *\/ *[0-9]+ *\].*/\1/' <<< "$counts")
            total=$(sed -E 's/.*\[ *[0-9]+ *\/ *([0-9]+) *\].*/\1/' <<< "$counts")
            [[ "$free" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || continue
            case "$queue" in
                gpulong) gpulong_free=$((gpulong_free + free)) ;;
                cmbgpu)  cmbgpu_free=$((cmbgpu_free + free)) ;;
                optgpu)  optgpu_free=$((optgpu_free + free)) ;;
            esac
        done < <(showgpus 2>/dev/null || true)
    elif command -v gpustat >/dev/null 2>&1; then
        while IFS= read -r line; do
            [[ "$line" =~ ^(gpulong|cmbgpu|optgpu)[[:space:]] ]] || continue
            queue="${BASH_REMATCH[1]}"
            counts=$(grep -oE '\[[0-9]+/[0-9]+\]' <<< "$line" | head -n 1 || true)
            [[ -n "$counts" ]] || continue
            free="${counts#\[}"
            free="${free%%/*}"
            total="${counts#*/}"
            total="${total%\]}"
            [[ "$free" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || continue
            case "$queue" in
                gpulong) gpulong_free=$((gpulong_free + free)) ;;
                cmbgpu)  cmbgpu_free=$((cmbgpu_free + free)) ;;
                optgpu)  optgpu_free=$((optgpu_free + free)) ;;
            esac
        done < <(gpustat 2>/dev/null || true)
    fi

    # Always submit at least one GPULONG batch, even if it has to queue.
    append_queue_slots gpulong 1
    if (( cmbgpu_free > 0 )); then
        append_queue_slots cmbgpu 1
    fi
    if (( optgpu_free > 0 )); then
        append_queue_slots optgpu 1
    fi

    if (( gpulong_free > 1 )); then
        append_queue_slots gpulong "$((gpulong_free - 1))"
    fi
    if (( cmbgpu_free > 1 )); then
        append_queue_slots cmbgpu "$((cmbgpu_free - 1))"
    fi
    if (( optgpu_free > 1 )); then
        append_queue_slots optgpu "$((optgpu_free - 1))"
    fi
}

print_injected_parameters() {
    local py="${CANDEL_PYTHON:-python3}"
    "$py" - "$ROOT/candel/mock/TRGB_mock.py" <<'PY' || {
import ast
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=path)

wanted = {"DEFAULT_TRUE_PARAMS", "DEFAULT_ANCHORS"}
found = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)

print("injected defaults:")
for name in ("DEFAULT_TRUE_PARAMS", "DEFAULT_ANCHORS"):
    values = found.get(name, {})
    print(f"  {name}:")
    for key, value in values.items():
        print(f"    {key:<20s} {value}")
PY
        echo "injected defaults: unavailable"
    }
}

usage() {
    cat <<EOF
usage: $(basename "$0") -q QUEUE [-n NCPU] [-m MEMORY] [--n-mocks N]
                        [--master-seed S] [--num-warmup N] [--num-samples N]
                        [--num-chains N] [--rhat-threshold X]
                        [--which-selection NAME] [--config PATH]
                        [--outdir PATH] [--infer-selection] [--no-field]
                        [--field-name NAME] [--single] [--local] [--dry]
                        [--gpu [--gpu-queues QUEUES] [--mocks-per-batch N]]

Submit TRGB mock closure test batch jobs. Runs with MPI (ranks = --ncpu).

defaults:
  Field sampling is ON by default using $field_name.
  Selection thresholds are fixed to the injected truth by default.
  Use --no-field for homogeneous no-reconstruction mocks.
  Use --infer-selection to infer the selection thresholds.

options:
  -q, --queue QUEUE       queue/partition (REQUIRED unless --local).
                          In --gpu mode this is the CPU merge-job queue;
                          GPU shard queues are chosen automatically or by
                          --gpu-queues.
  -n, --ncpu NCPU         MPI ranks (default: $ncpu)
  -m, --memory MEMORY     GB per job (default: $memory)
  --n-mocks N             mocks (default: $n_mocks)
  --master-seed S         master seed (default: $master_seed)
  --num-warmup N          NUTS warmup (default: $num_warmup)
  --num-samples N         NUTS samples (default: $num_samples)
  --num-chains N          NUTS chains per mock (default: $num_chains)
  --rhat-threshold X      warn when NumPyro R-hat exceeds X (default: 1.05)
  --which-selection NAME   TRGB_magnitude or redshift (default: $which_selection)
  --config PATH           base config (default: $config)
  --outdir PATH           output dir (default: $outdir)
  --infer-selection       infer selection thresholds instead of fixing truth
  --fix-selection         fix selection thresholds to truth (default)
  --fix-Vext              fix external velocity to injected truth
  --no-field              disable reconstruction-field sampling
  --use-field             enable reconstruction-field sampling (default)
  --field-name NAME       reconstruction field name (default: $field_name)
  --single                run without MPI
  --plot-only             with --single: generate and plot, skip inference
  --local                 run locally (mpirun / plain python), no submission
  --dry                   print submit command without submitting
  --gpu                   split mocks into independent single-GPU shard jobs
  --gpu-queues QUEUES     comma-separated GPU queues for shard jobs.
                          Use queue:N to weight a queue manually.
                          Default: gpulong plus free cmbgpu/optgpu GPUs.
  --mocks-per-batch N     mocks per GPU shard job (default: $mocks_per_batch)
  --gpu-shards N          advanced: set number of shard jobs directly
  --no-progress-bar       disable NumPyro progress bars in sequential jobs
  -h, --help

EOF
    print_injected_parameters
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)         usage ;;
        -q|--queue)        queue="$2"; shift 2 ;;
        -n|--ncpu)         ncpu="$2"; shift 2 ;;
        -m|--memory)       memory="$2"; shift 2 ;;
        --n-mocks)         n_mocks="$2"; shift 2 ;;
        --master-seed)     master_seed="$2"; shift 2 ;;
        --num-warmup)      num_warmup="$2"; shift 2 ;;
        --num-samples)     num_samples="$2"; shift 2 ;;
        --num-chains)      num_chains="$2"; shift 2 ;;
        --which-selection) which_selection="$2"; shift 2 ;;
        --config)          config="$2"; shift 2 ;;
        --outdir)          outdir="$2"; shift 2 ;;
        --infer-selection) fix_selection=false; shift ;;
        --fix-selection)   fix_selection=true; shift ;;
        --fix-Vext)        fix_Vext=true; shift ;;
        --no-field|--disable-field) use_field=false; shift ;;
        --use-field)       use_field=true; shift ;;
        --field-name)      field_name="$2"; shift 2 ;;
        --single)          single_mode=true; extra_args="$extra_args --single"; shift ;;
        --plot-only)       extra_args="$extra_args --plot-only"; shift ;;
        --local)           local_mode=true; shift ;;
        --dry)             dry=true; shift ;;
        --gpu)             gpu_mode=true; shift ;;
        --gpu-queues)      gpu_queues="$2"; shift 2 ;;
        --gpu-shards)      gpu_shards="$2"; shift 2 ;;
        --mocks-per-batch) mocks_per_batch="$2"; shift 2 ;;
        --no-progress-bar) extra_args="$extra_args --no-progress-bar"; shift ;;
        *)                 extra_args="$extra_args $1"; shift ;;
    esac
done

if ! $local_mode && [[ -z "$queue" ]]; then
    echo "[ERROR] -q QUEUE is required (cluster=$CANDEL_CLUSTER)"; exit 1
fi
if $gpu_mode; then
    if $local_mode; then
        echo "[ERROR] --gpu submission does not support --local"; exit 1
    fi
    if $single_mode; then
        echo "[ERROR] --gpu cannot be combined with --single"; exit 1
    fi
    if ! [[ "$mocks_per_batch" =~ ^[0-9]+$ ]] || [[ "$mocks_per_batch" -le 0 ]]; then
        echo "[ERROR] --mocks-per-batch must be positive"; exit 1
    fi
    if ! [[ "$gpu_shards" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] --gpu-shards must be a non-negative integer"; exit 1
    fi
fi

if ! $gpu_mode && ! $single_mode && [[ $ncpu -gt 1 ]]; then
    if ! "$CANDEL_PYTHON" - <<'PY' >/dev/null 2>&1; then
from mpi4py import MPI
PY
        echo "[ERROR] MPI batch mode requires mpi4py in $CANDEL_PYTHON" >&2
        echo "        Install it before submitting, or use --single / -n 1." >&2
        exit 1
    fi
fi

echo "TRGB mock closure test"
echo "============================================================"
echo "  Cluster:     $CANDEL_CLUSTER"
if $local_mode; then
    echo "  Mode:        LOCAL"
elif $gpu_mode; then
    echo "  Mode:        GPU shards"
else
    echo "  Mode:        SUBMIT (queue=$queue)"
fi
if $gpu_mode; then
    echo "  GPU shard:   1 GPU, 1 CPU core, ${memory} GB each"
    echo "  Merge job:   1 CPU core, ${memory} GB on $queue"
    echo "  GPU queues:  $([[ -n "$gpu_queues" ]] && echo "$gpu_queues" || echo auto)"
    echo "  Batch size:  $mocks_per_batch mocks"
else
    echo "  CPUs/ranks:  $ncpu"
fi
echo "  N_mocks:     $n_mocks"
echo "  Master seed: $master_seed"
echo "  Warmup:      $num_warmup"
echo "  Samples:     $num_samples"
echo "  Chains:      $num_chains"
echo "  Selection:   $which_selection"
echo "  Field:       $use_field"
if $use_field; then
    echo "  Field name:  $field_name"
fi
echo "  Sel params:  $($fix_selection && echo fixed-to-truth || echo inferred)"
echo "  Vext:        $($fix_Vext && echo fixed-to-truth || echo inferred)"
echo "  Config:      $config"
echo "  Output:      $outdir"
[[ -n "$extra_args" ]] && echo "  Extra args: $extra_args"
echo

read -rp "Proceed? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborting."; exit 1
fi

mkdir -p "$outdir"

selection_args=""
$fix_selection && selection_args="--fix-selection"
vext_args=""
$fix_Vext && vext_args="--fix-Vext"
field_args=""
$use_field && field_args="--use-field --field-name $field_name"

pycmd="$CANDEL_PYTHON -u $ROOT/scripts/mocks/mock_TRGB.py \
    --n-mocks $n_mocks \
    --master-seed $master_seed \
    --num-warmup $num_warmup \
    --num-samples $num_samples \
    --num-chains $num_chains \
    --which-selection $which_selection \
    --config $config \
    --outdir $outdir \
    $selection_args \
    $vext_args \
    $field_args \
    $extra_args"

if $gpu_mode; then
    if [[ -n "$gpu_queues" ]]; then
        parse_gpu_queue_specs "$gpu_queues"
    else
        detect_gpu_queue_slots
    fi
    if [[ ${#gpu_queue_list[@]} -eq 0 ]]; then
        echo "[ERROR] No GPU queues parsed from --gpu-queues"; exit 1
    fi
    if [[ $gpu_shards -le 0 ]]; then
        gpu_shards=$(((n_mocks + mocks_per_batch - 1) / mocks_per_batch))
    else
        mocks_per_batch=$(((n_mocks + gpu_shards - 1) / gpu_shards))
    fi

    dry_flag=()
    $dry && dry_flag=(--dry)
    field_tag="nofield"
    $use_field && field_tag="field_$(safe_tag "$field_name")"
    selection_tag="fixedsel"
    $fix_selection || selection_tag="infersel"
    vext_tag=""
    $fix_Vext && vext_tag="_fixedVext"
    mode_tag="$(safe_tag "$which_selection")_${field_tag}_${selection_tag}${vext_tag}"
    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    shard_root="$outdir/gpu_shards_${mode_tag}_seed_${master_seed}_${run_stamp}"
    merge_out="$outdir/mock_TRGB_biases_${mode_tag}_gpu_merged.npz"
    mkdir -p "$shard_root"

    echo "GPU queue slots: ${gpu_queue_list[*]}"
    echo "Shard root: $shard_root"
    echo "Submitting $gpu_shards GPU shard jobs..."
    job_ids=()
    remaining_mocks=$n_mocks
    for ((i = 0; i < gpu_shards; i++)); do
        shard_mocks=$mocks_per_batch
        if (( shard_mocks > remaining_mocks )); then
            shard_mocks=$remaining_mocks
        fi
        remaining_mocks=$((remaining_mocks - shard_mocks))
        shard_queue="${gpu_queue_list[$((i % ${#gpu_queue_list[@]}))]}"
        shard_seed=$((master_seed + i + 1))
        shard_dir="$shard_root/shard_$(printf '%03d' "$i")"
        mkdir -p "$shard_dir"
        shard_cmd="$CANDEL_PYTHON -u $ROOT/scripts/mocks/mock_TRGB.py \
            --n-mocks $shard_mocks \
            --master-seed $shard_seed \
            --num-warmup $num_warmup \
            --num-samples $num_samples \
            --num-chains $num_chains \
            --which-selection $which_selection \
            --config $config \
            --outdir $shard_dir \
            $selection_args \
            $vext_args \
            $field_args \
            $extra_args"
        echo "  shard $i: queue=$shard_queue mocks=$shard_mocks seed=$shard_seed"
        submit_out=$(submit_job --queue "$shard_queue" --mem "$memory" \
            --gpu --cpus 1 \
            --name "mock_TRGB_gpu_$(printf '%03d' "$i")" \
            "${dry_flag[@]}" -- $shard_cmd)
        echo "$submit_out"
        jid=$(echo "$submit_out" | grep -oP 'JOBID=\K[0-9]+' | tail -n 1 || true)
        if [[ -n "$jid" ]]; then
            job_ids+=("$jid")
        fi
    done

    if $dry; then
        echo "[dry] Merge job not submitted because shard job IDs are unavailable."
    else
        if [[ ${#job_ids[@]} -ne $gpu_shards ]]; then
            echo "[ERROR] Expected $gpu_shards shard job IDs, got ${#job_ids[@]}" >&2
            exit 1
        fi
        deps=$(IFS=:; echo "${job_ids[*]}")
        merge_cmd="$CANDEL_PYTHON -u $ROOT/scripts/mocks/merge_mock_TRGB_shards.py \
            $shard_root/shard_*/mock_TRGB_biases_*.npz \
            --out $merge_out \
            --delete-inputs"
        echo "Submitting merge job after $deps..."
        submit_job --queue "$queue" --mem "$memory" --cpus 1 --runafter "$deps" \
            --name "mock_TRGB_merge" --default-log -- $merge_cmd
    fi

    echo
    echo "Done."
    exit 0
fi

if $dry && $local_mode; then
    echo "Dry run command:"
    if $single_mode || [[ $ncpu -eq 1 ]]; then
        echo "$pycmd"
    else
        echo "mpirun -np $ncpu $pycmd"
    fi
    exit 0
fi

if $single_mode || [[ $ncpu -eq 1 ]]; then
    if $local_mode; then
        echo "Running locally without MPI..."
        eval "$pycmd"
    else
        dry_flag=()
        $dry && dry_flag=(--dry)
        submit_job --queue "$queue" --mem "$memory" --cpus 1 \
            --name "mock_TRGB" \
            --default-log \
            "${dry_flag[@]}" -- $pycmd
    fi
elif $local_mode; then
    echo "Running locally with MPI..."
    eval "mpirun -np $ncpu $pycmd"
else
    dry_flag=()
    $dry && dry_flag=(--dry)
    submit_job --queue "$queue" --mem "$memory" --mpi-n "$ncpu" \
        --name "mock_TRGB" \
        --default-log \
        "${dry_flag[@]}" -- $pycmd
fi

echo
echo "Done."
