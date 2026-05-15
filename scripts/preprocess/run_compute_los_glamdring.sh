#!/bin/bash -l
#
# Submit compute_los.py jobs to the Glamdring queue system and optionally
# warm the matching field cache afterwards.
#
# Machine-specific settings (python_exec, root_main) are read from
# local_config.toml at the project root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"
tmp_root="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$tmp_root/candel_mpl_${USER:-user}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$tmp_root/candel_numba_${USER:-user}}"

# Extract a TOML key from a config file, with fallback to local_config.toml
get_toml_key() {
    local key="$1"
    local config="$2"
    local val
    val=$(grep -E "^${key} *= *" "$config" 2>/dev/null | sed -E "s/^${key} *= *\"([^\"]+)\"$/\1/")
    if [[ -z "$val" ]]; then
        local local_config
        local_config="$(cd "$(dirname "$0")/../.." && pwd)/local_config.toml"
        val=$(grep -E "^${key} *= *" "$local_config" 2>/dev/null | sed -E "s/^${key} *= *\"([^\"]+)\"$/\1/")
    fi
    echo "$val"
}

resolve_preprocess_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    elif [[ -e "$path" ]]; then
        (cd "$(dirname "$path")" && printf '%s/%s\n' "$PWD" "$(basename "$path")")
    else
        echo "$ROOT/scripts/preprocess/$path"
    fi
}

looks_like_task_file() {
    local path="$1"
    [[ "$(basename "$path")" == tasks_*.txt ]]
}

infer_los_jobs() {
    local task_file="$1"
    local tasks="${2:-}"
    "$python_exec" - "$task_file" "$tasks" "$ROOT" <<'PY'
import re
import sys
import tomllib
from pathlib import Path

task_file = Path(sys.argv[1]).resolve()
task_spec = sys.argv[2]
root = Path(sys.argv[3]).resolve()


def parse_tasks(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def task_rows(path):
    wanted = parse_tasks(task_spec)
    for fallback_idx, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and re.fullmatch(r"\d+", parts[0]):
            idx = int(parts[0])
            rel = parts[1]
        else:
            idx = fallback_idx
            rel = parts[0]
        if wanted is not None and idx not in wanted:
            continue
        cfg = Path(rel)
        if not cfg.is_absolute():
            cfg = root / cfg
        yield idx, cfg.resolve()


def compute_catalogue(catalogue, los_template):
    if catalogue in {"CF4_W1", "CF4_i"}:
        return "CF4"
    if "los_CF4_TFR_" in los_template:
        return "CF4"
    return catalogue


seen = {}
for idx, cfg_path in task_rows(task_file):
    config = tomllib.loads(cfg_path.read_text())
    kind = config["pv_model"]["kind"]
    if not kind.startswith("precomputed_los_"):
        continue
    reconstruction = kind.removeprefix("precomputed_los_")
    catalogues = config["io"]["catalogue_name"]
    if isinstance(catalogues, str):
        catalogues = [catalogues]
    for catalogue in catalogues:
        io_section = config["io"].get(catalogue)
        if io_section is None or "los_file" not in io_section:
            raise KeyError(
                f"{cfg_path}: no io.{catalogue}.los_file section")
        los_template = io_section["los_file"]
        los_path = los_template.replace("<X>", reconstruction)
        catalogue_compute = compute_catalogue(catalogue, los_template)
        key = (catalogue_compute, reconstruction, los_path)
        seen.setdefault(key, cfg_path)

for (catalogue, reconstruction, los_path), cfg_path in sorted(seen.items()):
    print(f"{catalogue}\t{reconstruction}\t{cfg_path}\t{los_path}")
PY
}

los_grid_summary() {
    "$python_exec" - "$config" <<'PY'
import sys
import candel

config = candel.load_config(sys.argv[1])
grid = config["io"]["reconstruction_main"]
print(
    f"r=[{grid['rmin']}, {grid['rmax']}] Mpc/h, "
    f"num_steps={grid['num_steps']}")
PY
}

los_grid_values() {
    "$python_exec" - "$config" <<'PY'
import sys
import candel

config = candel.load_config(sys.argv[1])
grid = config["io"]["reconstruction_main"]
print(f"{grid['rmin']}\t{grid['rmax']}\t{grid['num_steps']}")
PY
}

los_file_status() {
    local path="$1"
    "$python_exec" - "$path" "$los_rmin" "$los_rmax" "$los_num_steps" <<'PY'
import sys
from pathlib import Path

import h5py
import numpy as np

path = Path(sys.argv[1])
rmin = float(sys.argv[2])
rmax = float(sys.argv[3])
num_steps = int(sys.argv[4])
expected = np.linspace(rmin, rmax, num_steps)

if not path.exists():
    print("missing")
    sys.exit(1)

try:
    with h5py.File(path, "r") as f:
        r = f["r"][:]
except Exception as exc:
    print(f"unreadable: {exc}")
    sys.exit(1)

if r.shape != expected.shape or not np.allclose(
        r, expected, rtol=0.0, atol=1e-5):
    print(
        f"grid mismatch: found r=[{r[0]:.6g}, {r[-1]:.6g}], "
        f"n={len(r)}; expected r=[{expected[0]:.6g}, "
        f"{expected[-1]:.6g}], n={len(expected)}")
    sys.exit(1)

print("ok")
PY
}

# ---- defaults ----
config="../runs/configs/config.toml"
queue="cmb"
ncpu=10
memory=32
smooth_target=0
warm_cache=true
task_file=""
task_ids=""
cache_queue=""
cache_ncpu=""
cache_memory=""
dry=false
force_los=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [TASK_FILE]

Submit compute_los.py jobs to the Glamdring queue for LOS files required by a
task file. By default, also submit a field-cache warmup job for the same task
file.

One task file is required, either positionally or via --task-file.

Options:
  -c, --config PATH           LOS TOML config with fixed radial grid
                              (default: $config)
  -q, --queue NAME            Queue name (default: $queue)
  -n, --ncpu N                Number of CPUs per job (default: $ncpu)
  -m, --memory GB             Memory per job in GB (default: $memory)
  -s, --smooth-target VALUE   Smoothing target (default: $smooth_target)
      --task-file PATH        Task file to inspect
      --tasks IDS             Task ids/ranges to inspect, e.g. 0,3-5
      --force-los             Submit LOS jobs even when output files exist
      --cache-queue NAME      Cache warmup queue (default: LOS queue)
      --cache-ncpu N          Cache warmup MPI ranks (default: LOS CPUs)
      --cache-memory GB       Cache warmup memory per rank (default: LOS memory)
      --no-warm-cache         Only submit LOS jobs
      --dry                   Print submission commands without submitting
  -h, --help                  Show this help message

Examples:
  $(basename "$0") ../runs/tasks_VFO.txt
  $(basename "$0") --tasks 0-5 ../runs/tasks_VFO.txt
EOF
    exit 0
}

# ---- parse arguments ----
inputs=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)             usage ;;
        -c|--config)           config="$2"; shift 2 ;;
        -q|--queue)            queue="$2"; shift 2 ;;
        -n|--ncpu)             ncpu="$2"; shift 2 ;;
        -m|--memory)           memory="$2"; shift 2 ;;
        -s|--smooth-target)    smooth_target="$2"; shift 2 ;;
        --task-file|--cache-task-file) task_file="$2"; shift 2 ;;
        --tasks)               task_ids="$2"; shift 2 ;;
        --force-los)           force_los=true; shift ;;
        --cache-queue)         cache_queue="$2"; shift 2 ;;
        --cache-ncpu)          cache_ncpu="$2"; shift 2 ;;
        --cache-memory)        cache_memory="$2"; shift 2 ;;
        --no-warm-cache)       warm_cache=false; shift ;;
        --dry)                 dry=true; shift ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            echo "Run $(basename "$0") --help for usage." >&2
            exit 1 ;;
        *)  inputs+=("$1"); shift ;;
    esac
done

config="$(resolve_preprocess_path "$config")"
if [[ ${#inputs[@]} -gt 1 ]]; then
    echo "[ERROR] Expected at most one positional argument: TASK_FILE." >&2
    exit 1
elif [[ ${#inputs[@]} -eq 1 ]]; then
    if [[ -n "$task_file" ]]; then
        echo "[ERROR] Provide TASK_FILE either positionally or with "
        echo "        --task-file, not both." >&2
        exit 1
    fi
    task_file="${inputs[0]}"
fi
if [[ -z "$task_file" ]]; then
    echo "[ERROR] A task file is required." >&2
    echo "        Example: $(basename "$0") ../runs/tasks_VFO.txt" >&2
    exit 1
fi
task_file="$(resolve_preprocess_path "$task_file")"
if ! looks_like_task_file "$task_file"; then
    echo "[ERROR] Expected a task list named tasks_*.txt, got: $task_file" >&2
    exit 1
fi

# Resolve python_exec and root_main from config / local_config.toml
python_exec=$(get_toml_key "python_exec" "$config")
root_main=$(get_toml_key "root_main" "$config")

if [[ -z "$python_exec" ]]; then
    python_exec="$CANDEL_PYTHON"
fi

if [[ -z "$cache_queue" ]]; then
    cache_queue="$queue"
fi
if [[ -z "$cache_ncpu" ]]; then
    cache_ncpu="$ncpu"
fi
if [[ -z "$cache_memory" ]]; then
    cache_memory="$memory"
fi

los_grid_display="$(los_grid_summary)"
IFS=$'\t' read -r los_rmin los_rmax los_num_steps < <(los_grid_values)

# ---- submit jobs ----
job_catalogues=()
job_reconstructions=()
job_configs=()
job_los_paths=()
skipped_los=()
stale_los=()

if [[ ! -f "$task_file" ]]; then
    echo "[ERROR] Task file not found: $task_file" >&2
    exit 1
fi
while IFS=$'\t' read -r catalogue reconstruction_i config_i los_path; do
    [[ -z "$catalogue" ]] && continue
    los_check_path="$los_path"
    if [[ "$los_check_path" != /* ]]; then
        los_check_path="$ROOT/$los_check_path"
    fi
    if [[ -f "$los_check_path" && $force_los == false ]]; then
        if status=$(los_file_status "$los_check_path"); then
            skipped_los+=("$catalogue / $reconstruction_i -> $los_path")
            continue
        else
            stale_los+=(
                "$catalogue / $reconstruction_i -> $los_path ($status)")
        fi
    fi
    job_catalogues+=("$catalogue")
    job_reconstructions+=("$reconstruction_i")
    job_configs+=("$config_i")
    job_los_paths+=("$los_path")
done < <(infer_los_jobs "$task_file" "$task_ids")

echo "LOS compute plan:"
if [[ ${#job_catalogues[@]} -gt 0 ]]; then
    for i in "${!job_catalogues[@]}"; do
        echo "  - ${job_catalogues[$i]} / ${job_reconstructions[$i]}"
        if [[ -n "${job_los_paths[$i]}" ]]; then
            echo "      ${job_los_paths[$i]}"
        fi
    done
else
    echo "  none"
fi
if [[ ${#skipped_los[@]} -gt 0 ]]; then
    echo "Existing LOS files skipped:"
    for item in "${skipped_los[@]}"; do
        echo "  - $item"
    done
fi
if [[ ${#stale_los[@]} -gt 0 ]]; then
    echo "Existing LOS files to recompute:"
    for item in "${stale_los[@]}"; do
        echo "  - $item"
    done
fi
echo

if [[ ${#job_catalogues[@]} -eq 0 ]]; then
    echo "No LOS jobs to submit."
    if ! $warm_cache; then
        exit 0
    fi
fi

echo "Settings:"
echo "  Queue: $queue"
echo "  CPUs: $ncpu"
echo "  Memory: ${memory} GB"
echo "  Python: $python_exec"
echo "  LOS config: $config"
echo "  LOS grid: $los_grid_display"
echo "  Force LOS recompute: $force_los"
echo "  Warm cache: $warm_cache"
if $warm_cache; then
    echo "  Cache task file: $task_file"
    [[ -n "$task_ids" ]] && echo "  Cache task ids: $task_ids"
    echo "  Cache queue: $cache_queue"
    echo "  Cache CPUs: $cache_ncpu"
    echo "  Cache memory: ${cache_memory} GB"
fi
echo

if [[ -t 0 && $dry == false ]]; then
    read -p "Submit these jobs? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

jobids=()
dry_flag=()
$dry && dry_flag=(--dry)
for i in "${!job_catalogues[@]}"; do
    catalogue="${job_catalogues[$i]}"
    reconstruction_i="${job_reconstructions[$i]}"
    config_i="${job_configs[$i]}"
    cmd=(
        "$python_exec" -u "$ROOT/scripts/preprocess/compute_los.py"
        --catalogue "$catalogue"
        --reconstruction "$reconstruction_i"
        --config "$config_i"
        --smooth_target "$smooth_target"
    )

    echo "Submitting catalogue: $catalogue / $reconstruction_i"
    submit_out=$(
        submit_job --queue "$queue" --mem "$memory" --mpi-n "$ncpu" \
            --name "compute_los_${catalogue}_${reconstruction_i}" \
            "${dry_flag[@]}" -- "${cmd[@]}" 2>&1
    )
    echo "$submit_out"
    jobid=$(
        echo "$submit_out" \
            | grep -oE 'JOBID=[0-9]+' \
            | tail -1 \
            | cut -d= -f2 \
            || true
    )
    if [[ -n "$jobid" ]]; then
        jobids+=("$jobid")
    fi
    echo
done

if $warm_cache; then
    if [[ ! -f "$task_file" ]]; then
        echo "[ERROR] Cache task file not found: $task_file" >&2
        exit 1
    fi

    cache_cmd=(
        "$python_exec" -u "$ROOT/scripts/preprocess/warm_field_cache.py"
        "$task_file"
    )
    if [[ -n "$task_ids" ]]; then
        cache_cmd+=(--tasks "$task_ids")
    fi
    runafter_args=()
    if [[ ${#jobids[@]} -gt 0 ]]; then
        runafter=$(IFS=:; echo "${jobids[*]}")
        runafter_args=(--runafter "$runafter")
    elif [[ ${#job_catalogues[@]} -gt 0 ]] && ! $dry; then
        echo "[WARN] Could not parse LOS job IDs; submitting cache warmup without dependency." >&2
    fi

    echo "Submitting field-cache warmup:"
    echo "  CANDEL_FIELD_CACHE_MPI=1"
    export CANDEL_FIELD_CACHE_MPI=1
    submit_job --queue "$cache_queue" --mem "$cache_memory" \
        --mpi-n "$cache_ncpu" --name "warm_field_cache_VFO" \
        "${runafter_args[@]}" "${dry_flag[@]}" -- "${cache_cmd[@]}"
fi
