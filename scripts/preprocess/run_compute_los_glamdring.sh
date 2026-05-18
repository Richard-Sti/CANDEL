#!/bin/bash -l
#
# Submit compute_los.py jobs to the Glamdring queue system.
#
# Machine-specific settings are read from local_config.toml at the project root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"
cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/candel"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$cache_root/matplotlib}"
tmp_root="${TMPDIR:-/tmp}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$tmp_root/candel_numba_${USER:-user}}"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

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


def get_nested(config, path, default=None):
    out = config
    for part in path.split("/"):
        if not isinstance(out, dict) or part not in out:
            return default
        out = out[part]
    return out


def h0_los_job(config, cfg_path):
    if not get_nested(config, "model/use_reconstruction", False):
        return None
    which_run = get_nested(config, "model/which_run", None)
    if which_run == "CH0":
        catalogue = "SH0ES"
        reconstruction = get_nested(config, "io/SH0ES/which_host_los", None)
        los_template = get_nested(config, "io/PV_main/SH0ES/los_file", None)
    elif which_run in {"CCHP", "CCHP_CSP"}:
        catalogue = "CCHP"
        reconstruction = get_nested(
            config, "io/which_host_los",
            get_nested(config, "io/CCHP/which_host_los", None))
        los_template = get_nested(config, "io/CCHP/los_file", None)
    elif which_run in {"EDD_TRGB", "EDD_TRGB_grouped"}:
        catalogue = which_run
        reconstruction = get_nested(
            config, "io/which_host_los",
            get_nested(
                config,
                f"io/PV_main/{which_run}/which_host_los", None))
        los_template = get_nested(
            config, f"io/PV_main/{which_run}/los_file", None)
    else:
        return None

    if reconstruction is None or los_template is None:
        raise KeyError(
            f"{cfg_path}: no LOS reconstruction/template for {which_run}")
    return catalogue, reconstruction, los_template


def iter_los_jobs(config, cfg_path):
    kind = get_nested(config, "pv_model/kind", "")
    if isinstance(kind, str) and kind.startswith("precomputed_los_"):
        reconstruction = kind.removeprefix("precomputed_los_")
        catalogues = config["io"]["catalogue_name"]
        if isinstance(catalogues, str):
            catalogues = [catalogues]
        for catalogue in catalogues:
            io_section = config["io"].get(catalogue)
            if io_section is None or "los_file" not in io_section:
                raise KeyError(
                    f"{cfg_path}: no io.{catalogue}.los_file section")
            yield catalogue, reconstruction, io_section["los_file"]
        return

    h0_job = h0_los_job(config, cfg_path)
    if h0_job is not None:
        yield h0_job


seen = {}
for idx, cfg_path in task_rows(task_file):
    config = tomllib.loads(cfg_path.read_text())
    for catalogue, reconstruction, los_template in iter_los_jobs(
            config, cfg_path):
        los_path = los_template.replace("<X>", reconstruction)
        catalogue_compute = compute_catalogue(catalogue, los_template)
        key = (catalogue_compute, reconstruction, los_path)
        seen.setdefault(key, cfg_path)

for (catalogue, reconstruction, los_path), cfg_path in sorted(seen.items()):
    print(f"{catalogue}\t{reconstruction}\t{cfg_path}\t{los_path}")
PY
}

los_grid_values() {
    local grid_config="$1"
    "$python_exec" - "$grid_config" <<'PY'
import sys
import tomllib
from pathlib import Path


def deep_merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as f:
        config = tomllib.load(f)

    base_paths = config.pop("base", None)
    if base_paths is None:
        return config
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    merged = {}
    for base in base_paths:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = deep_merge(merged, load_config(base_path))
    return deep_merge(merged, config)


config = load_config(sys.argv[1])
grid = config["io"]["reconstruction_main"]
print(f"{grid['rmin']}\t{grid['rmax']}\t{grid['num_steps']}")
PY
}

los_file_status() {
    local path="$1"
    local rmin="$2"
    local rmax="$3"
    local num_steps="$4"
    "$python_exec" - "$path" "$rmin" "$rmax" "$num_steps" <<'PY'
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
task_file=""
task_ids=""
dry=false
force_los=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [TASK_FILE]

Submit compute_los.py jobs to the Glamdring queue for LOS files required by a
task file.

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
        --task-file)           task_file="$2"; shift 2 ;;
        --tasks)               task_ids="$2"; shift 2 ;;
        --force-los)           force_los=true; shift ;;
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

# Resolve python_exec from config / local_config.toml
python_exec=$(get_toml_key "python_exec" "$config")

if [[ -z "$python_exec" ]]; then
    python_exec="$CANDEL_PYTHON"
fi

# ---- submit jobs ----
job_catalogues=()
job_reconstructions=()
job_configs=()
job_los_paths=()
job_grid_displays=()
skipped_los=()
stale_los=()

if [[ ! -f "$task_file" ]]; then
    echo "[ERROR] Task file not found: $task_file" >&2
    exit 1
fi
while IFS=$'\t' read -r catalogue reconstruction_i config_i los_path; do
    [[ -z "$catalogue" ]] && continue
    IFS=$'\t' read -r los_rmin los_rmax los_num_steps < <(
        los_grid_values "$config_i")
    los_grid_display="r=[$los_rmin, $los_rmax] Mpc/h, num_steps=$los_num_steps"
    los_check_path="$los_path"
    if [[ "$los_check_path" != /* ]]; then
        los_check_path="$ROOT/$los_check_path"
    fi
    if [[ -f "$los_check_path" && $force_los == false ]]; then
        if status=$(
            los_file_status "$los_check_path" \
                "$los_rmin" "$los_rmax" "$los_num_steps"); then
            skipped_los+=(
                "$catalogue / $reconstruction_i -> $los_path ($los_grid_display)")
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
    job_grid_displays+=("$los_grid_display")
done < <(infer_los_jobs "$task_file" "$task_ids")

echo "LOS compute plan:"
if [[ ${#job_catalogues[@]} -gt 0 ]]; then
    for i in "${!job_catalogues[@]}"; do
        echo "  - ${job_catalogues[$i]} / ${job_reconstructions[$i]}"
        if [[ -n "${job_los_paths[$i]}" ]]; then
            echo "      ${job_los_paths[$i]}"
        fi
        echo "      ${job_grid_displays[$i]}"
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
    exit 0
fi

echo "Settings:"
echo "  Queue: $queue"
echo "  CPUs: $ncpu"
echo "  Memory: ${memory} GB"
echo "  Python: $python_exec"
echo "  Default config: $config"
echo "  Force LOS recompute: $force_los"
echo

if [[ -t 0 && $dry == false ]]; then
    read -p "Submit these jobs? [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

dry_flag=()
$dry && dry_flag=(--dry)
for i in "${!job_catalogues[@]}"; do
    catalogue="${job_catalogues[$i]}"
    reconstruction_i="${job_reconstructions[$i]}"
    config_i="${job_configs[$i]}"
    cmd=(
        env
        "MPLCONFIGDIR=$MPLCONFIGDIR"
        "NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR"
        "$python_exec" -u "$ROOT/scripts/preprocess/compute_los.py"
        --catalogue "$catalogue"
        --reconstruction "$reconstruction_i"
        --config "$config_i"
        --smooth_target "$smooth_target"
    )

    echo "Submitting catalogue: $catalogue / $reconstruction_i"
    echo "  LOS grid: ${job_grid_displays[$i]}"
    submit_out=$(
        submit_job --queue "$queue" --mem "$memory" --mpi-n "$ncpu" \
            --name "compute_los_${catalogue}_${reconstruction_i}" \
            "${dry_flag[@]}" -- "${cmd[@]}" 2>&1
    )
    echo "$submit_out"
    echo
done
