#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PYTHON="${CANDEL_CONFIG_PYTHON:-$ROOT_DIR/venv_candel/bin/python}"
if [[ ! -x "$CONFIG_PYTHON" ]]; then
  CONFIG_PYTHON="${PYTHON:-python3}"
fi
CONFIG_EXPORTS="$(
  "$CONFIG_PYTHON" "$SCRIPT_DIR/borg_field_config.py" \
    --shell-env borg_python borg_forward cosmotool_sph plot_python srun \
      chain_name run_dir schedule field_output_dir
)"
eval "$CONFIG_EXPORTS"
RUN_PYTHON="${CANDEL_RUN_PYTHON:-$PYTHON_PATH}"

usage() {
  cat <<'EOF'
Usage:
  run_borg_fields.sh MCMC_FILE [run options]                         # submits to berg
  run_borg_fields.sh --steps [STEPS] [schedule options] [run options] # submits to berg
  run_borg_fields.sh -locally MCMC_FILE [run options]
  run_borg_fields.sh -locally --steps [STEPS] [schedule options] [run options]
  run_borg_fields.sh -locally validate-rsd CHAIN_DIR [validation options]
  run_borg_fields.sh --submit [submit options] MCMC_FILE [run options]
  run_borg_fields.sh --submit [submit options] --steps [STEPS] [schedule options] [run options]
  run_borg_fields.sh --submit [submit options] validate-rsd CHAIN_DIR [validation options]

Default behaviour:
  Without -locally, this wrapper submits one berg-node job with 28 CPUs.
  Field production is real-space unless --rsd is passed.
  Use -locally or --locally to run in the current shell instead.

Schedule options:
  --steps [STEPS]       Inclusive schedule steps, e.g. 0:50, 0-50, 0,5,10:12. Default if omitted after --steps: 0:50
  --schedule FILE       Default: active [borg_fields.chains.<name>].schedule
  --borg-run-dir DIR    Default: active [borg_fields.chains.<name>].run_dir
  --field-output-dir DIR
                         Default: active [borg_fields.chains.<name>].field_output_dir
  --pm-nsteps N         Override [gravity_chain_2] pm_nsteps. Default: 10

Submit options:
  -q, --queue QUEUE      Default: berg
  -n, --nprocs N         BORG MPI ranks. Default: 28
  --omp-threads N       BORG OpenMP threads per MPI rank. Default: 1
  --mem-gb GB           Default: 7
  --nodes N             Split --steps over N single-node jobs. Default: 1
  --name NAME           Default: borg-fields
  --log-dir DIR         Default: this directory
  --submit-dry-run      Print addqueue command without submitting

Pylians CIC gridding runs after BORG by default. Use --mas cic-borg for
BORG's CIC density/velocity output; this passes --vfield to borg_forward.
Use --mas pcs for Pylians PCS or --mas sph for CosmoTool SPH. The Python
runner uses RUN_PYTHON (default: the configured CANDEL Python) so MAS_library
is checked before BORG starts. SPH uses SPH_OMP_THREADS; on berg this defaults
to 28, so the SPH stage uses the whole node unless SPH_OMP_THREADS is set.
Pass --keep-particles to keep /<mode>/u_pos and /<mode>/u_vel in the packed
product and skip final slimming.
Slim final products expose /overdensity, /velocity, /vx, /vy, and /vz. They
also store BORG-reader attrs: Om from the MCMC cosmology, boxsize from the
BORG forward scalars, observer_position as the box centre, and frame=icrs.
Slice plots are optional diagnostics; pass --plots to write them to
PRODUCT_PARENT/plots.
BORG chain-specific defaults are read from local_config.toml.
For --steps runs, packed products are written to
BORG_FIELD_OUTPUT_DIR/<MAS>/mcmc_<schedule-step>.hdf5.
For direct MCMC runs, the default packed product is
BORG_FIELD_OUTPUT_DIR/<MAS>/mcmc_<iteration>.hdf5 unless --single-output is
passed. Pass multiple MAS values, e.g. --mas cic pcs, to write multiple
products from one BORG run.
For --steps runs, one requested sample is also run in RSD mode for validation
against /scalars/BORG_final_density, including a non-fatal Pylians
cross-correlation check.

Examples:
  run_borg_fields.sh --steps 0:2 --real-space --state state_6124
  run_borg_fields.sh --nodes 5 --steps 0:9
  run_borg_fields.sh --steps 0:2 --real-space --mas sph
  run_borg_fields.sh --steps 0:2 --real-space --mas pcs --keep-particles
  run_borg_fields.sh --steps 0:2 --real-space --mas cic-borg
  run_borg_fields.sh -locally MCMC.h5 --real-space --state state_6124
  run_borg_fields.sh --submit --steps 0:2 --real-space --state state_6124
  run_borg_fields.sh --submit MCMC.h5 --real-space --include-rsd --state state_6124
  run_borg_fields.sh --submit MCMC.h5 --real-space --state state_6124
  run_borg_fields.sh --submit validate-rsd /path/to/chain/l1_e_b000 --samples 1 --state state_6124
EOF
}

quote_cmd() {
  local quoted=()
  local arg
  for arg in "$@"; do
    quoted+=("$(printf "%q" "$arg")")
  done
  printf "%s\n" "${quoted[*]}"
}

has_help_arg() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        return 0
        ;;
    esac
    shift
  done
  return 1
}

has_steps_arg() {
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--steps" ]]; then
      return 0
    fi
    shift
  done
  return 1
}

has_option_arg() {
  local option="$1"
  shift
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "$option" || "$1" == "$option="* ]]; then
      return 0
    fi
    shift
  done
  return 1
}

normalize_run_args() {
  NORMALIZED_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n)
        NORMALIZED_ARGS+=(--nprocs "$2")
        shift 2
        ;;
      *)
        NORMALIZED_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

get_steps_arg() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --steps)
        if [[ $# -gt 1 && "${2:-}" != --* ]]; then
          printf "%s\n" "$2"
        else
          printf "0:50\n"
        fi
        return 0
        ;;
    esac
    shift
  done
  return 1
}

split_step_specs() {
  local nodes="$1"
  local steps="$2"
  "$BORG_PYTHON" - "$nodes" "$steps" <<'PY'
from __future__ import annotations

import sys


def parse_steps(spec: str) -> list[int]:
    steps: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, stop = item.split(":", 1)
        elif "-" in item:
            start, stop = item.split("-", 1)
        else:
            steps.add(int(item))
            continue
        start_i = int(start)
        stop_i = int(stop)
        if stop_i < start_i:
            raise SystemExit(f"Invalid descending step range: {item}")
        steps.update(range(start_i, stop_i + 1))
    if not steps:
        raise SystemExit("No schedule steps were requested.")
    return sorted(steps)


nodes = int(sys.argv[1])
if nodes < 1:
    raise SystemExit("--nodes must be >= 1")
steps = parse_steps(sys.argv[2])
nchunks = min(nodes, len(steps))
chunk_size, remainder = divmod(len(steps), nchunks)
start = 0
for i in range(nchunks):
    stop = start + chunk_size + (1 if i < remainder else 0)
    chunk = steps[start:stop]
    start = stop
    print(",".join(str(step) for step in chunk))
PY
}

replace_steps_arg() {
  local new_steps="$1"
  shift
  CHUNK_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --steps)
        CHUNK_ARGS+=("--steps" "$new_steps")
        if [[ $# -gt 1 && "${2:-}" != --* ]]; then
          shift 2
        else
          shift
        fi
        ;;
      *)
        CHUNK_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

resolve_step_targets() {
  local steps="$1"
  local borg_run_dir="$2"
  local schedule="$3"
  "$BORG_PYTHON" - "$steps" "$borg_run_dir" "$schedule" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_steps(spec: str) -> list[int]:
    steps: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, stop = item.split(":", 1)
        elif "-" in item:
            start, stop = item.split("-", 1)
        else:
            steps.add(int(item))
            continue
        start_i = int(start)
        stop_i = int(stop)
        if stop_i < start_i:
            raise SystemExit(f"Invalid descending step range: {item}")
        steps.update(range(start_i, stop_i + 1))
    if not steps:
        raise SystemExit("No schedule steps were requested.")
    return sorted(steps)


def read_schedule(path: Path) -> dict[int, tuple[str, int]]:
    step_re = re.compile(r"^(\d+):\s*$")
    subchain_re = re.compile(r"^\s{2}([^:\s]+):\s*$")
    mcmc_re = re.compile(r"^\s{4}mcmc_step:\s*(\d+)\s*$")
    schedule: dict[int, tuple[str, int]] = {}
    current_step: int | None = None
    current_subchain: str | None = None

    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        match = step_re.match(raw_line)
        if match:
            current_step = int(match.group(1))
            current_subchain = None
            continue
        match = subchain_re.match(raw_line)
        if match and current_step is not None:
            current_subchain = match.group(1)
            continue
        match = mcmc_re.match(raw_line)
        if match and current_step is not None and current_subchain is not None:
            schedule[current_step] = (current_subchain, int(match.group(1)))
            continue
        raise SystemExit(f"Could not parse {path}:{line_no}: {raw_line!r}")
    return schedule


steps_spec = sys.argv[1]
borg_run_dir = Path(sys.argv[2]).expanduser().resolve()
schedule_path = (
    Path(sys.argv[3]).expanduser().resolve()
    if sys.argv[3]
    else borg_run_dir / "schedule_final.yaml"
)
schedule = read_schedule(schedule_path)
steps = parse_steps(steps_spec)
missing = [step for step in steps if step not in schedule]
if missing:
    raise SystemExit(f"Schedule is missing requested steps {missing}: {schedule_path}")

for step in steps:
    subchain, mcmc = schedule[step]
    path = borg_run_dir / "chain" / subchain / f"mcmc_{mcmc}.h5"
    print(f"{step}\t{path}")
PY
}

run_schedule_steps() {
  local steps=""
  local schedule="$BORG_SCHEDULE"
  local borg_run_dir="$BORG_RUN_DIR"
  local field_output_dir="$BORG_FIELD_OUTPUT_DIR"
  local -a run_args=()
  local -a validation_args=()
  local explicit_single_output="false"

  if [[ "${1:-}" == "run" ]]; then
    shift
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --steps)
        if [[ $# -gt 1 && "${2:-}" != --* ]]; then
          steps="$2"
          shift 2
        else
          steps="0:50"
          shift
        fi
        ;;
      --schedule)
        schedule="$2"
        shift 2
        ;;
      --borg-run-dir)
        borg_run_dir="$2"
        shift 2
        ;;
      --field-output-dir)
        field_output_dir="$2"
        run_args+=("$1" "$2")
        shift 2
        ;;
      -n)
        run_args+=("--nprocs" "$2")
        validation_args+=("--nprocs" "$2")
        shift 2
        ;;
      --params|--state|--output-root|--borg-forward|--mpirun|--mpi-launcher|--nprocs|--omp-threads|--plot-python|--pm-nsteps|--output-index)
        run_args+=("$1" "$2")
        validation_args+=("$1" "$2")
        shift 2
        ;;
      --rsd-comparison-plot-dir|--rsd-comparison-plot-axis|--rsd-comparison-plot-index|--rsd-comparison-plot-script|--rsd-cross-output-dir|--rsd-cross-boxsize|--rsd-cross-axis|--rsd-cross-threads|--rsd-cross-kmax|--rsd-cross-min-mean-r|--rsd-cross-script)
        validation_args+=("$1" "$2")
        shift 2
        ;;
      --no-rsd-comparison-plot|--no-rsd-cross-correlation)
        validation_args+=("$1")
        shift
        ;;
      --single-output)
        explicit_single_output="true"
        run_args+=("$1" "$2")
        shift 2
        ;;
      --dry-run|--no-patch-missing-vobs)
        run_args+=("$1")
        validation_args+=("$1")
        shift
        ;;
      *)
        run_args+=("$1")
        shift
        ;;
    esac
  done

  if [[ -z "$steps" ]]; then
    return 42
  fi

  local -a targets=()
  mapfile -t targets < <(resolve_step_targets "$steps" "$borg_run_dir" "$schedule")
  echo "Resolved ${#targets[@]} schedule step(s)."
  local target step mcmc
  local status=0
  local -a step_run_args
  for target in "${targets[@]}"; do
    step="${target%%$'\t'*}"
    mcmc="${target#*$'\t'}"
    echo "Running schedule step ${step}: ${mcmc}"
    step_run_args=("${run_args[@]}")
    if [[ "$explicit_single_output" != "true" ]]; then
      step_run_args+=("--output-index" "$step")
    fi
    if ! "$SCRIPT_DIR/run_borg_fields.sh" -locally "$mcmc" "${step_run_args[@]}"; then
      status=1
    fi
  done

  if [[ "$status" -eq 0 ]]; then
    local validation_target validation_step validation_mcmc validation_chain validation_glob
    validation_target="$(
      printf "%s\n" "${targets[@]}" | "$BORG_PYTHON" -c 'import random, sys; lines=[line.strip() for line in sys.stdin if line.strip()]; random.seed(12345); print(random.choice(lines))'
    )"
    validation_step="${validation_target%%$'\t'*}"
    validation_mcmc="${validation_target#*$'\t'}"
    validation_chain="$(dirname "$validation_mcmc")"
    validation_glob="$(basename "$validation_mcmc")"
    echo "Running RSD validation for schedule step ${validation_step}: ${validation_mcmc}"
    if ! "$SCRIPT_DIR/run_borg_fields.sh" -locally validate-rsd "$validation_chain" --glob "$validation_glob" --samples 1 "${validation_args[@]}"; then
      status=1
    fi
  fi
  return "$status"
}

if [[ $# -lt 1 ]] || has_help_arg "$@"; then
  usage
  exit 0
fi

LOCAL_RUN="false"
if [[ "${1:-}" == "-locally" || "${1:-}" == "--locally" ]]; then
  LOCAL_RUN="true"
  shift
fi
if [[ $# -lt 1 ]] || has_help_arg "$@"; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--submit" ]]; then
  shift
  QUEUE="berg"
  NPROCS="${NPROCS:-28}"
  OMP_THREADS="${OMP_NUM_THREADS:-1}"
  MEM_GB="7"
  NODES="1"
  NAME="borg-fields"
  LOG_DIR="$SCRIPT_DIR"
  SUBMIT_DRY_RUN="false"
  RUN_ARGS=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -q|--queue) QUEUE="$2"; shift 2 ;;
      -n|--nprocs) NPROCS="$2"; shift 2 ;;
      --omp-threads) OMP_THREADS="$2"; shift 2 ;;
      --mem-gb) MEM_GB="$2"; shift 2 ;;
      --nodes) NODES="$2"; shift 2 ;;
      --name) NAME="$2"; shift 2 ;;
      --log-dir) LOG_DIR="$2"; shift 2 ;;
      --submit-dry-run) SUBMIT_DRY_RUN="true"; shift ;;
      --) shift; RUN_ARGS+=("$@"); break ;;
      *) RUN_ARGS+=("$1"); shift ;;
    esac
  done

  if [[ "${#RUN_ARGS[@]}" -lt 1 ]]; then
    usage >&2
    exit 2
  fi

  if [[ ! -x "$SRUN" ]]; then
    SRUN="srun"
  fi
  mkdir -p "$LOG_DIR"
  LOG_DIR="$(cd "$LOG_DIR" && pwd)"
  export BORG_PYTHON BORG_FORWARD PYTHON_PATH RUN_PYTHON NPROCS
  export BORG_OMP_THREADS="$OMP_THREADS"
  export OMP_NUM_THREADS="$OMP_THREADS"
  export SPH_OMP_THREADS="${SPH_OMP_THREADS:-$NPROCS}"
  if [[ -z "${BORG_MPI_LAUNCHER:-}" ]]; then
    BORG_MPI_LAUNCHER="$SRUN -u -n {nprocs} --mpi=pmix"
  fi
  export BORG_MPI_LAUNCHER

  if [[ "$NODES" -gt 1 ]]; then
    if ! has_steps_arg "${RUN_ARGS[@]}"; then
      echo "--nodes can only split schedule runs; pass --steps." >&2
      exit 2
    fi

    STEPS_SPEC="$(get_steps_arg "${RUN_ARGS[@]}")"
    mapfile -t STEP_CHUNKS < <(split_step_specs "$NODES" "$STEPS_SPEC")
    echo "Splitting steps '${STEPS_SPEC}' across ${#STEP_CHUNKS[@]} single-node job(s)."
    SUBMIT_STATUS=0
    CHUNK_INDEX=0
    for STEP_CHUNK in "${STEP_CHUNKS[@]}"; do
      replace_steps_arg "$STEP_CHUNK" "${RUN_ARGS[@]}"
      LOG_FILE="${LOG_DIR}/%j-${NAME}-chunk_${CHUNK_INDEX}.out"
      CMD=("$SCRIPT_DIR/run_borg_fields.sh" -locally "${CHUNK_ARGS[@]}")
      ADDQUEUE_FLAGS=(-q "$QUEUE" -m "$MEM_GB" -c "${NAME}-${CHUNK_INDEX}" -o "$LOG_FILE" -s -n "1x${NPROCS}")
      ADDQUEUE=(addqueue "${ADDQUEUE_FLAGS[@]}" "${CMD[@]}")

      echo "Chunk ${CHUNK_INDEX} steps: ${STEP_CHUNK}"
      echo "Log file: $LOG_FILE"
      echo "Command:"
      quote_cmd "${ADDQUEUE[@]}"

      if [[ "$SUBMIT_DRY_RUN" == "true" ]]; then
        echo "(dry: not submitting chunk ${CHUNK_INDEX})"
      else
        cd "$SCRIPT_DIR"
        if ! addqueue --sbatch "${ADDQUEUE_FLAGS[@]}" "${CMD[@]}"; then
          SUBMIT_STATUS=1
        fi
      fi
      CHUNK_INDEX=$((CHUNK_INDEX + 1))
    done
    exit "$SUBMIT_STATUS"
  fi

  LOG_FILE="${LOG_DIR}/%j-${NAME}.out"
  CMD=("$SCRIPT_DIR/run_borg_fields.sh" -locally "${RUN_ARGS[@]}")
  ADDQUEUE_FLAGS=(-q "$QUEUE" -m "$MEM_GB" -c "$NAME" -o "$LOG_FILE" -s -n "1x${NPROCS}")
  ADDQUEUE=(addqueue "${ADDQUEUE_FLAGS[@]}" "${CMD[@]}")

  echo "Queue: $QUEUE"
  echo "Single-node shape: 1x${NPROCS}"
  echo "Memory request (-m): ${MEM_GB} GB"
  echo "BORG OpenMP threads per rank: $OMP_THREADS"
  echo "BORG Python: $BORG_PYTHON"
  echo "BORG forward: $BORG_FORWARD"
  echo "BORG chain: $BORG_CHAIN_NAME"
  echo "BORG run directory: $BORG_RUN_DIR"
  echo "BORG schedule: $BORG_SCHEDULE"
  echo "BORG field output directory: $BORG_FIELD_OUTPUT_DIR"
  echo "Run Python: $RUN_PYTHON"
  echo "Plot Python: $PYTHON_PATH"
  echo "SPH OpenMP threads: $SPH_OMP_THREADS"
  echo "MPI launcher: $BORG_MPI_LAUNCHER"
  echo "Log file: $LOG_FILE"
  echo "Command:"
  quote_cmd "${ADDQUEUE[@]}"

  if [[ "$SUBMIT_DRY_RUN" == "true" ]]; then
    echo "(dry: not submitting)"
    exit 0
  fi

  cd "$SCRIPT_DIR"
  exec addqueue --sbatch "${ADDQUEUE_FLAGS[@]}" "${CMD[@]}"
fi

if [[ "$LOCAL_RUN" != "true" ]]; then
  exec "$SCRIPT_DIR/run_borg_fields.sh" --submit "$@"
fi

export NPROCS="${NPROCS:-8}"
export BORG_OMP_THREADS="${BORG_OMP_THREADS:-${OMP_NUM_THREADS:-1}}"
export OMP_NUM_THREADS="$BORG_OMP_THREADS"
export SPH_OMP_THREADS="${SPH_OMP_THREADS:-$NPROCS}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-^cm}"
export BORG_FORWARD
export PYTHON_PATH RUN_PYTHON

if run_schedule_steps "$@"; then
  exit 0
else
  step_status=$?
  if [[ "$step_status" -ne 42 ]]; then
    exit "$step_status"
  fi
fi

if [[ "${1:-}" != "run" && "${1:-}" != "validate-rsd" ]]; then
  set -- run "$@"
fi
normalize_run_args "$@"
set -- "${NORMALIZED_ARGS[@]}"

ulimit -l unlimited 2>/dev/null || true

EXTRA_ARGS=()
if ! has_option_arg "--borg-forward" "$@"; then
  EXTRA_ARGS+=(--borg-forward "$BORG_FORWARD")
fi
if ! has_option_arg "--nprocs" "$@"; then
  EXTRA_ARGS+=(--nprocs "$NPROCS")
fi
if ! has_option_arg "--omp-threads" "$@"; then
  EXTRA_ARGS+=(--omp-threads "$OMP_NUM_THREADS")
fi
if [[ "${1:-}" == "run" || "${1:-}" == "validate-rsd" ]]; then
  if ! has_option_arg "--plot-python" "$@"; then
    EXTRA_ARGS+=(--plot-python "$PYTHON_PATH")
  fi
fi
if [[ "${1:-}" == "run" ]]; then
  if ! has_option_arg "--field-output-dir" "$@"; then
    EXTRA_ARGS+=(--field-output-dir "$BORG_FIELD_OUTPUT_DIR")
  fi
fi

exec nice -n "${NICE:-10}" \
  "$RUN_PYTHON" -u "$SCRIPT_DIR/run_borg_fields.py" \
  "$@" \
  "${EXTRA_ARGS[@]}"
