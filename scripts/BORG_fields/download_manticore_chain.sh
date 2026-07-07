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
    --shell-env borg_python srun chain_name run_dir download_generation
)"
eval "$CONFIG_EXPORTS"

usage() {
  cat <<'EOF'
Usage:
  download_manticore_chain.sh [download options]                 # submits to berg
  download_manticore_chain.sh -locally [download options]        # runs here
  download_manticore_chain.sh --locally [download options]       # runs here
  download_manticore_chain.sh --submit [submit options] [download options]

Default behaviour:
  Without -locally/--locally, this wrapper submits one berg job with one CPU.

Submit options:
  --queue QUEUE          Default: berg
  --mem-gb GB           Default: 7
  --name NAME           Default: manticore-download
  --log-dir DIR         Default: this directory
  --submit-dry-run      Print addqueue command without submitting

Download examples:
  download_manticore_chain.sh --steps 0
  download_manticore_chain.sh --steps 0:50
  download_manticore_chain.sh --locally --steps 0 --dry-run
  download_manticore_chain.sh --locally --steps 50-79 --access-json public-keys.json
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

if has_help_arg "$@"; then
  usage
  "$BORG_PYTHON" -u "$SCRIPT_DIR/download_manticore_chain.py" --help
  exit 0
fi

LOCAL_RUN="false"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -locally|--locally)
      LOCAL_RUN="true"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${ARGS[@]}"

if [[ "${1:-}" == "--submit" ]]; then
  shift
  QUEUE="berg"
  MEM_GB="7"
  NAME="manticore-download"
  LOG_DIR="$SCRIPT_DIR"
  SUBMIT_DRY_RUN="false"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --queue) QUEUE="$2"; shift 2 ;;
      --mem-gb) MEM_GB="$2"; shift 2 ;;
      --name) NAME="$2"; shift 2 ;;
      --log-dir) LOG_DIR="$2"; shift 2 ;;
      --submit-dry-run) SUBMIT_DRY_RUN="true"; shift ;;
      --) shift; break ;;
      *) break ;;
    esac
  done

  mkdir -p "$LOG_DIR"
  LOG_DIR="$(cd "$LOG_DIR" && pwd)"
  export BORG_PYTHON

  LOG_FILE="${LOG_DIR}/%j-${NAME}.out"
  CMD=("$SCRIPT_DIR/download_manticore_chain.sh" -locally "$@")
  ADDQUEUE_FLAGS=(-q "$QUEUE" -m "$MEM_GB" -c "$NAME" -o "$LOG_FILE" -s -n "1x1")
  ADDQUEUE=(addqueue "${ADDQUEUE_FLAGS[@]}" "${CMD[@]}")

  echo "Queue: $QUEUE"
  echo "Single-node shape: 1x1"
  echo "Memory request (-m): ${MEM_GB} GB"
  echo "BORG Python: $BORG_PYTHON"
  echo "BORG chain: $BORG_CHAIN_NAME"
  echo "BORG run directory: $BORG_RUN_DIR"
  echo "BORG download generation: $BORG_DOWNLOAD_GENERATION"
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
  exec "$SCRIPT_DIR/download_manticore_chain.sh" --submit "$@"
fi

exec "$BORG_PYTHON" -u "$SCRIPT_DIR/download_manticore_chain.py" "$@"
