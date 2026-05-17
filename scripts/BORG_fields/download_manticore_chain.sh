#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATHS_FILE="$SCRIPT_DIR/paths.env"
if [[ -f "$PATHS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$PATHS_FILE"
fi
: "${BORG_PYTHON:?Missing BORG_PYTHON in $PATHS_FILE}"
: "${SRUN:?Missing SRUN in $PATHS_FILE}"

usage() {
  cat <<'EOF'
Usage:
  download_manticore_chain.sh [download options]              # submits to berg
  download_manticore_chain.sh -locally [download options]     # runs here
  download_manticore_chain.sh --submit [submit options] [download options]

Default behaviour:
  Without -locally, this wrapper submits one berg job with one CPU.

Submit options:
  --queue QUEUE          Default: berg
  --mem-gb GB           Default: 7
  --name NAME           Default: manticore-download
  --log-dir DIR         Default: this directory
  --submit-dry-run      Print addqueue command without submitting

Download examples:
  download_manticore_chain.sh --steps 0
  download_manticore_chain.sh --steps 0:50
  download_manticore_chain.sh -locally --steps 0 --dry-run
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

if [[ $# -gt 0 && ( "${1:-}" == "-h" || "${1:-}" == "--help" ) ]]; then
  usage
  "$BORG_PYTHON" -u "$SCRIPT_DIR/download_manticore_chain.py" --help
  exit 0
fi

LOCAL_RUN="false"
if [[ "${1:-}" == "-locally" || "${1:-}" == "--locally" ]]; then
  LOCAL_RUN="true"
  shift
fi

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
