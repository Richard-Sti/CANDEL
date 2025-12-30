#!/bin/bash
set -euo pipefail

skip_if_exists=true
positional=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-if-exists)
      skip_if_exists=true
      shift
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

set -- "${positional[@]}"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <tasks_file> <queue> [gputype] [gpus] [mem_gb] [--skip-if-exists]"
  exit 1
fi

tasks_file="$1"
queue="$2"
gputype="${3:-}"
gpus="${4:-1}"
mem_gb="${5:-15}"

if [ ! -f "$tasks_file" ]; then
  echo "Tasks file not found: $tasks_file"
  exit 2
fi

count=$(wc -l < "$tasks_file" | tr -d ' ')
if [ "$count" -le 0 ]; then
  echo "No tasks found in $tasks_file"
  exit 3
fi

max=$((count - 1))

addqueue_args=(-q "$queue" -s --gpus "$gpus" -m "$mem_gb")
if [ -n "$gputype" ]; then
  addqueue_args+=(--gputype "$gputype")
fi

extra_run_args=()
if [ "$skip_if_exists" = true ]; then
  extra_run_args+=(--skip-if-exists)
fi

addqueue "${addqueue_args[@]}" \
  --range 0,"$max" \
  ./scripts/cluster_runs/run_task_by_index.sh "$tasks_file" ARG_REPLACE \
  "${extra_run_args[@]}"
