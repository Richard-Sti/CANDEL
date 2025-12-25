#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <tasks_file> <queue> <gputype> [gpus] [mem_gb]"
  exit 1
fi

tasks_file="$1"
queue="$2"
gputype="$3"
gpus="${4:-1}"
mem_gb="${5:-32}"

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

addqueue -q "$queue" -s --gpus "$gpus" --gputype "$gputype" -m "$mem_gb" \
  --range 0,"$max" \
  ./scripts/cluster_runs/run_task_by_index.sh "$tasks_file" ARG_REPLACE
