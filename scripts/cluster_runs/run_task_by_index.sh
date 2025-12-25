#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <tasks_file> <index|count|max>"
  exit 1
fi

tasks_file="$1"
index="$2"

if [ "$index" = "count" ]; then
  wc -l < "$tasks_file" | tr -d ' '
  exit 0
fi

if [ "$index" = "max" ]; then
  count=$(wc -l < "$tasks_file" | tr -d ' ')
  if [ "$count" -eq 0 ]; then
    echo "-1"
  else
    echo $((count - 1))
  fi
  exit 0
fi

line=$(sed -n "$((index + 1))p" "$tasks_file")
if [ -z "$line" ]; then
  echo "No task found at index $index in $tasks_file"
  exit 2
fi

config_path=$(echo "$line" | cut -d' ' -f2-)
if [ -z "$config_path" ]; then
  echo "Could not parse config path from: $line"
  exit 3
fi

python scripts/runs/main.py --config "$config_path"
