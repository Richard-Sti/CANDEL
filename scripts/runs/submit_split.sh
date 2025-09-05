#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-}"
DIR="${2:-.}"

if [[ -z "$BASE" ]]; then
  echo "Usage: $0 BASE_ID [DIR]"
  echo "Example: $0 0       # looks for tasks_0X*.txt in current dir"
  exit 1
fi

PATTERN="${DIR%/}/tasks_${BASE}X*.txt"

# Expand and sort by numeric suffix after BASEX
mapfile -t matches < <(compgen -G "$PATTERN" | sort -tX -k2,2n)

if (( ${#matches[@]} == 0 )); then
  echo "No matches for: $PATTERN"
  exit 1
fi

echo "Found ${#matches[@]} subtasks (sorted):"
for f in "${matches[@]}"; do
  echo "  $(basename "$f")"
done

read -r -p "Submit these subtasks with sbatch? [y/N] " reply
if [[ "$reply" =~ ^([yY]|[yY][eE][sS])$ ]]; then
  for f in "${matches[@]}"; do
    base="$(basename "$f")"           # e.g., tasks_0X01.txt
    id="${base#tasks_}"               # -> 0X01.txt
    id="${id%.txt}"                   # -> 0X01
    echo "Submitting $id ..."
    sbatch submit.sh "$id"
  done
  echo "All matching subtasks submitted."
else
  echo "Aborted."
fi
