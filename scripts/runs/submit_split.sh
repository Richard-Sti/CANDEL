#!/usr/bin/env bash
# Expand tasks_<BASE>X*.txt and submit each split with the unified submit.sh.
# Extra args after BASE_ID are forwarded to submit.sh (e.g. -q QUEUE, -n N).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 BASE_ID [-- submit.sh args...]"
    echo "Example: $0 0 -q cmb       # submit tasks_0X*.txt on queue 'cmb'"
    exit 1
fi

BASE="$1"; shift
DIR="."
if [[ "${1:-}" != "" && "${1:0:1}" != "-" ]]; then
    DIR="$1"; shift
fi

PATTERN="${DIR%/}/tasks_${BASE}X*.txt"
mapfile -t matches < <(compgen -G "$PATTERN" | sort -tX -k2,2n)

if (( ${#matches[@]} == 0 )); then
    echo "No matches for: $PATTERN"; exit 1
fi

echo "Found ${#matches[@]} subtasks (sorted):"
for f in "${matches[@]}"; do
    echo "  $(basename "$f")"
done

read -r -p "Submit these subtasks? [y/N] " reply
if [[ ! "$reply" =~ ^([yY]|[yY][eE][sS])$ ]]; then
    echo "Aborted."; exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in "${matches[@]}"; do
    base="$(basename "$f")"   # tasks_0X01.txt
    id="${base#tasks_}"
    id="${id%.txt}"           # 0X01
    echo "Submitting $id ..."
    bash "$script_dir/submit.sh" "$@" "$id"
done
echo "All matching subtasks submitted."
