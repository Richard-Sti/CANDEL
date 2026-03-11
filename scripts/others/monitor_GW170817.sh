#!/bin/bash -l
# Monitor all running GW170817 PE jobs.
# Finds all progress.log files under results/GW170817_* and tails them.
# Usage: ./monitor_GW170817.sh

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

logs=()
for f in "$repo_root"/results/GW170817_*/progress.log; do
    [[ -f "$f" ]] || continue
    # Only show logs modified in the last 6 hours (i.e. active jobs)
    if find "$f" -mmin -360 -print -quit 2>/dev/null | grep -q .; then
        logs+=("$f")
    fi
done

if [[ ${#logs[@]} -eq 0 ]]; then
    echo "No active GW170817 progress logs found under $repo_root/results/"
    exit 1
fi

echo "Monitoring ${#logs[@]} active run(s)  (Ctrl-C to stop)"
echo "============================================================"
for f in "${logs[@]}"; do
    label=$(basename "$(dirname "$f")")
    echo "  $label  (last: $(tail -1 "$f" 2>/dev/null))"
done
echo "============================================================"

tail -f "${logs[@]}"
