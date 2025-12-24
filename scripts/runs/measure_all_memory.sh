#!/bin/bash
# Measure peak memory for all tasks and sort by usage
# Each config runs in a fresh process via /usr/bin/time

# Activate venv
source venv_candel/bin/activate

TASKS_FILE="${1:-tasks_0.txt}"
RESULTS_FILE="memory_usage.txt"

echo "Config,PeakMemoryMB" > "$RESULTS_FILE"

while read -r line; do
    # Extract config path (second field)
    config=$(echo "$line" | awk '{print $2}')
    [ -z "$config" ] && continue

    echo "Testing: $config"

    # Run and extract PEAK_MEMORY_MB from Python output
    output=$(python scripts/runs/measure_memory.py "$config" 2>&1)
    peak=$(echo "$output" | grep "PEAK_MEMORY_MB" | awk -F': ' '{print $2}')

    if [ -n "$peak" ]; then
        echo "$config,$peak" >> "$RESULTS_FILE"
        echo "  -> ${peak} MB"
    else
        echo "$config,FAILED" >> "$RESULTS_FILE"
        echo "  -> FAILED ($(echo "$output" | tail -1))"
    fi
done < "$TASKS_FILE"

echo ""
echo "=== Sorted by memory usage (top 20) ==="
tail -n +2 "$RESULTS_FILE" | sort -t',' -k2 -n -r | head -20
