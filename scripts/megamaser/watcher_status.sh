#!/bin/bash -l
# Show status of all running watcher processes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGDIR="$ROOT/scripts/megamaser/logs"

found=0
while IFS= read -r pid; do
    # Get the full command line
    cmdline=$(ps -p "$pid" -o args= 2>/dev/null) || continue
    # Only match our watcher
    [[ "$cmdline" == *watch_and_resubmit* ]] || continue

    # Find the log file for this PID
    logfile=$(grep -rl "PID $pid" "$LOGDIR"/watcher_*.log 2>/dev/null | head -1)
    if [[ -z "$logfile" ]]; then
        # Fallback: find by lsof
        logfile=$(ls -t "$LOGDIR"/watcher_*.log 2>/dev/null | while read -r f; do
            if lsof -p "$pid" 2>/dev/null | grep -q "$f"; then echo "$f"; break; fi
        done)
    fi

    # Get last line of the log for current status
    last=""
    if [[ -n "$logfile" && -f "$logfile" ]]; then
        last=$(grep -E '^\[watch\]' "$logfile" | tail -1)
    fi

    echo "PID $pid"
    [[ -n "$logfile" ]] && echo "  log: $logfile"
    [[ -n "$last" ]]    && echo "  status: $last"
    echo ""
    found=$((found + 1))
done < <(pgrep -u "$USER" -f watch_and_resubmit 2>/dev/null)

if [[ $found -eq 0 ]]; then
    echo "No active watchers found."
fi
