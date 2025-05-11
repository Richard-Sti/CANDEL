#!/bin/bash

set -e

if [[ -z "$1" ]]; then
    echo "Usage: $0 <task_index> (e.g., 0 for tasks_0.txt)"
    exit 1
fi

task_index="$1"
task_file="tasks_${task_index}.txt"

if [[ ! -f "$task_file" ]]; then
    echo "[ERROR] Task file not found: $task_file"
    exit 2
fi

# --- Read tasks into array ---
mapfile -t task_lines < "$task_file"

# --- Preview tasks ---
echo "[INFO] Preparing to run the following tasks:"
task_count=${#task_lines[@]}
for line in "${task_lines[@]}"; do
    idx=$(echo "$line" | cut -d' ' -f1)
    config_path=$(echo "$line" | cut -d' ' -f2-)
    echo "  - Task $idx: $config_path"
done
echo "[INFO] Total tasks to run: $task_count"
echo

# --- Run tasks ---
for line in "${task_lines[@]}"; do
    idx=$(echo "$line" | cut -d' ' -f1)
    config_path=$(echo "$line" | cut -d' ' -f2-)

    echo "[INFO] === Starting task $idx ==="
    echo "[INFO] Config: $config_path"

    if [[ ! -f "$config_path" ]]; then
        echo "[WARNING] Config file not found: $config_path"
        continue
    fi

    python_exec=$(grep -E '^python_exec *= *' "$config_path" | sed -E 's/^python_exec *= *"([^"]+)"$/\1/')
    machine=$(grep -E '^machine *= *' "$config_path" | sed -E 's/^machine *= *"([^"]+)"$/\1/')

    if [[ -z "$python_exec" || -z "$machine" ]]; then
        echo "[ERROR] Missing python_exec or machine in: $config_path"
        continue
    fi

    if [[ "$machine" == "rusty" ]]; then
        echo "[INFO] Loading modules for machine: rusty"
        module --force purge
        module load modules/2.2-20230808
        module load gcc
        module load cuda
        module load python
    fi

    echo "[INFO] Running main.py with config: $config_path"
    $python_exec main.py --config "$config_path"
    echo "[INFO] === Finished task $idx ==="
    echo
done
