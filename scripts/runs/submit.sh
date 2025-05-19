#!/bin/bash
#SBATCH -p gpu
#SBATCH --mail-user=rstiskalek@flatironinstitute.org
#SBATCH --mail-type=FAIL,END
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --time=00:30:00
#SBATCH --job-name=candel
#SBATCH --output=logs/logs-%j.out
#SBATCH --error=logs/logs-%j.err

# Report requested time
if [[ -n "$SLURM_TIMELIMIT" ]]; then
    hrs=$((SLURM_TIMELIMIT / 60))
    mins=$((SLURM_TIMELIMIT % 60))
    echo "[INFO] SLURM time limit requested: ${hrs}h ${mins}m"
fi

set -e

if [[ -z "$1" ]]; then
    echo "Usage: sbatch $0 <task_index> (e.g., 0 for tasks_0.txt)"
    exit 1
fi

task_index="$1"
task_file="tasks_${task_index}.txt"

if [[ ! -f "$task_file" ]]; then
    echo "[ERROR] Task file not found: $task_file"
    exit 2
fi

mapfile -t task_lines < "$task_file"

echo "[INFO] Preparing to run ${#task_lines[@]} tasks:"
for line in "${task_lines[@]}"; do
    idx=$(echo "$line" | cut -d' ' -f1)
    config_path=$(echo "$line" | cut -d' ' -f2-)
    echo "  - Task $idx: $config_path"
done
echo

# --- Run each task ---
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

    # Frozen package logic
    if [[ "$machine" == "rusty" ]]; then
        frozen_dir="/mnt/home/${USER}/frozen_candel/current"
    elif [[ "$machine" == "local" ]]; then
        frozen_dir="/Users/${USER}/Projects/CANDEL_frozen"
    else
        echo "[ERROR] Unknown machine: $machine"
        exit 3
    fi

    if [[ ! -d "$frozen_dir" ]]; then
        echo "[ERROR] Frozen package not found: $frozen_dir"
        echo "Run freeze_candel.sh first."
        exit 4
    fi

    export PYTHONPATH="$frozen_dir:$PYTHONPATH"

    echo "[INFO] Running main.py with config: $config_path"
    $python_exec main.py --config "$config_path"

    echo "[INFO] === Finished task $idx ==="
    echo
done
