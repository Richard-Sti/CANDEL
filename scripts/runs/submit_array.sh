#!/bin/bash
#SBATCH -p gpu
#SBATCH --mail-user=rstiskalek@flatironinstitute.org
#SBATCH --mail-type=FAIL,END
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=a100
#SBATCH --time=01:00:00
#SBATCH --job-name=candel
#SBATCH --output=logs/logs-%A_%a.out
#SBATCH --error=logs/logs-%A_%a.err
#SBATCH --array=0-3%4

set -e

# --- Read argument for task index ---
if [[ -z "$1" ]]; then
    echo "Usage: sbatch $0 <task_index> (e.g., 0 for tasks_0.txt)"
    exit 1
fi


task_index="$1"
task_file="tasks_${task_index}.txt"

echo "[DEBUG] Reading from $task_file for task ID $SLURM_ARRAY_TASK_ID"

# Safely extract line and config path
task_line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$task_file")
config_path=$(echo "$task_line" | cut -d' ' -f2)

# --- Determine machine from config file ---
machine=$(grep -E '^machine *= *' "$config_path" | sed -E 's/^machine *= *"([^"]+)"/\1/')
if [[ -z "$machine" ]]; then
    echo "[ERROR] Could not determine machine from config: $config_path"
    exit 1
fi

# --- Choose frozen path based on machine ---
if [[ "$machine" == "rusty" ]]; then
    frozen_dir="/mnt/home/${USER}/frozen_candel/current"
elif [[ "$machine" == "local" ]]; then
    frozen_dir="/Users/${USER}/Projects/CANDEL_frozen"
else
    echo "[ERROR] Unknown machine: $machine"
    exit 2
fi

mkdir -p "logs"

# --- Use frozen package ---
if [[ ! -d "$frozen_dir" ]]; then
    echo "[ERROR] Frozen package not found: $frozen_dir"
    echo "Run freeze_candel.sh first."
    exit 3
fi

echo "[INFO] Using frozen package from: $frozen_dir"
export PYTHONPATH="$frozen_dir:$PYTHONPATH"

# --- Validate config path ---
if [[ -z "$config_path" ]]; then
    echo "[ERROR] No config path found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# --- Extract python_exec from TOML config ---
python_exec=$(grep -E '^python_exec *= *' "$config_path" | sed -E 's/^python_exec *= *"([^"]+)"$/\1/')
if [[ -z "$python_exec" ]]; then
    echo "[ERROR] 'python_exec' not found in config file: $config_path"
    exit 2
fi

# --- Optionally load modules based on machine ---
if [[ "$machine" == "rusty" ]]; then
    echo "[INFO] Loading modules for machine: rusty"
    module --force purge
    module load modules/2.2-20230808
    module load gcc
    module load cuda
    module load python
    module list
fi

# --- Run ---
echo "[INFO] Submitting run with config: $config_path"
echo "[INFO] Using Python: $python_exec"
$python_exec main.py --config "$config_path"

# --- Clean up frozen package ---
echo "[INFO] Cleaning up frozen package at: $frozen_dir"
[[ "$frozen_dir" == /mnt/home/$USER/frozen_candel/* ]] && rm -rf "$frozen_dir"