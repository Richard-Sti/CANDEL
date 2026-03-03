#!/bin/bash
#SBATCH -p gpu
#SBATCH --mail-user=rstiskalek@flatironinstitute.org
#SBATCH --mail-type=FAIL,END
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --constraint=a100
#SBATCH --job-name=candel
#SBATCH --output=logs/logs-%A_%a.out
#SBATCH --error=logs/logs-%A_%a.err
#SBATCH --array=0-1%2

set -e

# Extract a TOML key from a config file, with fallback to local_config.toml
get_toml_key() {
    local key="$1"
    local config="$2"
    local val
    val=$(grep -E "^${key} *= *" "$config" 2>/dev/null | sed -E "s/^${key} *= *\"([^\"]+)\"$/\1/")
    if [[ -z "$val" ]]; then
        local local_config
        local_config="$(cd "$(dirname "$0")/../.." && pwd)/local_config.toml"
        val=$(grep -E "^${key} *= *" "$local_config" 2>/dev/null | sed -E "s/^${key} *= *\"([^\"]+)\"$/\1/")
    fi
    echo "$val"
}

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
machine=$(get_toml_key "machine" "$config_path")
if [[ -z "$machine" ]]; then
    echo "[ERROR] Could not determine machine from config: $config_path"
    exit 1
fi

# --- Choose frozen root based on machine ---
if [[ "$machine" == "rusty" ]]; then
    frozen_root="/mnt/home/${USER}/frozen_candel"
elif [[ "$machine" == "local" ]]; then
    frozen_root="/Users/${USER}/Projects/CANDEL_frozen"
else
    echo "[ERROR] Unknown machine: $machine"
    exit 2
fi

if [[ ! -f "$frozen_root/main.py" ]]; then
    echo "[ERROR] Frozen main.py not found in: $frozen_root"
    echo "Did you run freeze_candel.sh?"
    exit 3
fi

echo "[INFO] Using frozen package from: $frozen_root"
export PYTHONPATH="$frozen_root:$PYTHONPATH"

# --- Validate config path ---
if [[ -z "$config_path" ]]; then
    echo "[ERROR] No config path found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

# --- Extract python_exec from TOML config ---
python_exec=$(get_toml_key "python_exec" "$config_path")
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

    export XLA_FLAGS="--xla_hlo_profile=false --xla_dump_to=/tmp/nowhere"
fi

# --- Run ---
echo "[INFO] Submitting run with config: $config_path"
echo "[INFO] Using Python: $python_exec"
"$python_exec" "$frozen_root/main.py" --config "$config_path"
