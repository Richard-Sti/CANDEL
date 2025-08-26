#!/bin/bash
### ------------------ SLURM settings: choose cluster ------------------

# ### Rusty
# #SBATCH -p gpu
# #SBATCH --mail-user=rstiskalek@flatironinstitute.org
# #SBATCH --mail-type=BEGIN,FAIL,END
# #SBATCH --ntasks=1
# #SBATCH --gpus-per-task=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=32G
# #SBATCH --constraint=a100
# #SBATCH --time=01:00:00
# #SBATCH --job-name=candel
# #SBATCH --output=logs/logs-%j.out
# #SBATCH --error=logs/logs-%j.err

## ARC HTC

#SBATCH --partition=short
#SBATCH --mail-user=richard.stiskalek@physics.ox.ac.uk
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 --constraint="cpu_gen:Cascade_Lake|cpu_gen:Skylake"
#SBATCH --time=02:00:00
#SBATCH --mem=32G
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
    echo "Usage: sbatch $0 <task_index> [ncpus]"
    exit 1
fi

task_index="$1"
ncpus="${2:-1}"

echo "[INFO] Task index: $task_index"
echo "[INFO] Using $ncpus CPU threads (may be overwritten by SBATCH)"

task_file="tasks_${task_index}.txt"

if [[ ! -f "$task_file" ]]; then
    echo "[ERROR] Task file not found: $task_file"
    exit 2
fi

task_lines=()
if command -v mapfile >/dev/null 2>&1 && mapfile -t < <(echo test) 2>/dev/null; then
    mapfile -t task_lines < "$task_file"
else
    # Fallback: use portable read loop (for macOS or Bash <4)
    while IFS= read -r line || [[ -n "$line" ]]; do
        task_lines+=("$line")
    done < "$task_file"
fi

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

        # Override ncpus to what SLURM gave us
        if [[ -n "$SLURM_CPUS_PER_TASK" ]]; then
            ncpus="$SLURM_CPUS_PER_TASK"
            echo "[INFO] Overriding ncpus to SLURM_CPUS_PER_TASK=$ncpus"
        fi

        export XLA_FLAGS="--xla_hlo_profile=false --xla_dump_to=/tmp/nowhere"
    elif [[ "$machine" == "arc" ]]; then
        echo "[INFO] Loading modules for machine: arc"
        module --force purge
        module add Python/3.11.3-GCCcore-12.3.0

        # Override ncpus to what SLURM gave us
        if [[ -n "$SLURM_CPUS_PER_TASK" ]]; then
            ncpus="$SLURM_CPUS_PER_TASK"
            echo "[INFO] Overriding ncpus to SLURM_CPUS_PER_TASK=$ncpus"
        fi
    fi

    # Override ncpus from SLURM if running with SBATCH
    if [[ "$machine" == "rusty" || "$machine" == "arc" ]]; then
        if [[ -n "$SLURM_CPUS_PER_TASK" ]]; then
            ncpus="$SLURM_CPUS_PER_TASK"
            echo "[INFO] Overriding ncpus to SLURM_CPUS_PER_TASK=$ncpus"
        fi
    fi

    # Set root of frozen package
    if [[ "$machine" == "rusty" ]]; then
        frozen_root="/mnt/home/${USER}/frozen_candel"
    elif [[ "$machine" == "arc" ]]; then
        frozen_root="/home/${USER}/frozen_candel"
    elif [[ "$machine" == "local" ]]; then
        frozen_root="/Users/${USER}/Projects/CANDEL_frozen"
    else
        echo "[ERROR] Unknown machine: $machine"
        exit 3
    fi

    if [[ ! -d "$frozen_root" ]]; then
        echo "[ERROR] Frozen package not found: $frozen_root"
        echo "Run freeze_candel.sh first."
        exit 4
    fi

    echo "[INFO] Running main.py with config: $config_path"
    export PYTHONPATH="$frozen_root:$PYTHONPATH"
    "$python_exec" "$frozen_root/main.py" --config "$config_path" --host-devices "$ncpus"

    echo "[INFO] === Finished task $idx ==="
    echo
done
