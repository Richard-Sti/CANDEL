#!/bin/bash

# Exit on error
set -e

# --- Validate input ---
config_path="$1"
if [[ -z "$config_path" ]]; then
    echo "Usage: $0 path/to/config.toml"
    exit 1
fi

# --- Extract python_exec from TOML config ---
python_exec=$(grep -E '^python_exec *= *' "$config_path" | sed -E 's/^python_exec *= *"([^"]+)"$/\1/')
if [[ -z "$python_exec" ]]; then
    echo "Error: 'python_exec' not found in config file: $config_path"
    exit 2
fi

# --- Extract machine from TOML config ---
machine=$(grep -E '^machine *= *' "$config_path" | sed -E 's/^machine *= *"([^"]+)"$/\1/')
if [[ -z "$machine" ]]; then
    echo "Error: 'machine' not found in config file: $config_path"
    exit 3
fi

# --- Optionally load modules based on machine ---
if [[ "$machine" == "rusty" ]]; then
    echo "[INFO] Loading modules for machine: rusty"
    module --force purge
    module load modules/2.3-20240529
    module load python/3.11.7
    module load cuda
    module load gsl/2.7.1
    module list
fi


# --- Run the script ---
echo "Submitting run with config: $config_path"
echo "Python version: $python_exec"

$python_exec main.py --config "$config_path"
