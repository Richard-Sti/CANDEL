#!/bin/bash

# Exit on error
set -e

# --- Validate input ---
config_path="$1"
if [[ -z "$config_path" ]]; then
    echo "Usage: $0 path/to/config.toml"
    exit 1
fi

# --- Set Python interpreter ---
python_exec="/Users/rstiskalek/Projects/CANDEL/venv_candel/bin/python"

# --- Run the script ---
echo "Submitting run with config: $config_path"
echo "Python version: $python_exec"

$python_exec main.py --config "$config_path"
