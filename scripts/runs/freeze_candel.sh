#!/bin/bash
set -e

# --- Parse machine from config.toml ---
config_path="./config.toml"
if [[ ! -f "$config_path" ]]; then
    echo "[ERROR] config.toml not found at $config_path"
    exit 1
fi

machine=$(grep -E '^machine *= *' "$config_path" | sed -E 's/^machine *= *"([^"]+)"/\1/')
if [[ -z "$machine" ]]; then
    echo "[ERROR] Could not determine machine from config.toml"
    exit 2
fi

# --- Choose paths based on machine ---
if [[ "$machine" == "rusty" ]]; then
    src_dir="/mnt/home/${USER}/CANDEL/candel"
    main_script="/mnt/home/${USER}/CANDEL/scripts/runs/main.py"
    frozen_root="/mnt/home/${USER}/frozen_candel"
elif [[ "$machine" == "local" ]]; then
    src_dir="/Users/${USER}/Projects/CANDEL/candel"
    main_script="/Users/${USER}/Projects/CANDEL/scripts/runs/main.py"
    frozen_root="/Users/${USER}/Projects/CANDEL_frozen"
elif [[ "$machine" == "arc" ]]; then
    ARC_USER="${USER:-phys1997}"
    src_dir="/home/${ARC_USER}/CANDEL/candel"
    main_script="/home/${ARC_USER}/CANDEL/scripts/runs/main.py"
    frozen_root="/home/${ARC_USER}/frozen_candel"
else
    echo "[ERROR] Unknown machine: $machine"
    exit 3
fi

frozen_dir="${frozen_root}"

# Freeze
echo "[INFO] Freezing candel package + main.py"
echo "[INFO] From: $src_dir"
echo "[INFO] To:   $frozen_dir"
rm -rf "$frozen_dir"
mkdir -p "$frozen_root"

# Copy the full candel/ package
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$src_dir" "$frozen_dir"

# Copy or symlink main.py into frozen dir root
cp "$main_script" "$frozen_dir/main.py"

echo "[INFO] Frozen structure:"
if command -v tree >/dev/null 2>&1; then
    tree -L 2 "$frozen_dir"
else
    echo "[INFO] (Skipping tree output: 'tree' not found)"
    ls -l "$frozen_dir"
fi
