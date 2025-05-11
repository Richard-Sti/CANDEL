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
    frozen_dir="/mnt/home/${USER}/frozen_candel/current"
elif [[ "$machine" == "local" ]]; then
    src_dir="/Users/${USER}/Projects/CANDEL/candel"
    frozen_dir="/Users/${USER}/Projects/CANDEL_frozen"
else
    echo "[ERROR] Unknown machine: $machine"
    exit 3
fi

# --- Freeze package ---
echo "[INFO] Freezing package for machine: $machine"
echo "[INFO] From: $src_dir"
echo "[INFO] To:   $frozen_dir"

if [[ ! -d "$src_dir" ]]; then
    echo "[ERROR] Source directory not found: $src_dir"
    exit 4
fi

rm -rf "$frozen_dir"
mkdir -p "$frozen_dir"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$src_dir/" "$frozen_dir/"

echo "$frozen_dir"